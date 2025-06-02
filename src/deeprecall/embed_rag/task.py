import os
import logging
from dotenv import load_dotenv
import pathlib
import shutil
import uuid
import time
import random

from deeprecall.services.celery_app import celery_app
from tika import parser
from deeprecall.services.providers.vectorstore import get_vectorstore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents.base import Document
from langchain_community.embeddings import *
from deeprecall.services.providers.embeding import create_embedding

nil_namespace = uuid.UUID(int=0)

# Configure logging
logging.basicConfig(
    filename="process_data.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

load_dotenv()

# Configuration from environment variables
TIKA_SERVER_URL = os.getenv("TIKA_SERVER_URL")


@celery_app.task(bind=True)
def preprocess(
    self,
    job_id: str,
    input_folder_path: str,
    output_folder_path: str,
    embedding_provider: str,
    vectorstore_provider: str,
    embedding_model: str = None,
    embedding_key: str = None,
    embedding_url: str = None,
    embedding_provider_id: str = None,
    context_size: int = 8192,
):
    """
    Process files in a folder and store embeddings in Milvus.

    Args:
        self: Celery task instance (for task state updates)
        folder_path: Path to directory containing files to process
        embedding_provider: Name of the embedding provider to use
        embedding_model (str, optional): Specific model to use with the provider
        embedding_key (str, optional): Authentication key for the provider
        embedding_url (str, optional): Custom endpoint URL for the provider
        embedding_provider_id (str, optional): Provider-specific identifier
        context_size (int, optional): Context size for embedding generation. Defaults to 8192.

    Returns:
        bool: True if processing completed successfully, False otherwise
    """
    try:
        # Validate input
        if not os.path.exists(input_folder_path):
            error_msg = f"Folder path {input_folder_path} does not exist"
            logging.error(error_msg)
            self.update_state(state="FAILURE", meta={"error": error_msg})
            return False

        # Create output directory if it doesn't exist
        outpath = pathlib.Path(output_folder_path)
        doc_chunk_path = outpath / job_id / "doc_chunks"

        # Ensure the path exists
        if doc_chunk_path.exists() and doc_chunk_path.is_dir():
            # Remove the directory and all its contents
            shutil.rmtree(doc_chunk_path)

        doc_chunk_path.mkdir(parents=True, exist_ok=True)

        # Dynamically create an instance of the specified embedding provider
        try:
            embed_engine = create_embedding(
                provider_name=embedding_provider,
                model=embedding_model,
                key=embedding_key,
                url=embedding_url,
                provider_id=embedding_provider_id,
            )
        except ValueError as e:
            error_msg = str(e)
            logging.error(error_msg)
            self.update_state(state="FAILURE", meta={"error": error_msg})
            return False

        # Create vector store with dynamic provider and collection name
        vector_store = get_vectorstore(
            provider_name=vectorstore_provider,
            collection_name=job_id,
            embedding_function=embed_engine,
            init_collection=True,
        )

        # Helper function to retry document addition
        def retry_add_document(doc, max_retries=3):
            """Attempt to add document with exponential backoff and jitter"""
            base_delay = 1  # seconds
            for attempt in range(max_retries):
                try:
                    vector_store.add_documents([doc])
                    return True
                except Exception as e:
                    # Calculate jittered backoff (exponential with random factor)
                    jitter = 0.5 + random.random()  # Random between 0.5-1.5
                    delay = base_delay * (2**attempt) * jitter

                    logging.warning(
                        f"Attempt {attempt+1}/{max_retries} failed for chunk {doc.id}: {str(e)}. "
                        f"Retrying in {delay:.2f} seconds."
                    )
                    time.sleep(delay)

            # All retries failed
            logging.error(
                f"Failed to add document after {max_retries} retries: {doc.metadata}"
            )
            return False

        # Process files
        for root, dirs, files in os.walk(input_folder_path):
            # Skip hidden directories
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            # Skip hidden files
            files = [f for f in files if not f.startswith(".")]
            for file_name in files:
                file_path = os.path.join(root, file_name)

                # Parse file with Tika
                if not TIKA_SERVER_URL:
                    error_msg = "TIKA_SERVER_URL not configured"
                    logging.error(error_msg)
                    self.update_state(state="FAILURE", meta={"error": error_msg})
                    return False

                parsed = parser.from_file(
                    file_path, serverEndpoint=TIKA_SERVER_URL, service="all"
                )
                text = parsed.get("content", "").strip()

                if not text:
                    logging.warning(f"No content extracted from {file_path}")
                    continue

                # Split text into chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=context_size, chunk_overlap=200
                )
                chunks = text_splitter.split_text(text)

                # Save chunks to output folder and prepare documents
                for i, chunk in enumerate(chunks):
                    chunk_filename = f"{file_name}_chunk_{i}.txt"
                    chunk_path = doc_chunk_path / chunk_filename
                    with open(chunk_path, "w") as chunk_file:
                        chunk_file.write(chunk)

                    chunk_uuid = str(uuid.uuid5(nil_namespace, chunk_path.as_posix()))

                    doc = Document(
                        id=chunk_uuid,
                        page_content=chunk,
                        metadata={
                            "source": file_path,
                            "file_name": file_name,
                            "chunk_index": i,
                        },
                    )

                # Add document one by one with retry mechanism - will raise exception on failure
                retry_add_document(doc)

    except Exception as e:
        # Log and propagate critical errors to Celery
        logging.error(f"Critical error in process_data: {str(e)}", exc_info=True)
        self.update_state(state="FAILURE", meta={"error": str(e)})
        return False


# Example usage
if __name__ == "__main__":
    import sys

    if len(sys.argv) != 6:
        print(
            "Usage: python task.py <job_id> <input_folder_path> <output_folder_path> <vectorstore_provider> <embedding_provider>"
        )
        sys.exit(1)

    EMBEDDING_ID = os.getenv("EMBEDDING_ID")
    EMBEDDING_KEY = os.getenv("EMBEDDING_KEY")
    EMBEDDING_URL = os.getenv("EMBEDDING_URL")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
    EMBEDDING_CONTEXT_SIZE = int(os.getenv("EMBEDDING_CONTEXT_SIZE", "2048"))

    job_id = sys.argv[1]
    input_folder_path = sys.argv[2]
    output_folder_path = sys.argv[3]
    vectorstore_provider = sys.argv[4]
    embedding_provider = sys.argv[5]

    preprocess(
        job_id=job_id,
        input_folder_path=input_folder_path,
        output_folder_path=output_folder_path,
        embedding_provider=embedding_provider,
        vectorstore_provider=vectorstore_provider,
        embedding_key=EMBEDDING_KEY,
        embedding_model=EMBEDDING_MODEL,
        embedding_url=EMBEDDING_URL,
        embedding_provider_id=EMBEDDING_ID,
        context_size=EMBEDDING_CONTEXT_SIZE,
    )
