import os
import logging
from dotenv import load_dotenv

from deeprecall.services.celery_app import celery_app

from tika import parser
from langchain_community.embeddings import DeepInfraEmbeddings
from langchain_milvus import Milvus
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents.base import Document

# Configure logging
logging.basicConfig(
    filename="process_data.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

load_dotenv()

# Configuration from environment variables
TIKA_SERVER_URL = os.getenv("TIKA_SERVER_URL")
MILVUS_URL = os.getenv("MILVUS_URL")
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY")
EMBEDDING_API_URL = os.getenv("EMBEDDING_API_URL")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")
# Embedding configuration with context size
EMBEDDING_CONTEXT_SIZE = int(os.getenv("EMBEDDING_CONTEXT_SIZE", "2048"))


@celery_app.task(bind=True)
def process_data(self, folder_path, collection_name):
    """
    Process files in a folder and store embeddings in Milvus.

    Args:
        self: Celery task instance (for task state updates)
        folder_path: Path to directory containing files to process
        collection_name: Collection name for Milvus

    Returns:
        bool: True if processing completed successfully
    """
    try:
        # Validate input
        if not os.path.exists(folder_path):
            error_msg = f"Folder path {folder_path} does not exist"
            logging.error(error_msg)
            self.update_state(state="FAILURE", meta={"error": error_msg})
            return False

        embeddings = DeepInfraEmbeddings(
            model_id=EMBEDDING_MODEL_NAME,
            deepinfra_api_token=EMBEDDING_API_KEY,
        )

        # Create Milvus vector store with dynamic collection name
        vector_store = Milvus(
            embedding_function=embeddings,
            collection_name=collection_name,
            connection_args={
                "uri": MILVUS_URL,
            },
            drop_old=False,
            auto_id=True,
        )

        # Process files
        for root, dirs, files in os.walk(folder_path):
            for file_name in files:
                file_path = os.path.join(root, file_name)

                try:
                    # Parse file with Tika
                    if not TIKA_SERVER_URL:
                        error_msg = "TIKA_SERVER_URL not configured"
                        logging.error(error_msg)
                        self.update_state(state="FAILURE", meta={"error": error_msg})
                        return False

                    parsed = parser.from_file(file_path, serverEndpoint=TIKA_SERVER_URL)
                    text = parsed.get("content", "").strip()

                    if not text:
                        logging.warning(f"No content extracted from {file_path}")
                        continue

                    # Split text into chunks
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=EMBEDDING_CONTEXT_SIZE, chunk_overlap=200
                    )
                    chunks = text_splitter.split_text(text)

                    documents = [
                        Document(
                            page_content=chunk,
                            metadata={
                                "source": file_path,
                                "file_name": file_name,
                                "chunk_index": i,
                            },
                        )
                        for i, chunk in enumerate(chunks)
                    ]

                    # Add to vector store
                    vector_store.add_documents(documents)

                except Exception as e:
                    logging.error(f"Error processing {file_path}: {str(e)}")
                    continue

        return True

    except Exception as e:
        # Log and propagate critical errors to Celery
        logging.error(f"Critical error in process_data: {str(e)}", exc_info=True)
        self.update_state(state="FAILURE", meta={"error": str(e)})
        return False


# Example usage
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python task.py <folder_path> <collection_name>")
        sys.exit(1)

    folder_path = sys.argv[1]
    collection_name = sys.argv[2] if len(sys.argv) > 2 else None
    process_data(folder_path, collection_name)
