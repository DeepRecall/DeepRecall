import os
import logging
from dotenv import load_dotenv
from deeprecall.services.celery_app import celery_app
from tika import parser
from deeprecall.services.providers.vectorstore import get_vectorstore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents.base import Document
from langchain_community.embeddings import *
from deeprecall.services.providers.embeding import create_embedding

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
def create_rag(
    self,
    folder_path: str,
    collection_name: str,
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
        collection_name: Collection name for Milvus
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
        if not os.path.exists(folder_path):
            error_msg = f"Folder path {folder_path} does not exist"
            logging.error(error_msg)
            self.update_state(state="FAILURE", meta={"error": error_msg})
            return False

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
            collection_name=collection_name,
            embedding_function=embed_engine,
            init_collection=True,
        )

        # Process files
        for root, dirs, files in os.walk(folder_path):
            # Skip hidden directories
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            # Skip hidden files
            files = [f for f in files if not f.startswith(".")]
            for file_name in files:
                file_path = os.path.join(root, file_name)

                try:
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

    if len(sys.argv) != 5:
        print(
            "Usage: python task.py <folder_path> <vectorstore_provider> <collection_name> <embedding_provider>"
        )
        sys.exit(1)

    EMBEDDING_ID = os.getenv("EMBEDDING_ID")
    EMBEDDING_KEY = os.getenv("EMBEDDING_KEY")
    EMBEDDING_URL = os.getenv("EMBEDDING_URL")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
    EMBEDDING_CONTEXT_SIZE = int(os.getenv("EMBEDDING_CONTEXT_SIZE", "2048"))

    folder_path = sys.argv[1]
    vectorstore_provider = sys.argv[2]
    collection_name = sys.argv[3]
    embedding_provider = sys.argv[4]

    create_rag(
        folder_path=folder_path,
        collection_name=collection_name,
        embedding_provider=embedding_provider,
        vectorstore_provider=vectorstore_provider,
        embedding_key=EMBEDDING_KEY,
        embedding_model=EMBEDDING_MODEL,
        embedding_url=EMBEDDING_URL,
        embedding_provider_id=EMBEDDING_ID,
        context_size=EMBEDDING_CONTEXT_SIZE,
    )
