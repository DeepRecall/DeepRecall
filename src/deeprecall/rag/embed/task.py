from deeprecall.services.celery_app import celery_app
import os
from tika import parser
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Milvus
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Added import
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    filename='process_data.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

load_dotenv()

# Configuration from environment variables
TIKA_SERVER_URL = os.getenv('TIKA_SERVER_URL')
MILVUS_HOST = os.getenv('MILVUS_HOST')
MILVUS_PORT = os.getenv('MILVUS_PORT')
EMBEDDING_API_KEY = os.getenv('EMBEDDING_API_KEY')
EMBEDDING_API_URL = os.getenv('EMBEDDING_API_URL')  # Now used via base_url
EMBEDDING_MODEL_NAME = os.getenv('EMBEDDING_MODEL_NAME')  # New model name configuration
# Default collection name as fallback
DEFAULT_COLLECTION_NAME = os.getenv('DEFAULT_COLLECTION_NAME', 'tika_docs')
# Embedding configuration with context size
EMBEDDING_CONTEXT_SIZE = int(os.getenv('EMBEDDING_CONTEXT_SIZE', '2048'))
DIMENSIONS = int(os.getenv('EMBEDDING_DIMENSIONS', '1536'))

@celery_app.task(bind=True)
def process_data(self, folder_path, collection_name=None):
    """
    Process files in a folder and store embeddings in Milvus.
    
    Args:
        self: Celery task instance (for task state updates)
        folder_path: Path to directory containing files to process
        collection_name: Optional custom collection name for Milvus
    
    Returns:
        bool: True if processing completed successfully
    """
    try:
        # Validate input
        if not os.path.exists(folder_path):
            error_msg = f"Folder path {folder_path} does not exist"
            logging.error(error_msg)
            self.update_state(state='FAILURE', meta={'error': error_msg})
            return False

        # Use provided collection name or fallback to default
        collection_name = collection_name or DEFAULT_COLLECTION_NAME
        if not collection_name:
            error_msg = "No collection name provided and no default configured"
            logging.error(error_msg)
            self.update_state(state='FAILURE', meta={'error': error_msg})
            return False

        # Initialize embeddings with configurable context size
        embeddings = OpenAIEmbeddings(
            openai_api_key=EMBEDDING_API_KEY,
            openai_api_base=EMBEDDING_API_URL,
            model=EMBEDDING_MODEL_NAME,
        )

        # Create Milvus vector store with dynamic collection name
        vector_store = Milvus(
            embedding_function=embeddings,
            collection_name=collection_name,
            connection_args={"host": MILVUS_HOST, "port": MILVUS_PORT},
            dimension=DIMENSIONS,
            # Don't fail if collection exists
            drop_old=False
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
                        self.update_state(state='FAILURE', meta={'error': error_msg})
                        return False
                        
                    parsed = parser.from_file(file_path, server_endpoint=TIKA_SERVER_URL)
                    text = parsed.get('content', '')
                    
                    # Split text into chunks with overlap to maintain context
                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=EMBEDDING_CONTEXT_SIZE,  # Use configured context size
                        chunk_overlap=200  # Maintain 200 token overlap between chunks
                    )
                    chunks = splitter.split_text(text)
                    
                    # Generate and store embeddings for each chunk
                    try:
                        embeddings_list = embeddings.embed_documents(chunks)
                    except Exception as e:
                        logging.error(f"Error generating embeddings for {file_path}: {str(e)}")
                        continue
                        
                    if embeddings_list:
                        vector_store.add_embeddings(
                            text_embeddings=zip(chunks, embeddings_list),
                            metadatas=[{"source": file_path}] * len(chunks)
                        )
                    
                except Exception as e:
                    logging.error(f"Error processing {file_path}: {str(e)}")
                    continue

        return True

    except Exception as e:
        # Log and propagate critical errors to Celery
        logging.error(f"Critical error in process_data: {str(e)}", exc_info=True)
        self.update_state(state='FAILURE', meta={'error': str(e)})
        return False

# Example usage
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python task.py <folder_path> [collection_name]")
        sys.exit(1)
        
    folder_path = sys.argv[1]
    collection_name = sys.argv[2] if len(sys.argv) > 2 else None
    process_data(folder_path, collection_name)
