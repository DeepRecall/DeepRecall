import os
import logging
from typing import List, Dict, Any
from dotenv import load_dotenv
from deeprecall.services.celery_app import celery_app
from deeprecall.services.providers.embeding import create_embedding
from deeprecall.services.providers.vectorstore import get_vectorstore

# Configure logging
logging.basicConfig(
    filename="extract.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

load_dotenv()

# Configuration from environment variables
MILVUS_URL = os.getenv("MILVUS_URL")
EMBEDDING_ID = os.getenv("EMBEDDING_ID")
EMBEDDING_KEY = os.getenv("EMBEDDING_KEY")
EMBEDDING_URL = os.getenv("EMBEDDING_URL")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
EMBEDDING_CONTEXT_SIZE = int(os.getenv("EMBEDDING_CONTEXT_SIZE", "2048"))


@celery_app.task(bind=True)
def extract_relevant_documents(
    self,
    query: str,
    collection_name: str,
    embedding_provider: str,
    vectorstore_provider: str,
    k: int = 4,
) -> Dict[str, Any]:
    """
    Extract relevant documents from Milvus vector store for LLM context.

    Args:
        self: Celery task instance
        query: Search query text
        collection_name: Name of the Milvus collection to search
        embedding_provider: Name of the embedding provider to use
        k: Number of documents to retrieve (default: 4)

    Returns:
        Dict containing search results and metadata
    """
    try:
        # Validate required parameters
        if not all([query, collection_name, embedding_provider]):
            error_msg = "Missing required parameters: query, collection_name, and embedding_provider"
            logging.error(error_msg)
            self.update_state(state="FAILURE", meta={"error": error_msg})
            return {"success": False, "error": error_msg}

        # Get embedding provider
        embedding_provider = (
            embedding_provider.lower().replace(" ", "_").replace("-", "_")
        )
        # Create embedding engine using the same pattern as embed task
        embed_engine = create_embedding(
            provider_name=embedding_provider,
            model=EMBEDDING_MODEL,
            key=EMBEDDING_KEY,
            url=EMBEDDING_URL,
            provider_id=EMBEDDING_ID,
        )

        if embed_engine is None:
            error_msg = f"Unsupported embedding provider: {embedding_provider}"
            logging.error(error_msg)
            self.update_state(state="FAILURE", meta={"error": error_msg})
            return {"success": False, "error": error_msg}

        embed_engine = embed_engine()

        # Use vectorstore provider factory pattern like embed task
        vector_store = get_vectorstore(
            provider_name=vectorstore_provider,
            collection_name=collection_name,
            embedding_function=embed_engine,
            init_collection=False,
        )

        # Perform similarity search
        results = vector_store.similarity_search(query, k=k)

        # Format results for LLM context
        formatted_results = [
            {
                "text": doc.page_content,
                "metadata": doc.metadata,
                "source": doc.metadata.get("source", "unknown"),
            }
            for doc in results
        ]

        return {
            "success": True,
            "query": query,
            "collection": collection_name,
            "results": formatted_results,
            "count": len(formatted_results),
        }

    except Exception as e:
        error_msg = f"Critical error in extract_relevant_documents: {str(e)}"
        logging.error(error_msg, exc_info=True)
        self.update_state(state="FAILURE", meta={"error": str(e)})
        return {"success": False, "error": error_msg}
