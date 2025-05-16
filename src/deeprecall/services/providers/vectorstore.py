from langchain_core.vectorstores import VectorStore
from langchain_milvus import Milvus
from langchain_chroma import Chroma
import chromadb
from qdrant_client import QdrantClient
from langchain_elasticsearch import ElasticsearchStore
from elasticsearch import Elasticsearch
from langchain_qdrant import QdrantVectorStore

# from langchain_weaviate import WeaviateVectorStore
# from langchain_postgres import PGVector
from langchain_core.embeddings import Embeddings
import os


def get_vectorstore(
    provider_name: str,
    collection_name: str,
    embedding_function: Embeddings,
    init_collection: bool,
    **kwargs,
) -> VectorStore:
    """
    Create a vector store instance based on the provider name and configuration.

    Args:
        provider_name: Name of the vector store provider
        collection_name: Name of the collection/index
        embedding_function: Embedding function to use
        distance_strategy: Distance metric to use for similarity search
        **kwargs: Provider-specific connection parameters

    Returns:
        Vector store instance
    """
    provider = provider_name.lower().replace(" ", "_").replace("-", "_")

    # Common environment variables
    MILVUS_URL = os.getenv("MILVUS_URL")
    CHROMA_HOST = os.getenv("CHROMA_HOST")
    CHROMA_PORT = os.getenv("CHROMA_PORT")
    ELASTICSEARCH_URL = os.getenv("ELASTICSEARCH_URL")
    ELASTICSEARCH_USR = os.getenv("ELASTICSEARCH_USR")
    ELASTICSEARCH_PASS = os.getenv("ELASTICSEARCH_PASS")
    QDRANT_URL = os.getenv("QDRANT_URL")
    WEAVIATE_HOST = os.getenv("WEAVIATE_HOST")
    WEAVIATE_HTTP_PORT = os.getenv("WEAVIATE_HTTP_PORT")
    WEAVIATE_GRPC_PORT = os.getenv("WEAVIATE_GRPC_PORT")
    PGVECTOR_URL = os.getenv("PGVECTOR_URL")
    try:
        if provider == "milvus":
            return Milvus(
                embedding_function=embedding_function,
                collection_name=collection_name,
                connection_args={"uri": MILVUS_URL},
                drop_old=init_collection,
                auto_id=True,
            )
        elif provider == "chroma":

            client = chromadb.HttpClient(
                host=CHROMA_HOST,
                port=CHROMA_PORT,
                database=collection_name,
            )

            if init_collection:
                try:
                    # Delete the collection
                    client.delete_collection(collection_name)
                except:
                    pass

                client.create_collection(name=collection_name)

            return Chroma(
                client=client,
                collection_name=collection_name,
                embedding_function=embedding_function,
            )

        elif provider == "elasticsearch":
            # Create client
            client = Elasticsearch(
                hosts=[ELASTICSEARCH_URL],
                basic_auth=(ELASTICSEARCH_USR, ELASTICSEARCH_PASS),
            )

            if init_collection:
                # Check and delete index
                if client.indices.exists(index=collection_name):
                    client.indices.delete(index=collection_name)

            return ElasticsearchStore(
                index_name=collection_name,
                embedding=embedding_function,
                es_connection=client,
            )

        elif provider == "qdrant":

            client = QdrantClient(location=QDRANT_URL)

            if init_collection:

                if client.collection_exists(collection_name=collection_name):
                    client.delete_collection(collection_name=collection_name)

                client.create_collection(collection_name=collection_name)

            return QdrantVectorStore(
                client=client,
                collection_name=collection_name,
                embedding=embedding_function,
            )

            # elif provider == "weaviate":
            client = weaviate.connect_to_custom(
                http_host=WEAVIATE_HOST,  # Replace with your Weaviate host
                http_port=WEAVIATE_HTTP_PORT,
                http_secure=False,
                grpc_host=WEAVIATE_HOST,  # Replace with your Weaviate gRPC host
                grpc_port=WEAVIATE_GRPC_PORT,
                grpc_secure=False,
            )
            return WeaviateVectorStore(
                client_url=f"http://{WEAVIATE_HOST}:{WEAVIATE_HTTP_PORT}",
                index_name=collection_name,
                embedding=embedding_function,
                **kwargs,
            )
            # elif provider == "postgres":
            return PGVector(
                collection_name=collection_name,
                embedding_function=embedding_function,
                connection=PGVECTOR_URL,
                distance_strategy=distance_strategy,
                **kwargs,
            )
        else:
            raise ValueError(
                f"Unsupported vector store provider: {provider}.\n"
                + "Currently supported: milvus, chroma, elasticsearch, qdrant, weaviate, pgvector"
            )
    except Exception as e:
        raise ConnectionError(f"Cannot connect to {provider_name} vector store: {e}")
