import os
import logging
from dotenv import load_dotenv
from deeprecall.services.celery_app import celery_app
from tika import parser
from langchain_milvus import Milvus
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents.base import Document
from langchain_community.embeddings import *

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
EMBEDDING_ID = os.getenv("EMBEDDING_ID")
EMBEDDING_KEY = os.getenv("EMBEDDING_KEY")
EMBEDDING_URL = os.getenv("EMBEDDING_URL")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
EMBEDDING_CONTEXT_SIZE = int(os.getenv("EMBEDDING_CONTEXT_SIZE", "2048"))

EMBEDDING_PROVIDERS = {
    "aleph_alpha_asymmetric": lambda: AlephAlphaAsymmetricSemanticEmbedding(
        model=EMBEDDING_MODEL,
        aleph_alpha_api_key=EMBEDDING_KEY,
        host=EMBEDDING_URL or "https://api.aleph-alpha.com",
    ),
    "aleph_alpha_symmetric": lambda: AlephAlphaSymmetricSemanticEmbedding(
        model=EMBEDDING_MODEL,
        aleph_alpha_api_key=EMBEDDING_KEY,
        host=EMBEDDING_URL or "https://api.aleph-alpha.com",
    ),
    "anyscale": lambda: AnyscaleEmbeddings(
        model_name=EMBEDDING_MODEL,
        anyscale_api_key=EMBEDDING_KEY,
        anyscale_api_base=EMBEDDING_URL or "https://api.endpoints.anyscale.com/v1",
    ),
    "azure_openai": lambda: AzureOpenAIEmbeddings(
        azure_deployment=EMBEDDING_MODEL,
        openai_api_key=EMBEDDING_KEY,
        azure_endpoint=EMBEDDING_URL,
    ),
    "baichuan": lambda: BaichuanTextEmbeddings(
        model_name=EMBEDDING_MODEL, baichuan_api_key=EMBEDDING_KEY
    ),
    "bookend": lambda: BookendEmbeddings(
        model_id=EMBEDDING_MODEL, domain=EMBEDDING_ID, api_token=EMBEDDING_KEY
    ),
    "clarifai": lambda: ClarifaiEmbeddings(
        model_id=EMBEDDING_MODEL,
        token=EMBEDDING_KEY,
        user_id=EMBEDDING_ID,
        api_base=EMBEDDING_URL or "https://api.clarifai.com",
    ),
    "clova": lambda: ClovaEmbeddings(
        model=EMBEDDING_MODEL, app_id=EMBEDDING_ID, clova_emb_api_key=EMBEDDING_KEY
    ),
    "clova_x": lambda: ClovaXEmbeddings(
        model_name=EMBEDDING_MODEL,
        ncp_clovastudio_api_key=EMBEDDING_KEY,
        base_url=EMBEDDING_URL or "https://clovastudio.stream.ntruss.com",
    ),
    "cohere": lambda: CohereEmbeddings(
        model=EMBEDDING_MODEL, cohere_api_key=EMBEDDING_KEY
    ),
    "dashscope": lambda: DashScopeEmbeddings(
        model=EMBEDDING_MODEL, dashscope_api_key=EMBEDDING_KEY
    ),
    "databricks": lambda: DatabricksEmbeddings(target_uri=EMBEDDING_URL),
    "deepinfra": lambda: DeepInfraEmbeddings(
        model_id=EMBEDDING_MODEL, deepinfra_api_token=EMBEDDING_KEY
    ),
    "eden_ai": lambda: EdenAiEmbeddings(
        model=EMBEDDING_MODEL, edenai_api_key=EMBEDDING_KEY, provider=EMBEDDING_ID
    ),
    "embaas": lambda: EmbaasEmbeddings(
        model=EMBEDDING_MODEL,
        embaas_api_key=EMBEDDING_KEY,
        api_url=EMBEDDING_URL or "https://api.embaas.io/v1/embeddings/",
    ),
    "ernie": lambda: ErnieEmbeddings(
        ernie_client_id=EMBEDDING_ID,
        ernie_client_secret=EMBEDDING_KEY,
        ernie_api_base=EMBEDDING_URL or "https://aip.baidubce.com",
        model_name=EMBEDDING_MODEL,
    ),
    "gigachat": lambda: GigaChatEmbeddings(
        base_url=EMBEDDING_URL,
        model=EMBEDDING_MODEL,
        user=EMBEDDING_ID,
        password=EMBEDDING_KEY,
    ),
    "google_palm": lambda: GooglePalmEmbeddings(
        google_api_key=EMBEDDING_KEY,
        model_name=EMBEDDING_MODEL,
    ),
    "gradient": lambda: GradientEmbeddings(
        model=EMBEDDING_MODEL,
        gradient_workspace_id=EMBEDDING_ID,
        gradient_access_token=EMBEDDING_KEY,
        gradient_api_url=EMBEDDING_URL or "https://api.gradient.ai/api",
    ),
    "huggingface_hub": lambda: HuggingFaceHubEmbeddings(
        model=EMBEDDING_MODEL,
        huggingfacehub_api_token=EMBEDDING_KEY,
    ),
    "huggingface_inference_api": lambda: HuggingFaceInferenceAPIEmbeddings(
        api_key=EMBEDDING_KEY,
        model_name=EMBEDDING_MODEL,
    ),
    "hunyuan": lambda: HunyuanEmbeddings(
        hunyuan_secret_id=EMBEDDING_ID,
        hunyuan_secret_key=EMBEDDING_KEY,
    ),
    "javelin_ai_gateway": lambda: JavelinAIGatewayEmbeddings(
        gateway_uri=EMBEDDING_URL, route=EMBEDDING_MODEL, javelin_api_key=EMBEDDING_KEY
    ),
    "jina": lambda: JinaEmbeddings(
        model_name=EMBEDDING_MODEL, jina_api_key=EMBEDDING_KEY
    ),
    "llm_rails": lambda: LLMRailsEmbeddings(
        model=EMBEDDING_MODEL, api_key=EMBEDDING_KEY
    ),
    "minimax": lambda: MiniMaxEmbeddings(
        model=EMBEDDING_MODEL, group_id=EMBEDDING_ID, api_key=EMBEDDING_KEY
    ),
    "mosaicml_instructor": lambda: MosaicMLInstructorEmbeddings(
        model=EMBEDDING_MODEL,
        mosaicml_api_token=EMBEDDING_KEY,
    ),
    "nlpcloud": lambda: NLPCloudEmbeddings(
        model_name=EMBEDDING_MODEL, gpu=True, nlpcloud_api_key=EMBEDDING_KEY
    ),
    "octoai": lambda: OctoAIEmbeddings(
        model=EMBEDDING_MODEL,
        octoai_api_token=EMBEDDING_KEY,
        endpoint_url=EMBEDDING_URL or "https://text.octoai.run/v1/",
    ),
    "openai": lambda: OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=EMBEDDING_KEY,
        openai_api_base=EMBEDDING_URL or "https://api.openai.com/v1",
    ),
    "ovhcloud": lambda: OVHCloudEmbeddings(
        access_token=EMBEDDING_KEY, model_name=EMBEDDING_MODEL
    ),
    "premai": lambda: PremAIEmbeddings(
        project_id=int(EMBEDDING_ID),
        premai_api_key=EMBEDDING_KEY,
        model=EMBEDDING_MODEL,
    ),
    "qianfan": lambda: QianfanEmbeddingsEndpoint(
        qianfan_ak=EMBEDDING_ID,
        qianfan_sk=EMBEDDING_KEY,
        model=EMBEDDING_MODEL,
    ),
    "solar": lambda: SolarEmbeddings(
        endpoint_url=EMBEDDING_URL or "https://api.upstage.ai/v1/solar/embeddings",
        model=EMBEDDING_MODEL,
        solar_api_key=EMBEDDING_KEY,
    ),
    "textembed": lambda: TextEmbedEmbeddings(
        model=EMBEDDING_MODEL, api_url=EMBEDDING_URL, api_key=EMBEDDING_KEY
    ),
    "volcengine": lambda: VolcanoEmbeddings(
        volcano_ak=EMBEDDING_ID,
        volcano_sk=EMBEDDING_KEY,
        model=EMBEDDING_MODEL,
        host=EMBEDDING_URL or "maas-api.ml-platform-cn-beijing.volces.com",
    ),
    "voyageai": lambda: VoyageEmbeddings(
        model=EMBEDDING_MODEL,
        voyage_api_base=EMBEDDING_URL or "https://api.voyageai.com/v1/embeddings",
        voyage_api_key=EMBEDDING_KEY,
    ),
    "xinference": lambda: XinferenceEmbeddings(
        server_url=EMBEDDING_URL, model_uid=EMBEDDING_MODEL
    ),
    "yandex": lambda: YandexGPTEmbeddings(
        iam_token=EMBEDDING_KEY,
        model_uri=f"emb://{EMBEDDING_ID}/{EMBEDDING_MODEL}/latest",
        folder_id=EMBEDDING_ID,
    ),
    "zhipuai": lambda: ZhipuAIEmbeddings(model=EMBEDDING_MODEL, api_key=EMBEDDING_KEY),
}


@celery_app.task(bind=True)
def populate_rag(self, folder_path: str, collection_name: str, embedding_provider: str):
    """
    Process files in a folder and store embeddings in Milvus.

    Args:
        self: Celery task instance (for task state updates)
        folder_path: Path to directory containing files to process
        collection_name: Collection name for Milvus
        embedding_provider: Name of the embedding provider to use

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

        # Dynamically create an instance of the specified embedding provider
        embedding_provider = (
            embedding_provider.lower().replace(" ", "_").replace("-", "_")
        )
        embed_engine = EMBEDDING_PROVIDERS.get(embedding_provider)
        if embed_engine is None:
            error_msg = f"Unsupported embedding provider: {embedding_provider}"
            logging.error(error_msg)
            self.update_state(state="FAILURE", meta={"error": error_msg})
            return False
        embed_engine = embed_engine()

        # Create Milvus vector store with dynamic collection name
        vector_store = Milvus(
            embedding_function=embed_engine,
            collection_name=collection_name,
            connection_args={
                "uri": MILVUS_URL,
            },
            drop_old=False,
            auto_id=True,
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
        print(
            "Usage: python task.py <folder_path> <collection_name> <embedding_provider>"
        )
        sys.exit(1)

    folder_path = sys.argv[1]
    collection_name = sys.argv[2] if len(sys.argv) > 2 else None
    embedding_provider = sys.argv[3] if len(sys.argv) > 3 else None
    populate_rag(folder_path, collection_name, embedding_provider)
