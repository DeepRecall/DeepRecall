from langchain_community.embeddings import *
from langchain_core.embeddings import Embeddings


def create_embedding(
    provider_name: str,
    key: str,
    provider_id: str | None = None,
    url: str | None = None,
    model: str | None = None,
) -> Embeddings:
    """
    Create an embedding engine based on the provider name and configuration parameters.

    Args:
        provider_name: Name of the embedding provider
        model: Model name/identifier
        key: API key for authentication
        url: Base URL for the API (optional)
        provider_id: Provider-specific identifier (optional)
        context_size: Maximum context size for embeddings

    Returns:
        An embedding engine instance
    """
    provider = provider_name.lower().replace(" ", "_").replace("-", "_")

    if provider == "aleph_alpha_asymmetric":
        kwargs = {"aleph_alpha_api_key": key}
        if model is not None:
            kwargs["model"] = model
        if url is not None:
            kwargs["host"] = url
        return AlephAlphaAsymmetricSemanticEmbedding(**kwargs)

    elif provider == "aleph_alpha_symmetric":
        kwargs = {"aleph_alpha_api_key": key}
        if model is not None:
            kwargs["model"] = model
        if url is not None:
            kwargs["host"] = url
        return AlephAlphaSymmetricSemanticEmbedding(**kwargs)
    elif provider == "anyscale":
        kwargs = {"anyscale_api_key": key}
        if model is not None:
            kwargs["model_name"] = model
        if url is not None:
            kwargs["anyscale_api_base"] = url
        return AnyscaleEmbeddings(**kwargs)
    elif provider == "azure_openai":
        kwargs = {"openai_api_key": key}
        if model is not None:
            kwargs["azure_deployment"] = model
        if url is not None:
            kwargs["azure_endpoint"] = url
        return AzureOpenAIEmbeddings(**kwargs)
    elif provider == "baichuan":
        kwargs = {"baichuan_api_key": key}
        if model is not None:
            kwargs["model_name"] = model
        return BaichuanTextEmbeddings(**kwargs)
    elif provider == "bookend":
        assert isinstance(provider_id, str), "domain must be defined for bookend"
        kwargs = {"api_token": key, "domain": provider_id}
        if model is not None:
            kwargs["model_id"] = model
        return BookendEmbeddings(**kwargs)
    elif provider == "clarifai":
        assert isinstance(provider_id, str), "user id must be defined for clarifai"
        kwargs = {"token": key, "user_id": provider_id}
        if model is not None:
            kwargs["model_id"] = model
        if url is not None:
            kwargs["api_base"] = url
        return ClarifaiEmbeddings(**kwargs)
    elif provider == "clova":
        assert isinstance(provider_id, str), "app_id must be defined for clova"
        kwargs = {"clova_emb_api_key": key, "app_id": provider_id}
        if model is not None:
            kwargs["model"] = model
        return ClovaEmbeddings(**kwargs)
    elif provider == "clova_x":
        kwargs = {"ncp_clovastudio_api_key": key}
        if model is not None:
            kwargs["model_name"] = model
        if url is not None:
            kwargs["base_url"] = url
        return ClovaXEmbeddings(**kwargs)
    elif provider == "cohere":
        kwargs = {"cohere_api_key": key}
        if model is not None:
            kwargs["model"] = model
        return CohereEmbeddings(**kwargs)
    elif provider == "dashscope":
        kwargs = {"dashscope_api_key": key}
        if model is not None:
            kwargs["model"] = model
        return DashScopeEmbeddings(**kwargs)
    elif provider == "deepinfra":
        kwargs = {"deepinfra_api_token": key}
        if model is not None:
            kwargs["model_id"] = model
        return DeepInfraEmbeddings(**kwargs)
    elif provider == "eden_ai":
        assert isinstance(provider_id, str), "provider must be defined for eden_ai"
        kwargs = {"edenai_api_key": key, "provider": provider_id}
        if model is not None:
            kwargs["model"] = model
        return EdenAiEmbeddings(**kwargs)
    elif provider == "embaas":
        kwargs = {"embaas_api_key": key}
        if model is not None:
            kwargs["model"] = model
        if url is not None:
            kwargs["api_url"] = url
        return EmbaasEmbeddings(**kwargs)
    elif provider == "ernie":
        assert isinstance(provider_id, str), "ernie_client_id must be defined for ernie"
        kwargs = {"ernie_client_id": provider_id, "ernie_client_secret": key}
        if model is not None:
            kwargs["model_name"] = model
        if url is not None:
            kwargs["ernie_api_base"] = url
        return ErnieEmbeddings(**kwargs)
    elif provider == "gigachat":
        assert isinstance(provider_id, str), "user must be defined for gigachat"
        kwargs = {"user": provider_id, "password": key}
        if model is not None:
            kwargs["model"] = model
        if url is not None:
            kwargs["base_url"] = url
        return GigaChatEmbeddings(**kwargs)
    elif provider == "google_palm":
        kwargs = {"google_api_key": key}
        if model is not None:
            kwargs["model_name"] = model
        return GooglePalmEmbeddings(**kwargs)
    elif provider == "gradient":
        assert isinstance(
            provider_id, str
        ), "gradient_workspace_id must be defined for gradient"
        kwargs = {
            "gradient_access_token": key,
            "gradient_workspace_id": provider_id,
        }
        if model is not None:
            kwargs["model"] = model
        if url is not None:
            kwargs["gradient_api_url"] = url
        return GradientEmbeddings(**kwargs)
    elif provider == "huggingface_hub":
        kwargs = {"huggingfacehub_api_token": key}
        if model is not None:
            kwargs["model"] = model
        return HuggingFaceHubEmbeddings(**kwargs)
    elif provider == "huggingface_inference_api":
        kwargs = {"api_key": key}
        if model is not None:
            kwargs["model_name"] = model
        return HuggingFaceInferenceAPIEmbeddings(**kwargs)
    elif provider == "hunyuan":
        assert isinstance(
            provider_id, str
        ), "hunyuan_secret_id must be defined for hunyuan"
        kwargs = {"hunyuan_secret_key": key, "hunyuan_secret_id": provider_id}
        return HunyuanEmbeddings(**kwargs)
    elif provider == "javelin_ai_gateway":
        kwargs = {"javelin_api_key": key}
        if model is not None:
            kwargs["route"] = model
        if url is not None:
            kwargs["gateway_uri"] = url
        return JavelinAIGatewayEmbeddings(**kwargs)
    elif provider == "jina":
        kwargs = {"jina_api_key": key}
        if model is not None:
            kwargs["model_name"] = model
        return JinaEmbeddings(**kwargs)
    elif provider == "llm_rails":
        kwargs = {"api_key": key}
        if model is not None:
            kwargs["model"] = model
        return LLMRailsEmbeddings(**kwargs)
    elif provider == "minimax":
        assert isinstance(provider_id, str), "group_id must be defined for minimax"
        kwargs = {"api_key": key, "group_id": provider_id}
        if model is not None:
            kwargs["model"] = model
        return MiniMaxEmbeddings(**kwargs)
    elif provider == "mosaicml_instructor":
        kwargs = {"mosaicml_api_token": key}
        if model is not None:
            kwargs["model"] = model
        return MosaicMLInstructorEmbeddings(**kwargs)
    elif provider == "nlpcloud":
        kwargs = {"nlpcloud_api_key": key, "gpu": True}
        if model is not None:
            kwargs["model_name"] = model
        return NLPCloudEmbeddings(**kwargs)
    elif provider == "octoai":
        kwargs = {"octoai_api_token": key}
        if model is not None:
            kwargs["model"] = model
        if url is not None:
            kwargs["endpoint_url"] = url
        return OctoAIEmbeddings(**kwargs)
    elif provider == "openai":
        kwargs = {"openai_api_key": key}
        if model is not None:
            kwargs["model"] = model
        if url is not None:
            kwargs["openai_api_base"] = url
        return OpenAIEmbeddings(**kwargs)
    elif provider == "ovhcloud":
        kwargs = {"access_token": key}
        if model is not None:
            kwargs["model_name"] = model
        return OVHCloudEmbeddings(**kwargs)
    elif provider == "premai":
        assert isinstance(provider_id, str), "project_id must be defined for premai"
        kwargs = {"premai_api_key": key, "project_id": int(provider_id)}
        if model is not None:
            kwargs["model"] = model
        return PremAIEmbeddings(**kwargs)
    elif provider == "qianfan":
        assert isinstance(provider_id, str), "qianfan_ak must be defined for qianfan"
        kwargs = {"qianfan_ak": provider_id, "qianfan_sk": key}
        if model is not None:
            kwargs["model"] = model
        return QianfanEmbeddingsEndpoint(**kwargs)
    elif provider == "solar":
        kwargs = {"solar_api_key": key}
        if model is not None:
            kwargs["model"] = model
        if url is not None:
            kwargs["endpoint_url"] = url
        return SolarEmbeddings(**kwargs)
    elif provider == "textembed":
        kwargs = {"api_key": key}
        if model is not None:
            kwargs["model"] = model
        if url is not None:
            kwargs["api_url"] = url
        return TextEmbedEmbeddings(**kwargs)
    elif provider == "volcengine":
        assert isinstance(provider_id, str), "volcano_ak must be defined for volcengine"
        kwargs = {"volcano_sk": key, "volcano_ak": provider_id}
        if model is not None:
            kwargs["model"] = model
        if url is not None:
            kwargs["host"] = url
        return VolcanoEmbeddings(**kwargs)
    elif provider == "voyageai":
        kwargs = {"voyage_api_key": key}
        if model is not None:
            kwargs["model"] = model
        if url is not None:
            kwargs["voyage_api_base"] = url
        return VoyageEmbeddings(**kwargs)
    elif provider == "xinference":
        kwargs = {}
        if url is not None:
            kwargs["server_url"] = url
        if model is not None:
            kwargs["model_uid"] = model
        return XinferenceEmbeddings(**kwargs)
    elif provider == "yandex":
        assert isinstance(provider_id, str), "folder_id must be defined for yandex"
        assert isinstance(model, str), "model must be defined for yandex"
        kwargs = {
            "iam_token": key,
            "folder_id": provider_id,
            "model_uri": f"emb://{provider_id}/{model}/latest",
        }
        return YandexGPTEmbeddings(**kwargs)
    elif provider == "zhipuai":
        kwargs = {"api_key": key}
        if model is not None:
            kwargs["model"] = model
        return ZhipuAIEmbeddings(**kwargs)
    else:
        raise ValueError(f"Unsupported embedding provider: {provider}")
