from langchain_community.chat_models import (
    ChatOpenAI,
    AzureChatOpenAI,
    ChatAnthropic,
    ChatCohere,
    ChatGoogleGenerativeAI,
    ChatMistralAI,
    ChatGroq,
    ChatFireworks,
    ChatOllama,
    ChatTogether,
    ChatHuggingFace,
    ChatDeepInfra,
)

from langchain_core.language_models import BaseLanguageModel


def create_llm(
    provider_name: str,
    key: str,
    provider_id: str | None = None,
    url: str | None = None,
    model: str | None = None,
    **model_kwargs,
) -> BaseLanguageModel:
    """
    Create a language model based on the provider name and configuration parameters.

    Args:
        provider_name: Name of the LLM provider
        model: Model name/identifier
        key: API key for authentication
        url: Base URL for the API (optional)
        provider_id: Provider-specific identifier (optional)
        model_kwargs: Additional model-specific parameters (temperature, max_tokens, etc.)

    Returns:
        A language model instance
    """
    provider = provider_name.lower().replace(" ", "_").replace("-", "_")
    kwargs = {**model_kwargs}  # Unpack additional kwargs into new dict

    # OpenAI compatible providers
    if provider in ["openai", "openai_chat"]:
        kwargs["openai_api_key"] = key
        if model:
            kwargs["model"] = model  # Langchain uses 'model' for ChatOpenAI
        if url:
            kwargs["base_url"] = url  # Correct parameter is base_url
        return ChatOpenAI(**kwargs)

    elif provider == "azure_openai":
        kwargs["api_key"] = key  # Correct parameter is api_key
        if model:
            kwargs["deployment_name"] = model
        if url:
            kwargs["azure_endpoint"] = url
        if provider_id:  # Use provider_id as API version
            kwargs["api_version"] = provider_id  # Correct parameter is api_version
        return AzureChatOpenAI(**kwargs)

    # Anthropic
    elif provider == "anthropic":
        kwargs["anthropic_api_key"] = key
        if model:
            kwargs["model"] = model  # Correct parameter is model
        return ChatAnthropic(**kwargs)

    # Google providers
    elif provider == "google_gemini":
        kwargs["google_api_key"] = key
        if model:
            kwargs["model"] = model
        return ChatGoogleGenerativeAI(**kwargs)

    # Cohere
    elif provider == "cohere":
        kwargs["cohere_api_key"] = key
        if model:
            kwargs["model"] = model
        return ChatCohere(**kwargs)

    elif provider == "huggingface_chat":
        kwargs["huggingfacehub_api_token"] = key
        if model:
            kwargs["repo_id"] = model
        return ChatHuggingFace(**kwargs)

    # Other popular providers
    elif provider == "mistral":
        kwargs["mistral_api_key"] = key
        if model:
            kwargs["model"] = model
        if url:
            kwargs["endpoint"] = url
        return ChatMistralAI(**kwargs)

    elif provider == "groq":
        kwargs["groq_api_key"] = key
        if model:
            kwargs["model"] = model  # Correct parameter is model
        return ChatGroq(**kwargs)

    elif provider == "fireworks":
        kwargs["fireworks_api_key"] = key
        if model:
            kwargs["model"] = model
        if url:
            kwargs["base_url"] = url
        return ChatFireworks(**kwargs)

    elif provider == "together":
        kwargs["together_api_key"] = key
        if model:
            kwargs["model"] = model
        return ChatTogether(**kwargs)

    # DeepInfra provider
    elif provider == "deepinfra":
        kwargs["deepinfra_api_key"] = key
        if model:
            kwargs["model"] = model
        if url:
            kwargs["endpoint"] = url
        return ChatDeepInfra(**kwargs)

    # Self-hosted models
    elif provider == "ollama":
        if model:
            kwargs["model"] = model
        if url:
            kwargs["base_url"] = url
        return ChatOllama(**kwargs)

    # Fallback for unsupported providers
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")
