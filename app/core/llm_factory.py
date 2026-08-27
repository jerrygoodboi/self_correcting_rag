from typing import Any, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from app.core.config import settings
from app.core.logging import logger


def get_chat_llm(temperature: float = 0.0, model_name: Optional[str] = None) -> BaseChatModel:
    """
    Factory function to instantiate the configured Chat LLM (Gemini or Ollama via Tailscale/Local).
    """
    provider = settings.llm_provider.lower().strip()

    if provider == "ollama":
        try:
            from langchain_ollama import ChatOllama
            selected_model = model_name or settings.ollama_model
            logger.info(f"Using Ollama LLM '{selected_model}' at base_url='{settings.ollama_base_url}'")
            return ChatOllama(
                model=selected_model,
                base_url=settings.ollama_base_url,
                temperature=temperature,
            )
        except Exception as e:
            logger.error(f"Failed to initialize ChatOllama ({e}). Falling back to Gemini.")

    # Default to Gemini
    from langchain_google_genai import ChatGoogleGenerativeAI
    selected_model = model_name or settings.llm_model
    return ChatGoogleGenerativeAI(
        model=selected_model,
        temperature=temperature,
        google_api_key=settings.gemini_api_key,
    )


def get_embeddings() -> Embeddings:
    """
    Factory function to instantiate the configured Embedding model (Gemini or Ollama via Tailscale/Local).
    """
    provider = settings.embedding_provider.lower().strip()

    if provider == "ollama":
        try:
            from langchain_ollama import OllamaEmbeddings
            logger.info(f"Using Ollama Embeddings '{settings.ollama_embedding_model}' at base_url='{settings.ollama_base_url}'")
            return OllamaEmbeddings(
                model=settings.ollama_embedding_model,
                base_url=settings.ollama_base_url,
            )
        except Exception as e:
            logger.error(f"Failed to initialize OllamaEmbeddings ({e}). Falling back to Gemini.")

    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    return GoogleGenerativeAIEmbeddings(
        model=settings.embedding_model,
        google_api_key=settings.gemini_api_key,
    )
