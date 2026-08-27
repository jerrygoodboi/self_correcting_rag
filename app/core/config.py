import os
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Environment
    app_name: str = "Self-Correcting Agentic RAG"
    app_env: str = "development"
    log_level: str = "INFO"

    # LLM Provider Configuration ("gemini" | "ollama")
    llm_provider: str = "gemini"
    gemini_api_key: str = ""
    llm_model: str = "gemini-2.5-flash-lite"
    llm_temperature: float = 0.0
    
    # Ollama / Tailscale Configuration
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"
    
    # Embeddings Configuration ("gemini" | "ollama")
    embedding_provider: str = "gemini"
    embedding_model: str = "gemini-embedding-001"
    ollama_embedding_model: str = "nomic-embed-text"

    # Agent Parameters
    max_retries: int = 3
    top_k_documents: int = 4

    # Vector Store
    chroma_persist_dir: str = "./data/chromadb"
    chroma_collection_name: str = "knowledge_base"

    @property
    def chroma_persist_dir_abs(self) -> str:
        """Always return the resolved absolute path for ChromaDB."""
        if os.path.isabs(self.chroma_persist_dir):
            return self.chroma_persist_dir
        return os.path.abspath(os.path.join(ROOT_DIR, self.chroma_persist_dir))

    # PostgreSQL Checkpointer Settings
    use_postgres_checkpointer: bool = True
    postgres_uri: str = "postgresql://postgres:postgres@localhost:5433/self_correcting_rag"

    # Server Settings
    host: str = "0.0.0.0"
    port: int = 8000


settings = Settings()
