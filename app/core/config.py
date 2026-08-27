import os
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


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

    # LLM Settings
    gemini_api_key: str = ""
    llm_model: str = "gemini-2.5-flash-lite"
    llm_temperature: float = 0.0
    embedding_model: str = "gemini-embedding-001"

    # Agent Parameters
    max_retries: int = 3
    top_k_documents: int = 4

    # Vector Store
    chroma_persist_dir: str = "./data/chromadb"
    chroma_collection_name: str = "knowledge_base"

    # PostgreSQL Checkpointer Settings
    use_postgres_checkpointer: bool = True
    postgres_uri: str = "postgresql://postgres:postgres@localhost:5433/self_correcting_rag"

    # Server Settings
    host: str = "0.0.0.0"
    port: int = 8000


settings = Settings()
