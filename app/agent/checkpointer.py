from typing import Any, Optional
from psycopg_pool import ConnectionPool
from langgraph.checkpoint.memory import MemorySaver
from app.core.config import settings
from app.core.logging import logger


class CheckpointerFactory:
    """Factory for LangGraph checkpointer (PostgreSQL with MemorySaver fallback)."""
    _checkpointer: Optional[Any] = None
    _pool: Optional[ConnectionPool] = None

    @classmethod
    def get_checkpointer(cls) -> Any:
        if cls._checkpointer is not None:
            return cls._checkpointer

        if settings.use_postgres_checkpointer:
            try:
                from langgraph.checkpoint.postgres import PostgresSaver
                
                logger.info(f"Initializing PostgreSQL Checkpointer with URI: {settings.postgres_uri}")
                cls._pool = ConnectionPool(
                    conninfo=settings.postgres_uri,
                    max_size=10,
                    open=True,
                    kwargs={"autocommit": True}
                )
                saver = PostgresSaver(cls._pool)
                saver.setup()
                cls._checkpointer = saver
                logger.info("PostgreSQL Checkpointer successfully initialized and tables verified.")
                return cls._checkpointer
            except Exception as e:
                logger.warning(f"Failed to connect to PostgreSQL checkpointer ({e}). Falling back to in-memory MemorySaver.")

        cls._checkpointer = MemorySaver()
        logger.info("Using in-memory MemorySaver checkpointer.")
        return cls._checkpointer


get_checkpointer = CheckpointerFactory.get_checkpointer
