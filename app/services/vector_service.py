import os
from typing import List, Optional
from langchain_core.documents import Document
from langchain_chroma import Chroma
from app.core.llm_factory import get_embeddings
from app.core.config import settings
from app.core.logging import logger


class VectorService:
    """Service managing ChromaDB vectorstore and embeddings."""
    _instance: Optional["VectorService"] = None

    def __init__(self):
        os.makedirs(settings.chroma_persist_dir_abs, exist_ok=True)
        self.embeddings = get_embeddings()
        self._init_vector_store()

    def _init_vector_store(self):
        self.vector_store = Chroma(
            collection_name=settings.chroma_collection_name,
            embedding_function=self.embeddings,
            persist_directory=settings.chroma_persist_dir_abs,
        )
        logger.info(f"Initialized Chroma vector store in '{settings.chroma_persist_dir_abs}' with collection '{settings.chroma_collection_name}'")

    @classmethod
    def get_instance(cls) -> "VectorService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def similarity_search(self, query: str, k: Optional[int] = None) -> List[Document]:
        top_k = k or settings.top_k_documents
        try:
            results = self.vector_store.similarity_search(query, k=top_k)
            logger.info(f"Retrieved {len(results)} chunks for query: '{query}'")
            return results
        except Exception as e:
            logger.error(f"Error during vector search: {e}")
            return []

    def add_documents(self, documents: List[Document]) -> List[str]:
        if not documents:
            return []
        ids = self.vector_store.add_documents(documents)
        logger.info(f"Indexed {len(documents)} document chunks into Chroma")
        return ids

    def delete_by_source(self, source_name: str) -> None:
        """Deletes all chunks matching a specific source file to prevent duplicates."""
        try:
            self.vector_store.delete(where={"file_name": source_name})
            logger.info(f"Deleted existing vector chunks for source: '{source_name}'")
        except Exception as e:
            logger.debug(f"No existing chunks found or error deleting source '{source_name}': {e}")

    def reset_collection(self) -> None:
        """Completely wipes and re-initializes the ChromaDB collection."""
        try:
            self.vector_store.delete_collection()
            self._init_vector_store()
            logger.info("ChromaDB collection successfully reset and wiped clean.")
        except Exception as e:
            logger.error(f"Failed to reset ChromaDB collection: {e}")
            self._init_vector_store()

    def count(self) -> int:
        try:
            return self.vector_store._collection.count()
        except Exception:
            return 0


vector_service = VectorService.get_instance
