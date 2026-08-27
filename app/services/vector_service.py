import os
from typing import List, Optional
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.core.config import settings
from app.core.logging import logger


class VectorService:
    """Service managing ChromaDB vectorstore and embeddings."""
    _instance: Optional["VectorService"] = None

    def __init__(self):
        os.makedirs(settings.chroma_persist_dir, exist_ok=True)
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=settings.embedding_model,
            google_api_key=settings.gemini_api_key,
        )
        self.vector_store = Chroma(
            collection_name=settings.chroma_collection_name,
            embedding_function=self.embeddings,
            persist_directory=settings.chroma_persist_dir,
        )
        logger.info(f"Initialized Chroma vector store in '{settings.chroma_persist_dir}' with collection '{settings.chroma_collection_name}'")

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

    def count(self) -> int:
        try:
            return self.vector_store._collection.count()
        except Exception:
            return 0


vector_service = VectorService.get_instance
