import os
from typing import List, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.services.vector_service import vector_service
from app.core.logging import logger


class DocumentService:
    """Service for loading, parsing, chunking, and ingesting documents."""

    def __init__(self, chunk_size: int = 600, chunk_overlap: int = 100):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )

    def ingest_text(self, text: str, title: str, metadata: Optional[dict] = None) -> int:
        meta = metadata or {}
        meta["title"] = title
        meta["source"] = meta.get("source", title)

        chunks = self.text_splitter.create_documents([text], metadatas=[meta])
        logger.info(f"Split document '{title}' into {len(chunks)} chunks")
        
        v_service = vector_service()
        v_service.add_documents(chunks)
        return len(chunks)

    def ingest_file(self, file_path: str) -> int:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        filename = os.path.basename(file_path)
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        return self.ingest_text(
            text=content,
            title=filename,
            metadata={"source": file_path, "file_name": filename}
        )

    def ingest_directory(self, dir_path: str) -> int:
        if not os.path.exists(dir_path):
            return 0
        
        total_chunks = 0
        for root, _, files in os.walk(dir_path):
            for file in files:
                if file.endswith((".md", ".txt", ".json", ".rst")):
                    full_path = os.path.join(root, file)
                    try:
                        chunks = self.ingest_file(full_path)
                        total_chunks += chunks
                        logger.info(f"Ingested {chunks} chunks from '{file}'")
                    except Exception as e:
                        logger.error(f"Failed to ingest file '{file}': {e}")
        return total_chunks


document_service = DocumentService()
