import os
import io
from typing import List, Optional, Dict
import pypdf
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
        self._synced_file_mtimes: Dict[str, float] = {}

    def extract_text_from_bytes(self, content_bytes: bytes, filename: str = "") -> str:
        """Extract text from raw bytes, handling PDF, Markdown, Text, and JSON."""
        if filename.lower().endswith(".pdf") or content_bytes.startswith(b"%PDF"):
            try:
                reader = pypdf.PdfReader(io.BytesIO(content_bytes))
                text_pages = [page.extract_text() or "" for page in reader.pages]
                extracted_text = "\n\n".join(text_pages).strip()
                if not extracted_text:
                    raise ValueError("Could not extract any text from PDF.")
                return extracted_text
            except Exception as e:
                logger.error(f"Error parsing PDF bytes for '{filename}': {e}")
                raise ValueError(f"Failed to parse PDF: {e}")
        else:
            return content_bytes.decode("utf-8", errors="ignore")

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
        with open(file_path, "rb") as f:
            content_bytes = f.read()

        text = self.extract_text_from_bytes(content_bytes, filename=filename)

        chunks_count = self.ingest_text(
            text=text,
            title=filename,
            metadata={"source": file_path, "file_name": filename}
        )
        self._synced_file_mtimes[file_path] = os.path.getmtime(file_path)
        return chunks_count

    def ingest_directory(self, dir_path: str) -> int:
        if not os.path.exists(dir_path):
            return 0
        
        total_chunks = 0
        for root, _, files in os.walk(dir_path):
            for file in files:
                if file.endswith((".md", ".txt", ".json", ".rst", ".pdf")):
                    full_path = os.path.join(root, file)
                    try:
                        chunks = self.ingest_file(full_path)
                        total_chunks += chunks
                        logger.info(f"Ingested {chunks} chunks from '{file}'")
                    except Exception as e:
                        logger.error(f"Failed to ingest file '{file}': {e}")
        return total_chunks

    def auto_sync_sample_docs(self, dir_path: Optional[str] = None) -> int:
        """Checks for new or updated files in the sample docs directory and ingests them."""
        target_dir = dir_path or os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "data", "sample_docs")
        )
        if not os.path.exists(target_dir):
            return 0

        new_chunks = 0
        for root, _, files in os.walk(target_dir):
            for file in files:
                if file.endswith((".md", ".txt", ".json", ".rst", ".pdf")):
                    full_path = os.path.join(root, file)
                    current_mtime = os.path.getmtime(full_path)
                    last_mtime = self._synced_file_mtimes.get(full_path)

                    # Ingest if new or modified
                    if last_mtime is None or current_mtime > last_mtime:
                        try:
                            chunks = self.ingest_file(full_path)
                            new_chunks += chunks
                            logger.info(f"Auto-synced new/modified file '{file}' ({chunks} chunks)")
                        except Exception as e:
                            logger.error(f"Failed to auto-sync file '{file}': {e}")
        return new_chunks


document_service = DocumentService()
