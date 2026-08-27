import os
import sys

# Ensure root directory is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services.document_service import document_service
from app.services.vector_service import vector_service
from app.core.logging import logger


def seed():
    docs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "sample_docs"))
    logger.info(f"Seeding knowledge base from: {docs_dir}")
    
    if not os.path.exists(docs_dir):
        logger.error(f"Sample docs directory not found at: {docs_dir}")
        return

    chunks_indexed = document_service.ingest_directory(docs_dir)
    total_docs = vector_service().count()
    
    logger.info(f"Seeding complete! Indexed {chunks_indexed} chunks. Total in Chroma vectorstore: {total_docs}")


if __name__ == "__main__":
    seed()
