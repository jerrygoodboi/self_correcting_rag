import os
import sys
import argparse

# Ensure root directory is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services.document_service import document_service
from app.services.vector_service import vector_service
from app.core.logging import logger


def seed(clear_existing: bool = True):
    docs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "sample_docs"))
    
    if clear_existing:
        logger.info("Resetting Chroma vectorstore for clean re-indexing...")
        vector_service().reset_collection()

    logger.info(f"Seeding knowledge base from: {docs_dir}")
    
    if not os.path.exists(docs_dir):
        logger.error(f"Sample docs directory not found at: {docs_dir}")
        return

    chunks_indexed = document_service.ingest_directory(docs_dir)
    total_docs = vector_service().count()
    
    logger.info(f"Seeding complete! Indexed {chunks_indexed} chunks. Total in Chroma vectorstore: {total_docs}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed ChromaDB knowledge base with sample documents.")
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append new documents without wiping existing vector database collection.",
    )
    args = parser.parse_args()
    seed(clear_existing=not args.append)
