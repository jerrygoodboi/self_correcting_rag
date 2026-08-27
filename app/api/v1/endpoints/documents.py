from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import Optional
from app.schemas.rag import IngestTextRequest, IngestResponse
from app.services.document_service import document_service
from app.services.vector_service import vector_service
from app.core.logging import logger

router = APIRouter()


@router.post("/ingest/text", response_model=IngestResponse, summary="Ingest Raw Text Document")
def ingest_text_document(payload: IngestTextRequest):
    """Chunks and indexes raw text into Chroma vector database."""
    try:
        chunks_count = document_service.ingest_text(
            text=payload.content,
            title=payload.title,
            metadata=payload.metadata,
        )
        total_docs = vector_service().count()
        return IngestResponse(
            status="success",
            chunks_indexed=chunks_count,
            document_title=payload.title,
            total_collection_documents=total_docs,
        )
    except Exception as e:
        logger.exception(f"Failed to ingest text: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ingest/file", response_model=IngestResponse, summary="Upload and Ingest File")
async def ingest_file_document(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None)
):
    """Uploads a markdown, text, or json file, parses and chunks it into Chroma vector store."""
    try:
        content_bytes = await file.read()
        text = content_bytes.decode("utf-8", errors="ignore")
        doc_title = title or file.filename or "Uploaded File"
        
        chunks_count = document_service.ingest_text(
            text=text,
            title=doc_title,
            metadata={"source": file.filename, "file_name": file.filename},
        )
        total_docs = vector_service().count()
        return IngestResponse(
            status="success",
            chunks_indexed=chunks_count,
            document_title=doc_title,
            total_collection_documents=total_docs,
        )
    except Exception as e:
        logger.exception(f"Failed to ingest uploaded file: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats", summary="Vector Store Statistics")
def get_vectorstore_stats():
    """Returns vector database collection status and chunk count."""
    try:
        count = vector_service().count()
        return {
            "collection_name": vector_service().vector_store._collection.name,
            "total_chunks": count,
        }
    except Exception as e:
        logger.exception(f"Failed to get vector stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))
