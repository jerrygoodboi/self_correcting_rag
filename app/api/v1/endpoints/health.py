from fastapi import APIRouter
from app.core.config import settings
from app.services.vector_service import vector_service

router = APIRouter()


@router.get("/health", summary="Health Check")
def health_check():
    """Returns the operational status of the service, checkpointer mode, and vector count."""
    doc_count = 0
    try:
        doc_count = vector_service().count()
    except Exception:
        pass

    return {
        "status": "healthy",
        "app_name": settings.app_name,
        "environment": settings.app_env,
        "llm_model": settings.llm_model,
        "embedding_model": settings.embedding_model,
        "postgres_checkpointer_enabled": settings.use_postgres_checkpointer,
        "indexed_documents_count": doc_count,
    }
