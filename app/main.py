from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.core.logging import logger
from app.api.v1.router import api_router
from app.agent.checkpointer import get_checkpointer


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle event handler for startup and shutdown."""
    logger.info(f"Starting {settings.app_name} in {settings.app_env} mode...")
    
    # Initialize checkpointer (verifies PostgreSQL connection & tables)
    try:
        get_checkpointer()
    except Exception as e:
        logger.warning(f"Could not initialize checkpointer at startup: {e}")

    yield

    logger.info(f"Shutting down {settings.app_name}...")


def create_app() -> FastAPI:
    """FastAPI Application Factory."""
    app = FastAPI(
        title=settings.app_name,
        description=(
            "Self-Correcting Agentic RAG System utilizing LangGraph, Google Gemini, "
            "ChromaDB, and PostgreSQL state checkpointing with autonomous grading and query rewriting."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    # CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include V1 Router
    app.include_router(api_router, prefix="/api/v1")

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=True)
