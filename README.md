# Self-Correcting Agentic RAG System

An enterprise-ready **Self-Correcting Agentic RAG** system built using **FastAPI**, **LangGraph**, **Google Gemini**, **ChromaDB**, and **PostgreSQL** persistent state checkpointing.

---

## Key Features

- **Query Understanding & Routing**: Direct response for chit-chat / conversational queries, semantic vector search for technical knowledge questions.
- **Document Relevance Grading**: Evaluates retrieved document chunks and discards irrelevant noise.
- **Self-Correction & Query Rewriting**: Autonomously reformulates search queries when documents or answers fail quality or grounding checks.
- **Hallucination Detection / Grounding Check**: Validates that generated answers are strictly grounded in retrieved facts.
- **Persistent State Checkpointing**: Session-based multi-turn memory backed by PostgreSQL (`psycopg` pool).
- **FastAPI Enterprise Architecture**: Modular structure with `/api/v1` routes, Pydantic schemas, dependency injection, and health checks.
- **Interactive CLI & Full Test Suite**: Terminal REPL with rich trace logs and complete unit, integration, and E2E tests.

---

## Directory Structure

```
self_correcting_rag/
├── app/
│   ├── main.py                  # FastAPI application entrypoint & lifespan
│   ├── core/                    # Settings & structured logging
│   ├── schemas/                 # Pydantic request/response & evaluation schemas
│   ├── agent/                   # LangGraph Agentic RAG core
│   │   ├── state.py             # AgentState TypedDict
│   │   ├── graph.py             # LangGraph workflow & conditional edges
│   │   ├── checkpointer.py      # PostgreSQL & MemorySaver checkpointer
│   │   ├── prompts/             # System prompts per decision node
│   │   └── nodes/               # Router, Retriever, Graders, Rewriter, Generator
│   ├── services/                # Business services (RAG, Vector, Document)
│   └── api/v1/endpoints/        # FastAPI endpoints (health, rag, documents)
├── data/
│   ├── sample_docs/             # Initial knowledge base documents
│   └── chromadb/                # Persistent vector database
├── scripts/
│   ├── seed_data.py             # Ingest sample technical documents
│   └── cli.py                   # Interactive CLI REPL with reasoning trace
├── tests/                       # Unit, integration, and E2E test suite
├── .env.example
├── requirements.txt
└── README.md
```

---

## Getting Started

### 1. Configure Environment
```bash
cp .env.example .env
# Ensure GEMINI_API_KEY is set
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Seed Knowledge Base
```bash
python scripts/seed_data.py
```

### 4. Run the FastAPI Server
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive Swagger API documentation will be available at: `http://localhost:8000/docs`.

### 5. Interactive CLI
```bash
python scripts/cli.py
```

### 6. Run Test Suite
```bash
pytest tests/ -v
```
