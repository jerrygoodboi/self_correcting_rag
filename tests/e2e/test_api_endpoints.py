import io
import pytest
import pypdf
from fastapi.testclient import TestClient


def test_health_endpoint(test_client: TestClient):
    response = test_client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "postgres_checkpointer_enabled" in data


def test_ingest_text_endpoint(test_client: TestClient):
    payload = {
        "title": "Test Incident Manual",
        "content": "Kubernetes pods failing with CrashLoopBackOff should check container memory limits.",
        "metadata": {"category": "runbook"}
    }
    response = test_client.post("/api/v1/documents/ingest/text", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["chunks_indexed"] >= 1
    assert data["document_title"] == "Test Incident Manual"


def test_ingest_pdf_file_endpoint(test_client: TestClient):
    # Create a minimal in-memory PDF
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=72, height=72)
    pdf_bytes_io = io.BytesIO()
    
    # We can write text or use actual resume PDF if exists
    import os
    if os.path.exists("/home/jerry/JERRY_RESUME.pdf"):
        with open("/home/jerry/JERRY_RESUME.pdf", "rb") as f:
            pdf_data = f.read()
    else:
        writer.write(pdf_bytes_io)
        pdf_data = pdf_bytes_io.getvalue()

    files = {"file": ("test_resume.pdf", pdf_data, "application/pdf")}
    response = test_client.post("/api/v1/documents/ingest/file", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["chunks_indexed"] >= 1


def test_vector_stats_endpoint(test_client: TestClient):
    response = test_client.get("/api/v1/documents/stats")
    assert response.status_code == 200
    data = response.json()
    assert "total_chunks" in data
