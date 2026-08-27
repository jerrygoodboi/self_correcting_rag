import pytest
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


def test_vector_stats_endpoint(test_client: TestClient):
    response = test_client.get("/api/v1/documents/stats")
    assert response.status_code == 200
    data = response.json()
    assert "total_chunks" in data
