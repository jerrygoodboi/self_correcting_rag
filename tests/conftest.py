import os
import sys
import pytest
from fastapi.testclient import TestClient

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.main import app
from app.core.config import settings


@pytest.fixture(scope="session")
def test_client():
    with TestClient(app) as client:
        yield client
