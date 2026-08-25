import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from fpl_analytics.server.app import app


def test_health():
    client = TestClient(app)
    res = client.get("/api/health")
    assert res.status_code == 200
    body = res.json()
    assert body["ok"] is True
    assert "version" in body
