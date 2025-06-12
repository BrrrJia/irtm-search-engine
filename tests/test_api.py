from fastapi.testclient import TestClient
from src.api.main import app

# client = TestClient(app)
import sys

print("sys.path =", sys.path)


def test_search_endpoint():
    with TestClient(app) as client:
        response = client.get("/search?query=test&mode=term")
        assert response.status_code == 200
        assert "results" in response.json()


def test_classify_endpoint():
    with TestClient(app) as client:
        response = client.post(
            "/classify",
            json={
                "name": "Die Simpsons",
                "title": "Super",
                "review": "ich LIEBE dieses spiel",
            },
        )
        assert response.status_code == 200
        assert "label" in response.json()


def test_evaluate():
    with TestClient(app) as client:
        response = client.get("/classify/evaluate")
        assert response.status_code == 200
        assert "Accuracy" in response.json()
        assert "F1" in response.json()


def test_clustering():
    with TestClient(app) as client:
        response = client.get("/clustering")
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"
