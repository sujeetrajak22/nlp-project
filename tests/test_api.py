"""
test_api.py
-----------
Integration tests for the FastAPI backend.

Run with:
    pytest tests/test_api.py -v

Note: These tests start the FastAPI app in-process using httpx AsyncClient.
      Models do NOT need to be trained for basic endpoint tests to pass;
      prediction tests will be skipped if no model is loaded.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"


@pytest_asyncio.fixture(scope="session")
async def client():
    """Shared async HTTP client for the duration of the test session."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_health_endpoint(client):
    """GET /api/health should return 200 with status 'ok'."""
    response = await client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "models_loaded" in data
    assert "timestamp" in data


@pytest.mark.anyio
async def test_root_endpoint(client):
    """GET / should return service information."""
    response = await client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "service" in data
    assert data["service"] == "Automatic News Topic Classifier"


# ---------------------------------------------------------------------------
# Categories
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_categories_endpoint(client):
    """GET /api/categories should return a list of 4 categories."""
    response = await client.get("/api/categories")
    assert response.status_code == 200
    categories = response.json()
    assert isinstance(categories, list)
    assert len(categories) == 4
    assert "World" in categories
    assert "Sports" in categories
    assert "Business" in categories
    assert "Sci/Tech" in categories


# ---------------------------------------------------------------------------
# Examples
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_examples_endpoint(client):
    """GET /api/examples should return examples list."""
    response = await client.get("/api/examples")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0


# ---------------------------------------------------------------------------
# Dataset stats
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_dataset_stats_endpoint(client):
    """GET /api/dataset/stats should return 200 with status field."""
    response = await client.get("/api/dataset/stats")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data  # either 'available' or 'not_generated'


# ---------------------------------------------------------------------------
# Model metrics
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_model_metrics_endpoint(client):
    """GET /api/model/metrics should return 200 with status field."""
    response = await client.get("/api/model/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_preprocess_valid_text(client):
    """POST /api/preprocess should return pipeline stages."""
    response = await client.post(
        "/api/preprocess",
        json={"text": "Apple reports record quarterly revenue from iPhone sales."},
    )
    assert response.status_code == 200
    data = response.json()
    assert "stages" in data
    assert isinstance(data["stages"], list)
    assert len(data["stages"]) > 0
    assert "original_text" in data
    assert "tfidf_preprocessed" in data
    assert "transformer_preprocessed" in data


@pytest.mark.anyio
async def test_preprocess_empty_text(client):
    """POST /api/preprocess with empty text should return 422."""
    response = await client.post("/api/preprocess", json={"text": ""})
    assert response.status_code == 422


@pytest.mark.anyio
async def test_preprocess_whitespace_only(client):
    """POST /api/preprocess with whitespace-only text should return 422."""
    response = await client.post("/api/preprocess", json={"text": "   "})
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Prediction — input validation
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_predict_both_empty(client):
    """POST /api/predict with both headline and article empty should return 422."""
    response = await client.post(
        "/api/predict", json={"headline": "", "article": ""}
    )
    assert response.status_code == 422


@pytest.mark.anyio
async def test_predict_no_body(client):
    """POST /api/predict with no headline or article should return 422."""
    response = await client.post("/api/predict", json={})
    assert response.status_code == 422


@pytest.mark.anyio
async def test_predict_invalid_model_type(client):
    """POST /api/predict with unknown model_type should return 422."""
    response = await client.post(
        "/api/predict",
        json={
            "article": "Some news article",
            "model_type": "invalid_model_name",
        },
    )
    assert response.status_code == 422


@pytest.mark.anyio
async def test_predict_input_too_long(client):
    """POST /api/predict with input exceeding max length should return 422."""
    response = await client.post(
        "/api/predict",
        json={"article": "word " * 3000},  # ~15,000 chars > 10,000 limit
    )
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Prediction — model inference (skip if no model loaded)
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_predict_sports_article(client):
    """POST /api/predict with a clear Sports article should classify correctly."""
    response = await client.post(
        "/api/predict",
        json={
            "headline": "Manchester City defeats Arsenal 3-1 in Premier League",
            "article": (
                "Erling Haaland scored twice as Manchester City defeated Arsenal "
                "in a breathtaking Premier League encounter. The result extends "
                "City's lead at the top of the table."
            ),
            "model_type": "tfidf_lr",
        },
    )

    if response.status_code == 503:
        pytest.skip("TF-IDF model not loaded — run train_baseline.py first")

    assert response.status_code == 200
    data = response.json()

    assert "predicted_class" in data
    assert "confidence" in data
    assert "probabilities" in data
    assert "word_count" in data
    assert "char_count" in data
    assert "processing_time_ms" in data
    assert "model_used" in data

    assert isinstance(data["confidence"], float)
    assert 0.0 <= data["confidence"] <= 1.0
    assert data["model_used"] == "tfidf_lr"

    # The model should predict Sports for this obvious Sports article
    assert data["predicted_class"] == "Sports", (
        f"Expected Sports but got {data['predicted_class']} "
        f"(confidence: {data['confidence']:.2%})"
    )


@pytest.mark.anyio
async def test_predict_tech_article(client):
    """POST /api/predict with a Sci/Tech article."""
    response = await client.post(
        "/api/predict",
        json={
            "article": (
                "Researchers at MIT developed a new deep learning algorithm that "
                "can detect early-stage cancer using artificial intelligence. The "
                "model achieves 94 percent accuracy on benchmark datasets."
            ),
            "model_type": "tfidf_lr",
        },
    )

    if response.status_code == 503:
        pytest.skip("TF-IDF model not loaded — run train_baseline.py first")

    assert response.status_code == 200
    data = response.json()
    assert data["predicted_class"] in ["Sci/Tech", "World", "Business", "Sports"]


@pytest.mark.anyio
async def test_predict_response_structure(client):
    """POST /api/predict response should include all required fields."""
    response = await client.post(
        "/api/predict",
        json={"article": "The stock market rose significantly today on Wall Street."},
    )

    if response.status_code == 503:
        pytest.skip("No model loaded")

    assert response.status_code == 200
    data = response.json()

    required_fields = {
        "predicted_class",
        "confidence",
        "probabilities",
        "word_count",
        "char_count",
        "processing_time_ms",
        "model_used",
        "uncertain",
    }
    assert required_fields.issubset(data.keys()), (
        f"Missing fields: {required_fields - data.keys()}"
    )

    # All 4 categories should be in probabilities
    for cat in ["World", "Sports", "Business", "Sci/Tech"]:
        assert cat in data["probabilities"]

    # Probabilities should sum to ~1
    total = sum(data["probabilities"].values())
    assert abs(total - 1.0) < 0.01, f"Probabilities should sum to 1, got {total}"
