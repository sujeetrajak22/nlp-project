"""
main.py – FastAPI application entry-point for the News Topic Classifier backend.

Startup sequence
----------------
1. Download required NLTK data (if missing).
2. Try to load the TF-IDF model (graceful skip if not trained yet).
3. Try to load the DistilBERT model (graceful skip if not trained yet).
4. Mount all routers under the ``/api`` prefix.

Run with::

    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Dict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.preprocessing.text_preprocessor import _ensure_nltk
from app.routes import data as data_router
from app.routes import metrics as metrics_router
from app.routes import predict as predict_router
from app.services.dataset_service import DatasetService
from app.services.metrics_service import MetricsService
from app.services.prediction_service import PredictionService

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Application lifespan (startup / shutdown)
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application startup and shutdown lifecycle.

    On startup:
      * Downloads NLTK corpora if absent.
      * Instantiates all services and attaches them to ``app.state``.
      * Attempts to load pre-trained models (skips gracefully if unavailable).

    On shutdown:
      * Logs a shutdown message (resources are released by the GC).
    """
    logger.info("=== Automatic News Topic Classifier – startup ===")

    # 1. NLTK resources
    logger.info("Ensuring NLTK resources are available …")
    try:
        _ensure_nltk()
    except Exception as exc:  # pragma: no cover
        logger.warning("NLTK bootstrap warning (non-fatal): %s", exc)

    # 2. Instantiate services
    prediction_service = PredictionService()
    dataset_service = DatasetService()
    metrics_service = MetricsService()

    # 3. Attach to app state (accessible from routes via request.app.state)
    app.state.prediction_service = prediction_service
    app.state.dataset_service = dataset_service
    app.state.metrics_service = metrics_service

    # 4. Load models (each failure is non-fatal)
    logger.info("Loading TF-IDF model …")
    prediction_service.load_tfidf()

    logger.info("Loading DistilBERT model …")
    prediction_service.load_transformer()

    loaded = prediction_service.models_loaded()
    logger.info(
        "Models loaded – tfidf_lr=%s, distilbert=%s",
        loaded["tfidf_lr"],
        loaded["distilbert"],
    )

    if not any(loaded.values()):
        logger.warning(
            "No models are loaded. "
            "Run scripts/train_baseline.py (and optionally scripts/train_transformer.py) "
            "to train the models before making prediction requests."
        )

    logger.info("=== Startup complete – API is ready ===")

    yield  # ← application runs here

    logger.info("=== Shutting down Automatic News Topic Classifier ===")


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        A fully configured :class:`fastapi.FastAPI` instance.
    """
    application = FastAPI(
        title="Automatic News Topic Classifier",
        description=(
            "REST API for classifying news articles into AG News categories: "
            "World, Sports, Business, and Sci/Tech.  "
            "Supports TF-IDF + Logistic Regression and fine-tuned DistilBERT backends."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # CORS – allow all origins in development
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routers
    application.include_router(predict_router.router, prefix="/api")
    application.include_router(data_router.router, prefix="/api")
    application.include_router(metrics_router.router, prefix="/api")

    return application


app = create_app()


# ---------------------------------------------------------------------------
# Root endpoint
# ---------------------------------------------------------------------------


@app.get("/", summary="API root", tags=["Root"])
async def root() -> Dict[str, Any]:
    """Return basic information about the API.

    This endpoint can be used as a quick sanity-check to confirm the service
    is running.
    """
    return {
        "service": "Automatic News Topic Classifier",
        "version": "1.0.0",
        "status": "running",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "docs": "/docs",
        "health": "/api/health",
        "categories": settings.CATEGORIES,
        "default_model": settings.DEFAULT_MODEL,
    }
