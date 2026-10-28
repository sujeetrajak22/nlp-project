"""
metrics.py – FastAPI router for model evaluation metrics and health check.

Endpoints
---------
GET /model/metrics  – Return evaluation metrics for the trained models.
GET /health         – Return server health status and model availability.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict

from fastapi import APIRouter, HTTPException, Request, status

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Metrics & Health"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_metrics_service(request: Request):
    """Retrieve the singleton MetricsService from application state."""
    svc = getattr(request.app.state, "metrics_service", None)
    if svc is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Metrics service is not available.",
        )
    return svc


def _get_prediction_service(request: Request):
    """Retrieve the singleton PredictionService from application state (best-effort)."""
    return getattr(request.app.state, "prediction_service", None)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get(
    "/model/metrics",
    summary="Get model evaluation metrics",
    response_description="Accuracy, F1 scores, and confusion matrix (if available).",
)
async def get_model_metrics(request: Request) -> Dict[str, Any]:
    """Return evaluation metrics for all trained model backends.

    Results are read from ``models/evaluation_results.json``.  If this file
    does not exist (i.e. the models have not been evaluated yet), a
    placeholder response is returned with ``"status": "not_generated"`` and
    instructions on how to generate the metrics.
    """
    try:
        metrics_service = _get_metrics_service(request)
        return metrics_service.get_metrics()
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover
        logger.exception("Error fetching model metrics: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve model metrics.",
        ) from exc


@router.get(
    "/health",
    summary="Application health check",
    response_description="Server status, loaded models, and current timestamp.",
)
async def health_check(request: Request) -> Dict[str, Any]:
    """Return a health-check response indicating server and model readiness.

    This endpoint never raises an HTTP error – it is designed to be polled
    by load-balancers and monitoring tools.  The ``models_loaded`` field
    shows which model backends are currently in memory.

    Returns:
        JSON with:
        - ``status``        – always ``"ok"`` while the server is running
        - ``models_loaded`` – ``{"tfidf_lr": bool, "distilbert": bool}``
        - ``timestamp``     – current UTC time in ISO-8601 format
    """
    prediction_service = _get_prediction_service(request)

    if prediction_service is not None:
        models_loaded = prediction_service.models_loaded()
    else:
        models_loaded = {"tfidf_lr": False, "distilbert": False}

    return {
        "status": "ok",
        "models_loaded": models_loaded,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
