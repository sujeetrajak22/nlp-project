"""
data.py – FastAPI router for dataset-related endpoints.

Endpoints
---------
GET /dataset/stats   – Return AG News dataset statistics.
GET /examples        – Return one labelled example per category.
GET /categories      – Return the list of supported category names.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Request, status

from app.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Dataset"])


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _get_dataset_service(request: Request):
    """Retrieve the singleton DatasetService from application state."""
    svc = getattr(request.app.state, "dataset_service", None)
    if svc is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Dataset service is not available.",
        )
    return svc


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get(
    "/dataset/stats",
    summary="Get AG News dataset statistics",
    response_description="Statistics about the training/test dataset.",
)
async def get_dataset_stats(request: Request) -> Dict[str, Any]:
    """Return statistics about the AG News dataset.

    If the processed stats file (``data/processed/stats.json``) has not been
    generated yet, a placeholder response with ``"status": "not_generated"``
    and a helpful message is returned instead of an error.
    """
    try:
        dataset_service = _get_dataset_service(request)
        return dataset_service.get_stats()
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover
        logger.exception("Error fetching dataset stats: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve dataset statistics.",
        ) from exc


@router.get(
    "/examples",
    summary="Get example news items",
    response_description="One labelled example per category.",
)
async def get_examples(request: Request) -> List[Dict[str, str]]:
    """Return one representative news item for each AG News category.

    Examples are read from ``data/processed/examples.json`` when available;
    otherwise a set of hardcoded AG News–style examples is returned.
    """
    try:
        dataset_service = _get_dataset_service(request)
        return dataset_service.get_examples()
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover
        logger.exception("Error fetching examples: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve examples.",
        ) from exc


@router.get(
    "/categories",
    summary="List supported news categories",
    response_description="Ordered list of category name strings.",
)
async def get_categories() -> List[str]:
    """Return the ordered list of AG News category labels.

    Categories: ``World``, ``Sports``, ``Business``, ``Sci/Tech``.
    """
    return settings.CATEGORIES
