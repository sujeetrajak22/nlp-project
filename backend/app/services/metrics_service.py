"""
metrics_service.py – Reads and exposes model evaluation metrics.

Metrics are expected to live in ``models/evaluation_results.json`` (produced
by ``scripts/evaluate.py``).  When the file is absent a safe placeholder
response is returned.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

from app.config import settings

logger = logging.getLogger(__name__)

_NOT_GENERATED_RESPONSE: Dict[str, str] = {
    "status": "not_generated",
    "message": (
        "Evaluation results have not been generated yet. "
        "Run scripts/train_baseline.py and scripts/evaluate.py first."
    ),
}


class MetricsService:
    """Service for reading and exposing model evaluation metrics.

    Reads ``models/evaluation_results.json`` when available; returns a
    placeholder response otherwise.
    """

    def __init__(self) -> None:
        """Resolve the path to the evaluation results JSON file."""
        self._results_path: Path = settings.MODEL_DIR / "evaluation_results.json"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_metrics(self) -> Dict[str, Any]:
        """Return model evaluation metrics.

        Reads ``models/evaluation_results.json`` when it exists; otherwise
        returns a placeholder dict with ``"status": "not_generated"``.

        Returns:
            Dictionary containing evaluation metrics or a placeholder.
        """
        if not self._results_path.exists():
            return _NOT_GENERATED_RESPONSE

        try:
            with self._results_path.open("r", encoding="utf-8") as fh:
                metrics: Dict[str, Any] = json.load(fh)
            metrics.setdefault("status", "available")
            return metrics
        except (json.JSONDecodeError, OSError) as exc:
            logger.error("Failed to read evaluation_results.json: %s", exc)
            return {
                "status": "error",
                "message": f"Failed to read evaluation results: {exc}",
            }
