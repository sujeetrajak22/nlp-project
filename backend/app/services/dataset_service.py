"""
dataset_service.py – Reads and exposes AG News dataset statistics and examples.

Data is read from pre-generated JSON files in ``data/processed/``.  When the
files are absent a safe placeholder response is returned so the API still
starts correctly before training data has been prepared.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from app.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hardcoded fallback examples (AG News style, one per category)
# ---------------------------------------------------------------------------
_HARDCODED_EXAMPLES: List[Dict[str, str]] = [
    {
        "category": "World",
        "headline": "UN Security Council emergency session called over border tensions",
        "article": (
            "World leaders gather for crisis talks as diplomatic efforts to defuse "
            "escalating tensions intensify. The session, convened at the request of "
            "five member states, follows a series of cross-border incidents that have "
            "raised fears of open conflict in the region."
        ),
    },
    {
        "category": "Sports",
        "headline": "Manchester City defeats Arsenal 3-1 in Premier League thriller",
        "article": (
            "Erling Haaland scores twice as City extend their lead at the top of the "
            "table. The Norwegian striker now has 27 league goals this season, with "
            "Phil Foden adding a stunning third from distance in the 78th minute. "
            "Arsenal's Gabriel Martinelli pulled one back late on, but it was not "
            "enough to deny the champions."
        ),
    },
    {
        "category": "Business",
        "headline": "Apple reports record quarterly revenue of $119 billion",
        "article": (
            "Driven by strong iPhone 15 sales and a growing services segment including "
            "App Store and Apple Music, the company posted its highest-ever quarterly "
            "figures. CEO Tim Cook credited robust demand in emerging markets and the "
            "continued expansion of Apple's ecosystem of hardware, software, and "
            "subscription services for the record-breaking performance."
        ),
    },
    {
        "category": "Sci/Tech",
        "headline": "Google DeepMind announces breakthrough in protein structure prediction",
        "article": (
            "Using a new AI model, DeepMind has surpassed previous AlphaFold benchmarks "
            "in both accuracy and computational speed. The system can predict the 3-D "
            "structure of proteins from amino-acid sequences in seconds, a task that "
            "previously required years of laboratory work. Researchers say the advance "
            "could accelerate drug discovery and the design of novel enzymes."
        ),
    },
]


class DatasetService:
    """Service for exposing AG News dataset statistics and example records.

    Reads from pre-generated JSON files when available; returns safe
    placeholder responses otherwise.
    """

    def __init__(self) -> None:
        """Resolve the paths to stats and examples JSON files."""
        self._stats_path: Path = settings.DATA_DIR / "processed" / "stats.json"
        self._examples_path: Path = settings.DATA_DIR / "processed" / "examples.json"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Return dataset statistics.

        Reads ``data/processed/stats.json`` when it exists; otherwise
        returns a placeholder dict with ``"status": "not_generated"``.

        Returns:
            Dictionary containing dataset statistics or a placeholder.
        """
        if not self._stats_path.exists():
            return {
                "status": "not_generated",
                "message": (
                    "Dataset statistics have not been generated yet. "
                    "Run scripts/download_dataset.py first."
                ),
            }

        try:
            with self._stats_path.open("r", encoding="utf-8") as fh:
                stats: Dict[str, Any] = json.load(fh)
            stats["status"] = "available"
            return stats
        except (json.JSONDecodeError, OSError) as exc:
            logger.error("Failed to read stats.json: %s", exc)
            return {
                "status": "error",
                "message": f"Failed to read dataset statistics: {exc}",
            }

    def get_examples(self) -> List[Dict[str, str]]:
        """Return one example news item per category.

        Reads ``data/processed/examples.json`` when it exists; otherwise
        returns hardcoded AG News–style examples.

        Returns:
            List of example dicts, each with ``category``, ``headline``,
            and ``article`` keys.
        """
        if self._examples_path.exists():
            try:
                with self._examples_path.open("r", encoding="utf-8") as fh:
                    examples: List[Dict[str, str]] = json.load(fh)
                return examples
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning(
                    "Could not read examples.json (%s); using hardcoded examples.", exc
                )

        return _HARDCODED_EXAMPLES
