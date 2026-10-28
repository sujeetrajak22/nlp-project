"""
config.py – Application-wide settings for the Automatic News Topic Classification backend.

All paths are resolved relative to this file so they always point to the project root,
regardless of where uvicorn is launched from.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# Resolve the project root (two levels up from this file:
#   backend/app/config.py → backend/ → project-root/)
# ---------------------------------------------------------------------------
_THIS_FILE = Path(__file__).resolve()
_BACKEND_DIR = _THIS_FILE.parent.parent          # …/backend
_PROJECT_ROOT = _BACKEND_DIR.parent             # …/automatic-news-topic-classification


class Settings(BaseSettings):
    """Centralised application configuration.

    Values are read from environment variables first, then from a ``.env``
    file located next to the ``backend/`` folder, and finally from the
    defaults defined here.
    """

    model_config = SettingsConfigDict(
        env_file=str(_BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # Directories (absolute paths derived from the project root)
    # ------------------------------------------------------------------
    MODEL_DIR: Path = _PROJECT_ROOT / "models"
    DATA_DIR: Path = _PROJECT_ROOT / "data"

    # ------------------------------------------------------------------
    # Prediction limits
    # ------------------------------------------------------------------
    MAX_INPUT_LENGTH: int = 10_000
    CONFIDENCE_THRESHOLD: float = 0.5

    # ------------------------------------------------------------------
    # Model selection
    # ------------------------------------------------------------------
    DEFAULT_MODEL: str = "tfidf_lr"  # options: "tfidf_lr" | "distilbert"

    # ------------------------------------------------------------------
    # AG News categories
    # ------------------------------------------------------------------
    CATEGORIES: List[str] = ["World", "Sports", "Business", "Sci/Tech"]
    AG_NEWS_LABEL_MAP: Dict[int, str] = {
        0: "World",
        1: "Sports",
        2: "Business",
        3: "Sci/Tech",
    }

    # ------------------------------------------------------------------
    # Reproducibility
    # ------------------------------------------------------------------
    RANDOM_SEED: int = 42

    # ------------------------------------------------------------------
    # Transformer (DistilBERT) settings
    # ------------------------------------------------------------------
    TRANSFORMER_MODEL_NAME: str = "distilbert-base-uncased"
    MAX_SEQ_LENGTH: int = 128
    BATCH_SIZE: int = 32
    LEARNING_RATE: float = 2e-5
    NUM_EPOCHS: int = 3
    WEIGHT_DECAY: float = 0.01


# ---------------------------------------------------------------------------
# Singleton – import ``settings`` everywhere
# ---------------------------------------------------------------------------
settings = Settings()

# Ensure key directories exist at import time (harmless if already present)
settings.MODEL_DIR.mkdir(parents=True, exist_ok=True)
settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
