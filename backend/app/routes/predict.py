"""
predict.py – FastAPI router for classification and preprocessing endpoints.

Endpoints
---------
POST /predict
    Classify a news item (headline + article) using a chosen model backend.
POST /preprocess
    Return the full preprocessing pipeline stages for a given text.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, field_validator, model_validator

from app.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Prediction"])


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class PredictRequest(BaseModel):
    """Request body for ``POST /predict``."""

    headline: Optional[str] = None
    article: Optional[str] = None
    model_type: str = settings.DEFAULT_MODEL

    @model_validator(mode="after")
    def at_least_one_field(self) -> "PredictRequest":
        """Ensure at least one of headline or article is non-empty."""
        headline_ok = bool(self.headline and self.headline.strip())
        article_ok = bool(self.article and self.article.strip())
        if not headline_ok and not article_ok:
            raise ValueError(
                "At least one of 'headline' or 'article' must be provided and non-empty."
            )
        return self

    @model_validator(mode="after")
    def combined_length_limit(self) -> "PredictRequest":
        """Validate that the combined text length does not exceed the configured limit."""
        combined = " ".join(
            filter(None, [self.headline or "", self.article or ""])
        ).strip()
        if len(combined) > settings.MAX_INPUT_LENGTH:
            raise ValueError(
                f"Combined input length ({len(combined)} chars) exceeds the "
                f"maximum allowed ({settings.MAX_INPUT_LENGTH} chars)."
            )
        return self

    @field_validator("model_type")
    @classmethod
    def validate_model_type(cls, v: str) -> str:
        """Ensure model_type is one of the supported backends."""
        allowed = {"tfidf_lr", "distilbert"}
        if v not in allowed:
            raise ValueError(
                f"Invalid model_type '{v}'. Must be one of: {sorted(allowed)}"
            )
        return v


class PreprocessRequest(BaseModel):
    """Request body for ``POST /preprocess``."""

    text: str

    @field_validator("text")
    @classmethod
    def text_not_empty(cls, v: str) -> str:
        """Validate that text is non-empty."""
        if not v or not v.strip():
            raise ValueError("'text' must be a non-empty string.")
        return v


# ---------------------------------------------------------------------------
# Helper – access shared service from app state
# ---------------------------------------------------------------------------


def _get_prediction_service(request: Request):
    """Retrieve the singleton PredictionService from application state."""
    svc = getattr(request.app.state, "prediction_service", None)
    if svc is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction service is not available.",
        )
    return svc


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post(
    "/predict",
    summary="Classify a news article",
    response_description="Prediction result with class, confidence, and metadata.",
)
async def predict(
    body: PredictRequest,
    prediction_service=Depends(_get_prediction_service),
) -> Dict:
    """Classify a news item using the selected model backend.

    - Combines *headline* and *article* (both are optional but at least one
      must be provided).
    - Applies the appropriate preprocessing pipeline automatically.
    - Returns the predicted category, confidence score, per-class
      probabilities, keywords, and optional uncertainty flags.

    **Model types**
    - ``tfidf_lr``   – TF-IDF + Logistic Regression (fast, CPU-only)
    - ``distilbert`` – Fine-tuned DistilBERT (more accurate, GPU-optional)
    """
    try:
        result = prediction_service.classify(
            headline=body.headline,
            article=body.article,
            model_type=body.model_type,
        )
        return result
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except Exception as exc:  # pragma: no cover
        logger.exception("Unexpected error during prediction: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during classification.",
        ) from exc


@router.post(
    "/preprocess",
    summary="Inspect the preprocessing pipeline",
    response_description="All preprocessing stages with token/char counts.",
)
async def preprocess(
    body: PreprocessRequest,
    prediction_service=Depends(_get_prediction_service),
) -> Dict:
    """Return every intermediate step of the text preprocessing pipeline.

    Useful for understanding how raw text is transformed before classification
    and for debugging model behaviour.

    Returns a JSON object with:
    - ``stages``                – ordered list of ``{stage_name, text, token_count, char_count}``
    - ``original_text``         – the unmodified input
    - ``tfidf_preprocessed``    – final text after the full TF-IDF pipeline
    - ``transformer_preprocessed`` – final text after the light transformer pipeline
    """
    try:
        preprocessor = prediction_service.preprocessor
        stages = preprocessor.get_preprocessing_stages(body.text)
        tfidf_preprocessed = preprocessor.preprocess_for_tfidf(body.text)
        transformer_preprocessed = preprocessor.preprocess_for_transformer(body.text)

        return {
            "original_text": body.text,
            "tfidf_preprocessed": tfidf_preprocessed,
            "transformer_preprocessed": transformer_preprocessed,
            "stages": stages,
        }
    except Exception as exc:  # pragma: no cover
        logger.exception("Unexpected error during preprocessing: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during preprocessing.",
        ) from exc
