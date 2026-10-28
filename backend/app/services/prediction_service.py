"""
prediction_service.py – High-level orchestration layer for news classification.

``PredictionService`` wraps ``NewsClassifier`` and ``TextPreprocessor`` and
provides a single ``classify()`` entry-point used by the API routes.
"""

from __future__ import annotations

import logging
from typing import Optional

from app.config import settings
from app.models.classifier import NewsClassifier
from app.preprocessing.text_preprocessor import TextPreprocessor

logger = logging.getLogger(__name__)


class PredictionService:
    """Orchestrates text preprocessing and model inference for classification.

    This service is intended to be instantiated once at application startup
    (see ``main.py``) and reused across requests.

    Attributes:
        classifier: The underlying :class:`NewsClassifier` instance.
        preprocessor: The shared :class:`TextPreprocessor` instance.
    """

    def __init__(self) -> None:
        """Initialise the service with a classifier and a preprocessor."""
        self.classifier = NewsClassifier(model_type=settings.DEFAULT_MODEL)
        self.preprocessor = TextPreprocessor()
        logger.info("PredictionService initialised.")

    # ------------------------------------------------------------------
    # Model loading helpers (called from main.py lifespan)
    # ------------------------------------------------------------------

    def load_tfidf(self) -> None:
        """Attempt to load the TF-IDF model from ``settings.MODEL_DIR``."""
        try:
            self.classifier.load_tfidf_model(settings.MODEL_DIR)
            logger.info("TF-IDF model loaded successfully.")
        except FileNotFoundError as exc:
            logger.warning("TF-IDF model not found: %s", exc)
        except Exception as exc:  # pragma: no cover
            logger.error("Failed to load TF-IDF model: %s", exc, exc_info=True)

    def load_transformer(self, device: str = "cpu") -> None:
        """Attempt to load the DistilBERT model from ``settings.MODEL_DIR``."""
        try:
            self.classifier.load_transformer_model(settings.MODEL_DIR, device=device)
            logger.info("DistilBERT model loaded successfully.")
        except FileNotFoundError as exc:
            logger.warning("Transformer model not found: %s", exc)
        except Exception as exc:  # pragma: no cover
            logger.error(
                "Failed to load transformer model: %s", exc, exc_info=True
            )

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------

    def classify(
        self,
        headline: Optional[str],
        article: Optional[str],
        model_type: str = "tfidf_lr",
    ) -> dict:
        """Classify a news item given its headline and/or article body.

        The headline (if provided) is prepended to the article with a space
        separator, giving the model a richer signal.

        Post-processing rules:
          * If ``confidence < settings.CONFIDENCE_THRESHOLD``, the field
            ``uncertain`` is set to ``True`` and an explanatory message is
            added.

        Args:
            headline: Optional short headline string.
            article: Optional longer article body.
            model_type: Which model backend to use (``"tfidf_lr"`` or
                ``"distilbert"``).

        Returns:
            A dictionary with the full prediction response, including:
              - ``predicted_class``, ``confidence``, ``probabilities``
              - ``word_count``, ``char_count``, ``processing_time_ms``
              - ``model_used``, ``keywords``
              - ``uncertain`` (bool), ``uncertainty_message`` (str | None)
              - ``combined_text`` (the text that was actually classified)

        Raises:
            ValueError: If both *headline* and *article* are empty.
            RuntimeError: If no model is available to serve the request.
        """
        # Combine headline + article
        parts = []
        if headline and headline.strip():
            parts.append(headline.strip())
        if article and article.strip():
            parts.append(article.strip())

        if not parts:
            raise ValueError(
                "At least one of 'headline' or 'article' must be non-empty."
            )

        combined_text = " ".join(parts)

        # Enforce length limit
        if len(combined_text) > settings.MAX_INPUT_LENGTH:
            combined_text = combined_text[: settings.MAX_INPUT_LENGTH]
            logger.debug(
                "Input truncated to %d characters.", settings.MAX_INPUT_LENGTH
            )

        # Run inference
        result = self.classifier.predict(combined_text, model_type=model_type)

        # Post-processing: flag low-confidence predictions
        confidence: float = result.get("confidence", 0.0)
        uncertain = confidence < settings.CONFIDENCE_THRESHOLD
        result["uncertain"] = uncertain
        result["uncertainty_message"] = (
            (
                "The model is uncertain about this classification "
                f"(confidence {confidence:.1%} < threshold "
                f"{settings.CONFIDENCE_THRESHOLD:.1%}). "
                "Consider providing more context."
            )
            if uncertain
            else None
        )

        result["combined_text"] = combined_text
        return result

    # ------------------------------------------------------------------
    # Status helper
    # ------------------------------------------------------------------

    def models_loaded(self) -> dict:
        """Return a dict indicating which model backends are loaded.

        Returns:
            ``{"tfidf_lr": bool, "distilbert": bool}``
        """
        return {
            "tfidf_lr": self.classifier.is_tfidf_loaded,
            "distilbert": self.classifier.is_transformer_loaded,
        }
