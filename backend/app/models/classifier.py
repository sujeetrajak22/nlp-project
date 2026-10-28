"""
classifier.py – Core model wrapper for the Automatic News Topic Classification system.

Supports two backends:
  * ``tfidf_lr``   : TF-IDF vectoriser + Logistic Regression (scikit-learn)
  * ``distilbert`` : Fine-tuned DistilBERT classifier (HuggingFace Transformers)

Usage::

    from app.models.classifier import NewsClassifier
    clf = NewsClassifier()
    clf.load_tfidf_model("/path/to/models")
    result = clf.predict("Apple launches new iPhone", model_type="tfidf_lr")
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from app.config import settings
from app.preprocessing.text_preprocessor import TextPreprocessor

logger = logging.getLogger(__name__)


class NewsClassifier:
    """Unified interface over TF-IDF/LR and DistilBERT classifiers.

    Attributes:
        model_type: Default model backend (``"tfidf_lr"`` or ``"distilbert"``).
    """

    # Internal model artefacts
    _tfidf_vectorizer: Any = None
    _lr_model: Any = None
    _transformer_tokenizer: Any = None
    _transformer_model: Any = None
    _device: str = "cpu"

    def __init__(self, model_type: str = "tfidf_lr") -> None:
        """Initialise the classifier.

        Args:
            model_type: Default model to use when ``predict()`` is called
                without an explicit ``model_type`` argument.
        """
        self.model_type = model_type
        self._preprocessor = TextPreprocessor()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_tfidf_loaded(self) -> bool:
        """``True`` when the TF-IDF vectoriser and LR model are in memory."""
        return self._tfidf_vectorizer is not None and self._lr_model is not None

    @property
    def is_transformer_loaded(self) -> bool:
        """``True`` when the DistilBERT tokeniser and model are in memory."""
        return (
            self._transformer_tokenizer is not None
            and self._transformer_model is not None
        )

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    def load_tfidf_model(self, model_dir: str | Path) -> None:
        """Load TF-IDF vectoriser and Logistic Regression model from disk.

        Expects two joblib-serialised files inside *model_dir*:
          - ``tfidf_vectorizer.pkl``
          - ``logistic_regression.pkl``

        Args:
            model_dir: Directory that contains the serialised artefacts.

        Raises:
            FileNotFoundError: If either file is missing.
            Exception: Re-raises any joblib loading error.
        """
        import joblib  # type: ignore

        model_dir = Path(model_dir)
        vec_path = model_dir / "tfidf_vectorizer.pkl"
        lr_path = model_dir / "logistic_regression.pkl"

        if not vec_path.exists():
            raise FileNotFoundError(f"TF-IDF vectoriser not found: {vec_path}")
        if not lr_path.exists():
            raise FileNotFoundError(f"Logistic Regression model not found: {lr_path}")

        logger.info("Loading TF-IDF vectoriser from %s", vec_path)
        self._tfidf_vectorizer = joblib.load(vec_path)

        logger.info("Loading Logistic Regression model from %s", lr_path)
        self._lr_model = joblib.load(lr_path)

        logger.info("TF-IDF + LR model loaded successfully.")

    def load_transformer_model(
        self, model_dir: str | Path, device: str = "cpu"
    ) -> None:
        """Load a fine-tuned DistilBERT classification model from disk.

        Expects a HuggingFace ``save_pretrained`` directory layout inside
        *model_dir* / ``transformer/``.

        Args:
            model_dir: Parent directory that contains the ``transformer/``
                       sub-folder.
            device: Torch device string (e.g. ``"cpu"``, ``"cuda"``).

        Raises:
            FileNotFoundError: If the transformer directory is missing.
            Exception: Re-raises any HuggingFace loading error.
        """
        from transformers import (  # type: ignore
            DistilBertForSequenceClassification,
            DistilBertTokenizerFast,
        )

        transformer_dir = Path(model_dir) / "transformer"
        if not transformer_dir.exists():
            raise FileNotFoundError(
                f"Transformer model directory not found: {transformer_dir}"
            )

        logger.info("Loading DistilBERT tokeniser from %s", transformer_dir)
        self._transformer_tokenizer = DistilBertTokenizerFast.from_pretrained(
            str(transformer_dir)
        )

        logger.info("Loading DistilBERT model from %s", transformer_dir)
        self._transformer_model = (
            DistilBertForSequenceClassification.from_pretrained(str(transformer_dir))
        )
        self._transformer_model.to(device)
        self._transformer_model.eval()
        self._device = device

        logger.info("DistilBERT model loaded on device=%s.", device)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, text: str, model_type: Optional[str] = None) -> Dict:
        """Classify *text* and return a rich prediction dictionary.

        Args:
            text: Raw (uncleaned) news text.
            model_type: Override the instance default. One of
                ``"tfidf_lr"`` or ``"distilbert"``.

        Returns:
            Dictionary with keys:
              - ``predicted_class`` (str)
              - ``confidence``      (float 0-1)
              - ``probabilities``   (dict[str, float])
              - ``word_count``      (int)
              - ``char_count``      (int)
              - ``processing_time_ms`` (float)
              - ``model_used``      (str)
              - ``keywords``        (list[str])
              - ``fallback_note``   (str | None)
        """
        if not text or not text.strip():
            raise ValueError("Input text must not be empty.")

        effective_model = model_type or self.model_type
        fallback_note: Optional[str] = None

        # Graceful fallback: if distilbert requested but not loaded, use tfidf
        if effective_model == "distilbert" and not self.is_transformer_loaded:
            if self.is_tfidf_loaded:
                fallback_note = (
                    "DistilBERT model is not loaded. "
                    "Falling back to TF-IDF + Logistic Regression."
                )
                effective_model = "tfidf_lr"
            else:
                raise RuntimeError("No models are currently loaded.")

        if effective_model == "tfidf_lr" and not self.is_tfidf_loaded:
            raise RuntimeError(
                "TF-IDF model is not loaded. "
                "Run scripts/train_baseline.py first."
            )

        t0 = time.perf_counter()

        if effective_model == "tfidf_lr":
            result = self._predict_tfidf(text)
        else:
            result = self._predict_transformer(text)

        elapsed_ms = (time.perf_counter() - t0) * 1000

        result["processing_time_ms"] = round(elapsed_ms, 2)
        result["model_used"] = effective_model
        result["word_count"] = self._preprocessor.get_word_count(text)
        result["char_count"] = self._preprocessor.get_char_count(text)
        result["keywords"] = self._preprocessor.get_keywords(text, top_n=10)

        if fallback_note:
            result["fallback_note"] = fallback_note

        return result

    # ------------------------------------------------------------------
    # Internal prediction helpers
    # ------------------------------------------------------------------

    def _predict_tfidf(self, text: str) -> Dict:
        """Run TF-IDF + LR inference.

        Args:
            text: Raw text (preprocessing is applied internally).

        Returns:
            Partial prediction dict (no timing / meta fields yet).
        """
        preprocessed = self._preprocessor.preprocess_for_tfidf(text)
        if not preprocessed.strip():
            preprocessed = text  # fallback: use raw if everything is stripped

        vec = self._tfidf_vectorizer.transform([preprocessed])
        proba_arr: np.ndarray = self._lr_model.predict_proba(vec)[0]
        class_idx: int = int(np.argmax(proba_arr))
        confidence: float = float(proba_arr[class_idx])

        categories = settings.CATEGORIES
        probabilities = {
            cat: round(float(p), 4)
            for cat, p in zip(categories, proba_arr)
        }

        return {
            "predicted_class": categories[class_idx],
            "confidence": round(confidence, 4),
            "probabilities": probabilities,
        }

    def _predict_transformer(self, text: str) -> Dict:
        """Run DistilBERT inference.

        Args:
            text: Raw text (light preprocessing is applied internally).

        Returns:
            Partial prediction dict (no timing / meta fields yet).
        """
        import torch  # type: ignore
        import torch.nn.functional as F  # type: ignore

        preprocessed = self._preprocessor.preprocess_for_transformer(text)
        if not preprocessed.strip():
            preprocessed = text

        encoding = self._transformer_tokenizer(
            preprocessed,
            max_length=settings.MAX_SEQ_LENGTH,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].to(self._device)
        attention_mask = encoding["attention_mask"].to(self._device)

        with torch.no_grad():
            outputs = self._transformer_model(
                input_ids=input_ids, attention_mask=attention_mask
            )
            logits = outputs.logits
            proba_tensor = F.softmax(logits, dim=-1)[0]

        proba_arr: np.ndarray = proba_tensor.cpu().numpy()
        class_idx: int = int(np.argmax(proba_arr))
        confidence: float = float(proba_arr[class_idx])

        categories = settings.CATEGORIES
        probabilities = {
            cat: round(float(p), 4)
            for cat, p in zip(categories, proba_arr)
        }

        return {
            "predicted_class": categories[class_idx],
            "confidence": round(confidence, 4),
            "probabilities": probabilities,
        }

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    def get_feature_importance(
        self, text: str, top_n: int = 10
    ) -> List[Dict]:
        """Return the most influential TF-IDF features for *text*.

        Works only when the TF-IDF model is loaded.  The method:
          1. Preprocesses *text* through the TF-IDF pipeline.
          2. Transforms it to a TF-IDF feature vector.
          3. Multiplies element-wise with the LR coefficients for the
             predicted class to get a per-token importance score.
          4. Returns the top-N tokens by absolute importance.

        Args:
            text: Raw input text.
            top_n: Maximum number of features to return.

        Returns:
            List of ``{"word": str, "weight": float}`` dicts,
            ordered by descending absolute weight.

        Raises:
            RuntimeError: If the TF-IDF model is not loaded.
        """
        if not self.is_tfidf_loaded:
            raise RuntimeError(
                "TF-IDF model is not loaded; feature importance unavailable."
            )

        preprocessed = self._preprocessor.preprocess_for_tfidf(text)
        if not preprocessed.strip():
            return []

        vec = self._tfidf_vectorizer.transform([preprocessed])

        # Predict to know which class's coefficients to use
        proba_arr: np.ndarray = self._lr_model.predict_proba(vec)[0]
        class_idx: int = int(np.argmax(proba_arr))

        # LR coefficients for the predicted class
        coef: np.ndarray = self._lr_model.coef_[class_idx]

        # Dot element-wise: tfidf_score * lr_weight
        feature_names: List[str] = self._tfidf_vectorizer.get_feature_names_out().tolist()
        tfidf_dense: np.ndarray = vec.toarray()[0]
        importance: np.ndarray = tfidf_dense * coef

        # Non-zero entries only
        nonzero_idx = np.where(tfidf_dense > 0)[0]
        if len(nonzero_idx) == 0:
            return []

        sorted_idx = nonzero_idx[np.argsort(importance[nonzero_idx])[::-1]]
        top_idx = sorted_idx[:top_n]

        return [
            {
                "word": feature_names[i],
                "weight": round(float(importance[i]), 6),
            }
            for i in top_idx
        ]
