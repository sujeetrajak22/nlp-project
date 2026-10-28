"""
text_preprocessor.py – Utilities for cleaning and normalising news text.

Provides two preprocessing pipelines:
  * TF-IDF pipeline  : aggressive cleaning, stopword removal, lemmatisation
  * Transformer pipeline : light cleaning only (preserves sentence structure)

Also exposes a ``get_preprocessing_stages`` helper that returns every
intermediate step so the frontend can visualise the pipeline.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import Dict, List

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# NLTK bootstrap – downloaded once per interpreter session
# ---------------------------------------------------------------------------
_NLTK_READY = False


def _ensure_nltk() -> None:
    """Download required NLTK corpora if not already present."""
    global _NLTK_READY
    if _NLTK_READY:
        return

    import nltk  # local import so the module loads even without NLTK

    resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
        ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
    ]
    for path, name in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            logger.info("Downloading NLTK resource: %s", name)
            nltk.download(name, quiet=True)

    _NLTK_READY = True


# ---------------------------------------------------------------------------
# Regex patterns (compiled once)
# ---------------------------------------------------------------------------
_RE_HTML = re.compile(r"<[^>]+>")
_RE_URL = re.compile(
    r"https?://\S+|www\.\S+",
    re.IGNORECASE,
)
_RE_SPECIAL = re.compile(r"[^a-zA-Z0-9\s]")
_RE_MULTI_SPACE = re.compile(r"\s+")
_RE_NEWLINE = re.compile(r"[\r\n\t]+")


class TextPreprocessor:
    """Text preprocessing for the Automatic News Topic Classification system.

    Supports two pipelines:
      - TF-IDF  : full NLP cleaning (stopwords, lemmatisation, etc.)
      - Transformer : minimal cleaning (preserves casing and punctuation)
    """

    def __init__(self) -> None:
        """Initialise the preprocessor and warm up NLTK resources."""
        _ensure_nltk()
        self._stopwords: set[str] | None = None
        self._lemmatizer = None

    # ------------------------------------------------------------------
    # Lazy-loaded NLTK helpers
    # ------------------------------------------------------------------

    @property
    def _sw(self) -> set[str]:
        """Return the cached NLTK English stopword set."""
        if self._stopwords is None:
            from nltk.corpus import stopwords as _sw_corpus  # type: ignore

            self._stopwords = set(_sw_corpus.words("english"))
        return self._stopwords

    @property
    def _lem(self):
        """Return the cached WordNetLemmatizer instance."""
        if self._lemmatizer is None:
            from nltk.stem import WordNetLemmatizer  # type: ignore

            self._lemmatizer = WordNetLemmatizer()
        return self._lemmatizer

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _remove_html(text: str) -> str:
        """Strip HTML / XML tags from *text*."""
        return _RE_HTML.sub(" ", text)

    @staticmethod
    def _remove_urls(text: str) -> str:
        """Replace URLs with a single space."""
        return _RE_URL.sub(" ", text)

    @staticmethod
    def _remove_special_chars(text: str) -> str:
        """Remove characters that are not alphanumeric or whitespace."""
        return _RE_SPECIAL.sub(" ", text)

    @staticmethod
    def _normalise_whitespace(text: str) -> str:
        """Collapse runs of whitespace to a single space and strip edges."""
        text = _RE_NEWLINE.sub(" ", text)
        return _RE_MULTI_SPACE.sub(" ", text).strip()

    @staticmethod
    def _normalise_unicode(text: str) -> str:
        """Normalise unicode characters to their ASCII equivalents where possible."""
        return (
            unicodedata.normalize("NFKD", text)
            .encode("ascii", errors="ignore")
            .decode("ascii")
        )

    def _tokenize(self, text: str) -> List[str]:
        """Tokenise *text* using NLTK word_tokenize."""
        from nltk import word_tokenize  # type: ignore

        return word_tokenize(text)

    def _remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Remove English stopwords from *tokens*."""
        return [t for t in tokens if t.lower() not in self._sw]

    def _lemmatize(self, tokens: List[str]) -> List[str]:
        """Lemmatise *tokens* using NLTK WordNetLemmatizer."""
        return [self._lem.lemmatize(t) for t in tokens]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def preprocess_for_tfidf(self, text: str) -> str:
        """Full NLP pipeline for the TF-IDF/ML model.

        Steps:
          1. Remove HTML tags
          2. Remove URLs
          3. Lowercase
          4. Remove special characters & punctuation
          5. Normalise whitespace
          6. Tokenise
          7. Remove stopwords
          8. Lemmatise

        Args:
            text: Raw input text.

        Returns:
            Cleaned, lemmatised string ready for TF-IDF vectorisation.
        """
        if not text or not text.strip():
            return ""

        text = self._remove_html(text)
        text = self._remove_urls(text)
        text = text.lower()
        text = self._remove_special_chars(text)
        text = self._normalise_whitespace(text)

        tokens = self._tokenize(text)
        tokens = [t for t in tokens if t.isalpha()]  # keep alphabetic only
        tokens = self._remove_stopwords(tokens)
        tokens = self._lemmatize(tokens)

        return " ".join(tokens)

    def preprocess_for_transformer(self, text: str) -> str:
        """Light cleaning pipeline for transformer models.

        Preserves casing, punctuation, and sentence structure so the
        transformer tokeniser can exploit contextual information.

        Steps:
          1. Normalise unicode
          2. Remove HTML tags
          3. Remove URLs
          4. Normalise whitespace

        Args:
            text: Raw input text.

        Returns:
            Lightly cleaned string ready for HuggingFace tokenisation.
        """
        if not text or not text.strip():
            return ""

        text = self._normalise_unicode(text)
        text = self._remove_html(text)
        text = self._remove_urls(text)
        text = self._normalise_whitespace(text)
        return text

    def get_preprocessing_stages(self, text: str) -> List[Dict]:
        """Return every intermediate preprocessing stage for visualisation.

        Args:
            text: Raw input text.

        Returns:
            List of dicts, each with keys:
              - ``stage_name``  (str)
              - ``text``        (str) – text after this stage
              - ``token_count`` (int)
              - ``char_count``  (int)
        """
        stages: List[Dict] = []

        def _record(name: str, t: str) -> str:
            stages.append(
                {
                    "stage_name": name,
                    "text": t,
                    "token_count": len(t.split()) if t else 0,
                    "char_count": len(t),
                }
            )
            return t

        current = _record("raw", text)

        current = _record("html_removed", self._remove_html(current))
        current = _record("url_removed", self._remove_urls(current))
        current = _record("lowercased", current.lower())
        current = _record(
            "special_chars_removed", self._remove_special_chars(current)
        )

        tokens = self._tokenize(self._normalise_whitespace(current))
        current = _record("tokenized", " ".join(tokens))

        tokens_no_sw = self._remove_stopwords(tokens)
        current = _record("stopwords_removed", " ".join(tokens_no_sw))

        tokens_lemma = self._lemmatize(tokens_no_sw)
        current = _record("lemmatized", " ".join(tokens_lemma))  # noqa: F841

        return stages

    def get_word_count(self, text: str) -> int:
        """Return the number of whitespace-separated tokens in *text*.

        Args:
            text: Any string.

        Returns:
            Integer word count (0 for empty/None input).
        """
        if not text or not text.strip():
            return 0
        return len(text.split())

    def get_char_count(self, text: str) -> int:
        """Return the total number of characters in *text*.

        Args:
            text: Any string.

        Returns:
            Integer character count (0 for None input).
        """
        return len(text) if text else 0

    def get_keywords(self, text: str, top_n: int = 10) -> List[str]:
        """Extract the most significant content words from *text*.

        Uses TF-IDF-style heuristics on a single document: after applying
        the full TF-IDF preprocessing pipeline the remaining tokens are
        ranked by frequency and the top-N are returned.

        Args:
            text: Raw input text.
            top_n: Maximum number of keywords to return.

        Returns:
            Ordered list of keyword strings (most frequent first).
        """
        if not text or not text.strip():
            return []

        preprocessed = self.preprocess_for_tfidf(text)
        tokens = preprocessed.split()
        if not tokens:
            return []

        # Rank by frequency, then alphabetically for stable ordering
        from collections import Counter

        freq: Counter = Counter(tokens)
        keywords = [word for word, _ in freq.most_common(top_n)]
        return keywords
