"""
test_preprocessing.py
---------------------
pytest tests for the text preprocessing pipeline.
"""

import sys
from pathlib import Path

# Ensure the backend package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

import pytest
from app.preprocessing.text_preprocessor import TextPreprocessor


@pytest.fixture(scope="module")
def preprocessor():
    """Shared TextPreprocessor instance."""
    return TextPreprocessor()


class TestTfidfPreprocessing:
    """Tests for the TF-IDF (aggressive) preprocessing pipeline."""

    def test_lowercasing(self, preprocessor):
        """All output should be lowercase."""
        result = preprocessor.preprocess_for_tfidf("Apple REPORTS Record Revenue")
        assert result == result.lower(), "Output should be fully lowercase"

    def test_html_removal(self, preprocessor):
        """HTML tags should be stripped."""
        result = preprocessor.preprocess_for_tfidf("<b>Breaking news</b> today")
        assert "<" not in result and ">" not in result

    def test_url_removal(self, preprocessor):
        """URLs should be removed."""
        result = preprocessor.preprocess_for_tfidf(
            "Visit https://www.bbc.com for more details"
        )
        assert "https" not in result
        assert "bbc.com" not in result

    def test_special_character_removal(self, preprocessor):
        """Special characters and punctuation should be removed."""
        result = preprocessor.preprocess_for_tfidf("Apple's revenue: $119B!")
        assert "$" not in result
        assert ":" not in result
        assert "!" not in result

    def test_stopword_removal(self, preprocessor):
        """Common stopwords should not appear in TF-IDF output."""
        stopwords_to_check = ["the", "is", "and", "of", "in", "a", "an"]
        result = preprocessor.preprocess_for_tfidf(
            "The company is in the process of reporting its quarterly revenue"
        )
        tokens = result.split()
        for sw in stopwords_to_check:
            assert sw not in tokens, f"Stopword '{sw}' should have been removed"

    def test_lemmatization(self, preprocessor):
        """Inflected forms should be lemmatized."""
        result = preprocessor.preprocess_for_tfidf("companies are reporting revenues")
        tokens = result.split()
        # 'companies' → 'company', 'reporting' → 'report', 'revenues' → 'revenue'
        assert "compani" in tokens or "company" in tokens, "Should lemmatize 'companies'"

    def test_empty_input(self, preprocessor):
        """Empty / whitespace-only input should return empty string."""
        assert preprocessor.preprocess_for_tfidf("") == ""
        assert preprocessor.preprocess_for_tfidf("   ") == ""
        assert preprocessor.preprocess_for_tfidf("\n\t") == ""

    def test_non_empty_output(self, preprocessor):
        """Normal article text should produce non-empty output."""
        result = preprocessor.preprocess_for_tfidf(
            "Apple reports record quarterly revenue driven by iPhone sales"
        )
        assert len(result) > 0


class TestTransformerPreprocessing:
    """Tests for the transformer (light) preprocessing pipeline."""

    def test_html_removal(self, preprocessor):
        """HTML tags should be stripped."""
        result = preprocessor.preprocess_for_transformer("<p>Breaking news</p> today")
        assert "<" not in result

    def test_preserves_casing(self, preprocessor):
        """Transformer pipeline should preserve mixed casing."""
        result = preprocessor.preprocess_for_transformer("Apple Reports Revenue")
        assert "Apple" in result, "Should preserve capitalisation"

    def test_url_removal(self, preprocessor):
        """URLs should be removed."""
        result = preprocessor.preprocess_for_transformer(
            "Visit https://example.com for details"
        )
        assert "https" not in result

    def test_whitespace_normalised(self, preprocessor):
        """Multiple spaces/newlines should be collapsed."""
        result = preprocessor.preprocess_for_transformer("Hello   \n\n  world")
        assert "  " not in result, "Multiple spaces should be collapsed"

    def test_empty_input(self, preprocessor):
        """Empty input should return empty string."""
        assert preprocessor.preprocess_for_transformer("") == ""


class TestPreprocessingStages:
    """Tests for the get_preprocessing_stages() method."""

    def test_returns_list(self, preprocessor):
        """Should return a list of dicts."""
        stages = preprocessor.get_preprocessing_stages("Hello world")
        assert isinstance(stages, list)
        assert len(stages) > 0

    def test_stage_keys(self, preprocessor):
        """Each stage dict should have the required keys."""
        stages = preprocessor.get_preprocessing_stages("Test article text here")
        required_keys = {"stage_name", "text", "token_count", "char_count"}
        for stage in stages:
            assert required_keys.issubset(stage.keys()), \
                f"Stage {stage.get('stage_name')} missing keys"

    def test_first_stage_is_raw(self, preprocessor):
        """First stage should be the raw text."""
        text = "Original text"
        stages = preprocessor.get_preprocessing_stages(text)
        assert stages[0]["stage_name"] == "raw"
        assert stages[0]["text"] == text

    def test_stage_order(self, preprocessor):
        """Stages should appear in the documented order."""
        stages = preprocessor.get_preprocessing_stages("Test article with HTML<br>")
        names = [s["stage_name"] for s in stages]
        assert names.index("raw") < names.index("html_removed")
        assert names.index("html_removed") < names.index("url_removed")


class TestUtilityMethods:
    """Tests for word_count, char_count, and get_keywords."""

    def test_word_count(self, preprocessor):
        assert preprocessor.get_word_count("Hello world today") == 3
        assert preprocessor.get_word_count("") == 0
        assert preprocessor.get_word_count("   ") == 0

    def test_char_count(self, preprocessor):
        assert preprocessor.get_char_count("Hello") == 5
        assert preprocessor.get_char_count("") == 0

    def test_get_keywords_returns_list(self, preprocessor):
        result = preprocessor.get_keywords(
            "Apple reports record quarterly revenue from iPhone sales and services"
        )
        assert isinstance(result, list)

    def test_get_keywords_empty_input(self, preprocessor):
        result = preprocessor.get_keywords("")
        assert result == []
