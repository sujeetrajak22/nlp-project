"""
preprocess.py
-------------
Preprocessing pipeline for the AG News classification project.

Applies two preprocessing strategies:
  1. TF-IDF pipeline  : aggressive cleaning (lowercase, remove HTML/URLs,
                        punctuation, stop-words, lemmatization).
  2. Transformer pipeline : light cleaning (lowercase, remove HTML/URLs,
                            collapse whitespace) — keeps natural language
                            structure needed by BERT-style models.

Usage:
    python scripts/preprocess.py

Inputs:
    data/raw/train.csv
    data/raw/test.csv

Outputs:
    data/processed/train_preprocessed.csv   (TF-IDF cleaned)
    data/processed/val_preprocessed.csv
    data/processed/test_preprocessed.csv
    data/processed/train_transformer.csv    (light-cleaned)
    data/processed/val_transformer.csv
    data/processed/test_transformer.csv
"""

import re
import time
from pathlib import Path

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
VAL_SIZE = 0.20  # 20 % of train goes to validation
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"

LABEL_NAMES = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}

# ---------------------------------------------------------------------------
# NLTK bootstrap (download only if not already present)
# ---------------------------------------------------------------------------

def _download_nltk_resources() -> None:
    """Download required NLTK resources silently if not cached."""
    resources = [
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
        ("tokenizers/punkt", "punkt"),
    ]
    for path, name in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            print(f"  Downloading NLTK resource: {name}")
            nltk.download(name, quiet=True)


# ---------------------------------------------------------------------------
# TF-IDF Preprocessing
# ---------------------------------------------------------------------------

# Compile regex patterns once for performance
_RE_HTML = re.compile(r"<[^>]+>")
_RE_URL = re.compile(r"https?://\S+|www\.\S+")
_RE_NON_ALPHA = re.compile(r"[^a-zA-Z\s]")
_RE_WHITESPACE = re.compile(r"\s+")


def preprocess_for_tfidf(text: str, lemmatizer: WordNetLemmatizer, stop_words: set) -> str:
    """
    Full NLP preprocessing pipeline for TF-IDF features.

    Steps:
      1. Lowercase
      2. Remove HTML tags
      3. Remove URLs
      4. Remove punctuation and special characters
      5. Tokenize (whitespace split)
      6. Remove stop-words
      7. Lemmatize tokens
      8. Rejoin into a single string

    Parameters
    ----------
    text : str
        Raw document text.
    lemmatizer : WordNetLemmatizer
        Pre-initialised NLTK lemmatizer.
    stop_words : set
        Set of stop-word strings to filter out.

    Returns
    -------
    str
        Cleaned text ready for TF-IDF vectorisation.
    """
    # 1. Lowercase
    text = text.lower()
    # 2. Remove HTML
    text = _RE_HTML.sub(" ", text)
    # 3. Remove URLs
    text = _RE_URL.sub(" ", text)
    # 4. Remove non-alpha characters
    text = _RE_NON_ALPHA.sub(" ", text)
    # 5. Collapse whitespace
    text = _RE_WHITESPACE.sub(" ", text).strip()
    # 6. Tokenize, filter stop-words, lemmatize
    tokens = [
        lemmatizer.lemmatize(token)
        for token in text.split()
        if token not in stop_words and len(token) > 1
    ]
    return " ".join(tokens)


# ---------------------------------------------------------------------------
# Transformer Preprocessing (light-touch)
# ---------------------------------------------------------------------------

def preprocess_for_transformer(text: str) -> str:
    """
    Light preprocessing for transformer models.

    Transformer models (e.g. DistilBERT) benefit from natural-language
    input, so we only perform minimal cleaning:
      1. Remove HTML tags
      2. Remove URLs
      3. Normalise whitespace

    Casing is intentionally kept (DistilBERT-uncased lowercases internally).

    Parameters
    ----------
    text : str
        Raw document text.

    Returns
    -------
    str
        Lightly-cleaned text.
    """
    text = _RE_HTML.sub(" ", text)
    text = _RE_URL.sub(" ", text)
    text = _RE_WHITESPACE.sub(" ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Pipeline Helpers
# ---------------------------------------------------------------------------

def apply_preprocessing(
    df: pd.DataFrame,
    pipeline: str,
    lemmatizer: WordNetLemmatizer | None = None,
    stop_words: set | None = None,
) -> pd.DataFrame:
    """
    Apply a preprocessing pipeline to a DataFrame and return a new DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input data (must have 'text' and 'label' columns).
    pipeline : str
        Either 'tfidf' or 'transformer'.
    lemmatizer : WordNetLemmatizer, optional
        Required when pipeline='tfidf'.
    stop_words : set, optional
        Required when pipeline='tfidf'.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: text_<pipeline>, label, label_name.
    """
    col = f"text_{pipeline}"
    tqdm.pandas(desc=f"    Preprocessing [{pipeline}]")

    if pipeline == "tfidf":
        assert lemmatizer is not None and stop_words is not None
        processed = df["text"].progress_apply(
            lambda t: preprocess_for_tfidf(t, lemmatizer, stop_words)
        )
    else:
        processed = df["text"].progress_apply(preprocess_for_transformer)

    result = pd.DataFrame(
        {
            col: processed,
            "label": df["label"].values,
            "label_name": df["label"].map(LABEL_NAMES),
        }
    )
    return result


def compute_vocab_size(texts: pd.Series) -> int:
    """Return the number of unique whitespace-delimited tokens in a text series."""
    vocab: set = set()
    for text in texts:
        vocab.update(str(text).split())
    return len(vocab)


def avg_length(texts: pd.Series) -> float:
    """Return the mean word count across a text series."""
    return float(texts.apply(lambda t: len(str(t).split())).mean())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the full preprocessing pipeline."""
    np.random.seed(RANDOM_SEED)
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  AG News Preprocessing Pipeline")
    print("=" * 60)

    # 0. NLTK resources
    print("\n[0] Loading NLTK resources…")
    _download_nltk_resources()
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))
    print(f"    Stop-words loaded: {len(stop_words)}")

    # 1. Load raw data
    print("\n[1] Loading raw CSVs…")
    train_raw = pd.read_csv(DATA_RAW_DIR / "train.csv")
    test_raw = pd.read_csv(DATA_RAW_DIR / "test.csv")
    print(f"    Train rows : {len(train_raw):,}")
    print(f"    Test rows  : {len(test_raw):,}")

    # 2. Train / validation split (stratified)
    print(f"\n[2] Splitting train → train ({1 - VAL_SIZE:.0%}) / val ({VAL_SIZE:.0%})…")
    train_split, val_split = train_test_split(
        train_raw,
        test_size=VAL_SIZE,
        stratify=train_raw["label"],
        random_state=RANDOM_SEED,
    )
    train_split = train_split.reset_index(drop=True)
    val_split = val_split.reset_index(drop=True)
    print(f"    Train : {len(train_split):,}  |  Val : {len(val_split):,}")

    # 3. Statistics BEFORE preprocessing
    print("\n[3] Computing pre-processing statistics…")
    orig_vocab = compute_vocab_size(train_raw["text"])
    orig_avg_len = avg_length(train_raw["text"])
    print(f"    Original vocabulary size : {orig_vocab:,}")
    print(f"    Original avg word count  : {orig_avg_len:.1f}")

    # 4. TF-IDF preprocessing
    print("\n[4] TF-IDF preprocessing…")
    start = time.time()

    print("  → Train split:")
    train_tfidf = apply_preprocessing(train_split, "tfidf", lemmatizer, stop_words)
    print("  → Validation split:")
    val_tfidf = apply_preprocessing(val_split, "tfidf", lemmatizer, stop_words)
    print("  → Test split:")
    test_tfidf = apply_preprocessing(test_raw, "tfidf", lemmatizer, stop_words)

    elapsed_tfidf = time.time() - start
    print(f"    Done in {elapsed_tfidf:.1f}s")

    # 5. Transformer preprocessing
    print("\n[5] Transformer (light) preprocessing…")
    start = time.time()

    print("  → Train split:")
    train_transformer = apply_preprocessing(train_split, "transformer")
    print("  → Validation split:")
    val_transformer = apply_preprocessing(val_split, "transformer")
    print("  → Test split:")
    test_transformer = apply_preprocessing(test_raw, "transformer")

    elapsed_transformer = time.time() - start
    print(f"    Done in {elapsed_transformer:.1f}s")

    # 6. Save preprocessed files
    print("\n[6] Saving preprocessed files…")
    files = {
        "train_preprocessed.csv": train_tfidf,
        "val_preprocessed.csv": val_tfidf,
        "test_preprocessed.csv": test_tfidf,
        "train_transformer.csv": train_transformer,
        "val_transformer.csv": val_transformer,
        "test_transformer.csv": test_transformer,
    }
    for fname, df in files.items():
        path = DATA_PROCESSED_DIR / fname
        df.to_csv(path, index=False, encoding="utf-8")
        print(f"    ✓ {fname:35s} ({len(df):,} rows)")

    # 7. Statistics AFTER preprocessing
    print("\n[7] Post-processing statistics…")
    tfidf_vocab = compute_vocab_size(train_tfidf["text_tfidf"])
    tfidf_avg_len = avg_length(train_tfidf["text_tfidf"])
    transformer_vocab = compute_vocab_size(train_transformer["text_transformer"])
    transformer_avg_len = avg_length(train_transformer["text_transformer"])

    print(f"\n  {'Metric':<35} {'Before':>10} {'TF-IDF':>10} {'Transf.':>10}")
    print("  " + "-" * 65)
    print(f"  {'Vocabulary size':<35} {orig_vocab:>10,} {tfidf_vocab:>10,} {transformer_vocab:>10,}")
    print(f"  {'Avg word count':<35} {orig_avg_len:>10.1f} {tfidf_avg_len:>10.1f} {transformer_avg_len:>10.1f}")
    vocab_reduction = 100 * (1 - tfidf_vocab / orig_vocab)
    len_reduction = 100 * (1 - tfidf_avg_len / orig_avg_len)
    print(f"\n  TF-IDF vocabulary reduction : {vocab_reduction:.1f}%")
    print(f"  TF-IDF length reduction     : {len_reduction:.1f}%")

    print("\n✅ Preprocessing complete!")


if __name__ == "__main__":
    main()
