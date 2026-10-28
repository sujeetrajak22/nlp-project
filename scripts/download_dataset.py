"""
download_dataset.py
-------------------
Downloads the AG News dataset from HuggingFace, saves raw CSVs, and computes
comprehensive statistics for the dataset exploration phase.

Dataset: AG News (4 classes)
  - 0: World
  - 1: Sports
  - 2: Business
  - 3: Sci/Tech

Usage:
    python scripts/download_dataset.py

Outputs:
    data/raw/train.csv                  -- raw training split
    data/raw/test.csv                   -- raw test split
    data/processed/stats.json           -- comprehensive dataset statistics
    data/processed/examples.json        -- one real example per category
"""

import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"

LABEL_NAMES = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
NUM_HISTOGRAM_BINS = 10
TOP_WORDS_OVERALL = 20
TOP_WORDS_PER_CAT = 10

# Common English stop-words for top-word filtering (lightweight, no NLTK dep here)
_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "has", "have", "had", "it", "its", "that", "this", "as", "he", "she",
    "they", "we", "you", "i", "not", "no", "do", "did", "will", "would",
    "can", "could", "may", "might", "should", "up", "out", "if", "about",
    "said", "says", "also", "new", "more", "than", "after", "s", "their",
    "his", "her", "into", "over", "than", "then", "there", "who", "which",
}


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def ensure_dirs() -> None:
    """Create necessary output directories if they don't exist."""
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def download_ag_news() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Download the AG News dataset from HuggingFace.

    Returns
    -------
    train_df : pd.DataFrame
        Training split with columns ['text', 'label'].
    test_df : pd.DataFrame
        Test split with columns ['text', 'label'].

    Raises
    ------
    ConnectionError
        If the dataset cannot be downloaded.
    """
    print("Downloading AG News dataset from HuggingFace…")
    try:
        dataset = load_dataset("fancyzhx/ag_news")
    except Exception as exc:
        raise ConnectionError(
            f"Failed to download AG News dataset: {exc}\n"
            "Check your internet connection and try again."
        ) from exc

    print("  Converting to DataFrames…")
    train_df = pd.DataFrame(
        {"text": dataset["train"]["text"], "label": dataset["train"]["label"]}
    )
    test_df = pd.DataFrame(
        {"text": dataset["test"]["text"], "label": dataset["test"]["label"]}
    )
    return train_df, test_df


def save_csv(df: pd.DataFrame, path: Path, desc: str = "") -> None:
    """Save a DataFrame to CSV with a progress indicator."""
    print(f"  Saving {desc} -> {path} ...")
    df.to_csv(path, index=False, encoding="utf-8")
    print(f"    ✓ {len(df):,} rows written.")


def tokenize_text(text: str) -> list[str]:
    """
    Simple whitespace + punctuation tokenizer for statistics computation.

    Parameters
    ----------
    text : str
        Raw document string.

    Returns
    -------
    list[str]
        List of lowercase word tokens (alpha-only).
    """
    return [w for w in re.findall(r"\b[a-zA-Z]+\b", text.lower()) if len(w) > 1]


def compute_top_words(texts: list[str], n: int, exclude_stops: bool = True) -> list[str]:
    """
    Compute the top-n most frequent words across a list of documents.

    Parameters
    ----------
    texts : list[str]
        Collection of raw text strings.
    n : int
        Number of top words to return.
    exclude_stops : bool
        Whether to filter out common stop-words.

    Returns
    -------
    list[str]
        Ordered list of top words (most frequent first).
    """
    counter: Counter = Counter()
    for text in tqdm(texts, desc="    Counting words", leave=False):
        tokens = tokenize_text(text)
        if exclude_stops:
            tokens = [t for t in tokens if t not in _STOPWORDS]
        counter.update(tokens)
    return [word for word, _ in counter.most_common(n)]


def compute_histogram(values: list[float], n_bins: int) -> list[dict]:
    """
    Compute a histogram of numeric values.

    Parameters
    ----------
    values : list[float]
        Numeric values to bin.
    n_bins : int
        Number of histogram bins.

    Returns
    -------
    list[dict]
        Each element has keys 'bin_start', 'bin_end', and 'count'.
    """
    counts, bin_edges = np.histogram(values, bins=n_bins)
    return [
        {
            "bin_start": round(float(bin_edges[i]), 2),
            "bin_end": round(float(bin_edges[i + 1]), 2),
            "count": int(counts[i]),
        }
        for i in range(n_bins)
    ]


# ---------------------------------------------------------------------------
# Statistics Computation
# ---------------------------------------------------------------------------

def compute_stats(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    """
    Compute comprehensive statistics for the AG News dataset.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training split (columns: text, label).
    test_df : pd.DataFrame
        Test split (columns: text, label).

    Returns
    -------
    dict
        Nested dictionary with all statistics fields.
    """
    all_df = pd.concat([train_df, test_df], ignore_index=True)
    all_texts = all_df["text"].tolist()

    # --- Basic counts ---
    print("  Computing text lengths…")
    char_lengths = [len(t) for t in tqdm(all_texts, desc="    Char lengths", leave=False)]
    word_counts_list = [len(tokenize_text(t)) for t in tqdm(all_texts, desc="    Word counts", leave=False)]

    # --- Vocabulary ---
    print("  Building vocabulary…")
    vocab: set[str] = set()
    all_words: list[str] = []
    for text in tqdm(all_texts, desc="    Vocabulary", leave=False):
        tokens = tokenize_text(text)
        vocab.update(tokens)
        all_words.extend(tokens)

    # --- Label distribution (train + test separately) ---
    train_label_dist = train_df["label"].value_counts().sort_index().to_dict()
    test_label_dist = test_df["label"].value_counts().sort_index().to_dict()
    combined_dist = all_df["label"].value_counts().sort_index().to_dict()
    label_dist_named = {LABEL_NAMES[k]: v for k, v in combined_dist.items()}

    # --- Top words overall ---
    print("  Computing top words overall…")
    top_words = compute_top_words(all_texts, TOP_WORDS_OVERALL)

    # --- Top words per category ---
    print("  Computing top words per category…")
    top_words_per_cat: dict[str, list[str]] = {}
    for label_id, label_name in LABEL_NAMES.items():
        cat_texts = all_df[all_df["label"] == label_id]["text"].tolist()
        top_words_per_cat[label_name] = compute_top_words(cat_texts, TOP_WORDS_PER_CAT)

    # --- Text length histogram (chars) ---
    print("  Computing text length histogram…")
    hist = compute_histogram(char_lengths, NUM_HISTOGRAM_BINS)

    # --- Missing / duplicate checks ---
    missing = int(all_df["text"].isna().sum())
    duplicates = int(all_df["text"].duplicated().sum())

    stats = {
        "total_documents": len(all_df),
        "num_categories": len(LABEL_NAMES),
        "categories": list(LABEL_NAMES.values()),
        "train_size": len(train_df),
        "test_size": len(test_df),
        "label_distribution": label_dist_named,
        "train_label_distribution": {LABEL_NAMES[k]: v for k, v in train_label_dist.items()},
        "test_label_distribution": {LABEL_NAMES[k]: v for k, v in test_label_dist.items()},
        "avg_text_length": round(float(np.mean(char_lengths)), 2),
        "min_text_length": int(np.min(char_lengths)),
        "max_text_length": int(np.max(char_lengths)),
        "std_text_length": round(float(np.std(char_lengths)), 2),
        "total_words": int(np.sum(word_counts_list)),
        "avg_words": round(float(np.mean(word_counts_list)), 2),
        "min_words": int(np.min(word_counts_list)),
        "max_words": int(np.max(word_counts_list)),
        "vocabulary_size": len(vocab),
        "missing_values": missing,
        "duplicate_count": duplicates,
        "top_words": top_words,
        "top_words_per_category": top_words_per_cat,
        "text_length_distribution": hist,
    }
    return stats


# ---------------------------------------------------------------------------
# Examples extraction
# ---------------------------------------------------------------------------

def extract_examples(test_df: pd.DataFrame) -> dict:
    """
    Extract the first real example from each category in the test split.

    Parameters
    ----------
    test_df : pd.DataFrame
        Test split DataFrame (columns: text, label).

    Returns
    -------
    dict
        Mapping from category name → {'text': ..., 'label': ...}.
    """
    examples: dict = {}
    for label_id, label_name in LABEL_NAMES.items():
        subset = test_df[test_df["label"] == label_id]
        if len(subset) == 0:
            continue
        row = subset.iloc[0]
        examples[label_name] = {
            "text": row["text"],
            "label": int(row["label"]),
            "label_name": label_name,
        }
    return examples


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """End-to-end download and statistics pipeline."""
    np.random.seed(RANDOM_SEED)

    print("=" * 60)
    print("  AG News Dataset Downloader & Statistics Generator")
    print("=" * 60)

    ensure_dirs()

    # 1. Download dataset
    start = time.time()
    train_df, test_df = download_ag_news()
    elapsed = time.time() - start
    print(f"  Downloaded in {elapsed:.1f}s")

    # 2. Print basic info
    print("\n--- Dataset Info ---")
    print(f"  Train size : {len(train_df):,}")
    print(f"  Test size  : {len(test_df):,}")
    print("\n  Train label distribution:")
    for lbl, cnt in train_df["label"].value_counts().sort_index().items():
        print(f"    {LABEL_NAMES[lbl]:10s}: {cnt:,}")
    print("\n  Test label distribution:")
    for lbl, cnt in test_df["label"].value_counts().sort_index().items():
        print(f"    {LABEL_NAMES[lbl]:10s}: {cnt:,}")

    # 3. Save raw CSVs
    print("\n--- Saving Raw CSVs ---")
    save_csv(train_df, DATA_RAW_DIR / "train.csv", "train split")
    save_csv(test_df, DATA_RAW_DIR / "test.csv", "test split")

    # 4. Compute statistics
    print("\n--- Computing Statistics ---")
    stats = compute_stats(train_df, test_df)

    stats_path = DATA_PROCESSED_DIR / "stats.json"
    print(f"  Saving stats → {stats_path}")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print("    ✓ Stats saved.")

    # 5. Extract and save examples
    print("\n--- Extracting Examples ---")
    examples = extract_examples(test_df)
    examples_path = DATA_PROCESSED_DIR / "examples.json"
    with open(examples_path, "w", encoding="utf-8") as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)
    print(f"    ✓ Examples saved → {examples_path}")

    # 6. Print summary
    print("\n--- Summary ---")
    print(f"  Total documents : {stats['total_documents']:,}")
    print(f"  Categories      : {', '.join(stats['categories'])}")
    print(f"  Vocabulary size : {stats['vocabulary_size']:,}")
    print(f"  Avg text length : {stats['avg_text_length']:.1f} chars")
    print(f"  Avg word count  : {stats['avg_words']:.1f} words")
    print(f"  Missing values  : {stats['missing_values']}")
    print(f"  Duplicates      : {stats['duplicate_count']}")
    print(f"\n  Top words: {', '.join(stats['top_words'][:10])}")
    print("\n✅ Done! All outputs written to data/raw/ and data/processed/")


if __name__ == "__main__":
    try:
        main()
    except ConnectionError as e:
        print(f"\n❌ Network error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}", file=sys.stderr)
        raise
