"""
train_baseline.py
-----------------
Trains a TF-IDF + Logistic Regression baseline classifier on the AG News dataset.

Model choices:
  - TF-IDF  : max_features=50,000, ngram_range=(1,2), sublinear_tf=True
  - LogReg  : C=1.0, solver='lbfgs', multi_class='multinomial', max_iter=1000

Usage:
    python scripts/train_baseline.py

Inputs:
    data/processed/train_preprocessed.csv
    data/processed/val_preprocessed.csv

Outputs:
    models/tfidf_vectorizer.pkl
    models/logistic_regression.pkl
    models/baseline_results.json
"""

import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"

LABEL_NAMES = ["World", "Sports", "Business", "Sci/Tech"]

# TF-IDF hyperparameters
TFIDF_PARAMS = {
    "max_features": 50_000,
    "ngram_range": (1, 2),
    "min_df": 2,
    "max_df": 0.95,
    "sublinear_tf": True,
    "strip_accents": "unicode",
    "analyzer": "word",
    "token_pattern": r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
}

# Logistic Regression hyperparameters
LR_PARAMS = {
    "C": 1.0,
    "max_iter": 1000,
    "solver": "lbfgs",
    "multi_class": "multinomial",
    "random_state": RANDOM_SEED,
    "verbose": 0,
    "n_jobs": -1,
}


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load preprocessed train and validation splits.

    Returns
    -------
    train_df : pd.DataFrame
        Training data with 'text_tfidf' and 'label' columns.
    val_df : pd.DataFrame
        Validation data with 'text_tfidf' and 'label' columns.

    Raises
    ------
    FileNotFoundError
        If preprocessed files are missing.
    """
    train_path = DATA_PROCESSED_DIR / "train_preprocessed.csv"
    val_path = DATA_PROCESSED_DIR / "val_preprocessed.csv"

    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(
            "Preprocessed data not found. Run preprocess.py first.\n"
            f"  Expected: {train_path}\n  Expected: {val_path}"
        )

    print(f"  Loading train: {train_path}")
    train_df = pd.read_csv(train_path)
    print(f"  Loading val  : {val_path}")
    val_df = pd.read_csv(val_path)

    # Ensure text column contains strings (fill NaN with empty string)
    train_df["text_tfidf"] = train_df["text_tfidf"].fillna("").astype(str)
    val_df["text_tfidf"] = val_df["text_tfidf"].fillna("").astype(str)

    return train_df, val_df


# ---------------------------------------------------------------------------
# Model Training
# ---------------------------------------------------------------------------

def train_pipeline(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> tuple[TfidfVectorizer, LogisticRegression, dict]:
    """
    Fit TF-IDF vectorizer and Logistic Regression on training data,
    then evaluate on the validation set.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training split.
    val_df : pd.DataFrame
        Validation split.

    Returns
    -------
    vectorizer : TfidfVectorizer
        Fitted TF-IDF vectorizer.
    clf : LogisticRegression
        Fitted classifier.
    results : dict
        Validation metrics and training metadata.
    """
    X_train_raw = train_df["text_tfidf"].tolist()
    y_train = train_df["label"].tolist()
    X_val_raw = val_df["text_tfidf"].tolist()
    y_val = val_df["label"].tolist()

    # ---- TF-IDF Vectorisation ----
    print("\n[1] Fitting TF-IDF vectorizer…")
    vectorizer = TfidfVectorizer(**TFIDF_PARAMS)

    t0 = time.time()
    X_train = vectorizer.fit_transform(X_train_raw)
    tfidf_fit_time = time.time() - t0

    print(f"    Feature matrix shape : {X_train.shape}")
    print(f"    Actual vocabulary    : {len(vectorizer.vocabulary_):,}")
    print(f"    Fit time             : {tfidf_fit_time:.2f}s")

    t0 = time.time()
    X_val = vectorizer.transform(X_val_raw)
    tfidf_transform_time = time.time() - t0
    print(f"    Val transform time   : {tfidf_transform_time:.3f}s")

    # ---- Logistic Regression ----
    print("\n[2] Training Logistic Regression…")
    clf = LogisticRegression(**LR_PARAMS)

    t0 = time.time()
    clf.fit(X_train, y_train)
    lr_train_time = time.time() - t0
    print(f"    Training time        : {lr_train_time:.2f}s")
    print(f"    Iterations converged : {clf.n_iter_[0] if hasattr(clf, 'n_iter_') else 'N/A'}")

    # ---- Validation Evaluation ----
    print("\n[3] Evaluating on validation set…")
    t0 = time.time()
    y_pred = clf.predict(X_val)
    inference_time = (time.time() - t0) / len(X_val_raw) * 1000  # ms per sample

    val_accuracy = float(accuracy_score(y_val, y_pred))
    val_macro_f1 = float(f1_score(y_val, y_pred, average="macro"))
    val_weighted_f1 = float(f1_score(y_val, y_pred, average="weighted"))
    report = classification_report(y_val, y_pred, target_names=LABEL_NAMES, output_dict=True)

    results = {
        "model": "tfidf_logistic_regression",
        "val_accuracy": round(val_accuracy, 6),
        "val_macro_f1": round(val_macro_f1, 6),
        "val_weighted_f1": round(val_weighted_f1, 6),
        "classification_report": report,
        "inference_time_ms_per_sample": round(inference_time, 4),
        "tfidf_params": {
            "max_features": TFIDF_PARAMS["max_features"],
            "ngram_range": list(TFIDF_PARAMS["ngram_range"]),
            "min_df": TFIDF_PARAMS["min_df"],
            "max_df": TFIDF_PARAMS["max_df"],
            "sublinear_tf": TFIDF_PARAMS["sublinear_tf"],
            "actual_vocabulary_size": len(vectorizer.vocabulary_),
        },
        "lr_params": {
            "C": LR_PARAMS["C"],
            "solver": LR_PARAMS["solver"],
            "max_iter": LR_PARAMS["max_iter"],
            "multi_class": LR_PARAMS["multi_class"],
        },
        "training_metadata": {
            "train_size": len(X_train_raw),
            "val_size": len(X_val_raw),
            "tfidf_fit_time_s": round(tfidf_fit_time, 3),
            "lr_train_time_s": round(lr_train_time, 3),
            "total_training_time_s": round(tfidf_fit_time + lr_train_time, 3),
            "random_seed": RANDOM_SEED,
        },
    }

    return vectorizer, clf, results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(results: dict) -> None:
    """Print a human-readable evaluation summary to stdout."""
    print("\n" + "=" * 60)
    print("  BASELINE MODEL — VALIDATION RESULTS")
    print("=" * 60)
    print(f"  Accuracy    : {results['val_accuracy']:.4f}  ({results['val_accuracy'] * 100:.2f}%)")
    print(f"  Macro F1    : {results['val_macro_f1']:.4f}")
    print(f"  Weighted F1 : {results['val_weighted_f1']:.4f}")
    print(f"  Inference   : {results['inference_time_ms_per_sample']:.3f} ms/sample")
    print()

    # Per-class metrics
    report = results["classification_report"]
    print(f"  {'Class':<15} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("  " + "-" * 55)
    for cls_name in LABEL_NAMES:
        if cls_name in report:
            r = report[cls_name]
            print(
                f"  {cls_name:<15} {r['precision']:>10.4f} {r['recall']:>10.4f} "
                f"{r['f1-score']:>10.4f} {int(r['support']):>10,}"
            )
    print("  " + "-" * 55)
    macro = report.get("macro avg", {})
    print(
        f"  {'Macro avg':<15} {macro.get('precision', 0):>10.4f} "
        f"{macro.get('recall', 0):>10.4f} {macro.get('f1-score', 0):>10.4f}"
    )
    print()
    meta = results["training_metadata"]
    print(f"  Training time : {meta['total_training_time_s']:.2f}s")
    print(f"  Train size    : {meta['train_size']:,}")
    print(f"  Val size      : {meta['val_size']:,}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Full training pipeline: load → vectorise → train → evaluate → save."""
    np.random.seed(RANDOM_SEED)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  TF-IDF + Logistic Regression Baseline Trainer")
    print("=" * 60)

    # 1. Load data
    print("\n[0] Loading data…")
    train_df, val_df = load_data()
    print(f"    Train: {len(train_df):,} samples")
    print(f"    Val  : {len(val_df):,} samples")

    # 2. Train
    vectorizer, clf, results = train_pipeline(train_df, val_df)

    # 3. Print report
    print_report(results)

    # 4. Save artefacts
    print("\n[4] Saving model artefacts…")
    vec_path = MODELS_DIR / "tfidf_vectorizer.pkl"
    with open(vec_path, "wb") as f:
        pickle.dump(vectorizer, f)
    print(f"    ✓ TF-IDF vectorizer → {vec_path}")

    clf_path = MODELS_DIR / "logistic_regression.pkl"
    with open(clf_path, "wb") as f:
        pickle.dump(clf, f)
    print(f"    ✓ Logistic Regression → {clf_path}")

    results_path = MODELS_DIR / "baseline_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"    ✓ Results → {results_path}")

    print("\n✅ Baseline training complete!")


if __name__ == "__main__":
    main()
