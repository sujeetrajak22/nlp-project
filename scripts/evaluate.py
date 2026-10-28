"""
evaluate.py
-----------
Evaluates trained models on the AG News test set and saves results to
models/evaluation_results.json.

Usage:
    python scripts/evaluate.py

Inputs:
    data/processed/test_preprocessed.csv   (TF-IDF text)
    data/processed/test_transformer.csv    (transformer text, optional)
    models/tfidf_vectorizer.pkl
    models/logistic_regression.pkl
    models/transformer/                    (optional)

Outputs:
    models/evaluation_results.json
    visualizations/confusion_matrix_tfidf.png
    visualizations/confusion_matrix_distilbert.png  (if available)
    visualizations/model_comparison.png
    visualizations/per_class_f1.png
"""

import json
import pickle
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
VIS_DIR = BASE_DIR / "visualizations"

LABEL_NAMES = ["World", "Sports", "Business", "Sci/Tech"]
LABEL_MAP = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_test_data():
    """Load test split (TF-IDF and transformer preprocessed)."""
    test_path = DATA_PROCESSED_DIR / "test_preprocessed.csv"
    if not test_path.exists():
        raise FileNotFoundError(
            f"Test data not found: {test_path}\n"
            "Run scripts/preprocess.py first."
        )
    df = pd.read_csv(test_path)
    df["text_tfidf"] = df["text_tfidf"].fillna("").astype(str)
    print(f"  Test samples: {len(df):,}")
    return df


# ---------------------------------------------------------------------------
# TF-IDF + LR Evaluation
# ---------------------------------------------------------------------------

def evaluate_tfidf(df: pd.DataFrame) -> dict | None:
    """Evaluate TF-IDF + Logistic Regression on the test set."""
    vec_path = MODELS_DIR / "tfidf_vectorizer.pkl"
    lr_path = MODELS_DIR / "logistic_regression.pkl"

    if not vec_path.exists() or not lr_path.exists():
        print("  ⚠ TF-IDF model not found. Skipping.")
        return None

    print("  Loading TF-IDF model…")
    with open(vec_path, "rb") as f:
        vectorizer = pickle.load(f)
    with open(lr_path, "rb") as f:
        clf = pickle.load(f)

    X_test = vectorizer.transform(df["text_tfidf"].tolist())
    y_test = df["label"].tolist()

    print("  Running inference…")
    t0 = time.time()
    y_pred = clf.predict(X_test)
    elapsed = time.time() - t0
    inference_ms = (elapsed / len(y_test)) * 1000

    acc = float(accuracy_score(y_test, y_pred))
    macro_f1 = float(f1_score(y_test, y_pred, average="macro"))
    weighted_f1 = float(f1_score(y_test, y_pred, average="weighted"))
    macro_prec = float(precision_score(y_test, y_pred, average="macro"))
    macro_rec = float(recall_score(y_test, y_pred, average="macro"))

    report = classification_report(
        y_test, y_pred, target_names=LABEL_NAMES, output_dict=True
    )
    cm = confusion_matrix(y_test, y_pred).tolist()

    print(f"  ✓ Accuracy: {acc:.4f}  Macro F1: {macro_f1:.4f}  Inference: {inference_ms:.3f}ms/sample")
    print(classification_report(y_test, y_pred, target_names=LABEL_NAMES))

    return {
        "accuracy": round(acc, 6),
        "macro_f1": round(macro_f1, 6),
        "weighted_f1": round(weighted_f1, 6),
        "macro_precision": round(macro_prec, 6),
        "macro_recall": round(macro_rec, 6),
        "classification_report": report,
        "confusion_matrix": cm,
        "inference_time_ms": round(inference_ms, 4),
        "test_size": len(y_test),
        "categories": LABEL_NAMES,
    }


# ---------------------------------------------------------------------------
# DistilBERT Evaluation
# ---------------------------------------------------------------------------

def evaluate_transformer(df: pd.DataFrame) -> dict | None:
    """Evaluate fine-tuned DistilBERT on the test set."""
    transformer_dir = MODELS_DIR / "transformer"
    if not transformer_dir.exists():
        print("  ⚠ Transformer model not found. Skipping.")
        return None

    # Load transformer preprocessed text if available
    trans_path = DATA_PROCESSED_DIR / "test_transformer.csv"
    if trans_path.exists():
        trans_df = pd.read_csv(trans_path)
        texts = trans_df["text_transformer"].fillna("").astype(str).tolist()
    else:
        texts = df["text_tfidf"].tolist()  # fallback

    y_test = df["label"].tolist()

    try:
        import torch
        from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

        print("  Loading DistilBERT model…")
        tokenizer = DistilBertTokenizerFast.from_pretrained(str(transformer_dir))
        model = DistilBertForSequenceClassification.from_pretrained(str(transformer_dir))
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()

        print(f"  Running inference on {device}…")
        batch_size = 32
        all_preds = []
        t0 = time.time()

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = tokenizer(
                batch,
                max_length=128,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.no_grad():
                logits = model(**enc).logits
            preds = logits.argmax(dim=-1).cpu().numpy().tolist()
            all_preds.extend(preds)
            if (i // batch_size) % 10 == 0:
                print(f"    Batch {i // batch_size + 1}/{len(texts) // batch_size + 1}…")

        elapsed = time.time() - t0
        inference_ms = (elapsed / len(y_test)) * 1000
        y_pred = all_preds

    except Exception as e:
        print(f"  ✗ Transformer evaluation failed: {e}")
        return None

    acc = float(accuracy_score(y_test, y_pred))
    macro_f1 = float(f1_score(y_test, y_pred, average="macro"))
    weighted_f1 = float(f1_score(y_test, y_pred, average="weighted"))
    macro_prec = float(precision_score(y_test, y_pred, average="macro"))
    macro_rec = float(recall_score(y_test, y_pred, average="macro"))

    report = classification_report(
        y_test, y_pred, target_names=LABEL_NAMES, output_dict=True
    )
    cm = confusion_matrix(y_test, y_pred).tolist()

    print(f"  ✓ Accuracy: {acc:.4f}  Macro F1: {macro_f1:.4f}  Inference: {inference_ms:.1f}ms/sample")
    print(classification_report(y_test, y_pred, target_names=LABEL_NAMES))

    return {
        "accuracy": round(acc, 6),
        "macro_f1": round(macro_f1, 6),
        "weighted_f1": round(weighted_f1, 6),
        "macro_precision": round(macro_prec, 6),
        "macro_recall": round(macro_rec, 6),
        "classification_report": report,
        "confusion_matrix": cm,
        "inference_time_ms": round(inference_ms, 2),
        "test_size": len(y_test),
        "categories": LABEL_NAMES,
    }


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

def plot_confusion_matrix(cm_data: list, title: str, save_path: Path) -> None:
    """Plot and save a confusion matrix heatmap."""
    cm = np.array(cm_data)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=LABEL_NAMES,
        yticklabels=LABEL_NAMES,
        ax=ax,
        linewidths=0.5,
    )
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {save_path}")


def plot_model_comparison(results: dict, save_path: Path) -> None:
    """Plot overall metric comparison bar chart."""
    metrics = ["accuracy", "macro_f1", "weighted_f1"]
    labels = ["Accuracy", "Macro F1", "Weighted F1"]
    models_available = list(results.keys())

    x = np.arange(len(metrics))
    width = 0.35 if len(models_available) > 1 else 0.5
    colors = ["#6366f1", "#a855f7"]

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (model_key, color) in enumerate(zip(models_available, colors)):
        vals = [results[model_key].get(m, 0) * 100 for m in metrics]
        offset = (i - 0.5 * (len(models_available) - 1)) * width
        bars = ax.bar(x + offset, vals, width, label=model_key.replace("_", " + ").title(),
                      color=color, alpha=0.85, edgecolor="white", linewidth=0.8)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.3,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Score (%)", fontsize=12)
    ax.set_title("Model Performance Comparison — AG News Test Set", fontsize=14, fontweight="bold")
    ax.set_ylim(70, 102)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {save_path}")


def plot_per_class_f1(results: dict, save_path: Path) -> None:
    """Plot per-class F1 comparison."""
    models_available = list(results.keys())
    colors = ["#6366f1", "#a855f7"]

    x = np.arange(len(LABEL_NAMES))
    width = 0.35 if len(models_available) > 1 else 0.5

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (model_key, color) in enumerate(zip(models_available, colors)):
        report = results[model_key].get("classification_report", {})
        f1s = [report.get(cat, {}).get("f1-score", 0) * 100 for cat in LABEL_NAMES]
        offset = (i - 0.5 * (len(models_available) - 1)) * width
        bars = ax.bar(x + offset, f1s, width,
                      label=model_key.replace("_", " + ").title(),
                      color=color, alpha=0.85, edgecolor="white")
        for bar, val in zip(bars, f1s):
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.3,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(LABEL_NAMES, fontsize=12)
    ax.set_ylabel("F1 Score (%)", fontsize=12)
    ax.set_title("Per-Class F1 Score — AG News Test Set", fontsize=14, fontweight="bold")
    ax.set_ylim(70, 102)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run full evaluation pipeline."""
    np.random.seed(RANDOM_SEED)
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Model Evaluation — AG News Test Set")
    print("=" * 60)

    # Load test data
    print("\n[0] Loading test data…")
    df = load_test_data()

    results = {}

    # Evaluate TF-IDF + LR
    print("\n[1] Evaluating TF-IDF + Logistic Regression…")
    tfidf_results = evaluate_tfidf(df)
    if tfidf_results:
        results["tfidf_lr"] = tfidf_results

    # Evaluate DistilBERT
    print("\n[2] Evaluating DistilBERT…")
    bert_results = evaluate_transformer(df)
    if bert_results:
        results["distilbert"] = bert_results

    if not results:
        print("\n❌ No models available for evaluation. Train at least one model first.")
        return

    # Save results
    results_path = MODELS_DIR / "evaluation_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Evaluation results saved → {results_path}")

    # Generate visualizations
    print("\n[3] Generating visualizations…")
    if "tfidf_lr" in results and results["tfidf_lr"].get("confusion_matrix"):
        plot_confusion_matrix(
            results["tfidf_lr"]["confusion_matrix"],
            "Confusion Matrix — TF-IDF + Logistic Regression",
            VIS_DIR / "confusion_matrix_tfidf.png",
        )

    if "distilbert" in results and results["distilbert"].get("confusion_matrix"):
        plot_confusion_matrix(
            results["distilbert"]["confusion_matrix"],
            "Confusion Matrix — DistilBERT",
            VIS_DIR / "confusion_matrix_distilbert.png",
        )

    if results:
        plot_model_comparison(results, VIS_DIR / "model_comparison.png")
        plot_per_class_f1(results, VIS_DIR / "per_class_f1.png")

    # Summary
    print("\n" + "=" * 60)
    print("  EVALUATION SUMMARY")
    print("=" * 60)
    for model_key, res in results.items():
        print(f"\n  {model_key.upper()}")
        print(f"    Accuracy    : {res['accuracy']:.4f} ({res['accuracy']*100:.2f}%)")
        print(f"    Macro F1    : {res['macro_f1']:.4f}")
        print(f"    Weighted F1 : {res['weighted_f1']:.4f}")
        print(f"    Inference   : {res['inference_time_ms']:.3f} ms/sample")

    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()
