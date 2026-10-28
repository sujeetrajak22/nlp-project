"""
train_transformer.py
--------------------
Fine-tunes DistilBERT on the AG News dataset using HuggingFace Trainer API.

Architecture : distilbert-base-uncased (66M parameters)
Task         : Sequence classification (4 classes)

Usage:
    python scripts/train_transformer.py

Inputs:
    data/processed/train_transformer.csv
    data/processed/val_transformer.csv

Outputs:
    models/transformer/                   (model checkpoint + config)
    models/transformer_results.json       (training metrics)
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import accuracy_score, f1_score

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
TRANSFORMER_DIR = MODELS_DIR / "transformer"

MODEL_NAME = "distilbert-base-uncased"
NUM_LABELS = 4
LABEL_NAMES = ["World", "Sports", "Business", "Sci/Tech"]
ID2LABEL = {i: name for i, name in enumerate(LABEL_NAMES)}
LABEL2ID = {name: i for i, name in enumerate(LABEL_NAMES)}

# Training hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 500
MAX_SEQ_LENGTH = 128  # AG News articles are short; 128 covers >99% of tokens

# ---------------------------------------------------------------------------
# Device Detection
# ---------------------------------------------------------------------------

def get_device() -> str:
    """Auto-detect and return the best available compute device."""
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        print(f"  GPU detected: {gpu_name}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        print("  Apple Silicon MPS detected.")
    else:
        device = "cpu"
        print("  No GPU found — training on CPU (this will be slow).")
    return device


# ---------------------------------------------------------------------------
# Data Loading & Tokenisation
# ---------------------------------------------------------------------------

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load light-cleaned transformer CSVs.

    Returns
    -------
    train_df, val_df : pd.DataFrame
        DataFrames with 'text_transformer' and 'label' columns.

    Raises
    ------
    FileNotFoundError
        If preprocessed files are missing.
    """
    train_path = DATA_PROCESSED_DIR / "train_transformer.csv"
    val_path = DATA_PROCESSED_DIR / "val_transformer.csv"

    for p in (train_path, val_path):
        if not p.exists():
            raise FileNotFoundError(
                f"Missing: {p}\nRun preprocess.py first."
            )

    print(f"  Loading {train_path.name}")
    train_df = pd.read_csv(train_path)
    print(f"  Loading {val_path.name}")
    val_df = pd.read_csv(val_path)

    # Ensure text is string
    train_df["text_transformer"] = train_df["text_transformer"].fillna("").astype(str)
    val_df["text_transformer"] = val_df["text_transformer"].fillna("").astype(str)

    return train_df, val_df


def tokenize_dataset(
    df: pd.DataFrame,
    tokenizer: AutoTokenizer,
    max_length: int = MAX_SEQ_LENGTH,
) -> Dataset:
    """
    Convert a pandas DataFrame to a tokenised HuggingFace Dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Input data with 'text_transformer' and 'label' columns.
    tokenizer : AutoTokenizer
        Pre-loaded tokenizer.
    max_length : int
        Maximum token sequence length (shorter texts are padded dynamically).

    Returns
    -------
    Dataset
        HuggingFace dataset ready for Trainer.
    """
    hf_dataset = Dataset.from_pandas(
        df[["text_transformer", "label"]].rename(columns={"text_transformer": "text"})
    )

    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding=False,  # Dynamic padding via DataCollatorWithPadding
        )

    tokenized = hf_dataset.map(
        tokenize_fn,
        batched=True,
        desc="Tokenising",
        remove_columns=["text"],
    )
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format("torch")
    return tokenized


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(eval_pred) -> dict:
    """
    Compute accuracy and F1 metrics for the HuggingFace Trainer.

    Parameters
    ----------
    eval_pred : EvalPrediction
        NamedTuple with fields: predictions (logits), label_ids.

    Returns
    -------
    dict
        Dictionary with 'accuracy', 'macro_f1', and 'weighted_f1'.
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro")),
        "weighted_f1": float(f1_score(labels, preds, average="weighted")),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Full transformer fine-tuning pipeline."""
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    TRANSFORMER_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  DistilBERT Fine-Tuning — AG News Classification")
    print("=" * 60)

    # 1. Detect device
    print("\n[0] Device detection…")
    device = get_device()

    # 2. Load data
    print("\n[1] Loading data…")
    train_df, val_df = load_data()
    print(f"    Train: {len(train_df):,}  |  Val: {len(val_df):,}")

    # 3. Load tokenizer & model
    print(f"\n[2] Loading tokenizer & model: {MODEL_NAME}…")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    Total parameters     : {total_params:,}")
    print(f"    Trainable parameters : {trainable_params:,}")

    # 4. Tokenise
    print("\n[3] Tokenising datasets…")
    train_dataset = tokenize_dataset(train_df, tokenizer)
    val_dataset = tokenize_dataset(val_df, tokenizer)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 5. Training arguments
    use_fp16 = device == "cuda"
    training_args = TrainingArguments(
        output_dir=str(TRANSFORMER_DIR),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=WARMUP_STEPS,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        fp16=use_fp16,
        logging_steps=50,
        logging_dir=str(TRANSFORMER_DIR / "logs"),
        seed=RANDOM_SEED,
        report_to="none",  # Disable wandb / tensorboard
        dataloader_num_workers=0,  # Avoids Windows multiprocessing issues
    )

    print(f"\n[4] Training configuration:")
    print(f"    Epochs          : {NUM_EPOCHS}")
    print(f"    Batch size      : {BATCH_SIZE}")
    print(f"    Learning rate   : {LEARNING_RATE}")
    print(f"    Weight decay    : {WEIGHT_DECAY}")
    print(f"    Warmup steps    : {WARMUP_STEPS}")
    print(f"    Max seq length  : {MAX_SEQ_LENGTH}")
    print(f"    FP16 training   : {use_fp16}")
    print(f"    Device          : {device}")

    # 6. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # 7. Train
    print("\n[5] Starting training…")
    t0 = time.time()
    train_result = trainer.train()
    training_time = time.time() - t0
    print(f"\n    Training complete in {training_time / 60:.1f} minutes.")

    # 8. Final evaluation
    print("\n[6] Running final evaluation…")
    eval_result = trainer.evaluate()
    print(f"    Val accuracy  : {eval_result.get('eval_accuracy', 0):.4f}")
    print(f"    Val macro F1  : {eval_result.get('eval_macro_f1', 0):.4f}")

    # 9. Save model & tokenizer
    print(f"\n[7] Saving model → {TRANSFORMER_DIR}")
    trainer.save_model(str(TRANSFORMER_DIR))
    tokenizer.save_pretrained(str(TRANSFORMER_DIR))
    print("    ✓ Model saved.")

    # 10. Save training results
    history = []
    for log in trainer.state.log_history:
        if "eval_accuracy" in log:
            history.append({
                "epoch": log.get("epoch"),
                "eval_accuracy": log.get("eval_accuracy"),
                "eval_macro_f1": log.get("eval_macro_f1"),
                "eval_weighted_f1": log.get("eval_weighted_f1"),
                "eval_loss": log.get("eval_loss"),
            })

    results = {
        "model": MODEL_NAME,
        "num_labels": NUM_LABELS,
        "categories": LABEL_NAMES,
        "val_accuracy": eval_result.get("eval_accuracy"),
        "val_macro_f1": eval_result.get("eval_macro_f1"),
        "val_weighted_f1": eval_result.get("eval_weighted_f1"),
        "val_loss": eval_result.get("eval_loss"),
        "training_history": history,
        "training_metadata": {
            "num_epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "warmup_steps": WARMUP_STEPS,
            "max_seq_length": MAX_SEQ_LENGTH,
            "fp16": use_fp16,
            "device": device,
            "total_train_time_s": round(training_time, 2),
            "total_train_time_min": round(training_time / 60, 2),
            "train_size": len(train_df),
            "val_size": len(val_df),
            "total_params": total_params,
            "trainable_params": trainable_params,
            "train_samples_per_second": train_result.metrics.get("train_samples_per_second"),
            "random_seed": RANDOM_SEED,
        },
    }

    results_path = MODELS_DIR / "transformer_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"    ✓ Results → {results_path}")

    print("\n✅ DistilBERT fine-tuning complete!")
    print(f"   Best val accuracy : {eval_result.get('eval_accuracy', 0):.4f}")
    print(f"   Best val macro F1 : {eval_result.get('eval_macro_f1', 0):.4f}")


if __name__ == "__main__":
    main()
