# Implementation Documentation

## Architecture Overview

```
automatic-news-topic-classification/
├── frontend/           React + Vite + Tailwind (User Interface)
├── backend/            FastAPI (REST API + ML inference)
├── data/               Dataset storage
├── models/             Trained model artifacts
├── notebooks/          Jupyter notebooks (EDA, training, analysis)
├── scripts/            Training and evaluation scripts
├── visualizations/     Generated plots
├── docs/               Project documentation
└── tests/              pytest test suite
```

---

## Backend Implementation

### Entry Point: `backend/app/main.py`
- Creates FastAPI application
- Configures CORS
- Registers API routers
- Loads models on startup

### API Routes

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/health | Backend health check |
| POST | /api/predict | Classify news article |
| POST | /api/preprocess | Show preprocessing stages |
| GET | /api/dataset/stats | Dataset statistics |
| GET | /api/examples | Sample articles per category |
| GET | /api/model/metrics | Model evaluation results |
| GET | /api/categories | List of categories |

### Preprocessing Module (`backend/app/preprocessing/text_preprocessor.py`)

The `TextPreprocessor` class implements:
- `preprocess_for_tfidf()` — Full NLP preprocessing pipeline
- `preprocess_for_transformer()` — Light cleaning for DistilBERT
- `get_preprocessing_stages()` — Step-by-step breakdown for UI

### Classifier Module (`backend/app/models/classifier.py`)

The `NewsClassifier` class:
- Loads TF-IDF vectorizer + Logistic Regression from `models/`
- Loads DistilBERT from `models/transformer/`
- `predict()` — Returns full prediction including probabilities
- `get_feature_importance()` — TF-IDF feature weights for explanation

---

## TF-IDF + Logistic Regression Implementation

### Feature Extraction
```python
TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)
```

### Classifier
```python
LogisticRegression(
    C=1.0,
    max_iter=1000,
    solver='lbfgs',
    multi_class='multinomial',
    random_state=42
)
```

### Why Logistic Regression?
- Provides probability outputs (needed for confidence scores)
- Fast inference
- Interpretable coefficients (feature importance)
- Strong baseline for text classification

---

## DistilBERT Implementation

### Model Configuration
```
Model: distilbert-base-uncased
Max sequence length: 128
Batch size: 32
Learning rate: 2e-5
Epochs: 3
Weight decay: 0.01
Warmup steps: 500
Optimizer: AdamW
```

### Why DistilBERT?
- 40% smaller than BERT, 60% faster inference
- Retains 97% of BERT performance
- Practical for student hardware
- Pre-trained on large English corpus
- Fine-tuning takes ~2-3 hours on CPU, ~30 minutes on GPU

---

## Post-processing Pipeline

1. **Label mapping:** Convert model output index → category name
   - 0 → "World", 1 → "Sports", 2 → "Business", 3 → "Sci/Tech"

2. **Probability extraction:** `softmax(logits)` → probability array

3. **Confidence check:**
   - confidence ≥ 0.70: high confidence prediction
   - 0.50 ≤ confidence < 0.70: moderate confidence
   - confidence < 0.50: uncertain prediction → add warning message

4. **Keyword extraction:** Top TF-IDF features for predicted class

5. **Response formatting:** JSON with all fields required by frontend

---

## Frontend Implementation

### Pages
| Route | Component | Description |
|-------|-----------|-------------|
| / | HomePage | Landing page with overview |
| /classify | ClassifyPage | Main classification interface |
| /dataset | DatasetPage | Dataset insights dashboard |
| /preprocessing | PreprocessingPage | Interactive preprocessing viewer |
| /performance | PerformancePage | Model comparison dashboard |
| /about | AboutPage | Project information |

### Key Dependencies
- React 18 + React Router v6
- Recharts (charts and visualizations)
- Tailwind CSS (styling)
- Axios (API calls)
- Lucide React (icons)
- React Hot Toast (notifications)

---

## Explainability

### TF-IDF Model
Feature importance is extracted from the Logistic Regression coefficient matrix.
For the predicted class c, the top-k features are those with highest `coef_[c]` values that are also present in the input text's TF-IDF representation.

This gives **interpretable keywords** that directly influenced the classification decision.

### DistilBERT
Attention-based explanations are not provided in the UI as attention weights are not a reliable explanation method (Jain & Wallace, 2019). Instead, simple keyword extraction (from TF-IDF analysis on the input text) is shown as a practical approximation.

---

## Evaluation Methodology

1. **Train set:** 96,000 samples (80% of AG News train)
2. **Validation set:** 24,000 samples (20% of AG News train)
3. **Test set:** 7,600 samples (original AG News test split — never seen during training)

### Metrics Computed
- Accuracy
- Macro F1 (equal weight per class)
- Weighted F1 (weighted by class frequency)
- Per-class precision, recall, F1
- Confusion matrix
- Inference time (ms per sample)

All results stored in `models/evaluation_results.json` and served via `/api/model/metrics`.
