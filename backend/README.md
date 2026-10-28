# Automatic News Topic Classification – Backend

FastAPI backend for classifying news articles into one of four AG News categories:
**World**, **Sports**, **Business**, and **Sci/Tech**.

---

## Project Structure

```
backend/
├── app/
│   ├── config.py               # Centralised settings (Pydantic Settings)
│   ├── main.py                 # FastAPI application entry-point
│   ├── models/
│   │   └── classifier.py       # NewsClassifier (TF-IDF + LR, DistilBERT)
│   ├── preprocessing/
│   │   └── text_preprocessor.py
│   ├── routes/
│   │   ├── data.py             # Dataset & category routes
│   │   ├── metrics.py          # Metrics & health-check routes
│   │   └── predict.py          # Prediction & preprocessing routes
│   └── services/
│       ├── dataset_service.py
│       ├── metrics_service.py
│       └── prediction_service.py
├── .env.example
├── README.md
└── requirements.txt
```

---

## Prerequisites

- Python 3.10 or newer
- (Optional) CUDA-capable GPU for DistilBERT inference

---

## Setup

### 1. Create and activate a virtual environment

```bash
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# macOS / Linux
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r backend/requirements.txt
```

### 3. Copy environment file

```bash
cp backend/.env.example backend/.env
```

Edit `backend/.env` to override any defaults if needed.

---

## Train the models first

The API starts successfully even without trained models (it returns helpful
error messages), but you must train them before making prediction requests.

```bash
# From the project root:
python scripts/download_dataset.py   # download & preprocess AG News
python scripts/train_baseline.py     # train TF-IDF + LR
python scripts/evaluate.py           # generate evaluation_results.json

# Optional – fine-tune DistilBERT (requires GPU recommended):
python scripts/train_transformer.py
```

---

## Running the Server

```bash
# From the project root:
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000

# Or from inside the backend/ directory:
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The interactive API docs are available at:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Root info |
| POST | `/api/predict` | Classify a news item |
| POST | `/api/preprocess` | Inspect preprocessing pipeline |
| GET | `/api/dataset/stats` | Dataset statistics |
| GET | `/api/examples` | Example news items per category |
| GET | `/api/categories` | List of category names |
| GET | `/api/model/metrics` | Model evaluation metrics |
| GET | `/api/health` | Health check & model status |

### Example: Predict

```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "headline": "Apple reports record quarterly revenue",
    "article": "Strong iPhone 15 sales drove the result.",
    "model_type": "tfidf_lr"
  }'
```

### Example: Health check

```bash
curl http://localhost:8000/api/health
```

---

## Running Tests

```bash
# From the project root:
pytest backend/ -v
```

---

## Configuration

All settings live in `app/config.py` and can be overridden via environment
variables or the `backend/.env` file.

| Variable | Default | Description |
|----------|---------|-------------|
| `DEFAULT_MODEL` | `tfidf_lr` | Default model backend |
| `MAX_INPUT_LENGTH` | `10000` | Max combined text length (chars) |
| `CONFIDENCE_THRESHOLD` | `0.5` | Below this → uncertain flag |
