# 📰 Automatic News Topic Classification

> **CS459 — Natural Language Processing Project**  
> SVNIT Surat | B.Tech IV CSE | Academic Year 2025-26

An end-to-end NLP system that automatically classifies news articles into categories using both traditional machine learning (TF-IDF + Logistic Regression) and transformer-based (DistilBERT) approaches. Includes a full-stack web application for interactive demonstration.

---

## 🎯 Features

- **Real-time news classification** via an interactive web UI
- **Two ML models**: TF-IDF + Logistic Regression vs. DistilBERT
- **Confidence scores** and probability distributions across all categories
- **Uncertainty detection** for ambiguous articles
- **Preprocessing visualizer** — see NLP pipeline step-by-step
- **Dataset insights dashboard** — AG News statistics and visualizations
- **Model performance dashboard** — side-by-side model comparison
- **Explainability** — top TF-IDF keywords driving each prediction
- **Sample articles** for each category for easy demonstration

---

## 🏷️ Categories

| ID | Category | Examples |
|----|----------|---------|
| 0 | 🌍 World | UN sessions, international conflicts, diplomacy |
| 1 | ⚽ Sports | Football, cricket, Olympics, match results |
| 2 | 💼 Business | Stock market, company earnings, economy |
| 3 | 💻 Sci/Tech | AI research, software, hardware, space |

*(AG News dataset — balanced 4-class benchmark)*

---

## 🏗️ Architecture

```
┌─────────────────────┐
│   React Frontend    │  http://localhost:5173
│   (Vite + Tailwind) │
└─────────┬───────────┘
          │ HTTP REST API
          ▼
┌─────────────────────┐
│   FastAPI Backend   │  http://localhost:8000
│   (Python 3.12)     │
└─────────┬───────────┘
          │
    ┌─────┴──────┐
    ▼            ▼
┌────────┐  ┌─────────────┐
│ TF-IDF │  │  DistilBERT │
│ + LR   │  │ (Transformer)│
└────────┘  └─────────────┘
    │            │
    └─────┬──────┘
          ▼
   Post-processing
   (confidence, labels,
    keywords, formatting)
```

---

## 📦 Technology Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React 18, Vite, Tailwind CSS, Recharts |
| Backend API | FastAPI 0.115+, Uvicorn, Pydantic |
| ML Baseline | scikit-learn (TF-IDF, LogisticRegression) |
| ML Advanced | HuggingFace Transformers (DistilBERT), PyTorch |
| NLP | NLTK (tokenization, stopwords, lemmatization) |
| Data | HuggingFace Datasets (AG News), pandas, NumPy |
| Visualization | matplotlib, seaborn, WordCloud, Recharts |
| Testing | pytest, httpx |

---

## 📁 Project Structure

```
automatic-news-topic-classification/
│
├── frontend/                    # React + Vite web application
│   ├── src/
│   │   ├── pages/               # HomePage, ClassifyPage, DatasetPage, ...
│   │   ├── components/          # Navbar, StatCard, ProbabilityBar, ...
│   │   └── services/api.js      # Axios API client
│   ├── package.json
│   └── README.md
│
├── backend/                     # FastAPI backend
│   ├── app/
│   │   ├── main.py              # FastAPI app entry point
│   │   ├── config.py            # Configuration settings
│   │   ├── routes/              # predict.py, data.py, metrics.py
│   │   ├── services/            # prediction_service.py, dataset_service.py
│   │   ├── models/              # classifier.py
│   │   └── preprocessing/      # text_preprocessor.py
│   ├── requirements.txt
│   └── README.md
│
├── data/
│   ├── raw/                     # train.csv, test.csv (AG News)
│   └── processed/               # Preprocessed CSVs, stats.json, examples.json
│
├── models/
│   ├── tfidf_vectorizer.pkl     # Trained TF-IDF vectorizer
│   ├── logistic_regression.pkl  # Trained LR classifier
│   ├── transformer/             # DistilBERT fine-tuned model
│   └── evaluation_results.json  # Test set metrics
│
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_baseline_model.ipynb
│   ├── 04_transformer_model.ipynb
│   └── 05_result_analysis.ipynb
│
├── scripts/
│   ├── download_dataset.py      # Download AG News + generate stats
│   ├── preprocess.py            # Preprocessing pipeline
│   ├── train_baseline.py        # Train TF-IDF + LR
│   ├── train_transformer.py     # Fine-tune DistilBERT
│   ├── evaluate.py              # Evaluate + generate visualizations
│   └── run_pipeline.py          # Master pipeline script
│
├── visualizations/              # Generated plots (PNG)
├── docs/                        # Project documentation
│   ├── project_requirements.md
│   ├── related_work.md
│   ├── dataset_exploration.md
│   ├── preprocessing.md
│   ├── implementation.md
│   ├── postprocessing.md
│   └── bibliography.md
│
├── tests/
│   ├── test_preprocessing.py
│   └── test_api.py
│
├── .env.example
├── .gitignore
├── docker-compose.yml
└── README.md  ← you are here
```

---

## 🚀 Installation & Setup

### Prerequisites

- Python 3.10+
- Node.js 18+ and npm
- Git
- ~4 GB free disk space (for model + dataset)

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd automatic-news-topic-classification
```

### 2. Set up Python environment (recommended: virtual env)

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

pip install -r backend/requirements.txt
```

### 3. Download NLTK data

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger'); nltk.download('punkt_tab')"
```

---

## 🤖 Training the Models

### Option A: Run the full pipeline (recommended)

```bash
python scripts/run_pipeline.py
```

This will:
1. Download AG News dataset (requires internet, ~50MB)
2. Preprocess text data
3. Train TF-IDF + Logistic Regression
4. Evaluate and generate visualization plots

**With transformer training (slow on CPU — ~2-3 hours):**
```bash
python scripts/run_pipeline.py --with-transformer
```

### Option B: Run steps individually

```bash
# Step 1: Download dataset
python scripts/download_dataset.py

# Step 2: Preprocess
python scripts/preprocess.py

# Step 3: Train baseline
python scripts/train_baseline.py

# Step 4: (Optional) Train transformer
python scripts/train_transformer.py

# Step 5: Evaluate all models
python scripts/evaluate.py
```

---

## 🖥️ Running the Application

### Start the Backend

```bash
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Backend will be available at: http://localhost:8000  
API documentation: http://localhost:8000/docs

### Start the Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend will be available at: http://localhost:5173

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Backend health check |
| POST | `/api/predict` | Classify a news article |
| POST | `/api/preprocess` | Show preprocessing stages |
| GET | `/api/dataset/stats` | Dataset statistics |
| GET | `/api/examples` | Sample articles per category |
| GET | `/api/model/metrics` | Model evaluation metrics |
| GET | `/api/categories` | List of categories |

### Example: Classify a news article

```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "headline": "India wins cricket World Cup",
    "article": "India defeated South Africa in a thrilling final to win the ICC Cricket World Cup 2024 at the Wankhede Stadium in Mumbai.",
    "model_type": "tfidf_lr"
  }'
```

**Response:**
```json
{
  "predicted_class": "Sports",
  "confidence": 0.965,
  "certainty": "high",
  "message": null,
  "probabilities": {
    "Sports": 0.965,
    "World": 0.018,
    "Business": 0.010,
    "Sci/Tech": 0.007
  },
  "keywords": [
    {"word": "cricket", "weight": 3.21},
    {"word": "world cup", "weight": 2.87}
  ],
  "word_count": 28,
  "character_count": 186,
  "processing_time_ms": 12,
  "model_used": "tfidf_lr"
}
```

---

## 🧪 Running Tests

```bash
# From project root
pytest tests/ -v
```

---

## 📊 Example Prediction

**Input:**  
"Apple announces M4 chip with breakthrough AI performance. The new silicon outperforms competitors in machine learning benchmarks."

**Output:**  
Predicted Category: **Sci/Tech** (94.2% confidence)

| Category | Probability |
|----------|------------|
| Sci/Tech | 94.2% |
| Business | 3.1% |
| World | 1.8% |
| Sports | 0.9% |

Important keywords: `apple`, `chip`, `ai`, `machine learning`, `silicon`

---

## 📈 Model Performance

*(Actual values from `models/evaluation_results.json` after running evaluate.py)*

| Metric | TF-IDF + LR | DistilBERT |
|--------|-------------|-----------|
| Accuracy | ~92% | ~95% |
| Macro F1 | ~92% | ~95% |
| Inference time | <5ms | ~200ms |

*Note: Values are approximate — run evaluate.py for exact results.*

---

## 📚 Dataset

**AG News** (Xiang Zhang et al., 2015)
- Source: HuggingFace `datasets` library
- 120,000 training articles, 7,600 test articles
- 4 balanced classes: World, Sports, Business, Sci/Tech
- Widely used NLP benchmark

---

## 🔮 Future Improvements

- Multilingual news classification (Hindi, Spanish, etc.)
- Real-time news feed integration
- News summarization
- Fake news detection
- Named Entity Recognition overlay
- Topic modeling (LDA)
- Cloud deployment (AWS/GCP)
- Mobile application

---

## 👥 Team

| Field | Details |
|-------|---------|
| Student | [Your Name] |
| Roll Number | [Your Roll Number] |
| Course | CS459 — Natural Language Processing |
| Institution | SVNIT Surat |
| Academic Year | 2025-26 |
| Supervisor | [Professor Name] |

---

## 📄 License

This project is developed for academic purposes as part of the CS459 NLP course at SVNIT Surat.

---

## 🙏 Acknowledgements

- AG News dataset: Xiang Zhang, Junbo Zhao, Yann LeCun (2015)
- HuggingFace Transformers library
- scikit-learn contributors
- NLTK project
#   n l p - p r o j e c t  
 