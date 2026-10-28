# Project Requirements — Automatic News Topic Classification

## 1. Introduction

News is produced at an unprecedented volume in the digital age. Manually categorizing every news article into topics such as World News, Sports, Business, or Technology is infeasible at scale. Automated news classification using Natural Language Processing (NLP) techniques can solve this problem efficiently.

This project implements an end-to-end NLP system for **Automatic News Topic Classification** as part of the CS459 Natural Language Processing course at SVNIT Surat.

---

## 2. Problem Statement

Given a news article (headline and/or body text), automatically determine which of the predefined topic categories it belongs to.

This is a **multi-class text classification problem** in NLP, where:
- **Input:** Raw news text (headline + article)
- **Output:** Predicted news category + confidence score

---

## 3. Motivation

- **Volume:** Millions of news articles published daily across global platforms
- **Speed:** Manual categorization cannot keep pace with real-time news
- **Consistency:** Human annotators disagree on ambiguous articles
- **Applications:** News aggregators, recommendation systems, content moderation, media monitoring

---

## 4. Objectives

1. Design and implement a complete NLP preprocessing pipeline
2. Train and compare two classification approaches: traditional (TF-IDF + LR) and transformer-based (DistilBERT)
3. Develop a REST API serving real-time predictions
4. Build an interactive web application for demonstration
5. Perform comprehensive quantitative and qualitative evaluation
6. Document the entire pipeline for academic reporting

---

## 5. Functional Requirements

| ID | Requirement |
|----|-------------|
| FR-01 | System shall accept news headline and/or article body as input |
| FR-02 | System shall preprocess text through a configurable NLP pipeline |
| FR-03 | System shall classify news into one of 4 AG News categories |
| FR-04 | System shall return prediction confidence (0–100%) |
| FR-05 | System shall return probability distribution across all categories |
| FR-06 | System shall handle uncertain predictions with appropriate messaging |
| FR-07 | System shall expose a REST API with documented endpoints |
| FR-08 | System shall provide an interactive web UI |
| FR-09 | System shall display preprocessing stages visually |
| FR-10 | System shall display model evaluation metrics |
| FR-11 | System shall show dataset statistics and visualizations |
| FR-12 | System shall provide example articles for each category |
| FR-13 | System shall highlight important classification features |

---

## 6. Non-Functional Requirements

| ID | Requirement |
|----|-------------|
| NFR-01 | Inference time: < 500ms for TF-IDF model |
| NFR-02 | Inference time: < 3000ms for DistilBERT on CPU |
| NFR-03 | Maximum input length: 10,000 characters |
| NFR-04 | System must work without GPU (CPU fallback) |
| NFR-05 | System must handle invalid/empty inputs gracefully |
| NFR-06 | System must not expose internal server errors to users |
| NFR-07 | Frontend must be responsive (laptop, tablet) |
| NFR-08 | Results must be reproducible (random seeds set) |
| NFR-09 | Models must be loaded once at startup, not per-request |
| NFR-10 | System must be runnable by another CSE student following README |

---

## 7. Dataset

**AG News Corpus**
- Source: HuggingFace Datasets (`ag_news`)
- Categories: World (0), Sports (1), Business (2), Sci/Tech (3)
- Train: 120,000 articles | Test: 7,600 articles
- Originally collected by Xiang Zhang et al. (2015)

---

## 8. High-Level Architecture

```
[User Input] 
    ↓
[React Frontend (Vite)]
    ↓ HTTP POST /api/predict
[FastAPI Backend]
    ↓
[Text Preprocessor]
    ↓
[Trained ML Model] ← [models/tfidf_vectorizer.pkl + logistic_regression.pkl]
                    ← [models/transformer/]
    ↓
[Post-processor] → confidence check, category name mapping
    ↓
[JSON Response]
    ↓
[Frontend Visualization]
```

---

## 9. NLP Pipeline

1. **Data Collection** — AG News via HuggingFace datasets
2. **Exploration** — statistical analysis, word clouds, distribution plots
3. **Preprocessing** — tokenization, stopword removal, lemmatization (TF-IDF); light cleaning (Transformer)
4. **Feature Extraction** — TF-IDF vectorization OR DistilBERT tokenization
5. **Model Training** — Logistic Regression OR DistilBERT fine-tuning
6. **Evaluation** — accuracy, F1, confusion matrix, qualitative analysis
7. **Post-processing** — probability ranking, confidence thresholding
8. **Deployment** — FastAPI backend + React frontend

---

## 10. Major Challenges

1. **Vocabulary overlap between categories** — Business and Sci/Tech articles often share terminology
2. **Short articles** — Headlines alone provide very limited context
3. **Ambiguous articles** — A technology IPO story overlaps Business and Sci/Tech
4. **Computational constraints** — DistilBERT fine-tuning is slow on CPU
5. **Data imbalance** — AG News is balanced, but real-world news is not
6. **Preprocessing trade-offs** — Over-preprocessing hurts transformers; under-preprocessing hurts TF-IDF

---

## 11. Technology Stack

| Component | Technology |
|-----------|-----------|
| Frontend | React 18, Vite, Tailwind CSS, Recharts |
| Backend | Python 3.12, FastAPI, Uvicorn |
| ML | scikit-learn 1.6, PyTorch 2.14, HuggingFace Transformers |
| NLP | NLTK, WordCloud |
| Data | HuggingFace Datasets, pandas, NumPy |
| Visualization | matplotlib, seaborn |
| Testing | pytest, httpx |
