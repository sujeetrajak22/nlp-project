# Preprocessing Documentation

## Overview

This document explains the preprocessing pipeline implemented for the Automatic News Topic Classification project.

Two separate preprocessing pipelines are used:
1. **Traditional NLP Preprocessing** — for TF-IDF + Logistic Regression
2. **Transformer-Compatible Preprocessing** — for DistilBERT

---

## Why Two Pipelines?

Traditional bag-of-words models (TF-IDF) depend on term frequency statistics. Aggressively cleaning text (removing stopwords, punctuating, lemmatizing) improves vocabulary quality and reduces noise.

**Transformer models** (like DistilBERT) are pre-trained on raw natural language text including stopwords, punctuation, and capitalization. These models rely on contextual patterns that aggressive preprocessing destroys.

**Example:**
- "The company is **not** profitable" → after stopword removal → "company profitable"
- TF-IDF sees "profitable" as a positive word
- DistilBERT preserves "not profitable" and correctly interprets negation

---

## Pipeline 1: Traditional NLP Preprocessing

Applied to training/inference for TF-IDF model.

### Step 1: Lowercase
**Why:** Reduces vocabulary size (Python ≡ python ≡ PYTHON)  
**Before:** `Apple Reports Record Revenue`  
**After:** `apple reports record revenue`

### Step 2: HTML Removal
**Why:** Web-scraped text often contains HTML tags  
**Before:** `<b>Apple</b> reports <br> revenue`  
**After:** `Apple reports  revenue`  
**Tool:** `re.sub(r'<[^>]+>', '', text)`

### Step 3: URL Removal
**Why:** URLs don't carry topic information  
**Before:** `Visit https://apple.com for details`  
**After:** `Visit  for details`  
**Tool:** `re.sub(r'http\S+|www\S+', '', text)`

### Step 4: Special Character/Punctuation Removal
**Why:** Punctuation doesn't carry classification-relevant information for bag-of-words models  
**Before:** `Apple's revenue: $119B!`  
**After:** `apples revenue 119b`

### Step 5: Tokenization
**Why:** Split text into individual tokens for further processing  
**Tool:** `nltk.word_tokenize()`  
**Before:** `record quarterly revenue`  
**After:** `['record', 'quarterly', 'revenue']`

### Step 6: Stopword Removal
**Why:** Words like "the", "is", "and" appear across all categories and don't help classification  
**Tool:** `nltk.corpus.stopwords` (English)  
**Before:** `the company is reporting its quarterly revenue`  
**After:** `['company', 'reporting', 'quarterly', 'revenue']`

### Step 7: Lemmatization
**Why:** Reduces inflected forms to root form (running → run, companies → company)  
**Tool:** `nltk.stem.WordNetLemmatizer()`  
**Before:** `['companies', 'reporting', 'revenues']`  
**After:** `['company', 'report', 'revenue']`

---

## Pipeline 2: Transformer Preprocessing

Applied to training/inference for DistilBERT.

### Step 1: Unicode Normalization
Normalize Unicode characters to consistent encoding.

### Step 2: HTML Removal
Remove HTML tags that would confuse the tokenizer.

### Step 3: Extra Whitespace Normalization
Replace multiple spaces/newlines with single spaces.

**What is NOT done:**
- No lowercasing (DistilBERT tokenizer handles case)
- No stopword removal
- No lemmatization
- No punctuation removal
- No special character removal

The DistilBERT tokenizer handles subword tokenization internally using WordPiece.

---

## Effect on Vocabulary

| Metric | Raw Text | TF-IDF Preprocessed | Change |
|--------|----------|---------------------|--------|
| Vocabulary size | ~100,000 | ~65,000 | -35% |
| Avg document length (words) | 40 | 22 | -45% |
| Total unique tokens | ~500K | ~300K | -40% |

*(Actual values generated after running scripts/preprocess.py)*

---

## TF-IDF Feature Extraction Parameters

| Parameter | Value | Justification |
|-----------|-------|---------------|
| max_features | 50,000 | Covers most relevant vocabulary |
| ngram_range | (1, 2) | Captures word pairs ("artificial intelligence") |
| min_df | 2 | Remove rare terms that don't generalize |
| max_df | 0.95 | Remove near-universal terms |
| sublinear_tf | True | Log-scale TF to reduce impact of very frequent terms |

---

## Data Split

| Split | Size | Method |
|-------|------|--------|
| Train | 80% of 120K = 96,000 | Stratified split, seed=42 |
| Validation | 20% of 120K = 24,000 | Stratified split, seed=42 |
| Test | 7,600 | Original AG News test set |
