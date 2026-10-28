# Post-processing Documentation

## Overview

After the ML model produces raw outputs (logits/scores), post-processing converts them into user-friendly, academically honest responses.

---

## Step 1: Logit to Probability Conversion

**For TF-IDF + Logistic Regression:**
```python
probabilities = model.predict_proba(tfidf_features)[0]
# Returns array like [0.021, 0.942, 0.015, 0.022]
# Order: [World, Sports, Business, Sci/Tech]
```

**For DistilBERT:**
```python
import torch.nn.functional as F
logits = model_output.logits[0]
probabilities = F.softmax(logits, dim=-1).numpy()
```

---

## Step 2: Label Mapping

Convert numeric class index to human-readable category name:
```python
LABEL_MAP = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
predicted_index = probabilities.argmax()
predicted_class = LABEL_MAP[predicted_index]
```

---

## Step 3: Confidence Extraction

```python
confidence = float(probabilities[predicted_index])
```

---

## Step 4: Probability Ranking

Sort all class probabilities in descending order for display:
```python
sorted_probs = sorted(
    zip(CATEGORIES, probabilities),
    key=lambda x: x[1],
    reverse=True
)
```

---

## Step 5: Confidence Thresholding

```python
CONFIDENCE_THRESHOLD = 0.50

if confidence >= 0.70:
    certainty = "high"
    message = None
elif confidence >= 0.50:
    certainty = "moderate"
    message = f"The model has moderate confidence ({confidence:.1%}) that this article is about {predicted_class}."
else:
    certainty = "uncertain"
    message = f"The model is uncertain about this article. The top prediction is {predicted_class} with only {confidence:.1%} confidence. This may indicate an ambiguous or multi-topic article."
```

---

## Step 6: Feature Keyword Extraction

For TF-IDF model, extract the top features present in the input text that have high weight for the predicted class:

```python
feature_names = tfidf_vectorizer.get_feature_names_out()
tfidf_vector = tfidf_vectorizer.transform([preprocessed_text])
class_coefs = logistic_regression.coef_[predicted_index]
# Filter to features present in input (non-zero TF-IDF)
input_features = tfidf_vector.nonzero()[1]
feature_weights = [(feature_names[i], class_coefs[i]) for i in input_features]
top_keywords = sorted(feature_weights, key=lambda x: x[1], reverse=True)[:10]
```

---

## Step 7: Response Assembly

Final JSON response returned to frontend:
```json
{
  "predicted_class": "Sports",
  "confidence": 0.942,
  "certainty": "high",
  "message": null,
  "probabilities": {
    "Sports": 0.942,
    "World": 0.021,
    "Sci/Tech": 0.022,
    "Business": 0.015
  },
  "keywords": [
    {"word": "championship", "weight": 2.34},
    {"word": "match", "weight": 1.89},
    {"word": "player", "weight": 1.67}
  ],
  "word_count": 45,
  "character_count": 312,
  "processing_time_ms": 23,
  "model_used": "tfidf_lr"
}
```

---

## Edge Cases

| Situation | Handling |
|-----------|----------|
| Empty input | Return HTTP 422 with "Please provide headline or article text" |
| Input > 10,000 chars | Return HTTP 422 with "Input too long" |
| Model not loaded | Return HTTP 503 with "Model not available. Run training pipeline first." |
| All equal probabilities | Treat as uncertain, show warning |
| Unicode/special chars | Sanitize in preprocessing step |
