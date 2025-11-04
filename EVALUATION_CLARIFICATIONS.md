# Evaluation Against 'Like' Scores - Important Clarifications

## Your Questions and Answers

### 1. Are LLM scores scaled (k*x + b) before comparison with 'like'?

**SHORT ANSWER: Now they are! (After the improved script)**

**DETAILED EXPLANATION:**

#### Before (Original `evaluate_against_like_score.py`):
- ❌ **Individual LLM methods**: NO scaling - raw scores directly compared to 'like'
- ✅ **Ensemble methods**: YES scaling - Linear/Logistic Regression learned coefficients during training

#### After (Improved `evaluate_like_improved.py`):
- ✅ **ALL methods**: Train linear regression `like_combined = k * prediction + b` for fair comparison
- This finds optimal alignment between each method's score range and the 'like' scores

**Results Show the Difference:**
```
Method            | Raw Pearson | Scaled Pearson | Scaling Formula
------------------|-------------|----------------|---------------------------
Observer          | -0.1996     | +0.1996        | -0.3683*pred + 0.7224
Ensemble Linear   | +0.1012     | +0.1012        | +0.2339*pred + 0.3311
```

**Key Insight:** Observer's negative coefficient (-0.3683) means higher observer scores → lower 'like' scores! After accounting for this inverse relationship, correlation becomes positive.

---

### 2. What does "Binary prediction using like threshold" mean?

**EXPLANATION:**

The 'like' scores are continuous (0-10), but we also want to evaluate binary classification. Here's how:

#### Step 1: Convert 'like' scores to binary labels
```python
# Ground truth from dataset
person1_like = 8.0  # Person 1 rated partner 8/10
person2_like = 7.0  # Person 2 rated partner 7/10

# Binary conversion (threshold = 5.0)
if person1_like >= 5.0 AND person2_like >= 5.0:
    binary_label = 1  # MATCH
else:
    binary_label = 0  # NO MATCH

# In this example: 8≥5 AND 7≥5 → binary_label = 1 (Match)
```

#### Step 2: Use prediction scores for classification
```python
# Method's continuous prediction score
prediction_score = 0.63  # e.g., participant combined score

# Binary prediction (threshold = 0.5)
if prediction_score >= 0.5:
    predicted_binary = 1  # Predict MATCH
else:
    predicted_binary = 0  # Predict NO MATCH
```

#### Step 3: Calculate classification metrics
```python
# ROC AUC: Uses continuous scores vs binary labels
roc_auc = roc_auc_score(binary_labels, prediction_scores)

# F1, Accuracy: Uses binary predictions vs binary labels  
f1 = f1_score(binary_labels, predicted_binaries)
```

**Why both thresholds matter:**
- **Like threshold (5.0)**: Defines what counts as a "match" in ground truth
- **Prediction threshold (0.5)**: Converts continuous predictions to binary decisions

**Example Confusion Matrix:**
```
                  Predicted: No Match (pred<0.5)  |  Predicted: Match (pred≥0.5)
----------------|--------------------------------|---------------------------
Actual: No Match|  TN = 29                       |  FP = 20
(both_like<5)   |  (Correct rejection)           |  (False alarm)
----------------|--------------------------------|---------------------------
Actual: Match   |  FN = 25                       |  TP = 25
(both_like≥5)   |  (Missed match)                |  (Correct detection)
```

---

### 3. Why aren't baseline methods included in 'like' evaluation?

**SHORT ANSWER: Technical issue - baseline script doesn't save individual predictions**

**DETAILED EXPLANATION:**

#### Current Baseline Output (`baseline_comparison_v2.json`):
```json
{
  "similarity_v1": {
    "accuracy": 0.500,
    "precision": 0.500,
    "f1": 0.667,
    "auc": 0.562
    // ❌ NO "predictions" field with individual pair scores
  }
}
```

#### What We Need:
```json
{
  "similarity_v1": {
    "metrics": {...},
    "predictions": [  // ✅ This is missing!
      {"pair_id": "pair_296_319", "probability": 0.873},
      {"pair_id": "pair_261_276", "probability": 0.652},
      ...
    ]
  }
}
```

#### Solution Options:

**Option A: Modify `baseline_models_v2.py` to save predictions**
- Add code to save each pair's probability score
- Re-run baseline evaluation
- Then run improved 'like' evaluation

**Option B: Re-compute baseline predictions on-the-fly**
- Load trained models from disk
- Recompute predictions for 100 test pairs
- Requires saving trained models (currently not done)

**Option C: Use existing baseline_models_v2.py metrics**
- Can't correlate with 'like' scores without individual predictions
- Can only compare aggregate metrics (ROC AUC, F1, etc.)

**Recommendation:** Implement Option A - it's the cleanest solution.

---

## Summary of Improvements

### ✅ What the improved script (`evaluate_like_improved.py`) does:

1. **Linear Scaling for All Methods**
   - Trains `like_combined = k * prediction + b` for each method
   - Shows both scaled and raw correlations
   - Reveals methods with inverse relationships (negative k)

2. **Clear Binary Classification Explanation**
   - Documents threshold meanings clearly
   - Shows conversion logic in output
   - Separates regression vs classification metrics

3. **Baseline Integration (Partial)**
   - Framework ready to load baseline predictions
   - Currently blocked by missing prediction data
   - Easy to integrate once baseline script updated

### 📊 Current Best Results (with scaling):

**Regression (predicting continuous 'like' scores):**
- Observer: Pearson = 0.1996 (after accounting for inverse relationship)
- Ensemble Linear: Pearson = 0.1012 (direct positive relationship)

**Classification (binary match/no-match):**
- Observer: ROC AUC = 0.5840, F1 = 0.9121
- Best at identifying mutual matches

### 🔧 Next Steps:

1. Modify `baseline_models_v2.py` to save individual predictions
2. Re-run baseline evaluation
3. Run improved 'like' evaluation with all methods
4. Compare: LLM vs Ensemble vs Baselines on same ground truth
