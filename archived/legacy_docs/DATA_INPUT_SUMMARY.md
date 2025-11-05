# Data Input Summary - Complete Pipeline

## Overview

This document describes **what data is used as input** at each stage of the compatibility prediction pipeline.

---

## 🎯 Ground Truth (Prediction Target)

**Consistent across ALL methods:**

- **Label**: `mutual_match` (binary: 1 or 0)
- **Definition**: BOTH people want to see each other again
- **Formula**: `(person1.match == 1) AND (person2.match == 1)`
- **Distribution** (100 test pairs):
  - Positive cases (match=1): **50 pairs (50%)**
  - Negative cases (match=0): **50 pairs (50%)**

---

## 📊 Input Data Sources

### 1. **Original Dataset** (`results/personas.json`)

**Source**: Speed Dating dataset (Fisman & Iyengar, 2006)
**Size**: 100 randomly sampled pairs (200 individuals)

**Data Structure** per pair:
```json
{
  "pair_id": "pair_001",
  "person1": { ... },
  "person2": { ... }
}
```

**Per Person Data:**

#### **A. Pre-Event Data (Time 1)** - Collected BEFORE the date

| Category | Fields | Description |
|----------|--------|-------------|
| **Demographics** | `age`, `gender`, `race` | Basic demographic info |
| **Career** | `field_cd`, `career_c` | Field of study, career importance |
| **Dating Context** | `go_out`, `date`, `exphappy`, `goal` | How often they date, expectations |
| **Preferences** | `preferences_self`, `preferences_opposite`, `preferences_same` | What they value in a partner (attractiveness, sincerity, intelligence, fun, ambition, shared_interests) |
| **Self-Ratings** | `self_ratings` | How they rate themselves (attractiveness, sincerity, intelligence, fun, ambition) |
| **Interests** | `sports`, `tvsports`, `exercise`, `dining`, `museums`, `art`, `hiking`, `gaming`, `clubbing`, `reading`, `tv`, `theater`, `movies`, `concerts`, `music`, `shopping`, `yoga` | 17 interest categories (0-10 scale) |
| **Cultural Values** | `imprace`, `imprelig` | Importance of race and religion |

#### **B. Post-Date Data (Time 2)** - Collected AFTER the date

| Category | Fields | Description |
|----------|--------|-------------|
| **Satisfaction** | `time2_reflection.satisfaction.satis_2` | How satisfied with the date (1-10) |
| **Match Perception** | `time2_reflection.match_perception` | Do they think partner will match them? |
| **Updated Self-Ratings** | `time2_reflection.personality_changes` | Updated ratings after date experience |
| **Narrative** | `time2_reflection_narrative` | Text description of satisfaction level |

#### **C. Outcome Data** - The decision

| Field | Description |
|-------|-------------|
| `dec` | Individual decision (1=yes want to see again, 0=no) |
| `match` | Mutual match result (1=both said yes, 0=at least one said no) |

---

## 🤖 Method 1: LLM Evaluation - Stage 1 (Pre-date Only)

### Input Files
- **results/llm_score_evaluation_stage1.json**

### Input Data Used

**ONLY Time 1 (Pre-event) data:**

1. **Demographics**: age, gender, race
2. **Preferences**: What each person values in a partner
   - Attractiveness, sincerity, intelligence, fun, ambition, shared interests
   - Weights sum to 100 for each person
3. **Self-Ratings**: How each rates themselves (1-10 scale)
   - Attractiveness, sincerity, intelligence, fun, ambition
4. **Interests**: 17 categories (sports, dining, museums, reading, etc.)
5. **Career Info**: Field of study, career importance
6. **Dating Context**: How often they go out, dating experience

### What's Excluded
❌ **NO Time 2 data** (post-date reflections)
❌ **NO satisfaction scores**
❌ **NO match perception**

### Output
- **3 prediction methods**:
  1. **Participant Method**: Each person rates their own attraction (8.5, 7.0, etc.)
  2. **Observer Method**: Third-party observer predicts compatibility
  3. **Advanced Observer Method**: Enhanced observer with detailed reasoning

- **Scores**: 0-1 probability of mutual match
- **100 pairs** evaluated

### Example Input to LLM
```
Person 1: 24yo woman, consultant, loves sports/hiking/museums, 
         values: attractiveness=20%, sincerity=20%, intelligence=20%
         self-rates: 8/10 attractiveness, 8/10 intelligence

Person 2: 30yo man, engineer, enjoys gaming/reading/movies,
         values: intelligence=30%, fun=25%, attractiveness=20%
         self-rates: 7/10 intelligence, 6/10 fun

Question: Will they both want to see each other again?
```

---

## 🤖 Method 2: LLM Evaluation - Stage 2 (Pre + Post-date)

### Input Files
- **results/llm_score_evaluation_stage2.json**

### Input Data Used

**Everything from Stage 1, PLUS Time 2 (Post-date) data:**

1. **All Stage 1 data** (demographics, preferences, interests)
2. **✅ Post-Date Satisfaction**: How satisfied each person was (1-10)
   - Low (1-4), Moderate (5-7), High (8-10)
3. **✅ Match Perception**: Whether each thinks the other will match them
4. **✅ Personality Changes**: Updated self-ratings after the date
5. **✅ Reflection Narrative**: Text description of their experience

### What's Added
✅ **Time 2 reflections** (satisfaction, perception, updated ratings)
✅ **Richer context** about how the date actually went

### Output
- **Same 3 prediction methods** as Stage 1
- **Same 100 pairs**
- **Goal**: See if post-date info improves predictions

### Example Additional Input
```
After the date:
Person 1: Satisfaction = 8/10 (HIGH), thinks Person 2 will match them
Person 2: Satisfaction = 3/10 (LOW), doesn't think Person 1 will match

Question: Will this be a mutual match?
```

---

## 🔗 Method 3: Ensemble Models

### Input Files
- **results/ensemble_evaluation_stage1.json**
- **results/ensemble_evaluation_stage2.json**

### Input Data Used

**NOT raw features - uses LLM SCORES as meta-features:**

#### Stage 1 Ensemble
- **3 features** (LLM scores from Stage 1):
  1. `participant_score` (0-1)
  2. `observer_score` (0-1)
  3. `advanced_observer_score` (0-1)

#### Stage 2 Ensemble
- **Same 3 features** (but from Stage 2 LLM evaluation)

### Models
1. **Linear Regression**: Learns weighted combination of 3 scores
2. **Logistic Regression**: Binary classifier on 3 scores

### Training/Test Split
- **Training**: 80% of 100 pairs (80 pairs)
- **Testing**: 20% of 100 pairs (20 pairs)

### Goal
Combine multiple LLM perspectives to improve predictions

---

## 📈 Method 4: Baseline Models (Traditional ML)

### Input Files
- **results/baseline_comparison_v2.json**

### Input Data Used

#### **Version 1 (Time 1 only)** - 4 models
Engineered features from pre-event data:

1. **Preference Similarity**: Cosine similarity of what they want
   - Compare person1.preferences_opposite vs person2.preferences_opposite
2. **Age Difference**: |age1 - age2|
3. **Self-Rating Differences**: |rating1 - rating2| for each attribute
4. **Interest Overlap**: Count of shared interests (both rated >5)
5. **Career Compatibility**: Field similarity
6. **Gender**: 0=female, 1=male for each person

**Models**:
- Similarity V1 (cosine similarity only)
- Logistic Regression V1
- Random Forest V1
- XGBoost V1

#### **Version 2 (Time 1 + Time 2)** - 4 models
All V1 features, PLUS:

7. **Satisfaction Scores**: satis_2 for each person
8. **Match Perception**: Does each think the other will match?
9. **Satisfaction Match**: |satis1 - satis2|
10. **Updated Personality Ratings**: Post-date self-ratings

**Models**:
- Similarity V2
- Logistic Regression V2
- Random Forest V2
- XGBoost V2

### Training/Test Split
- **Training**: 2,398 pairs (full Speed Dating dataset)
- **Testing**: 100 pairs (same as LLMs)

### Goal
Establish strong baseline with traditional features + large training set

---

## 📊 Data Comparison Table

| Method | Input Data | # Features | Training Data | Test Data |
|--------|-----------|------------|---------------|-----------|
| **Stage 1 LLM** | Pre-event (Time 1) | ~40 raw fields | None (zero-shot) | 100 pairs |
| **Stage 2 LLM** | Pre + Post-event (T1+T2) | ~45 raw fields | None (zero-shot) | 100 pairs |
| **Ensemble Stage 1** | 3 LLM scores (Stage 1) | 3 features | 80 pairs | 20 pairs |
| **Ensemble Stage 2** | 3 LLM scores (Stage 2) | 3 features | 80 pairs | 20 pairs |
| **Baseline V1** | Engineered features (T1) | ~10 features | 2,398 pairs | 100 pairs |
| **Baseline V2** | Engineered features (T1+T2) | ~15 features | 2,398 pairs | 100 pairs |

---

## 🔑 Key Differences

### Time 1 vs Time 2 Data

| Feature Category | Time 1 (Pre-event) | Time 2 (Post-event) |
|------------------|-------------------|---------------------|
| **Demographics** | ✅ Age, gender, race | (same) |
| **Preferences** | ✅ What they want in partner | (same) |
| **Self-Ratings** | ✅ Initial ratings | ✅ Updated after date |
| **Interests** | ✅ 17 categories | (same) |
| **Satisfaction** | ❌ | ✅ 1-10 score |
| **Match Perception** | ❌ | ✅ Yes/No |
| **Date Experience** | ❌ | ✅ Narrative reflection |

### Raw vs Engineered Features

| Approach | Input Type | Example |
|----------|-----------|---------|
| **LLMs** | Raw data → Natural language | "Person 1 is 24yo consultant who values intelligence (20pts)..." |
| **Baselines** | Engineered features → Numeric | `preference_similarity=0.85, age_diff=6, interest_overlap=7` |
| **Ensembles** | Meta-features → LLM outputs | `[0.72, 0.65, 0.58]` (3 LLM scores) |

---

## 📁 File Locations

```
test/
├── results/
│   ├── personas.json                          # Original dataset (100 pairs)
│   ├── llm_score_evaluation_stage1.json      # Stage 1 LLM predictions
│   ├── llm_score_evaluation_stage2.json      # Stage 2 LLM predictions
│   ├── ensemble_evaluation_stage1.json       # Stage 1 ensemble results
│   ├── ensemble_evaluation_stage2.json       # Stage 2 ensemble results
│   ├── baseline_comparison_v2.json           # Baseline model results
│   ├── optimal_thresholds_stage1.json        # Optimal thresholds (Stage 1)
│   ├── optimal_thresholds_stage2.json        # Optimal thresholds (Stage 2)
│   └── fair_comparison_complete.json         # Complete fair comparison
```

---

## 🎯 Summary

### What Data is Used?

1. **Core Input**: 100 pairs from Speed Dating dataset
   - Each pair has 2 people with demographics, preferences, interests
   - Ground truth: mutual match (both want to see each other)

2. **Time 1 (Pre-event)**: Demographics, preferences, self-ratings, interests
   - Used by: Stage 1 LLM, Baseline V1

3. **Time 2 (Post-event)**: Satisfaction, match perception, reflections
   - Used by: Stage 2 LLM, Baseline V2

4. **LLM Scores**: Meta-features from LLM predictions
   - Used by: Ensemble models

5. **Training Data**:
   - LLMs: None (zero-shot) + optimal threshold tuning
   - Baselines: 2,398 pairs from full dataset
   - Ensembles: 80 pairs from test set

### Key Insight

**All methods predict the SAME target** (mutual match) on the **SAME 100 test pairs**, but use **DIFFERENT input representations**:
- LLMs use **raw natural language** descriptions
- Baselines use **engineered numeric features**
- Ensembles use **LLM predictions** as features
