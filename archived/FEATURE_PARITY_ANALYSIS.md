# Feature Parity Analysis: LLM vs Baseline Models

## ❌ Current Problem: Unfair Comparison

The LLM methods have access to **MUCH MORE** information than the baseline ML models, making the comparison unfair and biased toward the LLM.

## What Each Method Currently Sees

### LLM Methods (from `persona_narrative`)
**Total: ~50+ features of rich information**

✅ **Demographics (5 features)**:
- Age (e.g., 24 years old)
- Gender (woman/man)
- Race (European/Caucasian-American, Asian, etc.)
- Field of study (Business/Economics/Finance, etc.)
- Career plans (Banking/Consulting, etc.)

✅ **Preferences (6 features with exact points)**:
- attractiveness (20 points)
- sincerity (20 points)  
- intelligence (20 points)
- fun (20 points)
- ambition (20 points)
- shared interests (0 points)

✅ **Self-Ratings (5 features with exact scores)**:
- attractiveness (8/10)
- sincerity (8/10)
- intelligence (8/10)
- fun (8/10)
- ambition (8/10)

✅ **Others' Perception (5 features)**:
- How they expect others would rate them (8/10, 7/10, etc.)

✅ **Interests (17 features with ratings)**:
- sports (8/10), tvsports (4/10), exercise (9/10), dining (9/10), museums (9/10), art (9/10), hiking (8/10), gaming (5/10), clubbing (8/10), reading (9/10), TV (8/10), theater (9/10), movies (9/10), concerts (9/10), music (9/10), shopping (9/10), yoga (9/10)

✅ **Dating Behavior (4 features)**:
- go_out frequency (several times a week)
- date frequency (twice a week)
- expected happiness tonight (7/10)
- goal (find serious relationship, have fun, etc.)

✅ **Values (2 features)**:
- importance of same race (7/10)
- importance of same religion (4/10)

✅ **Additional Context**:
- What they think opposite sex looks for
- What they think same sex looks for
- Narrative description with personality hints

---

### Baseline ML Models (from `time2_reflection` only)
**Total: 24 features**

❌ **Only Has**:
- Person 1: 6 preferences + 5 self-ratings = 11 features
- Person 2: 6 preferences + 5 self-ratings = 11 features
- Interest overlap (Jaccard): 1 feature (but no individual interest ratings!)
- Preference alignment: 1 feature

❌ **Missing (that LLM has)**:
- Demographics: age, gender, race, field, career
- Individual interest ratings (17 features per person)
- Others' perception ratings (5 features per person)
- Dating behavior: go_out, date frequency, exphappy, goal
- Values: imprace, imprelig
- What they think opposite/same sex looks for

## 📊 Information Asymmetry

| Category | LLM Has? | Baseline Has? | Feature Count |
|----------|----------|---------------|---------------|
| Demographics | ✅ | ❌ | 5 per person = 10 total |
| Preferences | ✅ | ✅ | 6 per person = 12 total |
| Self-ratings | ✅ | ✅ | 5 per person = 10 total |
| Others' perception | ✅ | ❌ | 5 per person = 10 total |
| Interest ratings | ✅ | ❌ | 17 per person = 34 total |
| Dating behavior | ✅ | ❌ | 4 per person = 8 total |
| Values | ✅ | ❌ | 2 per person = 4 total |
| **TOTAL** | **~90 features** | **24 features** | **Gap: 66 features** |

## 🎯 Solution: Feature Parity

To make a fair comparison, the baseline models MUST have access to the **exact same raw data** that the LLM sees.

### Where is this data in personas.json?

The problem is that `personas.json` structure has:
```json
{
  "pair_id": "pair_001",
  "person1": {
    "iid": 467,
    "gender": 0,
    "age": 24,
    "persona_narrative": "I'm a 24-year-old woman...", // <-- LLM sees this (full text)
    "time2_reflection": {
      "updated_preferences_self": {...},    // <-- Baseline uses this
      "updated_self_ratings": {...}         // <-- Baseline uses this
    }
  }
}
```

The **demographics, interests, values, dating behavior** are encoded IN the `persona_narrative` text, but NOT in the structured `time2_reflection` data!

### Two Options to Fix This:

#### Option A: Parse persona_narrative (hacky)
- Use regex to extract age, race, interests from narrative text
- Error-prone and fragile

#### Option B: Add structured pre-event data (proper)
- Go back to `persona_generator.py`
- Add a new field `pre_event_data` with ALL structured features
- Include: age, gender, race, field, career, interests (with ratings), go_out, date, exphappy, goal, imprace, imprelig

**Option B is the correct approach.**

## 📝 Action Items

### 1. Update `persona_generator.py`
Add structured pre-event data to persona JSON:

```python
persona_pair = {
    'pair_id': pair_id,
    'person1': {
        'iid': person1_iid,
        'gender': person1_gender,
        'age': person1_age,
        'persona_narrative': narrative1,  # For LLM
        'pre_event_data': {               # NEW: For baseline models
            'age': age1,
            'gender': gender1,
            'race': race1,
            'field': field1,
            'career': career1,
            'go_out': go_out1,
            'date': date1,
            'exphappy': exphappy1,
            'goal': goal1,
            'imprace': imprace1,
            'imprelig': imprelig1,
            'interests': {
                'sports': 8, 'tvsports': 4, 'exercise': 9, ...
            },
            'attr1_1': ..., 'sinc1_1': ..., etc.
        },
        'time2_reflection': {...}  # Post-event (current)
    },
    'person2': {...}
}
```

### 2. Update `baseline_models.py`
Load features from `pre_event_data` instead of just `time2_reflection`:

```python
def _extract_features(self, person1_data: Dict, person2_data: Dict):
    features = []
    
    # Person 1 features (from pre_event_data)
    features.append(person1_data['age'] / 100)
    features.append(person1_data['gender'])
    features.append(1 if person1_data['race'] == 2 else 0)  # One-hot
    # ... all preferences, ratings, interests, etc.
    
    # Person 2 features (same)
    # ...
    
    # Derived features
    # Interest overlap, alignment, etc.
    
    return np.array(features)  # Now ~90 features like LLM
```

### 3. Re-run Experiments
Once feature parity is achieved:
1. Regenerate personas.json with structured data
2. Re-run baseline evaluation
3. Re-run LLM simulation (if needed)
4. Compare with fair feature access

## 🔬 Why This Matters

**Current comparison is like:**
- LLM: Full resume + cover letter + interview transcript
- Baseline ML: Just 2 numbers (preference score + self-rating)

**Fair comparison should be:**
- Both see: Full resume + cover letter + interview transcript
- Then we measure: Who predicts better given the SAME information?

This is critical for scientific validity! 🎯
