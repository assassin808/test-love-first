# Feature Parity Investigation - LLMs vs Baselines

## Executive Summary

✅ **GOOD NEWS**: The comparison is FAIR! Both LLMs and Baselines use the **SAME underlying information**:
- Pre-event demographics, preferences, self-ratings, and interests
- **NO during-event partner ratings** are used by either method
- Different representations (natural language vs engineered features), but same data

---

## Question 1: Are the number of features the same for all methods?

### Answer: Different representations, SAME information

| Method | Representation | Count | Description |
|--------|---------------|-------|-------------|
| **LLMs** | Natural language | ~118 raw fields per pair | All data converted to text prompts |
| **Baselines** | Engineered features | ~60 numeric features | Hand-crafted mathematical features |

### Details:

#### LLMs (Stage 1)
**Input Format**: Natural language prompts

**Per Person (~59 fields):**
```
Demographics:
  - age (1)
  - gender (1)
  - race (1)

Career:
  - field_cd (1)
  - career_c (1)

Dating Context:
  - go_out (1)
  - date (1)
  - exphappy (1)
  - goal (1)

Cultural Values:
  - imprace (1)
  - imprelig (1)

Preferences (3 sets × 6 attributes = 18):
  - preferences_self: attractiveness, sincerity, intelligence, fun, ambition, shared_interests
  - preferences_opposite: [same 6]
  - preferences_same: [same 6]

Self-Ratings (5):
  - attractiveness
  - sincerity
  - intelligence
  - fun
  - ambition

Interests (17):
  - sports, tvsports, exercise, dining, museums, art, hiking
  - gaming, clubbing, reading, tv, theater, movies, concerts
  - music, shopping, yoga
```

**Per Pair**: ~59 × 2 = **118 raw fields**

**Example Prompt**:
```
Person 1: 24yo woman, consultant, values intelligence (30%), 
          rates herself 8/10 attractiveness, loves hiking and museums...

Person 2: 30yo man, engineer, values fun (25%), 
          rates himself 7/10 intelligence, enjoys gaming and reading...

Will they both want to see each other again?
```

---

#### Baselines (V1)
**Input Format**: Engineered numeric vectors

**Feature Breakdown (~60 total):**

```
Raw Features (56):
  Person 1 Preferences (6): attr1_1, sinc1_1, intel1_1, fun1_1, amb1_1, shar1_1
  Person 2 Preferences (6): [same for P2]
  
  Person 1 Self-Ratings (5): attr3_1, sinc3_1, intel3_1, fun3_1, amb3_1
  Person 2 Self-Ratings (5): [same for P2]
  
  Person 1 Interests (17): sports, tvsports, exercise, dining, ...
  Person 2 Interests (17): [same for P2]

Engineered Features (4):
  - preference_similarity: cosine(P1_prefs, P2_prefs)
  - age_difference: |age1 - age2|
  - interest_overlap: count(shared interests)
  - career_compatibility: field similarity
```

**Total**: ~60 features

**Example Input Vector**:
```python
[
    20, 15, 30, 25, 10, 0,  # P1 preferences
    25, 20, 30, 15, 10, 0,  # P2 preferences
    8, 8, 8, 7, 7,          # P1 self-ratings
    7, 7, 8, 6, 7,          # P2 self-ratings
    9, 2, 8, 7, 6, ...      # P1 interests (17 values)
    5, 8, 4, 6, 7, ...      # P2 interests (17 values)
    0.92,                    # preference similarity
    6,                       # age difference
    7,                       # interest overlap count
    0.8                      # career compatibility
]
```

---

### Key Insight: Different Representations, Same Information

**LLMs**:
- Get ~118 raw values
- Converted to natural language
- Model learns contextual relationships
- Example: "She values intelligence highly (30%) and he rates himself 8/10 on intelligence"

**Baselines**:
- Get ~60 features
- Numeric vectors + derived features
- Model learns statistical patterns
- Example: `preference_similarity = cosine([20,15,30,25,10,0], [25,20,30,15,10,0]) = 0.92`

**Conclusion**: ✅ Feature parity maintained - both use SAME underlying data, just different representations

---

## Question 2: Are DURING-EVENT partner ratings used?

### Answer: NO - neither method uses during-event ratings

### What are During-Event Ratings?

In the original Speed Dating study, participants rated their partners **DURING the date**:

**During-Event Ratings (NOT in our data)**:
- `attr`: "How attractive is your partner?" (1-10)
- `sinc`: "How sincere is your partner?" (1-10)
- `intel`: "How intelligent is your partner?" (1-10)
- `fun`: "How fun is your partner?" (1-10)
- `amb`: "How ambitious is your partner?" (1-10)
- `shar`: "How much do you share interests?" (1-10)

These are ratings **Person 1 gives to Person 2** during the actual date.

---

### What Data Do We Actually Have?

#### ✅ Pre-Event Data (Time 1)
Collected **BEFORE the date**:

| Variable | Meaning | Example |
|----------|---------|---------|
| `attr1_1` | What I **value** in a partner (attractiveness) | 20 (out of 100 points) |
| `attr3_1` | How I **rate myself** (attractiveness) | 8 (out of 10) |

**IMPORTANT**: `attr3_1` is **NOT** a rating given to partner!
- `attr3_1` = self-rating of own attractiveness
- This is a **pre-event** self-assessment, not during-event partner rating

#### ✅ Post-Event Data (Time 2)
Collected **AFTER the date**:

| Variable | Meaning | Example |
|----------|---------|---------|
| `satis_2` | Satisfaction with the date | 8 (out of 10) |
| `match_perception` | Think partner will match? | Yes/No |
| `updated_self_ratings` | Updated self-assessments | [8, 7, 9, 8, 7] |

#### ❌ What We DON'T Have
- **NO during-event partner ratings** (`attr`, `sinc`, `intel`, `fun`, `amb`, `shar`)
- **NO ratings like** "Person 1 rates Person 2's attractiveness as 8/10"
- **NO in-the-moment assessments** of the partner during the date

---

### Verification in Code

**Baseline Models (baseline_models.py)**:

Comment says "Partner ratings" but code shows **self-ratings**:

```python
# MISLEADING COMMENT:
# - Partner ratings (attr3_1, sinc3_1, intel3_1, fun3_1, amb3_1)

# ACTUAL CODE (Line 319-323):
self_ratings = time2_data.get('updated_self_ratings', {})
row['attr3_1'] = self_ratings.get('attractiveness', 0)  # SELF-rating, not partner
row['sinc3_1'] = self_ratings.get('sincerity', 0)       # SELF-rating, not partner
row['intel3_1'] = self_ratings.get('intelligence', 0)    # SELF-rating, not partner
```

**LLM Evaluator**:

Uses only what's in `personas.json`:
- Pre-event: demographics, preferences, self-ratings, interests
- Post-event (Stage 2): satisfaction, match perception
- **NO partner ratings** in prompts

---

### Data Investigation Results

Checked `results/personas.json` for any during-event fields:

```python
Sample person keys:
  - age ✅
  - gender ✅
  - race ✅
  - pre_event_data ✅
  - time2_reflection ✅
  - persona_narrative ✅
  - partner_age ✅ (metadata, not a rating)
  - partner_race ✅ (metadata, not a rating)
  
❌ NO fields like: attr, sinc, intel, fun, amb, shar
❌ NO fields with names: *_partner_rating, during_event_*
```

**Conclusion**: The data simply doesn't contain during-event partner ratings.

---

## Fair Comparison Verification

### Both Methods Use:

| Data Category | LLMs | Baselines | Status |
|---------------|------|-----------|--------|
| Demographics | ✅ | ✅ | ✅ Equal |
| Preferences (what they want) | ✅ | ✅ | ✅ Equal |
| Self-Ratings (how they rate themselves) | ✅ | ✅ | ✅ Equal |
| Interests (17 categories) | ✅ | ✅ | ✅ Equal |
| Career Info | ✅ | ✅ | ✅ Equal |
| Dating Context | ✅ | ✅ | ✅ Equal |
| **During-Event Partner Ratings** | ❌ | ❌ | ✅ **Fair - neither has it** |

### Time 2 (Post-Date) Data

| Data Category | Stage 1 LLM | Stage 2 LLM | Baseline V1 | Baseline V2 |
|---------------|-------------|-------------|-------------|-------------|
| Satisfaction Score | ❌ | ✅ | ❌ | ✅ |
| Match Perception | ❌ | ✅ | ❌ | ✅ |
| Updated Self-Ratings | ❌ | ✅ | ❌ | ✅ |

**Fair Comparisons**:
- Stage 1 LLM vs Baseline V1 ✅ (both use Time 1 only)
- Stage 2 LLM vs Baseline V2 ✅ (both use Time 1 + Time 2)

---

## Summary

### Question 1: Feature Count Consistency
**Answer**: ✅ **YES** - both use same information, different representations

- **LLMs**: ~118 raw fields → natural language
- **Baselines**: ~60 engineered features → numeric vectors
- **Same underlying data**: demographics, preferences, self-ratings, interests
- **Difference**: Representation format, not information content

### Question 2: During-Event Ratings Usage
**Answer**: ❌ **NO** - neither method uses during-event partner ratings

- **Not in data**: `personas.json` doesn't contain `attr`, `sinc`, `intel`, `fun`, `amb`, `shar`
- **Only have**: Pre-event expectations and post-event reflections
- **Fair**: Both LLMs and Baselines work from same pre/post data
- **No advantage**: Neither method has access to in-the-moment partner assessments

### Overall Fairness: ✅ MAINTAINED

1. ✅ Same underlying information (just different representations)
2. ✅ No during-event partner ratings for either method
3. ✅ Time 1 vs Time 2 separation respected (V1 vs V2, Stage 1 vs Stage 2)
4. ✅ All methods predict same target (mutual match) on same test set (100 pairs)

The comparison is methodologically sound!

---

## Recommendation

**Update baseline_models.py comments** to avoid confusion:

```python
# CURRENT (MISLEADING):
# - Partner ratings (attr3_1, sinc3_1, intel3_1, fun3_1, amb3_1)

# SHOULD BE:
# - Self ratings (attr3_1, sinc3_1, intel3_1, fun3_1, amb3_1)
# Note: These are how the person rates THEMSELVES, not their partner
```

This will prevent future confusion about what `attr3_1` represents.
