# LLM Enhancement Implementation Status

**Date:** November 4, 2025  
**Status:** Phase 1 Complete - Ready for Execution

---

## ✅ Phase 1: Implementation Complete

### 1. Feature Encoder Created
**File:** `experiments/feature_encoder.py` ✅

**Key Features:**
- Uses **Gemini 2.0 Flash** via OpenRouter (large context window)
- Converts structured numeric/categorical data → natural language narratives
- Removes ALL numbers from output (qualitative descriptions only)
- Preserves ALL information from original data
- Async processing with rate limiting and progress tracking

**Functions:**
- `encode_with_gemini()` - Core encoding via OpenRouter API
- `create_encoding_prompt()` - Persona encoding prompt
- `create_time2_encoding_prompt()` - Time 2 reflection encoding
- `extract_person_data()` - Extract features from dataset
- `extract_time2_data()` - Extract post-date reflection data
- `process_dataset()` - Batch process all personas

**Usage:**
```bash
python experiments/feature_encoder.py \
  --input "Speed Dating Data.csv" \
  --output results/encoded_narratives.json \
  --max-concurrent 5
```

**Output:** `results/encoded_narratives.json`

---

### 2. Persona Generator Enhanced
**File:** `experiments/persona_generator.py` ✅

**Enhancements Added:**

#### Enhancement A: Partner Demographic Features
- Added `partner_age` to each person's data
- Added `partner_race` to each person's data
- Added `same_race` boolean (whether they share same race)
- Automatically computed from pair data

**New Fields in personas.json:**
```json
{
  "person1": {
    ...
    "race": 2,
    "partner_age": 28,
    "partner_race": 2,
    "same_race": true
  }
}
```

#### Enhancement B: Enhanced Speed Dating Context
**Updated System Prompt** - Now includes:
- "You will meet **20 different people** tonight"
- "This person is one of them"
- "~4 minutes for each chat"
- "You'll decide after each: want to see them again?"
- "You're evaluating them among 20 potential matches"

**Context:** Mimics real speed dating decision-making with comparative evaluation.

---

### 3. LLM Score Evaluator Enhanced
**File:** `experiments/llm_score_evaluator.py` ✅

**Enhancements Added:**

#### Enhancement C: Two-Question Evaluation Format

**Participant Prompt (Updated):**
```
Question 1: How much do you like this person?
Scale: 1 = don't like at all, 10 = like a lot
Your rating: X.X/10

Question 2: Would you like to see him or her again?
Answer: Yes or No

Format:
Like Score: X.X/10
See Again: Yes/No
Reasoning: [1-2 sentences]
```

**Observer Prompt (Updated):**
```
Question 1: How compatible do you think they are?
Scale: 1 = not compatible at all, 10 = extremely compatible
Your rating: X.X/10

Question 2: Do you think they should see each other again?
Answer: Yes or No

Format:
Compatibility Score: X.X/10
Should Meet Again: Yes/No
Reasoning: [2-3 sentences]
```

**Key Improvements:**
✅ Explicit scale labels: **(1=don't like at all, 10=like a lot)**
✅ Two distinct questions (continuous rating + binary decision)
✅ Matches original dataset structure ('like' + 'dec' fields)
✅ Better calibration with explicit scales

#### Enhancement D: Updated Score Extraction
Added new regex patterns to extract scores from two-question format:
- `like score: X.X/10`
- `compatibility score: X.X/10`
- Backward compatible with old format

---

## 📋 What's NOT Yet Implemented (Phase 2)

### 1. Two-Stage Evaluation System
**Status:** 🔄 NOT IMPLEMENTED YET

**Required:**
- Stage 1: Immediate post-conversation evaluation (current system)
- Stage 2: Reflection-based evaluation (with Time 2 data)
- Separate scoring methods for each stage

**Why Skipped:**
- User requested: **"no need Combined"**
- Will evaluate Stage 1 and Stage 2 **separately** (6 LLM methods total)

**Implementation Plan:**
```python
# Need to add to llm_score_evaluator.py:

def evaluate_stage2_with_reflection(person_data, conversation, time2_data):
    """
    Stage 2: Evaluation with Time 2 reflection data
    
    Additional context:
    - satis_2: Satisfaction with date
    - attr2_1, sinc2_1, etc.: Updated partner ratings
    - Natural language encoding of Time 2 data
    """
    reflection_text = encode_time2_to_narrative(time2_data)
    
    prompt = f"""
    After your conversation, you've had time to reflect on the experience.
    
    {reflection_text}
    
    Given this reflection, please answer:
    
    Question 1: How much do you like this person?
    Scale: 1 = don't like at all, 10 = like a lot
    Your rating: X.X/10
    
    Question 2: Would you like to see him or her again?
    Answer: Yes or No
    """
    # ... rest of evaluation
```

**New Methods After Stage 2 Implementation:**
1. Participant Stage 1 (immediate) ✅
2. Participant Stage 2 (reflected) 🔄
3. Observer Stage 1 (immediate) ✅
4. Observer Stage 2 (reflected) 🔄
5. Advanced Observer Stage 1 (ICL) ✅
6. Advanced Observer Stage 2 (ICL + reflection) 🔄

**Total:** 6 LLM methods (3 existing + 3 new Stage 2 variants)

---

## 🎯 Current Evaluation Methods

### Implemented (3 methods):
1. **Participant Stage 1** - Two-question format ✅
2. **Observer Stage 1** - Two-question format ✅
3. **Advanced Observer (ICL) Stage 1** - Two-question format ✅

### Ready to Add (3 methods):
4. **Participant Stage 2** - With Time 2 reflection 🔄
5. **Observer Stage 2** - With Time 2 reflection 🔄
6. **Advanced Observer Stage 2** - ICL + Time 2 reflection 🔄

---

## 🚀 Execution Plan

### Step 1: Feature Encoding (NEW)
```bash
cd /Users/assassin808/Desktop/research_2025_xuan/yan/test

python experiments/feature_encoder.py \
  --input "Speed Dating Data.csv" \
  --output results/encoded_narratives.json \
  --max-concurrent 5
```

**Expected:**
- Time: ~15-20 minutes
- Cost: ~$5-8 (Gemini 2.0 Flash via OpenRouter)
- Output: Natural language narratives for 551 participants

---

### Step 2: Generate Enhanced Personas
```bash
python experiments/persona_generator.py \
  --input processed_pairs.json \
  --output-dir results \
  --num-pairs 100
```

**Changes:**
- Now includes partner demographics (age, race, same_race)
- Enhanced system prompt (20-person context)
- Time 2 data preserved separately

**Output:** `results/personas.json` with enhanced features

---

### Step 3: Simulate Conversations
```bash
python experiments/speed_dating_simulator.py \
  --pairs results/personas.json \
  --output-dir results \
  --num-rounds 5 \
  --sample-size 100
```

**Changes:**
- Uses enhanced system prompts (20-person context)
- Same conversation logic

**Output:** `results/conversations.json`

---

### Step 4: LLM Evaluation (Two-Question Format)
```bash
python experiments/llm_score_evaluator.py \
  --conversations results/conversations.json \
  --output-dir results \
  --max-pair-workers 5 \
  --method both \
  --threshold 0.5 \
  --report-curves \
  --icl-examples results/icl_examples.json
```

**Changes:**
- Two-question evaluation format
- Explicit scale labels
- Better score extraction

**Output:**
- `results/llm_score_evaluation.json`
- `results/participant_scores.json`
- `results/observer_scores.json`

---

### Step 5: Ensemble Models (if needed)
```bash
python experiments/ensemble_model.py \
  --llm-results results/llm_score_evaluation.json \
  --output results/ensemble_evaluation.json
```

---

### Step 6: Baseline Models
```bash
python experiments/baseline_models_v2.py \
  --personas results/personas.json \
  --output-dir results
```

---

### Step 7: Comprehensive Evaluation
```bash
python experiments/evaluate_like_improved.py \
  --llm-results results/llm_score_evaluation.json \
  --ensemble-results results/ensemble_evaluation.json \
  --baseline-results results/baseline_comparison_v2.json \
  --output-dir results
```

**Evaluates:**
- 3 LLM methods (Stage 1 with two-question format)
- 2 Ensemble methods
- 8 Baseline methods
**Total: 13 methods** (same as before, but with enhanced features)

**After Stage 2 Implementation:** Will evaluate 19 methods (9 LLM + 2 ensemble + 8 baseline)

---

## 📊 Expected Improvements

### Hypothesis 1: Partner Demographics Help
**Current:** No partner context  
**Enhanced:** Age, race, same_race explicitly available

**Expected:**
- LLM can reason about age compatibility
- Same-race preference explicitly modeled
- Better alignment with human decision-making patterns

---

### Hypothesis 2: Two-Question Format Improves Calibration
**Current:** Single implicit rating  
**Enhanced:** Explicit continuous (1-10) + binary (Yes/No)

**Expected:**
- More informative evaluation
- Better correlation with 'like' scores
- Binary decision aligns with 'dec' field

---

### Hypothesis 3: Enhanced Context Improves Realism
**Current:** "You are at a speed dating event"  
**Enhanced:** "You'll meet 20 people, ~4 minutes each, decide after each"

**Expected:**
- More realistic comparative evaluation
- Better mimics human speed dating psychology
- Improved decision calibration

---

### Hypothesis 4: Explicit Scale Labels Reduce Ambiguity
**Current:** "Rate 0-10"  
**Enhanced:** "Scale: 1 = don't like at all, 10 = like a lot"

**Expected:**
- Better LLM calibration
- More consistent scoring
- Improved correlation with human ratings

---

## 🔍 Success Metrics

### Primary Goals:
1. ✅ **Eliminate inverse relationship:** k > 0 for all LLM methods
2. ✅ **Improve correlation:** Pearson r > 0.30 (50% improvement over 0.20)
3. ✅ **Maintain classification:** F1 > 0.85

### Secondary Goals:
4. ✅ **Two-question format adds value:** Binary decision complements continuous rating
5. 🔄 **Stage 2 better than Stage 1:** (After implementation) Reflection improves accuracy
6. ✅ **Enhanced features help:** Partner demographics improve predictions

---

## 📝 Key Changes Summary

| Component | Enhancement | Status |
|-----------|-------------|--------|
| **Feature Encoder** | Gemini 2.0 Flash encoding | ✅ Created |
| **Persona Generator** | Partner demographics (age, race, same_race) | ✅ Added |
| **Persona Generator** | Enhanced speed dating context (20 people) | ✅ Added |
| **LLM Evaluator** | Two-question format with explicit scales | ✅ Implemented |
| **LLM Evaluator** | Updated score extraction | ✅ Implemented |
| **LLM Evaluator** | Stage 2 evaluation (with Time 2 data) | 🔄 NOT YET |
| **Evaluation Pipeline** | Support for 6 LLM methods | 🔄 Partial (3/6) |

---

## 🎯 Next Steps

### Option A: Run Current Pipeline (3 LLM Methods)
**What:** Execute Steps 1-7 with current implementation  
**Result:** Evaluate enhancements without Stage 2 (13 methods total)  
**Time:** ~2-3 hours  
**Cost:** ~$10-16

### Option B: Implement Stage 2 First
**What:** Add Stage 2 evaluation system before running pipeline  
**Result:** Full 6 LLM method evaluation (19 methods total)  
**Time:** +30-60 min implementation, then 2-3 hours execution  
**Cost:** ~$15-25 (more evaluations)

### Option C: Incremental Approach
**What:** Run with 3 methods first, then add Stage 2 and re-run  
**Result:** Before/after comparison of Stage 1 vs Stage 1+2  
**Benefit:** Can validate enhancements work before full implementation

---

## 💡 Recommendation

**I recommend Option A: Run current pipeline first**

**Rationale:**
1. Core enhancements are implemented (demographics, context, two-questions)
2. Can validate inverse relationship fix with Stage 1 alone
3. Stage 2 can be added if Stage 1 shows improvement
4. Faster feedback loop
5. Less API cost initially

**Command to Start:**
```bash
# Step 1: Feature encoding
python experiments/feature_encoder.py \
  --input "Speed Dating Data.csv" \
  --output results/encoded_narratives.json \
  --max-concurrent 5
```

---

**Status:** ✅ Ready to Execute  
**Last Updated:** November 4, 2025
