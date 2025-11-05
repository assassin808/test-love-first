# LLM Method Enhancement Plan
## Addressing Inverse Relationship & Improving Prediction Accuracy

**Date:** November 4, 2025  
**Problem:** Current LLM methods show inverse relationship with human 'like' scores (k=-0.37)  
**Goal:** Enhance LLM methods with richer features and two-stage evaluation

---

## 🎯 Proposed Enhancements (5 Major Changes)

### Enhancement 1: Add Demographic Partner Features
**Current:** Only self-attributes in persona  
**New:** Add 3 partner-related features:
1. **Same Race:** Whether participant and partner share same race (boolean/categorical)
2. **Partner Age:** Age of the partner (numeric)
3. **Partner Race:** Race/ethnicity of the partner (categorical)

**Rationale:**
- Research shows same-race preference is significant in speed dating
- Age difference affects compatibility perception
- Explicit partner demographics may help LLM reasoning

**Implementation:**
- Add to `persona_generator.py`: Include partner info in persona
- Format: Both self and partner demographics visible to each participant

---

### Enhancement 2: Feature Encoding Pipeline (LLM-based)
**Current:** Structured data → Direct template → Persona narrative  
**New:** Structured data → Advanced LLM encoding → Natural language paragraph → Persona

**Process:**
```
Raw Features (numeric/categorical)
    ↓
Advanced LLM Encoder (GPT-4)
    ↓
Coherent Natural Language Paragraph
    ↓
Persona Narrative (ALL details preserved, zero numerical values)
```

**Example Transformation:**
```
BEFORE (Current):
"Age: 25, attractive: 7/10, sincere: 8/10, sports: 5/10"

AFTER (Enhanced):
"You're a 25-year-old woman who considers yourself quite attractive and 
values sincerity highly in relationships. You have a moderate interest 
in sports and enjoy staying active, though it's not your primary passion."
```

**Key Requirements:**
- **Keep ALL details** from original data
- **Remove ALL numbers** (convert to qualitative descriptions)
- **Coherent narrative flow** (not bullet points)
- **Natural language** (human-like storytelling)

**LLM Prompt Template:**
```
Convert the following structured profile into a natural, coherent narrative 
paragraph. Preserve ALL information but remove numeric values, converting 
them to qualitative descriptions (e.g., 7/10 → "quite" or "very").

Structured Data:
{json_data}

Write a natural, flowing paragraph that a person would use to describe 
themselves. No bullet points, no numbers, just narrative.
```

---

### Enhancement 3: Speed Dating Context Enhancement
**Current:** "You are on a speed dating event"  
**New:** "You are participating in a speed dating event where you will meet 20 different people throughout the evening"

**Additions to System Prompt:**
- Total number of dates: 20 people
- Context of multiple meetings (not just one)
- Time constraint awareness (speed dating = brief interactions)
- Decision-making context (need to choose from multiple options)

**Updated System Prompt:**
```
You are participating in a speed dating event tonight. Over the course of 
the evening, you will have brief 4-minute conversations with 20 different 
people. After each conversation, you'll indicate whether you'd like to see 
that person again. Remember, you're looking for someone special among many 
potential matches.

[Your persona details...]
```

**Rationale:**
- Mimics real speed dating decision-making
- Provides context for comparative evaluation
- Adds realism to LLM's perspective

---

### Enhancement 4: Two-Question Evaluation Format
**Current:** Single implicit evaluation  
**New:** Two explicit questions (matching original dataset)

**Question 1 - Continuous Rating:**
```
How much do you like this person?
Scale: 1 (don't like at all) to 10 (like a lot)

Your rating: [1-10]
```

**Question 2 - Binary Decision:**
```
Would you like to see him or her again?
Answer: Yes / No
```

**Benefits:**
- **Matches original dataset structure** (both 'like' and 'dec' fields)
- **Two perspectives:** Continuous affinity + Binary decision
- **Better calibration:** Explicit scale reduces ambiguity
- **Rich evaluation:** Can analyze both regression and classification

**Applies to:**
- Participant self-evaluation
- Observer evaluation (reworded: "Do you think they should meet again?")

---

### Enhancement 5: Two-Stage Rating System

#### Stage 1: Immediate Post-Dating Rating (Current)
**Timing:** Right after conversation ends  
**Input:** Only conversation content  
**Process:** Same as current implementation
```
[Conversation ends]
→ Ask Question 1 (like 1-10)
→ Ask Question 2 (see again? Y/N)
```

#### Stage 2: Reflection-Based Rating (New)
**Timing:** After Stage 1, with additional reflection  
**Input:** Conversation + Post-meeting perception changes

**New Features from Dataset:**
- `satis_2`: Satisfaction with opposite sex (after meeting)
- `attr2_1`, `sinc2_1`, `intel2_1`, `fun2_1`, `amb2_1`, `shar2_1`: Updated attribute ratings
- Perception changes after the date

**Process:**
```
[Stage 1 complete]
↓
Add Reflection Context:
"After meeting [name], your perception has evolved:
[Natural language paragraph encoding satis_2 and updated attributes]"
↓
Ask Questions Again:
→ Question 1 (like 1-10) - with reflection
→ Question 2 (see again? Y/N) - with reflection
```

**Example Reflection Prompt:**
```
After your conversation with [name], you've had time to reflect. 
Your overall satisfaction with how the date went is quite positive. 
You now see them as very attractive, quite sincere, and highly intelligent. 
Their fun personality really stood out, and you noticed strong shared 
interests between you two.

Given this reflection, please answer:
1. How much do you like this person? (1-10)
2. Would you like to see them again? (Yes/No)
```

**Applies to:**
- **Participant:** Uses their own `satis_2` and updated attributes
- **Observer:** Uses combined/averaged perception changes from both participants

---

## 📋 Implementation Plan

### Phase 1: Data Pipeline Enhancement

#### Task 1.1: Enhance Persona Generator
**File:** `experiments/persona_generator.py`

**Changes:**
1. Add partner demographic features:
   ```python
   def add_partner_features(pair_data):
       person1 = pair_data['person1']
       person2 = pair_data['person2']
       
       # Add to person1's knowledge
       person1['partner_age'] = person2['age']
       person1['partner_race'] = person2['race']
       person1['same_race'] = (person1['race'] == person2['race'])
       
       # Add to person2's knowledge
       person2['partner_age'] = person1['age']
       person2['partner_race'] = person1['race']
       person2['same_race'] = (person1['race'] == person2['race'])
   ```

2. Extract Time 2 reflection data:
   ```python
   def extract_time2_reflection(iid, csv_data):
       """Extract satis_2, attr2_1, sinc2_1, etc."""
       person_data = csv_data[csv_data['iid'] == iid].iloc[0]
       return {
           'satis_2': person_data.get('satis_2'),
           'attr2_1': person_data.get('attr2_1'),
           'sinc2_1': person_data.get('sinc2_1'),
           'intel2_1': person_data.get('intel2_1'),
           'fun2_1': person_data.get('fun2_1'),
           'amb2_1': person_data.get('amb2_1'),
           'shar2_1': person_data.get('shar2_1')
       }
   ```

**Output:** Enhanced `personas.json` with partner features + Time 2 reflections

---

#### Task 1.2: Create Feature Encoding Pipeline
**New File:** `experiments/feature_encoder.py`

**Purpose:** Convert structured data → natural language using GPT-4

**Key Function:**
```python
async def encode_persona_to_narrative(structured_data: dict) -> str:
    """
    Use advanced LLM (GPT-4) to convert structured profile 
    into coherent natural language paragraph.
    
    Input: {age: 25, attractive: 7, sports: 5, ...}
    Output: "You're a 25-year-old woman who..."
    
    Requirements:
    - ALL details preserved
    - ZERO numeric values in output
    - Coherent narrative (not bullet points)
    - Natural language flow
    """
    prompt = f"""
    Convert this structured dating profile into a natural, flowing narrative.
    
    Rules:
    1. Keep ALL information
    2. Remove ALL numbers - convert to qualitative words
    3. Write as a coherent paragraph (no bullets)
    4. Natural language like someone describing themselves
    
    Structured Data:
    {json.dumps(structured_data, indent=2)}
    
    Write the narrative paragraph:
    """
    
    response = await openai_client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    
    return response.choices[0].message.content
```

**Conversion Guidelines:**
- `1-2` → "very low" / "not at all interested"
- `3-4` → "somewhat low" / "mild interest"
- `5-6` → "moderate" / "decent interest"
- `7-8` → "quite high" / "strong interest"
- `9-10` → "very high" / "extremely interested"

**Output:** Natural language persona narratives

---

### Phase 2: Conversation Simulation Enhancement

#### Task 2.1: Update System Prompts
**File:** `experiments/speed_dating_simulator.py`

**Changes:**
```python
def create_enhanced_system_prompt(persona_narrative, partner_info):
    return f"""
    You are participating in a speed dating event tonight. Over the course 
    of the evening, you will have brief 4-minute conversations with 20 
    different people. After each conversation, you'll indicate whether you'd 
    like to see that person again. Remember, you're looking for someone 
    special among many potential matches.
    
    YOUR PROFILE:
    {persona_narrative}
    
    YOUR DATING PARTNER FOR THIS CONVERSATION:
    {partner_info['brief_description']}
    Age: {partner_info['age']} years old
    Background: {partner_info['race_description']}
    {"You share the same racial/ethnic background." if partner_info['same_race'] else "You have different racial/ethnic backgrounds."}
    
    Be natural, authentic, and engage genuinely in the conversation.
    """
```

**Output:** Enhanced conversation context

---

### Phase 3: Evaluation Enhancement

#### Task 3.1: Create Two-Question Evaluation
**File:** `experiments/llm_score_evaluator.py`

**New Evaluation Format:**
```python
def create_evaluation_prompt(stage: int, reflection_context: str = None):
    """
    Stage 1: Immediate post-conversation
    Stage 2: With reflection context
    """
    
    base_prompt = """
    Based on your conversation with {partner_name}, please answer:
    
    Question 1: How much do you like this person?
    Scale: 1 (don't like at all) to 10 (like a lot)
    Your rating: 
    
    Question 2: Would you like to see him or her again?
    Answer (Yes or No):
    """
    
    if stage == 2 and reflection_context:
        prompt = f"""
        After meeting {partner_name}, you've had time to reflect on the experience.
        
        {reflection_context}
        
        Given this reflection:
        """ + base_prompt
    else:
        prompt = base_prompt
    
    return prompt
```

**Observer Prompt Variation:**
```python
observer_prompt = """
Based on the conversation between {person1} and {person2}:

Question 1: How compatible do you think they are?
Scale: 1 (not compatible at all) to 10 (extremely compatible)
Your rating:

Question 2: Do you think they should see each other again?
Answer (Yes or No):
"""
```

---

#### Task 3.2: Implement Two-Stage Evaluation
**File:** `experiments/llm_score_evaluator.py`

**New Methods:**

1. **Stage 1 Evaluation** (Current + Two Questions)
```python
async def evaluate_stage1_participant(conversation, persona):
    """Immediate post-conversation evaluation"""
    response = await get_llm_response(
        conversation=conversation,
        persona=persona,
        prompt=create_evaluation_prompt(stage=1)
    )
    
    return {
        'like_score': extract_rating(response),  # 1-10
        'see_again': extract_decision(response),  # Yes/No
        'stage': 1
    }
```

2. **Stage 2 Evaluation** (With Reflection)
```python
async def evaluate_stage2_participant(conversation, persona, time2_data):
    """Reflection-based evaluation"""
    
    # Convert Time 2 data to natural language
    reflection_context = encode_time2_reflection(time2_data)
    
    response = await get_llm_response(
        conversation=conversation,
        persona=persona,
        prompt=create_evaluation_prompt(
            stage=2, 
            reflection_context=reflection_context
        )
    )
    
    return {
        'like_score': extract_rating(response),  # 1-10
        'see_again': extract_decision(response),  # Yes/No
        'stage': 2,
        'reflection_used': True
    }
```

3. **Combined Scoring**
```python
def combine_stage_scores(stage1, stage2, weight_stage1=0.4, weight_stage2=0.6):
    """
    Combine Stage 1 (immediate) and Stage 2 (reflected) scores
    Default: Stage 2 weighted higher (reflection is more thoughtful)
    """
    combined_like = (
        stage1['like_score'] * weight_stage1 + 
        stage2['like_score'] * weight_stage2
    )
    
    # Binary: Both stages must say Yes
    combined_decision = (stage1['see_again'] and stage2['see_again'])
    
    return {
        'stage1': stage1,
        'stage2': stage2,
        'combined_like': combined_like,
        'combined_decision': combined_decision
    }
```

---

### Phase 4: Evaluation Metrics Update

#### Task 4.1: Expand Evaluation Framework
**File:** `experiments/evaluate_like_improved.py`

**New Methods to Evaluate:**
1. **Participant Stage 1** (immediate)
2. **Participant Stage 2** (reflected)
3. **Participant Combined** (weighted average)
4. **Observer Stage 1** (immediate)
5. **Observer Stage 2** (reflected, using averaged Time 2 data)
6. **Observer Combined** (weighted average)
7. **Advanced Observer Stage 1** (with ICL)
8. **Advanced Observer Stage 2** (with ICL + reflection)
9. **Advanced Observer Combined**

**Total: 9 new LLM methods** (replacing 3 old ones)

---

## 🔄 Complete Pipeline Execution Order

### Step 0: Feature Encoding Preparation
```bash
# Create feature encoder
python experiments/feature_encoder.py \
  --input "Speed Dating Data.csv" \
  --output results/encoded_narratives.json \
  --model gpt-4
```
**Output:** Encoded natural language narratives  
**Time:** ~15-20 minutes  
**Cost:** ~$5-8 (GPT-4 calls for 200 personas)

---

### Step 1: Enhanced Persona Generation
```bash
python experiments/persona_generator_v2.py \
  --input "Speed Dating Data.csv" \
  --encoded-narratives results/encoded_narratives.json \
  --output results/personas_enhanced.json \
  --num-pairs 100
```
**Changes:**
- Load encoded narratives
- Add partner demographic features (age, race, same_race)
- Include Time 2 reflection data (satis_2, attr2_*, etc.)
- Enhanced system prompts (20-person context)

**Output:** `results/personas_enhanced.json`

---

### Step 2: Enhanced Conversation Simulation
```bash
python experiments/speed_dating_simulator_v2.py \
  --pairs results/personas_enhanced.json \
  --output-dir results \
  --num-rounds 5
```
**Changes:**
- Use enhanced system prompts
- Include partner demographic context
- Add speed dating context (20 people total)

**Output:** `results/conversations_enhanced.json`

---

### Step 3: Create ICL Examples
```bash
# Same as before, but using enhanced data
python experiments/create_icl_examples.py \
  --conversations results/conversations_enhanced.json \
  --personas results/personas_enhanced.json \
  --output results/icl_examples_enhanced.json
```

---

### Step 4: Two-Stage LLM Evaluation
```bash
python experiments/llm_score_evaluator_v2.py \
  --conversations results/conversations_enhanced.json \
  --personas results/personas_enhanced.json \
  --output-dir results \
  --method both \
  --two-stage \
  --icl-examples results/icl_examples_enhanced.json
```
**New Features:**
- Two-question format (like 1-10 + see again Y/N)
- Stage 1: Immediate evaluation
- Stage 2: Reflection-based evaluation
- Combined scoring (weighted)
- Separate results for each stage + combined

**Output:** `results/llm_score_evaluation_v2.json`

---

### Step 5: Ensemble Models (if needed)
```bash
python experiments/ensemble_model_v2.py \
  --llm-results results/llm_score_evaluation_v2.json \
  --output results/ensemble_evaluation_v2.json
```

---

### Step 6: Baseline Models (unchanged)
```bash
python experiments/baseline_models_v2.py \
  --personas results/personas_enhanced.json \
  --output-dir results
```

---

### Step 7: Comprehensive Evaluation
```bash
python experiments/evaluate_like_improved_v2.py \
  --llm-results results/llm_score_evaluation_v2.json \
  --ensemble-results results/ensemble_evaluation_v2.json \
  --baseline-results results/baseline_comparison_v2.json \
  --output-dir results
```
**Evaluates:**
- 9 LLM methods (3 base × 3 stages)
- 2 Ensemble methods
- 8 Baseline methods
**Total: 19 methods**

**Output:**
- `results/like_score_evaluation_v2.json`
- `results/like_score_comparison_v2.png`

---

## 📊 Expected Improvements

### Hypothesis 1: Inverse Relationship Resolution
**Current:** Observer k = -0.37 (inverse!)  
**Expected:** Positive correlation with enhanced features
- Partner demographics provide critical context
- Natural language encoding improves LLM understanding
- Two-stage evaluation captures reflection (like humans)

### Hypothesis 2: Improved Correlation
**Current:** Best Pearson r = 0.20 (Observer)  
**Expected:** Pearson r > 0.30 with enhancements
- Richer features → Better predictions
- Two-stage reflection → More human-like evaluation
- Natural language → Better LLM comprehension

### Hypothesis 3: Better Calibration
**Current:** High F1 (0.91) but weak correlation  
**Expected:** Improved correlation WITHOUT sacrificing F1
- Explicit rating scale (1-10) reduces ambiguity
- Binary question aligns with human decision-making

---

## 🎯 Success Metrics

### Primary Goals
1. **Eliminate inverse relationship:** k > 0 for all LLM methods
2. **Improve correlation:** Pearson r > 0.30 (50% improvement)
3. **Maintain classification:** F1 > 0.85

### Secondary Goals
4. **Stage 2 better than Stage 1:** Reflection improves accuracy
5. **Two-question format more informative:** Binary decision adds value
6. **Natural language encoding helps:** GPT-4 narratives improve GPT-4o-mini performance

---

## 📝 Implementation Checklist

### Phase 1: Data Enhancement
- [ ] Create `feature_encoder.py` with GPT-4 encoding
- [ ] Update `persona_generator.py` → `persona_generator_v2.py`
  - [ ] Add partner demographic features
  - [ ] Extract Time 2 reflection data
  - [ ] Integrate encoded narratives
  - [ ] Enhanced system prompts

### Phase 2: Simulation Enhancement
- [ ] Update `speed_dating_simulator.py` → `speed_dating_simulator_v2.py`
  - [ ] Enhanced system prompts (20-person context)
  - [ ] Partner demographic inclusion

### Phase 3: Evaluation Enhancement
- [ ] Update `llm_score_evaluator.py` → `llm_score_evaluator_v2.py`
  - [ ] Two-question evaluation format
  - [ ] Stage 1 evaluation (immediate)
  - [ ] Stage 2 evaluation (reflection)
  - [ ] Combined scoring logic
  - [ ] Time 2 data encoding to natural language

### Phase 4: Final Evaluation
- [ ] Update `evaluate_like_improved.py` → `evaluate_like_improved_v2.py`
  - [ ] Support 9 LLM methods (3 stages × 3 types)
  - [ ] Stage comparison analysis
  - [ ] Before/after comparison with v1

### Phase 5: Documentation
- [ ] Update `EXPERIMENTAL_PIPELINE.md` with v2 changes
- [ ] Create `ENHANCEMENT_RESULTS.md` comparing v1 vs v2
- [ ] Update `README.md` with new pipeline

---

## ⏱️ Estimated Timeline

| Phase | Tasks | Time | Cost |
|-------|-------|------|------|
| Feature Encoding | Encode 200 personas | 15-20 min | $5-8 |
| Persona Generation | Enhanced personas | 5 min | $0 |
| Conversation Sim | 100 pairs × 5 rounds | 30-45 min | $2-3 |
| Two-Stage Eval | Stage 1 + Stage 2 | 40-60 min | $3-5 |
| Baselines | Same as before | 30 sec | $0 |
| Final Evaluation | All 19 methods | 10 sec | $0 |
| **Total** | **Full pipeline** | **~2-3 hours** | **~$10-16** |

---

## 🚀 Ready to Execute?

**Next Steps:**
1. Review this enhancement plan
2. Approve changes
3. Begin implementation with Phase 1 (Feature Encoding)
4. Execute full pipeline
5. Compare v1 vs v2 results

**Expected Outcome:**
- Resolved inverse relationship issue
- Improved correlation with human ratings
- More interpretable LLM evaluations
- Comprehensive comparison of enhanced methods

---

**Status:** 📋 Plan Complete - Ready for Implementation  
**Last Updated:** November 4, 2025
