# Repetition Penalty Retry Strategy

## Problem: Repetitive Output
When LLMs generate repetitive patterns like:
```
*needs* *needs* *needs* *needs* *needs* *needs* *needs* *needs*...
```

This indicates the model is stuck in a loop, which can be caused by:
1. **Token truncation** (max_tokens too low)
2. **Low repetition penalty** (model not penalized for repeating tokens)

## New Solution: Progressive Strategy

Instead of just increasing `max_tokens`, we now use a **4-stage progressive strategy** that combines both approaches:

### Stage 1: Normal Request (Baseline)
```python
max_tokens = 300
repetition_penalty = 1.0  # No penalty (default)
```
- **When**: First attempt
- **Purpose**: Try with default settings
- **Output**: `[API] model=mistralai/mistral-nemo, temp=0.3, max_tokens=300`

### Stage 2: More Tokens (Handle Truncation)
```python
max_tokens = 500
repetition_penalty = 1.0  # No penalty
```
- **When**: First attempt failed or found repetitive pattern
- **Purpose**: Give model more space to complete response
- **Output**: `[API] model=mistralai/mistral-nemo, temp=0.3, max_tokens=500`

### Stage 3: More Tokens + Light Penalty
```python
max_tokens = 800
repetition_penalty = 1.1  # Light penalty
```
- **When**: Previous attempts still showing repetition
- **Purpose**: Penalize repetitive tokens while allowing natural repetition
- **Output**: `[API] model=mistralai/mistral-nemo, temp=0.3, max_tokens=800, rep_penalty=1.1`

### Stage 4: Strong Penalty (Nuclear Option)
```python
max_tokens = 800
repetition_penalty = 1.2  # Strong penalty
```
- **When**: All previous attempts failed
- **Purpose**: Strongly discourage any token repetition
- **Output**: `[API] model=mistralai/mistral-nemo, temp=0.3, max_tokens=800, rep_penalty=1.2`

## How Repetition Penalty Works

### Formula (Simplified):
```
token_probability = original_probability / (repetition_penalty ^ count)
```

Where `count` = number of times token already appeared

### Examples:

**repetition_penalty = 1.0 (No penalty):**
- Word appears 1st time: probability = 100%
- Word appears 5th time: probability = 100%
- Result: Can repeat infinitely

**repetition_penalty = 1.1 (Light):**
- Word appears 1st time: probability = 100%
- Word appears 5th time: probability ≈ 68%
- Result: Gentle discouragement

**repetition_penalty = 1.2 (Strong):**
- Word appears 1st time: probability = 100%
- Word appears 5th time: probability ≈ 40%
- Result: Strong discouragement

## Detection Logic

### Repetitive Pattern Detection:
```python
# Check last 10 words
words = response.split()
last_words = words[-10:]

# If any word appears 5+ times in last 10 words
for word in set(last_words):
    if last_words.count(word) >= 5:
        # REPETITIVE PATTERN DETECTED!
        # Trigger retry with higher repetition_penalty
```

### Example Detection:
```
Response: "Sarah and Jake gauge needs needs needs needs needs needs needs needs needs"
                                    ↑ "needs" appears 8 times in last 10 words
                                    ↑ DETECTED as repetitive pattern
                                    ↑ Triggers Stage 3 with rep_penalty=1.1
```

## Expected Behavior

### Example 1: Success on Stage 2 (Truncation Issue)
```
[1/100] Evaluating pair_241_248...
   Attempt 1/4 with max_tokens=300...
      [API] model=mistralai/mistral-nemo, temp=0.3, max_tokens=300
   ⚠️ Could not extract score from response, trying next strategy...
   Response preview: **Reasoning:** Sarah and Jake both gauge needs needs needs...
   
   Attempt 2/4 with max_tokens=500...
      [API] model=mistralai/mistral-nemo, temp=0.3, max_tokens=500
   ✅ Successfully extracted score on attempt 2
   Person 1 score: 7.5/10
```

### Example 2: Success on Stage 3 (Repetition Issue)
```
[2/100] Evaluating pair_271_282...
   Attempt 1/4 with max_tokens=300...
      [API] model=mistralai/mistral-nemo, temp=0.3, max_tokens=300
   ⚠️ Detected repetitive pattern ('needs' x8)
   Retrying with higher repetition penalty to fix repetition issue...
   
   Attempt 2/4 with max_tokens=500...
      [API] model=mistralai/mistral-nemo, temp=0.3, max_tokens=500
   ⚠️ Detected repetitive pattern ('needs' x6)
   Retrying with higher repetition penalty to fix repetition issue...
   
   Attempt 3/4 with max_tokens=800, repetition_penalty=1.1...
      [API] model=mistralai/mistral-nemo, temp=0.3, max_tokens=800, rep_penalty=1.1
   ✅ Successfully extracted score on attempt 3
   Observer score: 6.0/10
```

### Example 3: Nuclear Option (Stage 4)
```
[5/100] Evaluating pair_467_492...
   Attempt 1/4 with max_tokens=300...
   ⚠️ Detected repetitive pattern ('very' x7)
   
   Attempt 2/4 with max_tokens=500...
   ⚠️ Detected repetitive pattern ('very' x5)
   
   Attempt 3/4 with max_tokens=800, repetition_penalty=1.1...
   ⚠️ Detected repetitive pattern ('very' x5)
   
   Attempt 4/4 with max_tokens=800, repetition_penalty=1.2...
      [API] model=mistralai/mistral-nemo, temp=0.3, max_tokens=800, rep_penalty=1.2
   ✅ Successfully extracted score on attempt 4
   Person 2 score: 8.0/10
```

## Key Improvements

### Before:
```python
# Old approach: only increase tokens
strategies = [300, 500, 800]  # Just token increases
# Problem: Doesn't fix repetition loops, just gives them more space!
```

### After:
```python
# New approach: combine tokens + repetition penalty
strategies = [
    (300, 1.0),  # Normal
    (500, 1.0),  # More space
    (800, 1.1),  # More space + light penalty
    (800, 1.2)   # Nuclear option
]
# Solution: Fixes both truncation AND repetition issues!
```

## Technical Details

### API Parameter:
```json
{
  "model": "mistralai/mistral-nemo",
  "messages": [...],
  "temperature": 0.3,
  "max_tokens": 800,
  "repetition_penalty": 1.2  ← NEW PARAMETER
}
```

### OpenRouter Support:
✅ `repetition_penalty` is supported by OpenRouter API
✅ Mistral models support this parameter
✅ Default value is 1.0 (no penalty)
✅ Recommended range: 1.0 - 1.5

### Why Not Always Use High Penalty?
- **1.0**: Natural language, can repeat when appropriate
- **1.1**: Gentle correction, good for most cases
- **1.2**: Strong correction, may make text less natural
- **1.5+**: Too aggressive, may produce nonsensical output

## Validation

### Check it's working:
```bash
# Look for repetition_penalty in debug output
cd /Users/assassin808/Desktop/research_2025_xuan/yan/test
python experiments/llm_score_evaluator.py --conversations results/conversations.json \
  --output-dir results --max-pair-workers 5 --method both \
  --threshold 0.5 --report-curves --icl-examples results/icl_examples.json | \
  grep "rep_penalty"

# Should see lines like:
# [API] model=mistralai/mistral-nemo, temp=0.3, max_tokens=800, rep_penalty=1.1
# [API] model=mistralai/mistral-nemo, temp=0.3, max_tokens=800, rep_penalty=1.2
```

## Summary

✅ **4-stage progressive retry strategy**  
✅ **Detects repetitive patterns automatically**  
✅ **Increases repetition_penalty when needed**  
✅ **Combines token increase + penalty increase**  
✅ **Handles both truncation AND repetition issues**  
✅ **Debug output shows strategy used**  

The system now intelligently adapts to fix repetitive output by increasing both space (max_tokens) and repetition penalties!
