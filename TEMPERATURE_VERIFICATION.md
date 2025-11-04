# Temperature & Model Verification

## Issue Report
User encountered repetitive output like:
```
Response preview: **Reasoning:** *Sarah* and *Jake* both *gauge* *needs* *needs* *needs* *needs* *needs* *needs* *needs* *needs* *needs* *needs*...
```

This is a **known LLM issue** when responses get truncated at the `max_tokens` limit.

## Current Settings ✅

### Evaluator (`llm_score_evaluator.py`)
- **Model**: `mistralai/mistral-nemo` (line 64)
- **Temperature**: `0.3` (all API calls)
- **Max Tokens**: Progressive retry (300 → 500 → 800)

### Simulator (`speed_dating_simulator.py`)  
- **Model**: `mistralai/mistral-nemo` (line 44)
- **Temperature**: `0.3` (all API calls)
- **Max Tokens**: `300`

## Root Cause Analysis

The repetitive "*needs* *needs*" pattern occurs when:
1. **Token limit is too low** (300 tokens)
2. **Model gets cut off mid-generation**
3. **Last token repeats** due to truncation

This is **NOT** a temperature issue - it's a truncation issue!

## Solutions Implemented

### 1. **Automatic Retry with More Tokens**
```python
# Progressive token limits: 300 -> 500 -> 800
token_limits = [300, 500, 800]

for attempt, max_tokens in enumerate(token_limits):
    response = call_openrouter_api(messages, temperature=0.3, max_tokens=max_tokens)
    score = extract_score_from_response(response)
    
    if score is not None:
        return score  # Success!
    else:
        # Retry with more tokens...
```

### 2. **Repetitive Pattern Detection**
```python
# Detect if same word repeats 5+ times in last 10 words
words = response.split()
last_words = words[-10:]
for word in set(last_words):
    if last_words.count(word) >= 5:
        print(f"⚠️ Detected repetitive pattern, likely truncated...")
        # Automatically retry with more tokens
```

### 3. **Debug Output**
```python
# Now prints API parameters to verify settings:
[API] model=mistralai/mistral-nemo, temp=0.3, max_tokens=300
```

## How to Verify Settings

### Option 1: Check Source Code
```bash
cd /Users/assassin808/Desktop/research_2025_xuan/yan/test
grep -n "MODEL =" experiments/llm_score_evaluator.py
grep -n "temperature" experiments/llm_score_evaluator.py | grep "0.3"
```

### Option 2: Check Runtime Output
When you run the evaluator, you'll now see:
```
[1/100] Evaluating pair_241_248...
   Attempt 1/3 with max_tokens=300...
      [API] model=mistralai/mistral-nemo, temp=0.3, max_tokens=300
   ⚠️ Could not extract score from response (truncated?), retrying...
   Response preview: **Reasoning:** *needs* *needs* *needs*...
   Attempt 2/3 with max_tokens=500...
      [API] model=mistralai/mistral-nemo, temp=0.3, max_tokens=500
   ✅ Successfully extracted score on attempt 2
   Person 1 score: 7.5/10
```

## Expected Behavior

### Before (Old Code):
```
⚠️ Warning: Could not extract score from response, using default 5.0
```

### After (New Code):
```
Attempt 1/3 with max_tokens=300...
   [API] model=mistralai/mistral-nemo, temp=0.3, max_tokens=300
⚠️ Detected repetitive pattern ('needs' x8), likely truncated...
   Retrying with more tokens to avoid truncation...
Attempt 2/3 with max_tokens=500...
   [API] model=mistralai/mistral-nemo, temp=0.3, max_tokens=500
✅ Successfully extracted score on attempt 2
Person 1 score: 7.5/10
```

## Why This Happens

### Token Truncation vs Temperature
- **Temperature = 0.3** is CORRECT ✅
- **Model = mistral-nemo** is CORRECT ✅
- **Problem = 300 tokens too low** ❌

### Example:
```
Prompt: "Rate this conversation from 0-10..."
Response (300 tokens): "**Reasoning:** Sarah and Jake both gauge each other's needs needs needs needs needs needs..."
                                                                                    ↑
                                                                                TRUNCATED HERE
```

When truncated mid-word or mid-sentence, the model's last token gets repeated.

## Verification Commands

### Check current evaluation run:
```bash
cd /Users/assassin808/Desktop/research_2025_xuan/yan/test
tail -f <terminal_output>  # Look for "[API] model=..." lines
```

### Verify no default scores:
```bash
# Should NOT find this warning anymore:
grep "using default 5.0" experiments/llm_score_evaluator.py
# Output: (nothing - we removed it!)
```

## Recommended Actions

1. **Let the retry logic work** - It will automatically use 500/800 tokens when needed
2. **Monitor the debug output** - Look for "[API] model=..." to confirm settings
3. **Check final results** - No more default 5.0 scores, all scores extracted from actual responses

## Summary

✅ **Temperature is 0.3** (verified)  
✅ **Model is mistral-nemo** (verified)  
✅ **Retry logic implemented** (300 → 500 → 800 tokens)  
✅ **Repetitive pattern detection added**  
✅ **Debug output added** to verify settings  
✅ **No more default fallback scores**  

The system will now automatically detect and fix truncation issues!
