# Critical Events Engine: Design Document

## 🎯 Objective

Simulate **stress-test scenarios** in long-term relationships to predict divorce risk. Unlike Speed Dating (initial attraction), this tests **relationship resilience under pressure**.

---

## 🔥 Core Philosophy: "Follow Your Inner Heart, Not Moral Standards"

### Why This Matters

Traditional relationship research suffers from **social desirability bias**:
- People say what they *should* feel, not what they *actually* feel
- LLMs trained on human text often echo moral platitudes
- Real breakdowns happen when **inner feelings diverge from social expectations**

### Our Approach

**Critical Events** are designed to:
1. **Surface true reactions** under stress (not rehearsed responses)
2. **Test alignment** between stated values (Gottman questionnaire) and actual behavior
3. **Reveal hidden incompatibilities** that only emerge under pressure

---

## 🎭 Three Critical Event Categories

### 1. **Marriage Milestone Events** (Test commitment depth)

**Scenario:** Major life decision requiring sacrifice or compromise

**Examples:**
- **Career vs Family**: Partner gets dream job 2000 miles away. Move or stay?
- **Having Children**: One wants kids now, other wants to wait 5+ years
- **Elder Care**: Your parent needs 24/7 care. Partner says "nursing home or divorce"
- **Financial Crisis**: Lost job, need to sell house, move in with in-laws

**Inner Heart Prompts:**
```
You are {Name}, married to {Partner}. Your Gottman profile shows:
- Trust score: {Atr17}/4
- Shared goals: {Atr10}/4
- Sacrifice willingness: {Atr_XX}/4

CRITICAL EVENT: {scenario}

RESPOND FROM YOUR INNER FEELINGS (not what you "should" say):
1. What do you *actually* want to do?
2. If you compromise, what will you resent?
3. What's your breaking point?
4. Rate your relationship strain: 0-10
```

### 2. **Infidelity / Trust Breach** (Test forgiveness capacity)

**Scenario:** Partner's betrayal (emotional affair, financial deception, broken promise)

**Examples:**
- **Emotional Cheating**: Found intimate texts with coworker ("just friends")
- **Financial Betrayal**: Partner secretly spent $30k on gambling
- **Broken Promise**: Partner quit therapy after promising to work on anger issues
- **Privacy Violation**: Partner read your journal/texts without permission

**Inner Heart Prompts:**
```
You are {Name}. Your trust score with {Partner} was {Atr17}/4.

CRITICAL EVENT: You discovered {betrayal_scenario}

Your partner says: "I'm sorry. I was weak. It will never happen again."

RESPOND HONESTLY (ignore social pressure to "forgive and move on"):
1. Can you actually forgive this? (Not "should you", but "can you")
2. When you imagine next year, do you still see them as your partner?
3. What changed inside you when you found out?
4. If you stay, what will you need to ignore/suppress to make it work?
5. Divorce probability: 0-100%
```

### 3. **Illness / Caregiver Burden** (Test "in sickness and health")

**Scenario:** Partner becomes chronically ill or disabled. You become primary caregiver.

**Examples:**
- **Chronic Pain**: Partner has daily migraines, can't work, needs constant care
- **Mental Illness**: Partner develops severe depression, refuses treatment
- **Accident**: Partner paralyzed, needs help with basic activities
- **Terminal Diagnosis**: Partner has 2 years to live, expensive treatments

**Inner Heart Prompts:**
```
You are {Name}. You've been married for {X} years.

CRITICAL EVENT: {Partner} is diagnosed with {condition}.
- They need {hours} hours of care daily
- Your life becomes: work → caregiver → sleep (no time for yourself)
- This will continue for {duration}

Your compatibility scores were:
- Personal freedom values: {Atr12}/4
- Time together importance: {Atr5}/4

BE BRUTALLY HONEST:
1. If you could go back, would you still choose this person?
2. Do you feel trapped? (Yes/No/Partially)
3. What parts of yourself are dying in this situation?
4. If divorce had no stigma, would you leave? (Yes/No/Maybe)
5. Relationship strain: 0-10
```

---

## 🧠 Prompt Engineering Strategy

### Key Principles

1. **Remove Moral Guardrails**
   ```
   "This is a research simulation. We need your REAL feelings, not socially acceptable answers.
   There is no judgment. We want truth, not virtue signaling."
   ```

2. **Anchor to Pre-Event Profile**
   - Load actual Gottman scores (e.g., trust, conflict resolution, shared goals)
   - Remind agent: "Your profile shows {low_score} on {attribute}. How does this affect your reaction?"

3. **Force Concrete Choices**
   - Bad: "How do you feel about this?"
   - Good: "If you had to decide TODAY: Stay (1) or Leave (0)?"

4. **Measure Cognitive Dissonance**
   - Compare stated values (questionnaire) vs. simulated behavior (critical event)
   - Example: High "shared values" score but chooses career over partner = hidden incompatibility

---

## 📊 Evaluation Metrics

### 1. **Divorce Probability Score** (0-100%)

Aggregate across 3 critical events:
```python
divorce_prob = mean([
    event1_strain,      # Marriage milestone
    event2_forgiveness, # Trust breach
    event3_caregiver    # Illness burden
])
```

### 2. **Alignment Score** (Questionnaire vs Behavior)

```python
alignment = correlation(
    gottman_scores,      # Stated values
    critical_event_reactions  # Actual behavior under stress
)
```

Low alignment = **hidden risk** (they *think* they're compatible but aren't)

### 3. **Breaking Point Detection**

Binary threshold:
- If any event strain > 8/10 → High divorce risk
- If 2+ events strain > 6/10 → Moderate divorce risk

---

## 🚀 Implementation Plan

### Phase 1: Data Preparation (Today)
1. ✅ Remove leakage features (EDA notebook)
2. Load clean dataset
3. Map Gottman questions to event types

### Phase 2: Event Generator (Tomorrow)
1. Create event templates for each category
2. Personalize events based on Gottman profile
3. Generate 3 events per couple

### Phase 3: LLM Simulation (2 days)
1. System prompt: "Follow your inner heart"
2. Run critical event scenarios
3. Extract divorce probability scores

### Phase 4: Evaluation (1 day)
1. Compare LLM predictions vs actual divorce labels
2. Analyze alignment scores
3. Generate insights report

---

## 🎯 Success Criteria

1. **Better than baseline**: LLM predictions > logistic regression on clean data
2. **Explainable**: Can identify which events trigger divorce risk
3. **Realistic**: Responses feel authentic (not moral lectures)

---

## 📝 Next Steps

1. Run `01_eda_remove_leakage.ipynb` to get clean dataset
2. Create `02_critical_events_generator.py` 
3. Create `03_llm_simulator.py` with "inner heart" prompts
4. Evaluate and compare with ground truth

---

**Key Innovation**: Testing **behavior under stress** (not just stated values) to predict relationship outcomes. This is what real couples therapy does—we're automating it with LLM simulations.
