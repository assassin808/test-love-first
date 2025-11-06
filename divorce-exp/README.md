# Divorce Prediction with Critical Events Engine

**Research Question:** Can LLMs predict divorce by simulating how couples react to stress-test scenarios?

## 🎯 Core Innovation

Unlike traditional ML (using questionnaire scores), we:
1. **Generate personalized critical events** based on Gottman profile
2. **Prompt LLM to follow "inner heart" not "moral standards"**
3. **Measure divorce risk** from simulated reactions under stress

This tests **behavior under pressure**, not just stated values.

---

## 📊 Dataset

- **Source:** Gottman Divorce Predictors Scale (DPS)
- **Size:** 150 couples (86 divorced, 64 married)
- **Features:** 54 questions on 0-4 scale (0=Never, 4=Always)
- **Challenge:** 4 features have data leakage (98% accuracy) → removed

---

## 🔥 Critical Events (3 Categories)

### 1. Marriage Milestone
Tests **commitment depth** with major life decisions:
- Career vs Family (dream job requires relocation)
- Children timing (partner wants baby NOW, you want to wait)
- Financial crisis (lost job, must move in with in-laws)

### 2. Trust Breach  
Tests **forgiveness capacity** after betrayal:
- Emotional affair (intimate texts with coworker)
- Financial deception (secret gambling, hidden debt)
- Broken promise (quit therapy after promising to change)

### 3. Illness/Caregiver Burden
Tests **"in sickness and health"** endurance:
- Chronic pain (daily migraines, can't work, needs constant care)
- Mental illness (severe depression, refuses treatment)
- Accident (paralysis, wheelchair-bound, angry caregiver burden)

**Key:** Events are **personalized** based on Gottman profile (e.g., high "trust" score → bigger betrayal needed to test forgiveness).

---

## 🛠 Pipeline

### Step 1: EDA + Remove Leakage Features

```bash
# Run notebook to identify and remove 4 leakage features
jupyter notebook 01_eda_remove_leakage.ipynb
```

**Output:** `divorce_clean.csv` (50 features instead of 54)

### Step 2: Generate Critical Events

```bash
python 02_critical_events_generator.py
```

**Output:** `critical_events.json` (450 events = 150 couples × 3 events)

### Step 3: LLM Simulation (TODO)

```bash
python 03_llm_simulator.py \
  --events critical_events.json \
  --output simulated_responses.json \
  --model mistralai/mistral-nemo
```

**Prompt Strategy:**
```
You are {Person}, married to {Spouse}. Your Gottman profile shows:
- Trust: {score}/4
- Shared goals: {score}/4
- Conflict resolution: {score}/4

CRITICAL EVENT: {scenario}

RESPOND FROM YOUR INNER FEELINGS (not what you "should" say):
1. What do you ACTUALLY want to do?
2. If you compromise, what will you resent?
3. What's your breaking point?
4. Rate relationship strain: 0-10
5. Divorce probability if this continues: 0-100%
```

### Step 4: Evaluation (TODO)

```bash
python 04_evaluate_predictions.py \
  --predictions simulated_responses.json \
  --ground_truth divorce_clean.csv \
  --output evaluation_report.json
```

**Metrics:**
- Accuracy, Precision, Recall, F1, AUC-ROC
- Compare vs baseline ML (logistic regression, random forest)
- Alignment score (questionnaire vs simulated behavior)

---

## 📁 Project Structure

```
divorce-exp/
├── README.md                           # This file
├── CRITICAL_EVENTS_DESIGN.md           # Detailed design rationale
├── doc.txt                             # Original dataset documentation
│
├── divorce.csv                         # Raw dataset (54 features)
├── divorce_clean.csv                   # Clean dataset (50 features, leakage removed)
├── leakage_features.txt                # List of removed features
│
├── 01_eda_remove_leakage.ipynb         # [Step 1] EDA + remove leakage
├── 02_critical_events_generator.py     # [Step 2] Generate personalized events
├── 03_llm_simulator.py                 # [Step 3] Simulate with LLM (TODO)
├── 04_evaluate_predictions.py          # [Step 4] Evaluate vs ground truth (TODO)
│
├── critical_events.json                # Generated events (450 scenarios)
├── simulated_responses.json            # LLM responses (TODO)
└── evaluation_report.json              # Final results (TODO)
```

---

## 🔬 Key Research Questions

1. **Can LLMs predict divorce from simulated behavior?**
   - Hypothesis: Reactions to critical events > questionnaire scores alone

2. **Which events are most predictive?**
   - Marriage milestone? Trust breach? Illness burden?

3. **Do "inner heart" prompts improve prediction?**
   - Compare standard prompts vs "follow your inner feelings" prompts

4. **Alignment Score:**
   - Do couples with high stated compatibility (Gottman scores) but poor simulated reactions have higher divorce risk?

---

## 🎯 Success Criteria

1. **Beat baseline ML:** LLM predictions > logistic regression on clean data
2. **Explainability:** Can identify which events trigger divorce risk
3. **Authenticity:** Responses feel real (not moral platitudes)
4. **Actionable insights:** Reveal hidden incompatibilities missed by questionnaires

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn openai python-dotenv

# 2. Set up API key
cp .env.example .env
# Edit .env and add OPENROUTER_API_KEY

# 3. Run EDA (Jupyter)
jupyter notebook 01_eda_remove_leakage.ipynb

# 4. Generate events
python 02_critical_events_generator.py

# 5. Simulate (TODO - coming next)
python 03_llm_simulator.py
```

---

## 📊 Expected Results

### Baseline (Traditional ML on Gottman Scores)
- Logistic Regression: ~85% accuracy
- Random Forest: ~88% accuracy
- XGBoost: ~90% accuracy

### Our Approach (LLM + Critical Events)
- **Target:** >90% accuracy
- **Bonus:** Explainable predictions (which event revealed the risk)

---

## 💡 Key Innovation

**Testing behavior under stress, not just stated values.**

Traditional research asks: "Do you trust your spouse?" (0-4 scale)

We test: "Your spouse betrayed you. Can you forgive them?" (simulated reaction)

This is closer to **real couples therapy** than questionnaires.

---

## 🔗 Related Work

- **Speed Dating Experiment** (same project, different stage of relationship)
  - Initial attraction prediction (first 5 minutes)
  - This project: Long-term resilience prediction (years of marriage)

---

## 📝 Next Steps

1. ✅ EDA + remove leakage features
2. ✅ Generate critical events
3. ⏳ Implement LLM simulator with "inner heart" prompts
4. ⏳ Run simulations on all 150 couples
5. ⏳ Evaluate predictions vs ground truth
6. ⏳ Analyze which events are most predictive
7. ⏳ Compare "inner heart" vs standard prompts

---

**Status:** 🟡 In Progress (Step 2/4 complete)

**Last Updated:** November 5, 2025
