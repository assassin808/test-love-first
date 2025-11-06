# Divorce Prediction via Critical Event Simulation with LLMs

**Research Project:** Can Large Language Models predict divorce by simulating couple interactions under relationship stress tests?

## 🎯 Project Overview

This research explores whether **Large Language Models (LLMs)** can predict divorce outcomes by:
1. Converting structured relationship survey data into rich couple personas
2. Simulating critical life events that stress-test relationship resilience
3. Analyzing couple interactions through multi-round conversations
4. Using LLM-based evaluation with Observer and Participant perspectives
5. Comparing predictions with In-Context Learning (ICL) calibration

### 🔬 Novel Approach: Critical Event Simulation

**Traditional Approach:** Predict divorce from static questionnaire responses  
**Our Approach:** Simulate couples facing critical events and observe behavioral patterns

**Core Philosophy:** "Follow Your Inner Heart, Not Social Norms"
- Reduces social desirability bias in responses
- Tests alignment between stated values and behavior under pressure
- Surfaces hidden incompatibilities that only emerge during crises

---

## 📊 Key Results

### 🏆 Best Performance: Observer with In-Context Learning

| Method | Accuracy | AUC | Samples |
|--------|----------|-----|---------|
| **Observer-ICL** | **0.90** | **0.918** | 30 |
| Observer | 0.50 | 0.551 | 30 |
| Participant | 0.43 | N/A | 30 |

### 🔍 Key Findings

1. **In-Context Learning Dramatically Improves Performance**
   - Observer-ICL: 90% accuracy, 0.918 AUC
   - 80% improvement over basic Observer method
   - ICL examples help LLM calibrate relationship patterns

2. **Observer > Participant Perspective**
   - Third-party evaluation outperforms self-assessment
   - Participants show self-assessment bias (underestimate divorce risk)
   - External perspective captures hidden patterns better

3. **Agent ICL During Simulation Matters**
   - Injecting ICL examples into agent prompts during simulation
   - Agents follow learned interaction patterns from examples
   - More realistic conversation dynamics

4. **Behavioral Simulation > Static Features**
   - Observing stress responses reveals compatibility better than questionnaires
   - Critical events surface hidden incompatibilities
   - Dynamic interactions capture nuance static data misses

---

## 🛠 Technical Architecture

### Data Pipeline

```
Divorce Prediction Dataset (170 couples, 27 survey questions)
           ↓
    [Data Cleaning & Leakage Removal]
           ↓
    Stratified Sampling (30 couples: 15 divorced, 15 married)
           ↓
    [Persona Generation]
    - Convert survey responses → natural language personas
    - Husband & Wife perspectives with relationship dynamics
           ↓
    [Critical Events Generation]
    - 3 event types per couple (marriage milestone, trust breach, illness)
    - Personalized based on compatibility scores
           ↓
    [Interaction Simulation]
    - Multi-round conversations (6 rounds default)
    - Agent ICL injection (optional)
    - World engine narration with personas + ICL
           ↓
    [Multi-Method Evaluation]
    - Observer: Third-party LLM rates compatibility
    - Observer-ICL: Observer + 10 labeled examples
    - Participant: Self-assessment from each spouse
    - Participant-ICL: Self-assessment + ICL examples
           ↓
    [Calibration & Analysis]
    - Logistic regression calibration
    - AUC computation
    - Confusion matrices
```

### LLM Configuration

| Component | Model | Purpose | Temperature |
|-----------|-------|---------|-------------|
| **Persona Generation** | GPT-4 | Survey → Narratives | 0.7 |
| **Event Generation** | GPT-4 | Critical scenarios | 0.7 |
| **Interaction Sim** | Mistral Nemo | Couple conversations | 0.7 |
| **Evaluation** | Gemini 2.5 Flash Lite | Compatibility scoring | 0.6 |

---

## 📁 Repository Structure

```
divorce-exp/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── .env.example                       # Environment template
│
├── scripts/                           # All pipeline scripts
│   ├── 00_clean_cache.py             # Clear cached files
│   ├── 01_generate_personas.py       # Create couple personas
│   ├── 02_generate_events.py         # Generate critical events
│   ├── 03_simulate_interactions.py   # Simulate conversations
│   ├── 04_evaluate_predictions.py    # Multi-method evaluation
│   ├── 05_calibrate_scores.py        # Logistic calibration
│   └── divorce_questions.py          # Survey question utilities
│
├── data/                              # Input datasets
│   ├── divorce.csv                    # Raw dataset (170 couples)
│   ├── divorce_clean.csv              # Cleaned data (no leakage)
│   ├── divorce_personas.json          # Generated personas
│   └── critical_events.json           # Generated events
│
├── results/                           # Experiment outputs
│   ├── divorce_simulations_*.json     # Simulation logs
│   ├── divorce_evaluation_*.json      # Evaluation results
│   ├── calibration_*.json             # Calibrated scores
│   ├── correlation_heatmap_clean.png  # Feature analysis
│   └── divorce_personas_sample.txt    # Persona examples
│
└── docs/                              # Documentation
    ├── CRITICAL_EVENTS_DESIGN.md      # Event design philosophy
    ├── 01_eda_remove_leakage.ipynb    # EDA + leakage analysis
    ├── leakage_features.txt           # Features removed
    └── doc.txt                        # Additional notes
```

---

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.8+
pip install -r requirements.txt

# Set up API keys
cp .env.example .env
# Edit .env and add your OPENROUTER_API_KEY
```

### View Latest Results (No Running Required)

```bash
# Best results: Observer-ICL on 30 couples
cat results/divorce_evaluation_results_icl_strat30_v4_numeric.json

# Calibration metrics
cat results/calibration_results_icl_strat30_10train.json

# View sample personas
cat results/divorce_personas_sample.txt
```

### Run Quick Test (3 couples, ~2 min)

```bash
cd scripts

# 1. Generate personas (fast, rule-based)
python 01_generate_personas.py \
  --input ../data/divorce_clean.csv \
  --output ../data/divorce_personas.json

# 2. Generate critical events (fast, rule-based)
python 02_generate_events.py \
  --personas ../data/divorce_personas.json \
  --output ../data/critical_events.json

# 3. Simulate interactions (3 couples × 3 events × 6 rounds)
python 03_simulate_interactions.py \
  --personas ../data/divorce_personas.json \
  --events ../data/critical_events.json \
  --output ../results/divorce_simulations_test3.json \
  --num-couples 3 \
  --rounds 6

# 4. Evaluate all methods
python 04_evaluate_predictions.py \
  --simulations ../results/divorce_simulations_test3.json \
  --clean-data ../data/divorce_clean.csv \
  --personas ../data/divorce_personas.json \
  --output ../results/divorce_evaluation_test3.json \
  --method all \
  --threshold 5.0
```

---

## 📝 Full Pipeline Execution

### Step 1: Generate Personas (~1 min, free)

```bash
python scripts/01_generate_personas.py \
  --input data/divorce_clean.csv \
  --output data/divorce_personas.json \
  --num-couples 170
```

**Output:** `data/divorce_personas.json` with 170 couple personas (husband + wife narratives)

### Step 2: Generate Critical Events (~1 min, free)

```bash
python scripts/02_generate_events.py \
  --personas data/divorce_personas.json \
  --output data/critical_events.json \
  --events-per-couple 3
```

**Output:** `data/critical_events.json` with 510 events (170 couples × 3 event types)

### Step 3: Simulate Interactions with Agent ICL (~30-45 min, ~$3-5)

```bash
# Option A: Full 30-couple stratified sample with agent ICL
OPENROUTER_MAX_CONCURRENCY=6 \
OPENROUTER_RPS=0.8 \
OPENROUTER_MODEL=mistralai/mistral-nemo \
python scripts/03_simulate_interactions.py \
  --personas data/divorce_personas.json \
  --events data/critical_events.json \
  --clean-data data/divorce_clean.csv \
  --enable-agent-icl \
  --output results/divorce_simulations_agent_icl_strat30.json \
  --subset-stratified 30 \
  --subset-seed 42 \
  --rounds 6 \
  --max-workers 6

# Option B: Full dataset (slow, expensive)
python scripts/03_simulate_interactions.py \
  --personas data/divorce_personas.json \
  --events data/critical_events.json \
  --output results/divorce_simulations.json \
  --num-couples 170 \
  --rounds 6 \
  --max-workers 10
```

**Output:** `results/divorce_simulations_*.json` with conversation logs

**Key Flags:**
- `--enable-agent-icl`: Inject ICL examples into agent prompts during simulation
- `--subset-stratified N`: Stratified sampling of N couples (balanced divorced/married)
- `--subset-seed`: Random seed for reproducibility
- `SCENE_REACT_EVERY=3`: World engine reacts every 3 rounds (default: disabled)

### Step 4: Evaluate All Methods (~15-20 min, ~$1-2)

```bash
OPENROUTER_MAX_CONCURRENCY=6 \
OPENROUTER_RPS=0.8 \
OPENROUTER_MODEL=google/gemini-2.5-flash-lite \
python scripts/04_evaluate_predictions.py \
  --simulations results/divorce_simulations_agent_icl_strat30.json \
  --clean-data data/divorce_clean.csv \
  --personas data/divorce_personas.json \
  --output results/divorce_evaluation_agent_icl_strat30.json \
  --method all \
  --max-workers 6 \
  --threshold 5.0 \
  --skip-baseline
```

**Output:** `results/divorce_evaluation_*.json` with accuracy, AUC, confusion matrices

**Methods Evaluated:**
- `observer`: Third-party LLM evaluation
- `observer-icl`: Observer + 10 ICL examples
- `participant`: Self-assessment (husband + wife average)
- `participant-icl`: Participant + ICL examples

### Step 5: Calibrate Scores (Optional, ~5 sec, free)

```bash
python scripts/05_calibrate_scores.py \
  --input results/divorce_evaluation_agent_icl_strat30.json \
  --output results/calibration_results_agent_icl_strat30.json \
  --n-train 10
```

**Output:** `results/calibration_*.json` with logistic regression calibrated scores

---

## 🔬 Methods Evaluated

### LLM Methods

1. **Observer**
   - Third-party LLM observes couple interactions
   - Rates divorce likelihood on 0-10 scale
   - Uses conversation history + observation summaries
   - Best correlation: Uses ICL with 10 labeled examples

2. **Observer-ICL** ⭐ **Best Performer**
   - Observer + In-Context Learning
   - 10 labeled examples (5 divorced, 5 married) in prompt
   - Includes survey Q&A + personas + outcomes
   - Dramatically improves calibration: 90% accuracy, 0.918 AUC

3. **Participant**
   - Each spouse self-assesses divorce likelihood
   - Average of husband + wife scores
   - Suffers from self-assessment bias
   - Lower accuracy: 43%

4. **Participant-ICL**
   - Participant + ICL examples
   - Self-assessment with outcome patterns
   - Still biased but slightly better than base participant

### Evaluation Metrics

**Classification (Binary divorce prediction):**
- **Accuracy**: Overall correctness
- **ROC AUC**: Ranking quality across all thresholds
- **Confusion Matrix**: True positives/negatives breakdown
- **Threshold**: Score > 5.0 → divorced; ≤ 5.0 → married

---

## 📈 Critical Events Design

### Three Event Categories

| Event Type | Purpose | Examples |
|------------|---------|----------|
| **Marriage Milestone** | Test commitment depth | Career relocation, having children, financial crisis |
| **Trust Breach** | Test forgiveness capacity | Emotional affair, financial deception, broken promise |
| **Illness Burden** | Test "in sickness and health" | Chronic illness, disability, caregiver role |

### Core Philosophy

**"Follow Your Inner Heart, Not Moral Standards"**

Traditional surveys suffer from social desirability bias. Critical events:
- Surface true reactions under stress
- Test alignment between stated values and behavior
- Reveal hidden incompatibilities
- Remove moral guardrails from LLM responses

**Prompt Example:**
```
You are {Name}, married to {Partner}. Your trust score is {X}/4.

CRITICAL EVENT: You discovered your partner had an emotional affair.

RESPOND HONESTLY (ignore social pressure to "forgive and move on"):
1. Can you actually forgive this? (Not "should you", but "can you")
2. When you imagine next year, do you still see them as your partner?
3. If divorce had no stigma, would you leave? (Yes/No/Maybe)
```

---

## 📊 Dataset

**Source:** Divorce Predictors Scale (Yöntem & İlhan, 2019)  
**Size:** 170 couples (Turkish university students/graduates)  
**Features:** 27 questions scored 0-4 (Never → Always)  
**Ground Truth:** Binary divorce outcome (54 attributes total)  
**Test Set:** 30 couples stratified by outcome (15 divorced, 15 married)

### Key Features
- **Communication:** Expression, understanding, conflict resolution
- **Trust & Respect:** Mutual trust, respect, boundaries
- **Shared Values:** Common goals, lifestyle compatibility
- **Conflict Patterns:** Arguing style, defensiveness, criticism
- **Emotional Connection:** Affection, intimacy, time together

### Data Cleaning
- ✅ Removed leakage features (features that directly reveal outcome)
- ✅ Kept only behavioral/attitudinal questions (27 core features)
- ✅ See `docs/01_eda_remove_leakage.ipynb` for analysis

---

## 💻 Technical Requirements

### Dependencies
```
Python >= 3.8
openai >= 1.0.0          # For OpenRouter API
pandas >= 2.0.0
numpy >= 1.24.0
scikit-learn >= 1.3.0
tqdm >= 4.65.0
python-dotenv >= 1.0.0
aiohttp >= 3.9.0         # Async HTTP
```

### API Keys Required
- **OpenRouter API Key** (for Mistral Nemo, Gemini 2.5 Flash Lite, GPT-4)
  - Get key: https://openrouter.ai/
  - Set in `.env`: `OPENROUTER_API_KEY=sk-or-v1-...`

### Environment Variables
```bash
OPENROUTER_API_KEY=sk-or-v1-xxx           # Required
OPENROUTER_MODEL=mistralai/mistral-nemo   # Default model
OPENROUTER_MAX_CONCURRENCY=6              # Concurrent requests
OPENROUTER_RPS=0.8                        # Rate limit (requests/sec)
SCENE_REACT_EVERY=3                       # World engine reaction frequency
```

### Estimated Costs (per full 30-couple run)
- **Persona Generation:** Free (rule-based)
- **Event Generation:** Free (rule-based)
- **Interaction Simulation:** ~$3-5 (Mistral Nemo, 30 couples × 3 events × 6 rounds)
- **Evaluation:** ~$1-2 (Gemini 2.5 Flash Lite, 30 couples × 4 methods)
- **Total:** ~$5-7 per complete experiment run

---

## 🔍 Research Findings Summary

### 1. In-Context Learning is Critical
- **90% accuracy** with Observer-ICL vs 50% without ICL
- 10 labeled examples dramatically improve calibration
- LLM learns relationship outcome patterns from examples

### 2. Observer > Participant Perspective
- Third-party evaluation (50-90%) > self-assessment (43%)
- Participants underestimate divorce risk (self-assessment bias)
- External observers capture hidden patterns better

### 3. Agent ICL During Simulation Improves Realism
- Injecting ICL into agent prompts during simulation
- Agents follow learned behavioral patterns
- More authentic conversation dynamics
- Better alignment with ground truth outcomes

### 4. Behavioral Simulation Reveals Hidden Risk
- Static questionnaires miss stress responses
- Critical events surface incompatibilities
- Conversation analysis captures nuanced patterns
- Dynamic interactions > static features

### 5. World Engine Context Matters
- Including personas + ICL in environment narration
- Consistent world evolution based on couple patterns
- Richer contextual grounding for agents

---

## 🚧 Limitations & Future Work

### Current Limitations
1. **Small dataset** - Only 170 couples (30 for testing)
2. **Simulated interactions** - Not real couple conversations
3. **Turkish population** - Cultural specificity
4. **Binary outcome** - Doesn't predict timing or severity
5. **Self-reported data** - Survey responses may be biased

### Future Directions
1. **Larger validation** - Test on 500+ couples from diverse datasets
2. **Real conversation data** - Incorporate actual therapy transcripts
3. **Multimodal inputs** - Add voice tone, facial expressions
4. **Longitudinal outcomes** - Predict divorce timing, not just binary
5. **Explainability** - Which events/patterns drive predictions?
6. **Cross-cultural validation** - Test on Western, Asian, Latin American populations
7. **Temporal dynamics** - Track relationship evolution over multiple simulations
8. **Intervention testing** - Simulate couples therapy interventions

---

## 📚 Key Documentation

| Document | Purpose | Content |
|----------|---------|---------|
| **README.md** | Project overview | Quick start, results, architecture |
| **CRITICAL_EVENTS_DESIGN.md** | Event philosophy | Design principles, prompt strategies |
| **01_eda_remove_leakage.ipynb** | Data analysis | EDA, leakage removal, cleaning |
| **leakage_features.txt** | Removed features | List of features causing data leakage |

---

## 🤝 Contributing

**Want to improve this research?**
- Test on different datasets (cross-cultural validation)
- Improve ICL example selection strategies
- Add more event types (e.g., substance abuse, infidelity recovery)
- Enhance explainability (which patterns drive predictions?)
- Scale to larger test sets (100+ couples)

---

## 📝 Citation

If you use this work, please cite:

```bibtex
@misc{divorce_llm_simulation_2025,
  title={Divorce Prediction via Critical Event Simulation with LLMs},
  author={Research Team},
  year={2025},
  note={Dataset: Yöntem, M. K., & İlhan, T. (2019). Divorce Predictors Scale.}
}
```

**Dataset Reference:**
```bibtex
@article{yontem2019divorce,
  title={The predictive power of the divorce predictors scale on marital quality},
  author={Y{\"o}ntem, Mustafa Kemal and {\.I}lhan, Tahsin},
  journal={European Journal of Education Studies},
  year={2019}
}
```

---

## 🎓 Acknowledgments

- **Dataset:** Mustafa Kemal Yöntem & Tahsin İlhan (2019)
- **Models:** Mistral AI (Mistral Nemo), Google (Gemini 2.5 Flash Lite), OpenAI (GPT-4)
- **Infrastructure:** OpenRouter API aggregation
- **Inspiration:** Gottman Method, Critical Incident Technique, Speed Dating LLM research

---

**Last Updated:** November 6, 2025  
**Status:** ✅ **Complete** - Observer-ICL achieves 90% accuracy (0.918 AUC) on 30-couple test set

---

**⭐ Key Insight:** LLMs can predict divorce by simulating couple behavior under stress, especially when calibrated with In-Context Learning. Observer perspective with 10 labeled examples achieves 90% accuracy—significantly better than self-assessment or uncalibrated methods. This validates the "behavioral simulation" approach over static questionnaire analysis.
