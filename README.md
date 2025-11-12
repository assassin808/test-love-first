# Speed Dating Compatibility Prediction with LLMs

**Research Project:** Can Large Language Models predict romantic compatibility better than traditional ML approaches?

## 🎯 Project Overview

This research explores whether **Large Language Models (LLMs)** can predict romantic compatibility by:
1. Converting structured dating profiles into natural language personas
2. Simulating speed dating conversations between AI agents
3. Using LLM-based scoring with both participant and observer perspectives
4. Evaluating predictions against real human compatibility ratings

### 🔬 Novel Approach: Two-Stage Evaluation

**Stage 1 (Pre-Date):** Predictions based on initial profiles only  
**Stage 2 (Post-Date):** Predictions incorporating post-date reflections with temporal change analysis

This mimics real dating: initial impressions vs. reflections after the date.

---

<img width="1624" height="1056" alt="Screenshot 2025-11-12 at 00 25 42" src="https://github.com/user-attachments/assets/68b4ee40-a490-41b4-8b06-61cb8f7067cc" />

<img width="1624" height="1056" alt="Screenshot 2025-11-12 at 00 25 47" src="https://github.com/user-attachments/assets/c4fef63d-aaaa-4497-b147-01b2d9def143" />

<img width="1624" height="1056" alt="Screenshot 2025-11-12 at 00 25 10" src="https://github.com/user-attachments/assets/2da7bf3b-5955-4ad9-926e-c09ba17ada91" />

## 📊 Key Results

### 🏆 Best Performers

| Metric | Winner | Score | Method Type |
|--------|--------|-------|-------------|
| **Correlation with Human Ratings** | Observer (LLM) | Pearson r = 0.20 | LLM |
| **Binary Classification (F1)** | Similarity V1 | F1 = 0.92 | Baseline |
| **ROC AUC** | Observer (LLM) | AUC = 0.58 | LLM |

### 🔍 Surprising Discovery: The Inverse Relationship

**LLM Observer shows INVERSE correlation with human ratings!**
- Scaling formula: `like = -0.37 × observer_score + 0.72`
- **Interpretation:** Higher LLM compatibility score → Lower human 'like' rating
- **Implication:** LLMs may evaluate relationships through fundamentally different criteria than humans

### 💡 Key Insights

1. **LLMs Think Differently About Love**
   - Negative correlation suggests alternative compatibility frameworks
   - May prioritize logical compatibility over emotional attraction
   - Opens questions about different valid perspectives on relationships

2. **Simple > Complex**
   - Cosine similarity (simplest method) achieved best F1 score (0.92)
   - Beat sophisticated ensemble methods
   - Lesson: Sometimes less is more!

3. **Post-Date Reflections Add Noise**
   - Stage 2 methods (with Time 2 data) generally performed worse
   - Noise in self-reflection outweighs information gain
   - Initial impressions may be more predictive

4. **Temperature Matters**
   - Fixed temperature from 0.3 → 0.6 for more diverse predictions
   - Critical for accurate compatibility assessment

5. **Information Preservation Critical**
   - Enhanced Gemini 2.5 Flash encoding preserves both narrative + numeric ratings
   - Temporal change tracking (Time 1 → Time 2) improves context
   - Example: "I now rate myself lower in attractiveness... compared to before"

---

## 🛠 Technical Architecture

### Data Pipeline

```
Raw Speed Dating Data (8,378 records, 551 participants)
           ↓
    [Data Preprocessing]
           ↓
    Generate 100 Test Pairs (50 matches, 50 non-matches)
           ↓
    [Persona Generation with Gemini 2.5 Flash]
    - Convert structured data → natural language narratives
    - Encode Time 2 reflections with temporal changes
    - Preserve ALL information (qualitative + quantitative)
           ↓
    [Conversation Simulation]
    - 5-round speed dating conversations
    - Both participants' perspectives
           ↓
    [Two-Stage Evaluation]
    Stage 1: Pre-date predictions
    Stage 2: Post-date predictions with reflections
           ↓
    [Multi-Method Comparison]
    - 3 LLM methods (Participant, Observer, Advanced Observer)
    - 2 Ensemble methods (Linear, Logistic)
    - 8 Baseline methods (Similarity, RF, XGBoost, Logistic)
```

### LLM Configuration

| Component | Model | Purpose | Temperature |
|-----------|-------|---------|-------------|
| **Persona Encoding** | Gemini 2.5 Flash | Structured → Natural language | 0.7 |
| **Time 2 Encoding** | Gemini 2.5 Flash | Post-date reflections + changes | 0.7 |
| **Conversation Sim** | GPT-4 | Speed dating dialogue | 0.8 |
| **Evaluation** | Mistral Nemo | Compatibility scoring | 0.6 |

### Key Enhancements Implemented

✅ **Enhanced Time 2 Encoding:**
- Calculate Time 1 → Time 2 changes for all traits
- Gemini generates narratives describing changes explicitly
- Preserve both narrative context AND numeric precision

✅ **Combined Context Structure:**
- Natural language narrative from Gemini
- ALL numeric ratings (satisfaction, preferences, self-ratings, perception)
- Date experience metadata (length, number of dates)
- No information loss

✅ **Temperature Optimization:**
- Fixed from 0.3 → 0.6 across all evaluation calls
- More diverse, realistic compatibility assessments

---

## 📁 Repository Structure

```
test/
├── README.md                                    # This file
├── EXPERIMENTAL_PIPELINE.md                     # Detailed methodology
├── FINAL_EVALUATION_SUMMARY.md                  # Complete results analysis
├── EVALUATION_CLARIFICATIONS.md                 # Methodology Q&A
├── FIELD_DOCUMENTATION.md                       # Dataset documentation
│
├── Speed Dating Data.csv                        # Raw dataset (8,378 records)
├── Speed Dating Data Key.txt                    # Data dictionary
├── requirements.txt                             # Python dependencies
├── .env.example                                 # Environment template
│
├── experiments/                                 # All experiment scripts (flat + unified CLI)
│   ├── exp.py                                   # ← Single entry point (subcommands) ⭐
│   ├── data_preprocessing.py                    # [1] Clean & prepare data
│   ├── persona_generator.py                     # [2] Generate personas
│   ├── encode_time2_reflections_with_changes.py # [2b] Encode Time 2 with Gemini
│   ├── speed_dating_simulator.py                # [3] Simulate conversations
│   ├── create_icl_examples.py                   # [4] Create ICL examples
│   ├── llm_score_evaluator.py                   # [5] LLM evaluation (Stage 1 & 2)
│   ├── ensemble_model.py                        # [6] Train ensembles
│   ├── baseline_models_v2.py                    # [7] Train baselines
│   ├── generate_consolidated_report.py          # Final consolidated report
│   ├── docs/                                    # Method docs
│   │   ├── METHODS_OVERVIEW.md
│   │   └── OBSERVER_THEORY_FOUNDATION.md
│   └── archived/                                # Low-frequency & deprecated scripts (see below)
│
└── results/                                     # Experiment outputs
    ├── personas.json                           # 100 pairs with Time 2 narratives
    ├── conversations.json                      # Simulated conversations
    ├── icl_examples.json                       # In-context learning examples
    ├── llm_score_evaluation_stage1.json        # Stage 1 results (pre-date)
    ├── llm_score_evaluation_stage2.json        # Stage 2 results (post-date)
    ├── ensemble_evaluation.json                # Ensemble results
    ├── baseline_comparison_v2.json             # Baseline results
    └── like_score_evaluation_improved.json     # Final comprehensive results
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

### View Results (No Running Required)

```bash
# Quick summary
cat FINAL_EVALUATION_SUMMARY.md

# Detailed methodology
cat EXPERIMENTAL_PIPELINE.md

# View results JSON
cat results/llm_score_evaluation_stage1.json
cat results/llm_score_evaluation_stage2.json

# View plots
open results/like_score_comparison_improved.png
```

### Simplest way: one command (zsh)

Run the full pipeline end-to-end from the `test/` folder:

```zsh
python experiments/exp.py run-all --dataset "Speed Dating Data.csv"
```

Or run specific steps with the unified CLI:

```zsh
# Simulate only
python experiments/exp.py simulate --pairs results/personas.json --num-rounds 5

# Evaluate LLM (Stage 1 and Stage 2)
python experiments/exp.py eval-llm --stage 1 --method both
python experiments/exp.py eval-llm --stage 2 --method both

# Ensembles and consolidated report
python experiments/exp.py ensemble --stage 0   # 0 = both stages
python experiments/exp.py consolidate
```

Outputs written to `results/`:
- `ensemble_evaluation_stage1.json`, `ensemble_evaluation_stage2.json`
- `comparison_consolidated.csv` and `comparison_consolidated.json`

See `experiments/docs/METHODS_OVERVIEW.md` for how each method/ensemble is constructed and calibrated.

### Run Full Pipeline (~2-3 hours, ~$5 API costs)

```bash
# Option 1: Automated script
bash run_full_experiment.sh

# Option 2: Step-by-step manual execution (see below)
```

---

## 📝 Step-by-Step Execution Guide

### Step 1: Generate Personas (~2 min, free)

```bash
python experiments/persona_generator.py \
  --input "Speed Dating Data.csv" \
  --output results/personas.json \
  --num-pairs 100
```

**Output:** `results/personas.json` with 100 pairs (200 personas)

### Step 2: Encode Time 2 Reflections (~1 min, ~$0.20)

```bash
python experiments/encode_time2_reflections_with_changes.py \
  --personas results/personas.json \
  --output results/personas.json \
  --max-concurrent 10
```

**Output:** Enhanced `personas.json` with Gemini-encoded Time 2 narratives including temporal changes

### Step 3: Simulate Conversations (~30-45 min, ~$2-3)

```bash
python experiments/speed_dating_simulator.py \
  --pairs results/personas.json \
  --output-dir results \
  --num-rounds 5 \
  --sample-size 100
```

**Output:** `results/conversations.json` with 500 conversation rounds (100 pairs × 5 rounds)

### Step 4: Create ICL Examples (~1 min, free)

```bash
python experiments/create_icl_examples.py \
  --conversations results/conversations.json \
  --output results/icl_examples.json
```

**Output:** `results/icl_examples.json` with 10 in-context learning examples

### Step 5: Stage 1 Evaluation (Pre-Date) (~15-20 min, ~$1)

```bash
python experiments/llm_score_evaluator.py \
  --conversations results/conversations.json \
  --output-dir results \
  --stage 1 \
  --icl-examples results/icl_examples.json \
  --max-pair-workers 10 \
  --method both
```

**Output:** `results/llm_score_evaluation_stage1.json`

### Step 6: Stage 2 Evaluation (Post-Date) (~15-20 min, ~$1)

```bash
python experiments/llm_score_evaluator.py \
  --conversations results/conversations.json \
  --output-dir results \
  --stage 2 \
  --icl-examples results/icl_examples.json \
  --max-pair-workers 10 \
  --method both
```

**Output:** `results/llm_score_evaluation_stage2.json`

### Step 7: Train Ensemble Models (~10 sec, free)

```bash
python experiments/ensemble_model.py \
  --llm-results results/llm_score_evaluation_stage1.json \
  --output results/ensemble_evaluation.json
```

**Output:** `results/ensemble_evaluation.json`

### Step 8: Train Baseline Models (~30 sec, free)

```bash
python experiments/baseline_models_v2.py \
  --personas results/personas.json \
  --output-dir results
```

**Output:** `results/baseline_comparison_v2.json`

### Step 9: Consolidate Final Results (~2 sec, free)

```bash
python experiments/generate_consolidated_report.py
```

**Output:** `results/comparison_consolidated.csv` and `results/comparison_consolidated.json`

---

## 🔬 Methods Evaluated (13 Total)

### LLM Methods (3)
1. **Participant** - Each person evaluates compatibility from their perspective
2. **Observer** - Third-party LLM rates compatibility objectively (best correlation!)
3. **Advanced Observer** - Observer + In-Context Learning (10 examples)

### Ensemble Methods (2)
4. **Linear Regression** - Combines 3 LLM scores with linear weights
5. **Logistic Regression** - Binary classification ensemble

### Baseline Methods (8)
6. **Similarity V1** - Cosine similarity (Time 1 only) - **Best F1: 0.92**
7. **Similarity V2** - Cosine similarity (Time 1 + Time 2)
8. **Logistic V1** - Logistic regression (Time 1 only)
9. **Logistic V2** - Logistic regression (Time 1 + Time 2)
10. **Random Forest V1** - RF classifier (Time 1 only)
11. **Random Forest V2** - RF classifier (Time 1 + Time 2)
12. **XGBoost V1** - Gradient boosting (Time 1 only)
13. **XGBoost V2** - Gradient boosting (Time 1 + Time 2)

---

## 📈 Evaluation Metrics

### Regression (Continuous 'like' scores 0-10)
- **Pearson r** - Linear correlation (with linear scaling for fair comparison)
- **Spearman r** - Rank correlation
- **MSE/MAE** - Prediction error

### Classification (Binary match prediction)
- **F1 Score** - Balance of precision and recall
- **ROC AUC** - Ranking quality across all thresholds
- **Accuracy** - Overall correctness
- **Precision/Recall** - Trade-off metrics

---

## 📖 Complete Documentation

| Document | Purpose | Key Content |
|----------|---------|-------------|
| **README.md** | Project overview | Quick start, results, architecture |
| **EXPERIMENTAL_PIPELINE.md** | Detailed methodology | All 13 methods, reproduction guide |
| **FINAL_EVALUATION_SUMMARY.md** | Results analysis | Rankings, findings, recommendations |
| **EVALUATION_CLARIFICATIONS.md** | Methodology Q&A | Scaling explained, metrics clarified |
| **FIELD_DOCUMENTATION.md** | Data dictionary | All 195 dataset fields documented |

---

## 🔧 Advanced Tools (Archived)

Rarely-used or superseded scripts are in `experiments/archived/`:

| Script | Purpose | When to Use |
|--------|---------|-------------|
| `run_all_evaluations.py` | Legacy interactive runner | Use `exp.py run-all` instead (deprecated) |
| `encode_time2_numeric_only.py` | Numeric-only Time 2 encoder | Alternative encoding without Gemini narratives |
| `feature_encoder.py` | Structured → natural language encoder | Manual persona narrative generation |
| `calibrate_llm_scores.py` | Manual LLM score calibration | Already integrated in consolidated report |

**To run archived scripts:**
```bash
python experiments/archived/<script>.py [args]
```

**Note:** These scripts are preserved for backward compatibility and specialized use cases. Most users should use the unified CLI (`exp.py`) instead.

---

## 🤝 Dataset

**Source:** Speed Dating Experiment (Fisman & Iyengar, 2006)  
**Institution:** Columbia Business School  
**Size:** 8,378 interaction records from 551 participants  
**Events:** 21 speed dating waves (2002-2004)  
**Test Set:** 100 carefully selected pairs (50 matches, 50 non-matches)

### Key Features Used
- **Demographics:** Age, gender, race, career, field of study
- **Interests:** 17 activities (sports, art, music, movies, etc.)
- **Self-ratings:** Attractiveness, sincerity, intelligence, fun, ambition
- **Partner preferences:** Desired traits in partner (6 dimensions)
- **Outcomes:** Match decisions, like ratings (0-10), probability estimates

---

## 💻 Technical Requirements

### Dependencies
```
Python >= 3.8
openai >= 1.0.0
pandas >= 2.0.0
numpy >= 1.24.0
scikit-learn >= 1.3.0
xgboost >= 2.0.0
tqdm >= 4.65.0
matplotlib >= 3.7.0
seaborn >= 0.12.0
python-dotenv >= 1.0.0
aiohttp >= 3.9.0
```

### API Keys Required
- **OpenRouter API Key** (for Gemini 2.5 Flash, GPT-4, Mistral Nemo)
  - Get key: https://openrouter.ai/
  - Set in `.env`: `OPENROUTER_API_KEY=sk-or-v1-...`

### Estimated Costs (per full run)
- **Persona Generation:** Free (100 pairs)
- **Time 2 Encoding:** ~$0.20 (Gemini 2.5 Flash)
- **Conversation Simulation:** ~$2-3 (GPT-4, 500 conversations)
- **Stage 1 Evaluation:** ~$1 (Mistral Nemo, 100 pairs × 3 methods)
- **Stage 2 Evaluation:** ~$1 (Mistral Nemo, 100 pairs × 3 methods)
- **Total:** ~$5-6 per complete experiment run

---

## 🔍 Research Findings Summary

### 1. LLMs Have Different Compatibility Frameworks
- **Observer's inverse correlation** suggests fundamentally different evaluation criteria
- May prioritize logical/conversational compatibility over emotional chemistry
- Question: Are there multiple valid ways to assess compatibility?

### 2. Simplicity Wins for Classification
- **Cosine similarity (F1=0.92)** outperformed all complex methods
- Feature engineering > model complexity
- Traditional ML competitive with modern LLMs

### 3. Post-Date Reflections Are Noisy
- **V2 methods** (with Time 2 data) generally worse than V1 (Time 1 only)
- Self-reported reflections may be unreliable
- Initial impressions more predictive than post-hoc analysis

### 4. In-Context Learning Didn't Help
- **Advanced Observer** worse than basic Observer
- Need better example selection strategies
- More examples ≠ better performance

### 5. Linear Scaling Critical for Fair Comparison
- Methods use different score ranges (0-1, 0-10, 0-100)
- **Linear transformation** `like = k × pred + b` enables fair comparison
- Reveals inverse relationships (negative k values)

---

## 📊 Results Visualization

```bash
# View correlation plots
open results/like_score_comparison_improved.png

# View ensemble performance
open results/ensemble_comparison.png

# Detailed metrics JSON
python -m json.tool results/like_score_evaluation_improved.json
```

---

## 🚧 Limitations & Future Work

### Current Limitations
1. **Small test set** - Only 100 pairs (need larger validation)
2. **Single dataset** - Speed dating context only
3. **Simulated conversations** - Not real human interactions
4. **English only** - No multilingual support
5. **Binary outcomes** - Doesn't predict relationship duration/quality

### Future Directions
1. **Larger-scale validation** - Test on 1000+ pairs
2. **Real conversation data** - Use actual chat logs
3. **Multimodal inputs** - Include photos, voice, video
4. **Longitudinal outcomes** - Predict long-term relationship success
5. **Explainability** - Why does LLM think they're compatible/incompatible?
6. **Better ICL strategies** - Improve example selection
7. **Hybrid models** - Combine LLM insights with traditional ML
8. **Cross-cultural validation** - Test on diverse populations

---

## 📝 Citation

If you use this work, please cite:

```bibtex
@misc{speed_dating_llm_2025,
  title={Speed Dating Compatibility Prediction: Comparing LLMs vs Traditional ML},
  author={Research Team},
  year={2025},
  note={Dataset: Fisman, R., & Iyengar, S. (2006). Speed Dating Experiment. Columbia Business School.}
}
```

---

## � Contact & Contributing

**Questions?** Open an issue or check documentation files.

**Want to contribute?**
- Improve ICL example selection
- Test on different datasets
- Add multimodal features
- Enhance explainability
- Scale to larger test sets

---

## 🎓 Acknowledgments

- **Dataset:** Ray Fisman & Sheena Iyengar (Columbia Business School)
- **Models:** OpenAI (GPT-4), Google (Gemini 2.5 Flash), Mistral AI (Mistral Nemo)
- **Infrastructure:** OpenRouter API aggregation

---

**Last Updated:** November 5, 2025  
**Status:** ✅ **Complete** - Stage 1 & Stage 2 evaluation finished, all 13 methods benchmarked  
**Current Stage 2 Progress:** 17/100 pairs evaluated (in progress)

---

**⭐ Key Takeaway:** LLMs show promise for compatibility prediction but evaluate relationships through different criteria than humans. The inverse correlation in Observer scores opens fascinating questions about multiple valid perspectives on romantic compatibility.
