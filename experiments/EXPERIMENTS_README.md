# Speed Dating LLM Prediction Experiments

This project compares **LLM-based matching predictions** with **traditional machine learning baselines** using the Columbia Speed Dating dataset.

## 🎯 Research Question

**Can Large Language Models (LLMs) predict speed dating matches better than traditional machine learning algorithms?**

## 📊 Methods Compared

### LLM-Based Approaches
1. **Self-Evaluation**: Both participants decide YES/NO after conversation
2. **Observer Evaluation**: AI "恋爱观察员" (dating observer) predicts compatibility

### Traditional ML Baselines
1. **Similarity Baseline**: Cosine similarity on preference/interest vectors (dummy baseline)
2. **Logistic Regression**: Simple interpretable model
3. **Random Forest**: Ensemble tree-based model
4. **XGBoost**: Gradient boosting (SOTA for tabular data)

*Baseline algorithms based on: "Predicting Speed Dating" - Tilburg University thesis*

## 🚀 Quick Start

### Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt
```

Required packages:
- `openai` - LLM API
- `scikit-learn` - ML baselines
- `xgboost` - Gradient boosting
- `pandas`, `numpy` - Data processing
- `matplotlib`, `seaborn` - Visualizations

### Step 1: Run Baseline Evaluation

Train and evaluate traditional ML models:

```bash
cd experiments
python baseline_models.py
```

This will:
- Load the 100 speed dating pairs
- Split into 70% train / 30% test
- Train 4 baseline models
- Evaluate on test set
- Save results to `results/baseline_comparison.json`

Expected runtime: ~30 seconds

### Step 2: Run LLM Speed Dating Simulation

Simulate conversations with LLM evaluations:

```bash
cd experiments
echo "3" | python speed_dating_simulator.py
```

Mode 3 settings:
- **100 pairs** of speed dating conversations
- **10 rounds** per conversation
- **5 parallel threads** for faster processing
- **Post-date evaluations** (self + observer)

Expected runtime: ~30-40 minutes  
Expected cost: ~$0.034 (using qwen/qwq-32b-preview)

Outputs:
- `results/conversations.json` - Full conversation logs with evaluations
- Auto-checkpoint every 5 pairs

### Step 3: Compare All Methods

Generate comprehensive comparison report:

```bash
cd experiments
python comprehensive_comparison.py
```

This will:
- Load LLM results from conversations.json
- Load baseline results from baseline_comparison.json
- Calculate metrics for all methods
- Generate comparison tables and plots
- Save report to `results/comprehensive_comparison.json`

Outputs:
- `results/comprehensive_comparison.json` - Full comparison report
- `results/comparison_metrics.png` - Multi-metric bar charts
- `results/comparison_f1.png` - F1 score comparison

## 📁 Project Structure

```
experiments/
├── baseline_models.py           # Traditional ML baselines
├── comprehensive_comparison.py  # Compare all methods
├── speed_dating_simulator.py    # LLM conversation simulator
├── persona_generator.py         # Generate dating personas
└── analyze_results.py           # LLM accuracy analysis

results/
├── personas.json                 # 100 generated personas
├── conversations.json            # LLM conversation logs
├── baseline_comparison.json      # ML baseline results
├── comprehensive_comparison.json # Full comparison report
└── comparison_*.png              # Visualization plots
```

## 🔬 Experimental Design

### Data Preprocessing

- **Dataset**: Columbia Speed Dating Experiment (2002-2004)
- **Selection**: 100 balanced pairs (50 matches, 50 non-matches)
- **Features Used**:
  - Preferences: attr, sinc, intel, fun, amb, shar
  - Self-ratings: attr3_1, sinc3_1, intel3_1, fun3_1, amb3_1
  - Interests: 15 binary interest indicators
  - Demographics: age, race, field of study

### LLM Conversation System

**Persona Generation:**
- Fixed names: Emma/Sarah (women), Jake/Ryan (men)
- Narrative from Time 1 data (before date)
- Stranger awareness and gradual warmup
- Concise inner thoughts (1-2 sentences)

**Conversation Format:**
```
<INNER_THOUGHT>
[Concise strategy: 1-2 sentences]
</INNER_THOUGHT>

<RESPONSE>
[Spoken words with [gesture tags]]
</RESPONSE>
```

**Post-Date Evaluations:**
1. Person 1 self-evaluation → YES/NO + reasoning
2. Person 2 self-evaluation → YES/NO + reasoning
3. Observer evaluation → compatibility score (0-10) + MATCH/NO_MATCH

### Baseline Feature Engineering

**Similarity Baseline:**
- Cosine similarity on concatenated feature vector
- Threshold tuned on training set

**ML Models:**
- Features: 24 engineered features per pair
  - Person 1: preferences (6) + self-ratings (5)
  - Person 2: preferences (6) + self-ratings (5)
  - Interest overlap (Jaccard similarity)
  - Preference-rating alignment (correlation)
- Standardization with StandardScaler
- 5-fold cross-validation during training

## 📈 Evaluation Metrics

- **Accuracy**: (TP + TN) / Total
- **Precision**: TP / (TP + FP) - How many predicted matches are correct?
- **Recall**: TP / (TP + FN) - How many actual matches did we find?
- **F1 Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under ROC curve (ML models only)
- **Confusion Matrix**: TP, TN, FP, FN breakdown

## 🎯 Key Features

### LLM System
- ✅ Realistic stranger-aware conversations
- ✅ Concise inner thought process
- ✅ Gesture tags instead of emoji spam
- ✅ Dual evaluation: self + observer
- ✅ Multi-threading (5 parallel workers)
- ✅ Retry logic with exponential backoff
- ✅ Auto-checkpoint every 5 pairs

### Baseline System
- ✅ Multiple algorithm comparison
- ✅ Proper train/test split (70/30)
- ✅ Hyperparameter tuning (threshold, CV)
- ✅ Feature engineering (alignment, overlap)
- ✅ Comprehensive metrics

### Comparison
- ✅ Unified evaluation framework
- ✅ Statistical comparison
- ✅ Bias analysis (optimistic/pessimistic)
- ✅ Visualization plots
- ✅ Detailed JSON reports

## 🔍 Analysis Examples

### View Baseline Results

```python
import json

with open('results/baseline_comparison.json', 'r') as f:
    results = json.load(f)

# Print model comparison
for model_name, metrics in results.items():
    if model_name != 'summary':
        print(f"{metrics['model']:20s} - F1: {metrics['f1']:.3f}")
```

### View LLM vs ML Comparison

```python
with open('results/comprehensive_comparison.json', 'r') as f:
    report = json.load(f)

# Best models
print(f"Best Accuracy: {report['key_findings']['best_accuracy_model']}")
print(f"Best F1 Score: {report['key_findings']['best_f1_model']}")

# Prediction patterns
patterns = report['prediction_patterns']
print(f"LLM Self-Eval Bias: {patterns['self_bias']:+.1%}")
print(f"LLM Observer Bias: {patterns['observer_bias']:+.1%}")
```

## 🛠️ Testing & Debugging

### Quick Test (1 pair, 2 rounds)

```bash
cd experiments
echo "1" | python speed_dating_simulator.py
```

### Analyze Single Conversation

```python
import json

with open('results/conversations.json', 'r') as f:
    conversations = json.load(f)

# View first conversation
conv = conversations[0]
print(f"Pair ID: {conv['pair_id']}")
print(f"Ground Truth: {conv['ground_truth']['match']}")

# Self-evaluations
p1_eval = conv['evaluations']['person1_self_evaluation']
print(f"Person 1: {p1_eval['decision']} - {p1_eval['reasoning']}")

p2_eval = conv['evaluations']['person2_self_evaluation']
print(f"Person 2: {p2_eval['decision']} - {p2_eval['reasoning']}")

# Observer evaluation
obs_eval = conv['evaluations']['observer_evaluation']
print(f"Observer: {obs_eval['prediction']} (score: {obs_eval['compatibility_score']}/10)")
print(f"Analysis: {obs_eval['analysis']}")
```

## 📊 Expected Results

Based on preliminary testing:

**Typical Performance Ranges:**
- Similarity Baseline: ~50-60% accuracy
- Logistic Regression: ~55-65% accuracy
- Random Forest: ~60-70% accuracy
- XGBoost: ~65-75% accuracy
- LLM Self-Eval: ~?% accuracy (to be tested)
- LLM Observer: ~?% accuracy (to be tested)

**Research Questions:**
1. Does LLM conversation analysis outperform feature-based ML?
2. Is observer evaluation more accurate than participant self-evaluation?
3. What prediction biases exist (optimistic vs pessimistic)?
4. Which features matter most for matching?

## 🔧 Troubleshooting

### "Import xgboost could not be resolved"

Install missing dependencies:
```bash
pip install scikit-learn xgboost pandas numpy matplotlib seaborn
```

### "Personas.json not found"

Generate personas first:
```bash
cd experiments
python persona_generator.py
```

### "API rate limit exceeded"

- Reduce `max_workers` in speed_dating_simulator.py
- Add longer delays between API calls
- Switch to a different model with higher rate limits

### Out of memory during baseline training

- Reduce training set size in baseline_models.py
- Use simpler models (Logistic Regression only)

## 📝 Citation

If you use this work, please cite:

```bibtex
@misc{speed_dating_llm_2025,
  title={Can LLMs Predict Speed Dating Matches? A Comparison with Traditional ML},
  author={[Your Name]},
  year={2025},
  note={Based on Columbia Speed Dating Experiment dataset}
}
```

Original dataset:
- Fisman, R., Iyengar, S. S., Kamenica, E., & Simonson, I. (2006). Gender differences in mate selection: Evidence from a speed dating experiment. The Quarterly Journal of Economics, 121(2), 673-697.

## 📧 Contact

For questions or issues, please contact [your email] or open an issue on GitHub.

## 📄 License

MIT License - see LICENSE file for details

---

**Happy Experimenting! 🚀**
