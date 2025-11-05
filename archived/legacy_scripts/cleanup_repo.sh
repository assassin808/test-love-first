#!/bin/bash
# Repository Cleanup Script
# Removes deprecated/test files and organizes repository structure

set -e  # Exit on error

echo "🧹 Starting repository cleanup..."

# Create archive directory for deprecated files
ARCHIVE_DIR="archived"
mkdir -p "$ARCHIVE_DIR"

echo ""
echo "📦 Step 1: Archiving deprecated experiment scripts..."

# Archive old baseline script (replaced by baseline_models_v2.py)
if [ -f "experiments/baseline_models.py" ]; then
    mv experiments/baseline_models.py "$ARCHIVE_DIR/"
    echo "  ✓ Archived experiments/baseline_models.py"
fi

# Archive old evaluation script (replaced by evaluate_like_improved.py)
if [ -f "experiments/evaluate_against_like_score.py" ]; then
    mv experiments/evaluate_against_like_score.py "$ARCHIVE_DIR/"
    echo "  ✓ Archived experiments/evaluate_against_like_score.py"
fi

# Archive comprehensive comparison (outdated)
if [ -f "experiments/comprehensive_comparison.py" ]; then
    mv experiments/comprehensive_comparison.py "$ARCHIVE_DIR/"
    echo "  ✓ Archived experiments/comprehensive_comparison.py"
fi

echo ""
echo "📦 Step 2: Archiving test/utility scripts..."

# Archive test scripts
for file in test_api_key.py test_repetition_strategy.py analyze_results.py quick_start.py; do
    if [ -f "$file" ]; then
        mv "$file" "$ARCHIVE_DIR/"
        echo "  ✓ Archived $file"
    fi
done

echo ""
echo "📦 Step 3: Archiving outdated documentation..."

# Archive old documentation (keep only essential docs)
DOCS_TO_ARCHIVE=(
    "BEFORE_AFTER_COMPARISON.md"
    "TEMPERATURE_VERIFICATION.md"
    "REPETITION_PENALTY_STRATEGY.md"
    "SIMULATOR_README.md"
    "EXPERIMENT_DESIGN.md"
    "experiments/FEATURE_PARITY_ANALYSIS.md"
    "experiments/FEATURE_PARITY_ACHIEVED.md"
    "experiments/EXPERIMENTS_README.md"
)

for doc in "${DOCS_TO_ARCHIVE[@]}"; do
    if [ -f "$doc" ]; then
        mv "$doc" "$ARCHIVE_DIR/"
        echo "  ✓ Archived $doc"
    fi
done

echo ""
echo "📦 Step 4: Organizing core documentation..."

# Keep these essential docs in root
ESSENTIAL_DOCS=(
    "EXPERIMENTAL_PIPELINE.md"      # Complete pipeline documentation
    "FINAL_EVALUATION_SUMMARY.md"   # Results summary
    "EVALUATION_CLARIFICATIONS.md"  # Methodology Q&A
    "FIELD_DOCUMENTATION.md"        # Data dictionary
)

echo "  ✓ Keeping essential documentation:"
for doc in "${ESSENTIAL_DOCS[@]}"; do
    if [ -f "$doc" ]; then
        echo "    - $doc"
    fi
done

echo ""
echo "📦 Step 5: Creating README.md..."

# Create main README
cat > README.md << 'EOF'
# Speed Dating Compatibility Prediction

**Research Project:** Comparing LLM-based vs Traditional ML approaches for romantic compatibility prediction

## 🎯 Quick Start

### View Results
```bash
# View comprehensive results summary
cat FINAL_EVALUATION_SUMMARY.md

# View detailed experimental pipeline
cat EXPERIMENTAL_PIPELINE.md
```

### Run Full Pipeline
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up API key
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# 3. Run complete pipeline (~2-3 hours)
bash run_full_experiment.sh
```

## 📊 Key Results

### Best Performers
- **Best Correlation with Human Ratings:** Observer (LLM) - Pearson r = 0.20
- **Best Binary Classification:** Similarity V1 (Baseline) - F1 = 0.92
- **Best ROC AUC:** Observer (LLM) - AUC = 0.58

### Surprising Discovery
LLM Observer has **inverse relationship** with human 'like' scores!
- Higher LLM compatibility score → Lower human rating
- Suggests LLMs evaluate relationships differently than humans

## 📁 Repository Structure

```
test/
├── README.md                           # This file
├── EXPERIMENTAL_PIPELINE.md            # Complete methodology & pipeline
├── FINAL_EVALUATION_SUMMARY.md         # Results & findings
├── EVALUATION_CLARIFICATIONS.md        # Methodology Q&A
│
├── experiments/                        # Experiment scripts (run in order)
│   ├── persona_generator.py           # [1] Generate test personas
│   ├── speed_dating_simulator.py      # [2] Simulate conversations
│   ├── create_icl_examples.py         # [3] Create ICL examples
│   ├── llm_score_evaluator.py         # [4] Evaluate LLM methods
│   ├── ensemble_model.py              # [5] Train ensemble models
│   ├── baseline_models_v2.py          # [6] Train baseline models
│   └── evaluate_like_improved.py      # [7] Comprehensive evaluation
│
└── results/                            # Experiment outputs
    ├── personas.json                  # Test set (100 pairs)
    ├── conversations.json             # Simulated conversations
    ├── llm_score_evaluation.json      # LLM results
    ├── ensemble_evaluation.json       # Ensemble results
    ├── baseline_comparison_v2.json    # Baseline results
    └── like_score_evaluation_improved.json  # Final comprehensive results
```

## 🔬 Methods Evaluated (13 total)

### LLM Methods (3)
1. **Participant:** Each person evaluates if they'd date the other
2. **Observer:** Third-party LLM rates compatibility (best correlation)
3. **Advanced Observer:** Observer + In-Context Learning

### Ensemble Methods (2)
4. **Linear Regression:** Combines 3 LLM scores
5. **Logistic Regression:** Binary classification ensemble

### Baseline Methods (8)
6. **Similarity V1:** Cosine similarity (Time 1 only) - best F1!
7. **Similarity V2:** Cosine similarity (Time 1 + Time 2)
8. **Logistic V1:** Logistic regression (Time 1 only)
9. **Logistic V2:** Logistic regression (Time 1 + Time 2)
10. **Random Forest V1:** RF classifier (Time 1 only)
11. **Random Forest V2:** RF classifier (Time 1 + Time 2)
12. **XGBoost V1:** Gradient boosting (Time 1 only)
13. **XGBoost V2:** Gradient boosting (Time 1 + Time 2)

## 📈 Evaluation Metrics

### Regression (Continuous 'like' scores 0-10)
- **Pearson r:** Linear correlation (with linear scaling for fair comparison)
- **Spearman r:** Rank correlation
- **MSE/MAE:** Prediction error

### Classification (Binary match prediction)
- **F1 Score:** Balance of precision and recall
- **ROC AUC:** Ranking quality
- **Accuracy:** Overall correctness

## 🔍 Key Findings

1. **Linear Scaling is Critical**
   - Different methods use different scales
   - Linear transformation enables fair comparison
   - Reveals inverse relationships (negative coefficients)

2. **Inverse Relationships Discovered**
   - LLM Observer: Higher score → Lower human 'like' rating (k=-0.37)
   - LLMs may evaluate compatibility differently than humans
   - Opens questions about different valid perspectives

3. **Simple Methods Surprisingly Effective**
   - Cosine similarity achieved best F1 score (0.92)
   - Beat complex ensemble methods
   - Sometimes less is more!

4. **Ensemble Methods Underperformed**
   - Combining contradictory signals degrades performance
   - Need smarter combination strategies

5. **In-Context Learning Decreased Performance**
   - Advanced Observer worse than basic Observer
   - Better example selection needed

6. **Time 2 Reflections Add Noise**
   - V2 methods (with post-event data) generally worse
   - Post-event information doesn't help predictions

## 📖 Documentation

- **EXPERIMENTAL_PIPELINE.md:** Complete methodology, all 13 methods, reproduction guide
- **FINAL_EVALUATION_SUMMARY.md:** Results, rankings, key findings, recommendations
- **EVALUATION_CLARIFICATIONS.md:** Answers to methodology questions
- **FIELD_DOCUMENTATION.md:** Speed dating dataset data dictionary

## 🚀 Step-by-Step Execution

```bash
# Step 1: Generate personas (2 min)
python experiments/persona_generator.py --input "Speed Dating Data.csv" --output results/personas.json

# Step 2: Simulate conversations (30-45 min, $2-3 cost)
python experiments/speed_dating_simulator.py --pairs results/personas.json --output-dir results

# Step 3: Create ICL examples (1 min)
python experiments/create_icl_examples.py --conversations results/conversations.json --output results/icl_examples.json

# Step 4: Evaluate LLM methods (20-30 min, $1-2 cost)
python experiments/llm_score_evaluator.py --conversations results/conversations.json --output-dir results --method both --icl-examples results/icl_examples.json

# Step 5: Train ensembles (10 sec)
python experiments/ensemble_model.py --llm-results results/llm_score_evaluation.json --output results/ensemble_evaluation.json

# Step 6: Train baselines (30 sec)
python experiments/baseline_models_v2.py --personas results/personas.json --output-dir results

# Step 7: Comprehensive evaluation (5 sec)
python experiments/evaluate_like_improved.py
```

## 📊 Viewing Results

```bash
# Summary
cat FINAL_EVALUATION_SUMMARY.md

# Detailed metrics JSON
cat results/like_score_evaluation_improved.json

# Comparison plots
open results/like_score_comparison_improved.png
```

## 🤝 Dataset

**Source:** Speed Dating Experiment (Fisman & Iyengar, 2006)  
**Size:** 8,378 records, 551 participants  
**Test Set:** 100 pairs (50 matches, 50 non-matches)

## 📝 Citation

If you use this work, please cite:
```
Speed Dating Compatibility Prediction using LLMs vs Traditional ML (2025)
Dataset: Fisman, R., & Iyengar, S. (2006). Speed Dating Experiment. Columbia Business School.
```

## 🔗 Related Files

- **Speed Dating Data.csv:** Raw dataset
- **Speed Dating Data Key.txt:** Data dictionary
- **requirements.txt:** Python dependencies
- **.env.example:** Environment variable template

---

**Last Updated:** November 4, 2025  
**Status:** ✅ Complete - 13 methods evaluated, results documented
EOF

echo "  ✓ Created README.md"

echo ""
echo "📦 Step 6: Cleaning up pycache..."

# Remove __pycache__ directories
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
echo "  ✓ Removed __pycache__ directories"

echo ""
echo "📦 Step 7: Creating archive README..."

cat > "$ARCHIVE_DIR/README.md" << 'EOF'
# Archived Files

This directory contains deprecated scripts and documentation that have been superseded by newer versions.

## Archived Experiment Scripts

- **baseline_models.py** → Replaced by `baseline_models_v2.py`
  - Old version without proper train/test split
  - Missing individual predictions

- **evaluate_against_like_score.py** → Replaced by `evaluate_like_improved.py`
  - Missing linear scaling
  - No baseline method integration

- **comprehensive_comparison.py** → Replaced by integrated pipeline
  - Outdated comparison logic
  - Use `evaluate_like_improved.py` instead

## Archived Test/Utility Scripts

- **test_api_key.py** - API key validation (one-time use)
- **test_repetition_strategy.py** - Repetition penalty testing
- **analyze_results.py** - Ad-hoc analysis script
- **quick_start.py** - Superseded by full pipeline

## Archived Documentation

- **BEFORE_AFTER_COMPARISON.md** - Temperature tuning comparison
- **TEMPERATURE_VERIFICATION.md** - Temperature testing results
- **REPETITION_PENALTY_STRATEGY.md** - Repetition penalty strategy
- **SIMULATOR_README.md** - Old simulator docs (see EXPERIMENTAL_PIPELINE.md)
- **EXPERIMENT_DESIGN.md** - Old experiment design (see EXPERIMENTAL_PIPELINE.md)
- **FEATURE_PARITY_ANALYSIS.md** - Feature parity documentation
- **FEATURE_PARITY_ACHIEVED.md** - Feature parity achievement notes
- **EXPERIMENTS_README.md** - Old experiment docs

## Current Documentation

For up-to-date documentation, see:
- **README.md** - Quick start and overview
- **EXPERIMENTAL_PIPELINE.md** - Complete methodology and pipeline
- **FINAL_EVALUATION_SUMMARY.md** - Results and findings
- **EVALUATION_CLARIFICATIONS.md** - Methodology Q&A

---

**Note:** These files are kept for historical reference but should not be used in production.
EOF

echo "  ✓ Created archived/README.md"

echo ""
echo "✨ Cleanup complete!"
echo ""
echo "📊 Summary:"
echo "  - Archived scripts: experiments/baseline_models.py, evaluate_against_like_score.py, comprehensive_comparison.py"
echo "  - Archived test files: test_*.py, analyze_results.py, quick_start.py"
echo "  - Archived docs: 9 old documentation files"
echo "  - Created: README.md (main documentation)"
echo "  - Created: archived/README.md (archive index)"
echo "  - Cleaned: __pycache__ directories"
echo ""
echo "📁 Essential documentation kept:"
echo "  ✓ README.md - Quick start guide"
echo "  ✓ EXPERIMENTAL_PIPELINE.md - Complete methodology"
echo "  ✓ FINAL_EVALUATION_SUMMARY.md - Results summary"
echo "  ✓ EVALUATION_CLARIFICATIONS.md - Methodology Q&A"
echo "  ✓ FIELD_DOCUMENTATION.md - Data dictionary"
echo ""
echo "🎯 Active experiment scripts (experiments/):"
echo "  ✓ persona_generator.py"
echo "  ✓ speed_dating_simulator.py"
echo "  ✓ create_icl_examples.py"
echo "  ✓ llm_score_evaluator.py"
echo "  ✓ ensemble_model.py"
echo "  ✓ baseline_models_v2.py"
echo "  ✓ evaluate_like_improved.py"
echo ""
echo "📦 All deprecated files moved to: archived/"
echo ""
echo "✅ Repository is now clean and well-organized!"
