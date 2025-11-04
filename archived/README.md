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
