# Divorce-Exp Repository Reorganization Summary

**Date:** November 6, 2025  
**Status:** ✅ Complete

---

## 📁 New Directory Structure

```
divorce-exp/
├── README.md                    # Comprehensive project documentation
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment variable template
├── .gitignore                   # Git ignore rules
│
├── scripts/                     # All executable scripts (7 files)
│   ├── 00_clean_cache.py
│   ├── 01_generate_personas.py
│   ├── 02_generate_events.py
│   ├── 03_simulate_interactions.py
│   ├── 04_evaluate_predictions.py
│   ├── 05_calibrate_scores.py
│   └── divorce_questions.py
│
├── data/                        # Input datasets (6 files)
│   ├── divorce.csv              # Raw dataset
│   ├── divorce_clean.csv        # Cleaned (no leakage)
│   ├── divorce_test10.csv       # Test subset
│   ├── divorce_personas.json    # Generated personas
│   ├── divorce_personas_test10.json
│   └── critical_events.json     # Generated events
│
├── results/                     # Experiment outputs (30+ files)
│   ├── divorce_simulations_*.json      # Simulation logs
│   ├── divorce_evaluation_*.json       # Evaluation results
│   ├── calibration_*.json              # Calibrated scores
│   ├── *.png                           # Visualizations
│   └── *_sample.txt                    # Sample outputs
│
└── docs/                        # Documentation (4 files)
    ├── CRITICAL_EVENTS_DESIGN.md
    ├── 01_eda_remove_leakage.ipynb
    ├── leakage_features.txt
    └── doc.txt
```

---

## 🔄 Changes Made

### 1. Created Clean Folder Structure
- **scripts/**: All executable Python scripts
- **data/**: Raw and processed datasets
- **results/**: All experimental outputs
- **docs/**: Documentation and analysis

### 2. Renamed Files for Clarity
| Old Name | New Name | Reason |
|----------|----------|--------|
| `divorce_persona_generator.py` | `01_generate_personas.py` | Sequential ordering |
| `02_critical_events_generator.py` | `02_generate_events.py` | Shorter name |
| `03_critical_events_simulator.py` | `03_simulate_interactions.py` | More descriptive |
| `04_evaluate_predictions.py` | *(unchanged)* | Already clear |
| `05_calibrate_llm_scores.py` | `05_calibrate_scores.py` | Shorter name |

### 3. Organized Data Files
- Moved all `.csv` files to `data/`
- Moved all personas/events JSON to `data/`
- Kept original filenames for traceability

### 4. Organized Results
- Moved all simulation outputs to `results/`
- Moved all evaluation JSONs to `results/`
- Moved all calibration files to `results/`
- Moved all visualizations (`.png`) to `results/`
- Kept sample text files for quick reference

### 5. Organized Documentation
- Moved design docs to `docs/`
- Moved Jupyter notebooks to `docs/`
- Moved leakage analysis to `docs/`

### 6. Added Infrastructure Files
- **README.md**: Comprehensive 500+ line documentation
- **.gitignore**: Python, cache, large files
- **.env.example**: Environment template

---

## 📊 Key Results Preserved

All evaluation results remain accessible in `results/`:

| File | Description | Key Metrics |
|------|-------------|-------------|
| `divorce_evaluation_results_icl_strat30_v4_numeric.json` | **Best results** | Observer-ICL: 90% acc, 0.918 AUC |
| `divorce_simulations_agent_icl_strat30.json` | Agent ICL simulations | 30 couples, 1.8MB |
| `calibration_results_icl_strat30_10train.json` | Calibrated scores | Logistic regression |

---

## 🚀 Quick Start Commands (Updated Paths)

All commands now reference the new structure:

```bash
# Generate personas
python scripts/01_generate_personas.py \
  --input data/divorce_clean.csv \
  --output data/divorce_personas.json

# Generate events
python scripts/02_generate_events.py \
  --personas data/divorce_personas.json \
  --output data/critical_events.json

# Simulate interactions
python scripts/03_simulate_interactions.py \
  --personas data/divorce_personas.json \
  --events data/critical_events.json \
  --output results/divorce_simulations.json

# Evaluate
python scripts/04_evaluate_predictions.py \
  --simulations results/divorce_simulations.json \
  --clean-data data/divorce_clean.csv \
  --personas data/divorce_personas.json \
  --output results/divorce_evaluation.json
```

---

## ✅ Benefits of Reorganization

1. **Clarity**: Clear separation of concerns (scripts/data/results/docs)
2. **Discoverability**: Files organized by purpose, not chronology
3. **Maintainability**: Easy to find and update specific components
4. **Onboarding**: New contributors can understand structure quickly
5. **Best Practices**: Follows standard Python project layout
6. **Documentation**: Comprehensive README matching speed dating project quality

---

## 🔄 Migration Notes

### For Existing Users

If you have old scripts referencing the original file paths, update:

```python
# Old paths
"divorce_clean.csv"
"divorce_personas.json"
"divorce_simulations.json"

# New paths
"data/divorce_clean.csv"
"data/divorce_personas.json"
"results/divorce_simulations.json"
```

### Script Imports

The `divorce_questions.py` utility moved to `scripts/`:

```python
# If running from project root
from scripts.divorce_questions import format_couple_features

# If running from scripts/
from divorce_questions import format_couple_features
```

---

## 📝 Next Steps

1. ✅ Repository reorganized
2. ✅ README.md created
3. ✅ .gitignore added
4. ✅ .env.example added
5. 🔄 Update any external documentation pointing to old paths
6. 🔄 Test all scripts with new paths
7. 🔄 Commit changes to git

---

**Reorganization Complete!** 🎉

The repository now follows industry best practices and matches the quality standards of the speed dating project. All files are logically organized, well-documented, and ready for external sharing or publication.
