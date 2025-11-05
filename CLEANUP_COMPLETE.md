# Test Folder Cleanup Complete

**Date**: November 4, 2025

## ✅ Cleaned Up Test Folder

Removed non-experiment files:
- ❌ Web application files (app.py, socket_events.py, Database.py, utils.py, website/)
- ❌ Jupyter notebooks (*.ipynb)
- ❌ Verbose documentation (AI_DESIGN, API_KEY_MIGRATION, SECURITY, etc.)
- ❌ Interview data files (Female/Male Interview.json, interview-list.json)
- ❌ Application files (chat-history.json, report.json, example.txt, result.txt)
- ❌ Helper scripts (setup_baseline.sh, validate_stage2.py)

## 📁 Current Test Folder Structure

```
test/
├── experiments/          # All experiment scripts ✅
│   ├── llm_score_evaluator.py (with Stage 2)
│   ├── persona_generator.py
│   ├── speed_dating_simulator.py
│   ├── create_icl_examples.py
│   ├── baseline_models.py
│   ├── baseline_models_v2.py
│   ├── comprehensive_comparison.py
│   ├── data_preprocessing.py
│   ├── ensemble_model.py
│   ├── evaluate_like_improved.py
│   └── feature_encoder.py
│
├── results/              # Experiment outputs ✅
│   ├── personas.json
│   ├── conversations.json
│   ├── icl_examples.json
│   ├── llm_score_evaluation_stage1.json
│   ├── llm_score_evaluation_stage2.json
│   └── ...
│
├── archived/             # Old files ✅
│
├── Data Files:           # Experiment data ✅
│   ├── Speed Dating Data.csv
│   ├── Speed Dating Data Key.txt
│   └── Speed Dating Data Key.doc
│
├── Documentation:        # Experiment docs only ✅
│   ├── README.md
│   ├── ENHANCEMENT_PLAN.md
│   ├── EXECUTION_READY.md
│   ├── IMPLEMENTATION_STATUS.md
│   ├── EXPERIMENTAL_PIPELINE.md
│   ├── EVALUATION_CLARIFICATIONS.md
│   ├── FIELD_DOCUMENTATION.md
│   ├── FINAL_EVALUATION_SUMMARY.md
│   ├── CLEANUP_SUMMARY.md
│   └── MIGRATION_AND_STAGE2_COMPLETE.md
│
├── Scripts:              # Experiment scripts ✅
│   ├── run_full_experiment.sh
│   └── cleanup_repo.sh
│
└── Config Files:         # Essential config ✅
    ├── requirements.txt
    ├── .env
    ├── .env.example
    ├── .gitignore
    └── column_analysis.json
```

## 🎯 Purpose

The `test` folder now contains **ONLY experiment-related files**:
- ✅ Experiment scripts (`experiments/`)
- ✅ Experiment results (`results/`)
- ✅ Experiment data (Speed Dating Data.csv)
- ✅ Experiment documentation (ENHANCEMENT_PLAN.md, EXECUTION_READY.md, etc.)
- ✅ Experiment configuration (requirements.txt, .env)

## 🚀 Usage

All experiment commands should be run from the `test` folder:

```bash
cd /Users/assassin808/Desktop/research_2025_xuan/yan/test

# Run Stage 1 evaluation
python experiments/llm_score_evaluator.py --conversations results/conversations.json --stage 1

# Run Stage 2 evaluation
python experiments/llm_score_evaluator.py --conversations results/conversations.json --stage 2
```

## 📝 What Was Removed

- Web application files → Stay in `cupid/` folder
- Database/app files → Stay in `cupid/` folder  
- Notebooks → Stay in `cupid/` folder
- Verbose docs → Stay in `cupid/` folder
- Interview JSON files → Stay in `cupid/` folder

The `test` folder is now **clean and focused on experiments only**!
