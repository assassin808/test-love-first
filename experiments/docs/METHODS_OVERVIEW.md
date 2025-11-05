# Methods Overview: How Each Method Is Constructed

This document explains how every evaluated method is built and scored, including the LLM ensemble mean and all baselines. It covers inputs, formulas, training/calibration steps, and evaluation splits so results are interpretable and reproducible.

## Notation and inputs

- Pairs and label
  - Each pair i has a binary ground truth label y_i ∈ {0,1} indicating match.
- LLM-derived per-pair scores (where available):
  - r1 = participant_combined ∈ [0,1] (from `participant_method.scores[].combined_score`)
  - r2 = observer_normalized ∈ [0,1] (from `observer_method.scores[].normalized_score`)
  - r3 = advanced_observer_normalized ∈ [0,1] (from `advanced_observer_method.scores[].normalized_score` if present)

Unless stated otherwise, scores are treated as continuous for AUC/PR-AUC and thresholded for Acc/P/R/F1.

## LLM single-method scores

- LLM Participant (Product of scores)
  - Input: each person’s per-pair rating/score in [0,1]
  - Score: s_part = s_person1 × s_person2
  - Metrics: AUC/PR-AUC computed on s_part; classification metrics via thresholding (default 0.5 unless explicitly tuned elsewhere)

- LLM Observer (scores)
  - Input: third-party/observer’s normalized pair score r2 ∈ [0,1]
  - Score: s_obs = r2
  - Stage 1/2: Produced separately for Stage 1 (Time 1) and Stage 2 (Time 2-oriented prompt)
  - Metrics: AUC/PR-AUC on s_obs; classification metrics via thresholding as reported in `llm_score_evaluation_stage*.json`

- LLM Observer (calibrated; Stage 2)
  - Fit linear calibration on Stage 1 observer scores only: y ≈ k x + b via least squares
  - Apply to Stage 2 observer: s_obs_cal = clip(k · r2_stage2 + b, 0, 1)
  - Select threshold τ on the Stage 1 calibrated train split to maximize F1, then apply τ on Stage 2 for classification; AUC/PR-AUC based on the continuous s_obs_cal

## LLM ensembles

Let the available features be X = [r1, r2, (r3)] as columns when present.

- Linear Regression
  - Model: ŷ = β0 + βᵀ X trained with sklearn LinearRegression on the train split
  - Scores: s_lin = ŷ (continuous)
  - Classification: threshold at 0.5 → ĉ = [s_lin ≥ 0.5]
  - Metrics: AUC/PR-AUC on s_lin; Acc/P/R/F1 on ĉ

- Logistic Regression
  - Model: p = σ(β0 + βᵀ X) with sklearn LogisticRegression
  - Scores: s_log = p (probability)
  - Classification: threshold at 0.5 → ĉ = [s_log ≥ 0.5]
  - Metrics: AUC/PR-AUC on s_log; Acc/P/R/F1 on ĉ

- Scaled Mean (calibrated threshold)
  - Idea: simple, monotonic, parameter-free combiner with a tuned decision threshold
  - Score: s_mean = mean(available scores among {r1, r2, r3})
  - Threshold τ: selected on the train split by maximizing F1 over a grid t ∈ [0,1]
  - Classification: ĉ = [s_mean ≥ τ]; AUC/PR-AUC on s_mean

- Multiplicative-Additive (Mul+Add)
  - Purpose: capture interaction between participant and observer while allowing a linear contribution from advanced observer
  - Score: s_mul = k1 · (r1 · r2) + k2 · r3
  - Training (train split): grid search k1, k2 ∈ [−2, 2] to maximize F1 after tuning threshold τ ∈ [0,1] on s_mul; keep the best (k1, k2, τ)
  - Classification: ĉ = [s_mul ≥ τ]; AUC/PR-AUC on s_mul
  - Note: evaluated only when r3 is available

### Threshold calibration routine (used by Scaled Mean and Mul+Add)

- Given scores s on the train split and labels y, find τ that maximizes F1 over a uniform grid in [0,1]. If scores are outside [0,1], they are min-max normalized only for searching τ; the final τ is mapped back to the original scale.

## Baselines

Two versions aligned to the LLM stages:
- V1 (Stage 1): Time 1 only — same information the LLM has before the event
- V2 (Stage 2): Time 1 + Time 2 — includes post-event reflections (“future knowledge”)

### Similarity Baseline (Cosine)

- Features (V1):
  - Self preferences: attr1_1, sinc1_1, intel1_1, fun1_1, amb1_1, shar1_1
  - Self ratings: attr3_1, sinc3_1, intel3_1, fun3_1, amb3_1
  - Interests: sports, tvsports, exercise, dining, museums, art, hiking, gaming, clubbing, reading, tv, theater, movies, concerts, music
- Similarity: cosine similarity between person1 and person2 feature vectors
- Threshold: tuned on the training split to maximize F1; evaluated on test
- Calibrated variants: fit y ≈ kx + b on train similarities; also tune τ on calibrated train; apply to test (metrics based on calibrated scores)
- V2 adds Time 2 updated preferences (e.g., attr1_2, …, shar1_2) and uses the same process

### ML Baselines (Logistic Regression, Random Forest, XGBoost)

- V1 (Time 1 only): demographic one-hot, preferences, self-ratings, others’ perception, interests, dating behavior, values, mirrored for both persons; strict parity with LLM prompt fields; scaled with StandardScaler
- V2 (Time 1 + Time 2): V1 features plus Time 2 updated preferences/ratings and satisfaction
- Class imbalance handling: class_weight='balanced' for logistic; SMOTE guarded for tree/boosting when minority ratio is low (different sampling targets for V1/V2)
- Thresholding (trees/boosting): try mean, median, and top-50% probability thresholds on test probabilities and pick the one with highest F1
- Linear calibration: fit y ≈ kx + b on training probabilities, calibrate test probabilities and tune a calibrated threshold on train; report both calibrated and uncalibrated metrics

## Splits, training, and evaluation

- LLM ensembles: 30% train / 70% test stratified split over the 100 pairs
- Baselines V2 pipeline: builds train features from the CSV excluding test participants, then evaluates on the same 100 pairs used by LLM; for strict parity in our consolidated comparison we also construct a direct 30/70 split from personas.json aligned with LLM
- Metrics reported: Accuracy, Precision, Recall, F1, ROC AUC (auc_roc), PR-AUC (pr_auc); confusion matrices saved per method where applicable

## Consolidated report

- Script: `test/experiments/generate_consolidated_report.py`
- Aggregates: baselines (V1/V2), LLM single-score methods (Stage 1/2), LLM ensembles (Stage 1/2), calibrated LLM observer (Stage 2)
- Mapping: V1 → Stage 1, V2 → Stage 2
- Output: `test/results/comparison_consolidated.csv` and `.json` (AUC filtering disabled per latest requirement)
- Leaderboards printed (Top 10 by F1 per stage) plus an LLM-only section per stage

## Quick reference (equations)

- Linear calibration:  
  y ≈ k x + b,   s_cal = clip(k x + b, 0, 1)

- Linear ensemble score:  
  s_lin = β0 + βᵀ [r1, r2, (r3)]

- Logistic ensemble score:  
  s_log = σ(β0 + βᵀ [r1, r2, (r3)])

- Scaled mean score:  
  s_mean = mean({r1, r2, (r3) that are available})

- Multiplicative-additive score (when r3 available):  
  s_mul = k1 · (r1 · r2) + k2 · r3

- Classification decision: ĉ = [s ≥ τ] with τ learned on the train split (F1-optimal) where applicable; else τ = 0.5 by default

## Artifacts and where to look

- LLM score evaluations: `test/results/llm_score_evaluation_stage1.json`, `test/results/llm_score_evaluation_stage2.json`, `test/results/llm_score_evaluation_stage2_calibrated.json`
- Ensemble evaluations: `test/results/ensemble_evaluation_stage1.json`, `test/results/ensemble_evaluation_stage2.json` (store thresholds and parameters)
- Baselines: `test/results/baseline_comparison_v2.json` (contains V1 and V2, calibrated variants, and predictions)
- Consolidated: `test/results/comparison_consolidated.csv` and `.json`

If you want this overview embedded in the top-level README or linked from other docs, say the word and I’ll wire it in.
