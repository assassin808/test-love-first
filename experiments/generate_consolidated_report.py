import json
import os
from typing import Any, Dict, List, Optional

import pandas as pd

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
RESULTS_DIR = os.path.abspath(RESULTS_DIR)


def _safe_load(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _row(method: str, mtype: str, stage: Optional[int], mode: Optional[str], metrics: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "method": method,
        "type": mtype,
        "stage": stage,
        "mode": mode,
        "accuracy": metrics.get("accuracy"),
        "precision": metrics.get("precision"),
        "recall": metrics.get("recall"),
        "f1": metrics.get("f1"),
        "auc_roc": metrics.get("auc") or metrics.get("auc_roc"),
        "pr_auc": metrics.get("pr_auc"),
        "threshold": metrics.get("threshold"),
    }


def collect_baselines(rows: List[Dict[str, Any]]):
    path = os.path.join(RESULTS_DIR, "baseline_comparison_v2.json")
    data = _safe_load(path)
    if not data:
        return
    # Known keys include similarity_v1, similarity_v1_calibrated, ..., logistic_v2, ..., etc.
    for key, obj in data.items():
        # Skip non-dict values
        if not isinstance(obj, dict):
            continue
        model = obj.get("model", key)
        feature_mode = obj.get("feature_mode")  # v1 or v2 or None
        # Map V1/V2 -> Stage 1/2
        stage = None
        if feature_mode == "v1":
            stage = 1
        elif feature_mode == "v2":
            stage = 2
        rows.append(
            _row(
                method=model,
                mtype="Baseline",
                stage=stage,
                mode=feature_mode,
                metrics={
                    "accuracy": obj.get("accuracy"),
                    "precision": obj.get("precision"),
                    "recall": obj.get("recall"),
                    "f1": obj.get("f1"),
                    "auc": obj.get("auc"),
                    "pr_auc": obj.get("pr_auc"),
                },
            )
        )


def collect_llm_ensembles(rows: List[Dict[str, Any]]):
    # Stage 1
    s1 = _safe_load(os.path.join(RESULTS_DIR, "ensemble_evaluation_stage1.json"))
    if s1 and "evaluation_results" in s1:
        er = s1["evaluation_results"]
        for name in [
            "linear_regression",
            "logistic_regression",
            "scaled_mean",
            "mul_add",
            "individual_participant",
            "individual_observer",
            "individual_advanced_observer",
        ]:
            if name in er:
                rows.append(
                    _row(
                        method=f"Ensemble {name}",
                        mtype="LLM (ensemble)",
                        stage=1,
                        mode=None,
                        metrics={
                            "accuracy": er[name].get("accuracy"),
                            "precision": er[name].get("precision"),
                            "recall": er[name].get("recall"),
                            "f1": er[name].get("f1"),
                            "auc_roc": er[name].get("roc_auc"),
                            "pr_auc": er[name].get("pr_auc"),
                        },
                    )
                )
    # Stage 2
    s2 = _safe_load(os.path.join(RESULTS_DIR, "ensemble_evaluation_stage2.json"))
    if s2 and "evaluation_results" in s2:
        er = s2["evaluation_results"]
        for name in [
            "linear_regression",
            "logistic_regression",
            "scaled_mean",
            "mul_add",
            "individual_participant",
            "individual_observer",
            "individual_advanced_observer",
        ]:
            if name in er:
                rows.append(
                    _row(
                        method=f"Ensemble {name}",
                        mtype="LLM (ensemble)",
                        stage=2,
                        mode=None,
                        metrics={
                            "accuracy": er[name].get("accuracy"),
                            "precision": er[name].get("precision"),
                            "recall": er[name].get("recall"),
                            "f1": er[name].get("f1"),
                            "auc_roc": er[name].get("roc_auc"),
                            "pr_auc": er[name].get("pr_auc"),
                        },
                    )
                )


def collect_llm_observer_calibrated(rows: List[Dict[str, Any]]):
    cal = _safe_load(os.path.join(RESULTS_DIR, "llm_score_evaluation_stage2_calibrated.json"))
    if not cal:
        return
    obs = cal.get("calibrated_results", {}).get("observer")
    if not obs:
        return
    metrics = obs.get("metrics", {})
    rows.append(
        _row(
            method="LLM Observer (calibrated)",
            mtype="LLM (scores)",
            stage=2,
            mode=None,
            metrics=metrics,
        )
    )


def collect_llm_score_methods(rows: List[Dict[str, Any]]):
    # Use baseline_comparison_v2.json appended LLM score-based metrics if available
    path = os.path.join(RESULTS_DIR, "baseline_comparison_v2.json")
    data = _safe_load(path)
    if not data:
        return
    # This file includes two appended keys in the printed summary only, but not guaranteed in JSON.
    # Try to read an explicit llm_score_evaluation_stage2.json for observer/participant metrics.
    s2 = _safe_load(os.path.join(RESULTS_DIR, "llm_score_evaluation_stage2.json"))
    if s2:
        # Observer-only metrics likely inside s2 under a top-level key
        # Try observer metrics
        try:
            obs_scores = s2.get("observer_method", {}).get("metrics")
            if obs_scores and obs_scores.get("f1") is not None:
                rows.append(
                    _row(
                        method="LLM Observer (scores)",
                        mtype="LLM (scores)",
                        stage=2,
                        mode=None,
                        metrics={
                            "accuracy": obs_scores.get("accuracy"),
                            "precision": obs_scores.get("precision"),
                            "recall": obs_scores.get("recall"),
                            "f1": obs_scores.get("f1"),
                            "auc_roc": obs_scores.get("auc_roc"),
                            "pr_auc": obs_scores.get("pr_auc"),
                        },
                    )
                )
        except Exception:
            pass
    # Stage 1 observer metrics
    s1 = _safe_load(os.path.join(RESULTS_DIR, "llm_score_evaluation_stage1.json"))
    if s1:
        try:
            obs_scores = s1.get("observer_method", {}).get("metrics")
            if obs_scores and obs_scores.get("f1") is not None:
                rows.append(
                    _row(
                        method="LLM Observer (scores)",
                        mtype="LLM (scores)",
                        stage=1,
                        mode=None,
                        metrics={
                            "accuracy": obs_scores.get("accuracy"),
                            "precision": obs_scores.get("precision"),
                            "recall": obs_scores.get("recall"),
                            "f1": obs_scores.get("f1"),
                            "auc_roc": obs_scores.get("auc_roc"),
                            "pr_auc": obs_scores.get("pr_auc"),
                        },
                    )
                )
        except Exception:
            pass


def main():
    rows: List[Dict[str, Any]] = []
    collect_baselines(rows)
    collect_llm_score_methods(rows)
    collect_llm_ensembles(rows)
    collect_llm_observer_calibrated(rows)

    df = pd.DataFrame(rows)
    # Sort for readability
    df = df.sort_values(by=["type", "stage", "method", "mode"], na_position="last")

    # AUC filtering disabled per request to provide complete results

    out_csv = os.path.join(RESULTS_DIR, "comparison_consolidated.csv")
    out_json = os.path.join(RESULTS_DIR, "comparison_consolidated.json")
    df.to_csv(out_csv, index=False)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(df.to_dict(orient="records"), f, indent=2)

    # Brief stdout summary (top by F1 per category/stage)
    def topk(group_df, k=10):
        g = group_df.dropna(subset=["f1"]).sort_values(by="f1", ascending=False).head(k)
        return g[["method", "type", "stage", "mode", "f1", "accuracy", "auc_roc", "pr_auc"]]

    print("Consolidated report written:")
    print(f"  CSV: {out_csv}")
    print(f"  JSON: {out_json}")

    # Small leaderboard prints per stage (1 and 2)
    for stage in [1, 2]:
        stage_df = df[df["stage"] == stage]
        if stage_df.empty:
            continue
        print(f"\nTop 10 by F1 (Stage {stage}):")
        print(topk(stage_df).to_string(index=False))

        # Always include all LLM methods in a separate view
        llm_df = stage_df[stage_df["type"].str.startswith("LLM", na=False)]
        if not llm_df.empty:
            print(f"\nAll LLM methods (Stage {stage}), sorted by F1:")
            print(
                llm_df
                .dropna(subset=["f1"]) 
                .sort_values(by="f1", ascending=False)[["method", "type", "stage", "mode", "f1", "accuracy", "auc_roc", "pr_auc"]]
                .to_string(index=False)
            )


if __name__ == "__main__":
    main()
