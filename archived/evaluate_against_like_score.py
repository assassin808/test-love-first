"""
Comprehensive Evaluation Against Ground Truth 'like' Score

Evaluates all methods (LLM + Baselines + Ensemble) against the original 'like' score (0-10)
from the speed dating dataset. This provides a unified comparison using the actual
human ratings as ground truth.

Methods evaluated:
1. Participant LLM method
2. Observer LLM method  
3. Advanced Observer LLM method (with ICL)
4. Linear Regression Ensemble
5. Logistic Regression Ensemble
6. Baseline methods from baseline_models_v2.py

Metrics computed:
- Correlation (Pearson, Spearman) with 'like' scores
- MSE, MAE for regression quality
- ROC AUC, PR-AUC using binary threshold on 'like' scores
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error,
    roc_auc_score, average_precision_score,
    accuracy_score, precision_score, recall_score, f1_score
)
import matplotlib.pyplot as plt


def load_like_scores(personas_path: str) -> Dict[str, Tuple[float, float]]:
    """
    Load ground truth 'like' scores from personas.json
    
    Returns:
        Dictionary mapping pair_id -> (person1_like, person2_like)
    """
    with open(personas_path, 'r', encoding='utf-8') as f:
        personas = json.load(f)
    
    like_scores = {}
    for entry in personas:
        p1 = entry['person1']
        p2 = entry['person2']
        pair_id = f"pair_{p1['iid']}_{p2['iid']}"
        
        # Extract 'like' scores from ground_truth ratings (0-10 scale)
        ground_truth = entry.get('ground_truth', {})
        p1_ratings = ground_truth.get('person1_ratings', {})
        p2_ratings = ground_truth.get('person2_ratings', {})
        
        p1_like = p1_ratings.get('like', None)
        p2_like = p2_ratings.get('like', None)
        
        if p1_like is not None and p2_like is not None:
            like_scores[pair_id] = (float(p1_like), float(p2_like))
    
    print(f"✅ Loaded 'like' scores for {len(like_scores)} pairs")
    return like_scores


def load_llm_predictions(llm_eval_path: str) -> Dict[str, Dict[str, float]]:
    """
    Load LLM method predictions from llm_score_evaluation.json
    
    Returns:
        Dictionary with keys: 'participant', 'observer', 'advanced_observer'
        Each maps pair_id -> normalized_score (0-1)
    """
    with open(llm_eval_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    predictions = {
        'participant': {},
        'observer': {},
        'advanced_observer': {}
    }
    
    # Participant scores
    for entry in data.get('participant_method', {}).get('scores', []):
        pair_id = entry['pair_id']
        predictions['participant'][pair_id] = entry['combined_score']
    
    # Observer scores
    for entry in data.get('observer_method', {}).get('scores', []):
        pair_id = entry['pair_id']
        predictions['observer'][pair_id] = entry['normalized_score']
    
    # Advanced Observer scores
    for entry in data.get('advanced_observer_method', {}).get('scores', []):
        pair_id = entry['pair_id']
        predictions['advanced_observer'][pair_id] = entry['normalized_score']
    
    print(f"✅ Loaded LLM predictions:")
    print(f"   Participant: {len(predictions['participant'])} pairs")
    print(f"   Observer: {len(predictions['observer'])} pairs")
    print(f"   Advanced Observer: {len(predictions['advanced_observer'])} pairs")
    
    return predictions


def load_ensemble_predictions(ensemble_path: str) -> Dict[str, Dict[str, float]]:
    """
    Load ensemble model predictions from ensemble_evaluation.json
    
    Returns:
        Dictionary with keys: 'linear_regression', 'logistic_regression'
        Each maps pair_id -> score
    """
    with open(ensemble_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    pair_ids = data.get('pair_ids', [])
    
    predictions = {
        'linear_regression': {},
        'logistic_regression': {}
    }
    
    # Linear regression scores
    lr_results = data.get('evaluation_results', {}).get('linear_regression', {})
    lr_scores = lr_results.get('y_scores', [])
    for pair_id, score in zip(pair_ids, lr_scores):
        predictions['linear_regression'][pair_id] = score
    
    # Logistic regression scores
    logreg_results = data.get('evaluation_results', {}).get('logistic_regression', {})
    logreg_scores = logreg_results.get('y_scores', [])
    for pair_id, score in zip(pair_ids, logreg_scores):
        predictions['logistic_regression'][pair_id] = score
    
    print(f"✅ Loaded ensemble predictions:")
    print(f"   Linear Regression: {len(predictions['linear_regression'])} pairs")
    print(f"   Logistic Regression: {len(predictions['logistic_regression'])} pairs")
    
    return predictions


def load_baseline_predictions(baseline_path: str) -> Dict[str, Dict[str, float]]:
    """
    Load baseline method predictions from baseline_comparison_v2.json
    
    Returns:
        Dictionary with baseline method names as keys
        Each maps pair_id -> probability score (0-1)
    """
    with open(baseline_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    predictions = {}
    
    # Extract predictions from each baseline method
    for method_name, method_data in data.items():
        if isinstance(method_data, dict) and 'predictions' in method_data:
            predictions[method_name] = {}
            for entry in method_data['predictions']:
                pair_id = entry['pair_id']
                prob = entry['probability']
                predictions[method_name][pair_id] = prob
    
    print(f"✅ Loaded baseline predictions:")
    for method_name, preds in predictions.items():
        print(f"   {method_name}: {len(preds)} pairs")
    
    return predictions


def compute_metrics_against_like(
    predictions: Dict[str, float],
    like_scores: Dict[str, Tuple[float, float]],
    method_name: str,
    like_threshold: float = 5.0,
    use_combined_like: bool = True
) -> Dict:
    """
    Compute metrics comparing predictions against 'like' scores
    
    Args:
        predictions: Dictionary mapping pair_id -> predicted_score (0-1 scale)
        like_scores: Dictionary mapping pair_id -> (person1_like, person2_like)
        method_name: Name of the method for reporting
        like_threshold: Threshold for binarizing 'like' scores (default 5.0)
        use_combined_like: If True, use (like1 * like2) / 100, else use average
    
    Returns:
        Dictionary with all computed metrics
    """
    # Align predictions with like scores
    common_pairs = set(predictions.keys()) & set(like_scores.keys())
    
    if not common_pairs:
        print(f"⚠️  No common pairs for {method_name}")
        return None
    
    y_pred = []
    y_like = []
    y_like_binary = []
    
    for pair_id in common_pairs:
        pred = predictions[pair_id]
        like1, like2 = like_scores[pair_id]
        
        # Compute combined 'like' score
        if use_combined_like:
            # Similar to participant combined score: (like1 * like2) / 100
            like_combined = (like1 * like2) / 100.0
        else:
            # Alternative: average
            like_combined = (like1 + like2) / 2.0 / 10.0  # Normalize to 0-1
        
        y_pred.append(pred)
        y_like.append(like_combined)
        
        # Binary label: both people like >= threshold
        y_like_binary.append(1 if (like1 >= like_threshold and like2 >= like_threshold) else 0)
    
    y_pred = np.array(y_pred)
    y_like = np.array(y_like)
    y_like_binary = np.array(y_like_binary)
    
    # Regression metrics (comparing continuous scores)
    pearson_corr, pearson_p = pearsonr(y_pred, y_like)
    spearman_corr, spearman_p = spearmanr(y_pred, y_like)
    mse = mean_squared_error(y_like, y_pred)
    mae = mean_absolute_error(y_like, y_pred)
    
    # Classification metrics (using binary 'like')
    try:
        roc_auc = roc_auc_score(y_like_binary, y_pred)
        pr_auc = average_precision_score(y_like_binary, y_pred)
    except ValueError:
        # Handle case where only one class present
        roc_auc = None
        pr_auc = None
    
    # Binary predictions at 0.5 threshold
    y_pred_binary = (y_pred >= 0.5).astype(int)
    accuracy = accuracy_score(y_like_binary, y_pred_binary)
    precision = precision_score(y_like_binary, y_pred_binary, zero_division=0)
    recall = recall_score(y_like_binary, y_pred_binary, zero_division=0)
    f1 = f1_score(y_like_binary, y_pred_binary, zero_division=0)
    
    return {
        'method': method_name,
        'n_pairs': len(common_pairs),
        'regression': {
            'pearson_r': float(pearson_corr),
            'pearson_p': float(pearson_p),
            'spearman_r': float(spearman_corr),
            'spearman_p': float(spearman_p),
            'mse': float(mse),
            'mae': float(mae),
        },
        'classification': {
            'roc_auc': float(roc_auc) if roc_auc is not None else None,
            'pr_auc': float(pr_auc) if pr_auc is not None else None,
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
        },
        'like_stats': {
            'like_threshold': like_threshold,
            'positive_pairs': int(y_like_binary.sum()),
            'negative_pairs': int((1 - y_like_binary).sum()),
        }
    }


def print_metrics_table(all_metrics: List[Dict]):
    """Print comprehensive metrics table"""
    print("\n" + "=" * 120)
    print("COMPREHENSIVE EVALUATION AGAINST GROUND TRUTH 'LIKE' SCORES")
    print("=" * 120)
    
    # Regression metrics
    print("\n📊 REGRESSION METRICS (Correlation with 'like' scores)")
    print("-" * 120)
    print(f"{'Method':<30} {'Pearson r':<12} {'Spearman r':<12} {'MSE':<12} {'MAE':<12}")
    print("-" * 120)
    
    for metrics in all_metrics:
        if metrics is None:
            continue
        method = metrics['method']
        reg = metrics['regression']
        print(f"{method:<30} {reg['pearson_r']:>11.4f} {reg['spearman_r']:>11.4f} {reg['mse']:>11.4f} {reg['mae']:>11.4f}")
    
    # Classification metrics
    print("\n📊 CLASSIFICATION METRICS (Binary prediction using like threshold)")
    print("-" * 120)
    print(f"{'Method':<30} {'ROC AUC':<12} {'PR-AUC':<12} {'Accuracy':<12} {'F1':<12}")
    print("-" * 120)
    
    for metrics in all_metrics:
        if metrics is None:
            continue
        method = metrics['method']
        cls = metrics['classification']
        roc_str = f"{cls['roc_auc']:.4f}" if cls['roc_auc'] is not None else "N/A"
        pr_str = f"{cls['pr_auc']:.4f}" if cls['pr_auc'] is not None else "N/A"
        print(f"{method:<30} {roc_str:>11} {pr_str:>11} {cls['accuracy']:>11.4f} {cls['f1']:>11.4f}")
    
    print("=" * 120)


def plot_comparison(all_metrics: List[Dict], output_dir: str = "results"):
    """Generate comparison plots"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Extract method names and metrics
    methods = []
    pearson_r = []
    spearman_r = []
    roc_auc = []
    f1 = []
    
    for metrics in all_metrics:
        if metrics is None:
            continue
        methods.append(metrics['method'])
        pearson_r.append(metrics['regression']['pearson_r'])
        spearman_r.append(metrics['regression']['spearman_r'])
        roc_auc.append(metrics['classification']['roc_auc'] if metrics['classification']['roc_auc'] is not None else 0)
        f1.append(metrics['classification']['f1'])
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Pearson correlation
    ax = axes[0, 0]
    bars = ax.barh(methods, pearson_r, color='steelblue')
    ax.set_xlabel('Pearson Correlation', fontsize=12)
    ax.set_title('Pearson Correlation with Like Scores', fontsize=14, fontweight='bold')
    ax.set_xlim([-0.2, 1.0])
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + 0.02, bar.get_y() + bar.get_height()/2, f'{pearson_r[i]:.3f}',
               ha='left', va='center', fontsize=9)
    
    # Spearman correlation
    ax = axes[0, 1]
    bars = ax.barh(methods, spearman_r, color='darkorange')
    ax.set_xlabel('Spearman Correlation', fontsize=12)
    ax.set_title('Spearman Correlation with Like Scores', fontsize=14, fontweight='bold')
    ax.set_xlim([-0.2, 1.0])
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + 0.02, bar.get_y() + bar.get_height()/2, f'{spearman_r[i]:.3f}',
               ha='left', va='center', fontsize=9)
    
    # ROC AUC
    ax = axes[1, 0]
    bars = ax.barh(methods, roc_auc, color='forestgreen')
    ax.set_xlabel('ROC AUC', fontsize=12)
    ax.set_title('ROC AUC (Binary Classification)', fontsize=14, fontweight='bold')
    ax.set_xlim([0, 1.0])
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + 0.02, bar.get_y() + bar.get_height()/2, f'{roc_auc[i]:.3f}',
               ha='left', va='center', fontsize=9)
    
    # F1 Score
    ax = axes[1, 1]
    bars = ax.barh(methods, f1, color='mediumvioletred')
    ax.set_xlabel('F1 Score', fontsize=12)
    ax.set_title('F1 Score (Binary Classification)', fontsize=14, fontweight='bold')
    ax.set_xlim([0, 1.0])
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + 0.02, bar.get_y() + bar.get_height()/2, f'{f1[i]:.3f}',
               ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    plot_path = output_path / "like_score_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved comparison plot to: {plot_path}")
    plt.close()


def main():
    """Main evaluation function"""
    print("🎯 Comprehensive Evaluation Against Ground Truth 'Like' Scores")
    print("=" * 70)
    
    # Paths
    personas_path = "results/personas.json"
    llm_eval_path = "results/llm_score_evaluation.json"
    ensemble_path = "results/ensemble_evaluation.json"
    baseline_path = "results/baseline_comparison_v2.json"
    output_path = "results/like_score_evaluation.json"
    
    # Load ground truth 'like' scores
    print("\n📂 Loading ground truth 'like' scores...")
    like_scores = load_like_scores(personas_path)
    
    # Load all predictions
    print("\n📂 Loading predictions from all methods...")
    llm_predictions = load_llm_predictions(llm_eval_path)
    ensemble_predictions = load_ensemble_predictions(ensemble_path)
    
    # Try to load baseline predictions
    try:
        baseline_predictions = load_baseline_predictions(baseline_path)
    except FileNotFoundError:
        print("⚠️  baseline_evaluation.json not found, skipping baseline methods")
        baseline_predictions = {}
    
    # Combine all predictions
    all_predictions = {**llm_predictions, **ensemble_predictions, **baseline_predictions}
    
    # Compute metrics for each method
    print("\n📊 Computing metrics for all methods...")
    all_metrics = []
    
    for method_name, predictions in all_predictions.items():
        print(f"\n   Processing: {method_name}...")
        metrics = compute_metrics_against_like(
            predictions,
            like_scores,
            method_name,
            like_threshold=5.0,
            use_combined_like=True
        )
        if metrics:
            all_metrics.append(metrics)
    
    # Print results table
    print_metrics_table(all_metrics)
    
    # Generate plots
    print("\n📊 Generating comparison plots...")
    plot_comparison(all_metrics, output_dir="results")
    
    # Save results
    print(f"\n💾 Saving evaluation results to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'evaluation_summary': 'Comprehensive evaluation against ground truth like scores',
            'like_threshold': 5.0,
            'use_combined_like': True,
            'metrics': all_metrics
        }, f, indent=2)
    
    # Find best methods
    print("\n" + "=" * 70)
    print("🏆 BEST METHODS BY METRIC")
    print("=" * 70)
    
    best_pearson = max(all_metrics, key=lambda x: x['regression']['pearson_r'])
    print(f"\n📈 Best Pearson Correlation: {best_pearson['method']}")
    print(f"   Pearson r = {best_pearson['regression']['pearson_r']:.4f}")
    
    best_spearman = max(all_metrics, key=lambda x: x['regression']['spearman_r'])
    print(f"\n📈 Best Spearman Correlation: {best_spearman['method']}")
    print(f"   Spearman r = {best_spearman['regression']['spearman_r']:.4f}")
    
    # Filter out None ROC AUC values
    valid_roc = [m for m in all_metrics if m['classification']['roc_auc'] is not None]
    if valid_roc:
        best_roc = max(valid_roc, key=lambda x: x['classification']['roc_auc'])
        print(f"\n📈 Best ROC AUC: {best_roc['method']}")
        print(f"   ROC AUC = {best_roc['classification']['roc_auc']:.4f}")
    
    best_f1 = max(all_metrics, key=lambda x: x['classification']['f1'])
    print(f"\n📈 Best F1 Score: {best_f1['method']}")
    print(f"   F1 = {best_f1['classification']['f1']:.4f}")
    
    print("\n✅ Evaluation completed!")


if __name__ == "__main__":
    main()
