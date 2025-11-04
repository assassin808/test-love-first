"""
Improved Evaluation Against Ground Truth 'like' Scores

Key improvements:
1. Train linear scaling (kx + b) for each method to align with 'like' scores
2. Clearer explanation of binary classification
3. Include baseline methods from baseline_comparison_v2.json
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error,
    roc_auc_score, average_precision_score,
    accuracy_score, precision_score, recall_score, f1_score
)
import matplotlib.pyplot as plt


def load_like_scores(personas_path: str) -> Dict[str, Tuple[float, float]]:
    """Load ground truth 'like' scores from personas.json"""
    with open(personas_path, 'r', encoding='utf-8') as f:
        personas = json.load(f)
    
    like_scores = {}
    for entry in personas:
        p1 = entry['person1']
        p2 = entry['person2']
        pair_id = f"pair_{p1['iid']}_{p2['iid']}"
        
        ground_truth = entry.get('ground_truth', {})
        p1_ratings = ground_truth.get('person1_ratings', {})
        p2_ratings = ground_truth.get('person2_ratings', {})
        
        p1_like = p1_ratings.get('like', None)
        p2_like = p2_ratings.get('like', None)
        
        if p1_like is not None and p2_like is not None:
            like_scores[pair_id] = (float(p1_like), float(p2_like))
    
    print(f"✅ Loaded 'like' scores for {len(like_scores)} pairs")
    return like_scores


def load_all_predictions(llm_path: str, ensemble_path: str, baseline_path: str) -> Dict[str, Dict[str, float]]:
    """Load predictions from all methods"""
    all_predictions = {}
    
    # Load LLM predictions
    with open(llm_path, 'r', encoding='utf-8') as f:
        llm_data = json.load(f)
    
    for entry in llm_data.get('participant_method', {}).get('scores', []):
        pair_id = entry['pair_id']
        if 'participant' not in all_predictions:
            all_predictions['participant'] = {}
        all_predictions['participant'][pair_id] = entry['combined_score']
    
    for entry in llm_data.get('observer_method', {}).get('scores', []):
        pair_id = entry['pair_id']
        if 'observer' not in all_predictions:
            all_predictions['observer'] = {}
        all_predictions['observer'][pair_id] = entry['normalized_score']
    
    for entry in llm_data.get('advanced_observer_method', {}).get('scores', []):
        pair_id = entry['pair_id']
        if 'advanced_observer' not in all_predictions:
            all_predictions['advanced_observer'] = {}
        all_predictions['advanced_observer'][pair_id] = entry['normalized_score']
    
    print(f"✅ Loaded LLM predictions:")
    print(f"   Participant: {len(all_predictions.get('participant', {}))} pairs")
    print(f"   Observer: {len(all_predictions.get('observer', {}))} pairs")
    print(f"   Advanced Observer: {len(all_predictions.get('advanced_observer', {}))} pairs")
    
    # Load ensemble predictions
    with open(ensemble_path, 'r', encoding='utf-8') as f:
        ensemble_data = json.load(f)
    
    pair_ids = ensemble_data.get('pair_ids', [])
    lr_results = ensemble_data.get('evaluation_results', {}).get('linear_regression', {})
    logreg_results = ensemble_data.get('evaluation_results', {}).get('logistic_regression', {})
    
    all_predictions['ensemble_linear'] = {}
    all_predictions['ensemble_logistic'] = {}
    
    for pair_id, score in zip(pair_ids, lr_results.get('y_scores', [])):
        all_predictions['ensemble_linear'][pair_id] = score
    
    for pair_id, score in zip(pair_ids, logreg_results.get('y_scores', [])):
        all_predictions['ensemble_logistic'][pair_id] = score
    
    print(f"✅ Loaded ensemble predictions:")
    print(f"   Linear Ensemble: {len(all_predictions['ensemble_linear'])} pairs")
    print(f"   Logistic Ensemble: {len(all_predictions['ensemble_logistic'])} pairs")
    
    # Load baseline predictions
    with open(baseline_path, 'r', encoding='utf-8') as f:
        baseline_data = json.load(f)
    
    for method_name, method_data in baseline_data.items():
        if isinstance(method_data, dict) and 'predictions' in method_data:
            all_predictions[f'baseline_{method_name}'] = {}
            for entry in method_data['predictions']:
                pair_id = entry['pair_id']
                prob = entry['probability']
                all_predictions[f'baseline_{method_name}'][pair_id] = prob
    
    baseline_count = sum(1 for k in all_predictions.keys() if k.startswith('baseline_'))
    print(f"✅ Loaded {baseline_count} baseline methods")
    
    return all_predictions


def train_linear_scaling(predictions: Dict[str, float], like_scores: Dict[str, Tuple[float, float]]) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Train linear scaling: like_combined = k * prediction + b
    
    Returns:
        k (slope), b (intercept), scaled_predictions, like_combined_scores
    """
    common_pairs = set(predictions.keys()) & set(like_scores.keys())
    
    X = []
    y = []
    
    for pair_id in common_pairs:
        pred = predictions[pair_id]
        like1, like2 = like_scores[pair_id]
        like_combined = (like1 * like2) / 100.0  # Normalize to 0-1
        
        X.append([pred])
        y.append(like_combined)
    
    X = np.array(X)
    y = np.array(y)
    
    # Train linear regression
    lr = LinearRegression()
    lr.fit(X, y)
    
    k = lr.coef_[0]
    b = lr.intercept_
    
    # Apply scaling
    scaled_pred = k * X.flatten() + b
    
    return k, b, scaled_pred, y


def evaluate_method_with_scaling(
    predictions: Dict[str, float],
    like_scores: Dict[str, Tuple[float, float]],
    method_name: str,
    like_threshold: float = 5.0
) -> Dict:
    """
    Evaluate method against 'like' scores with trained linear scaling
    """
    common_pairs = set(predictions.keys()) & set(like_scores.keys())
    
    if not common_pairs:
        print(f"⚠️  No common pairs for {method_name}")
        return None
    
    # Train linear scaling
    k, b, scaled_pred, y_like = train_linear_scaling(predictions, like_scores)
    
    # Original predictions (no scaling)
    y_pred_raw = np.array([predictions[pid] for pid in common_pairs])
    
    # Binary labels: both people like >= threshold
    y_like_binary = []
    for pair_id in common_pairs:
        like1, like2 = like_scores[pair_id]
        y_like_binary.append(1 if (like1 >= like_threshold and like2 >= like_threshold) else 0)
    y_like_binary = np.array(y_like_binary)
    
    # Regression metrics (with scaling)
    pearson_scaled, _ = pearsonr(scaled_pred, y_like)
    spearman_scaled, _ = spearmanr(scaled_pred, y_like)
    mse_scaled = mean_squared_error(y_like, scaled_pred)
    mae_scaled = mean_absolute_error(y_like, scaled_pred)
    
    # Regression metrics (without scaling)
    pearson_raw, _ = pearsonr(y_pred_raw, y_like)
    spearman_raw, _ = spearmanr(y_pred_raw, y_like)
    mse_raw = mean_squared_error(y_like, y_pred_raw)
    mae_raw = mean_absolute_error(y_like, y_pred_raw)
    
    # Classification metrics (using raw predictions for ROC AUC)
    try:
        roc_auc = roc_auc_score(y_like_binary, y_pred_raw)
        pr_auc = average_precision_score(y_like_binary, y_pred_raw)
    except ValueError:
        roc_auc = None
        pr_auc = None
    
    # Binary predictions at 0.5 threshold (raw)
    y_pred_binary = (y_pred_raw >= 0.5).astype(int)
    accuracy = accuracy_score(y_like_binary, y_pred_binary)
    precision = precision_score(y_like_binary, y_pred_binary, zero_division=0)
    recall = recall_score(y_like_binary, y_pred_binary, zero_division=0)
    f1 = f1_score(y_like_binary, y_pred_binary, zero_division=0)
    
    return {
        'method': method_name,
        'n_pairs': len(common_pairs),
        'linear_scaling': {
            'slope_k': float(k),
            'intercept_b': float(b),
            'formula': f'like_combined = {k:.4f} * prediction + {b:.4f}'
        },
        'regression_scaled': {
            'pearson_r': float(pearson_scaled),
            'spearman_r': float(spearman_scaled),
            'mse': float(mse_scaled),
            'mae': float(mae_scaled),
        },
        'regression_raw': {
            'pearson_r': float(pearson_raw),
            'spearman_r': float(spearman_raw),
            'mse': float(mse_raw),
            'mae': float(mae_raw),
        },
        'classification': {
            'roc_auc': float(roc_auc) if roc_auc is not None else None,
            'pr_auc': float(pr_auc) if pr_auc is not None else None,
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'note': f'Binary labels: like1≥{like_threshold} AND like2≥{like_threshold} → Match=1, else No Match=0'
        },
        'like_stats': {
            'like_threshold': like_threshold,
            'positive_pairs': int(y_like_binary.sum()),
            'negative_pairs': int((1 - y_like_binary).sum()),
        }
    }


def print_results_table(all_metrics: List[Dict]):
    """Print comprehensive results table"""
    print("\n" + "=" * 150)
    print("COMPREHENSIVE EVALUATION AGAINST GROUND TRUTH 'LIKE' SCORES")
    print("=" * 150)
    
    print("\n📊 REGRESSION METRICS (With Trained Linear Scaling: like_combined = k*pred + b)")
    print("-" * 150)
    print(f"{'Method':<35} {'k (slope)':<12} {'b (intercept)':<15} {'Pearson r':<12} {'Spearman r':<12} {'MSE':<12} {'MAE':<12}")
    print("-" * 150)
    
    for metrics in all_metrics:
        if metrics is None:
            continue
        method = metrics['method']
        scaling = metrics['linear_scaling']
        reg = metrics['regression_scaled']
        print(f"{method:<35} {scaling['slope_k']:>11.4f} {scaling['intercept_b']:>14.4f} {reg['pearson_r']:>11.4f} {reg['spearman_r']:>11.4f} {reg['mse']:>11.4f} {reg['mae']:>11.4f}")
    
    print("\n📊 REGRESSION METRICS (Raw predictions, no scaling)")
    print("-" * 150)
    print(f"{'Method':<35} {'Pearson r':<12} {'Spearman r':<12} {'MSE':<12} {'MAE':<12}")
    print("-" * 150)
    
    for metrics in all_metrics:
        if metrics is None:
            continue
        method = metrics['method']
        reg = metrics['regression_raw']
        print(f"{method:<35} {reg['pearson_r']:>11.4f} {reg['spearman_r']:>11.4f} {reg['mse']:>11.4f} {reg['mae']:>11.4f}")
    
    print("\n📊 CLASSIFICATION METRICS (Binary: both_like≥5.0 → Match=1)")
    print("-" * 150)
    print(f"{'Method':<35} {'ROC AUC':<12} {'PR-AUC':<12} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-" * 150)
    
    for metrics in all_metrics:
        if metrics is None:
            continue
        method = metrics['method']
        cls = metrics['classification']
        roc_str = f"{cls['roc_auc']:.4f}" if cls['roc_auc'] is not None else "N/A"
        pr_str = f"{cls['pr_auc']:.4f}" if cls['pr_auc'] is not None else "N/A"
        print(f"{method:<35} {roc_str:>11} {pr_str:>11} {cls['accuracy']:>11.4f} {cls['precision']:>11.4f} {cls['recall']:>11.4f} {cls['f1']:>11.4f}")
    
    print("=" * 150)


def plot_comparison(all_metrics: List[Dict], output_dir: str = "results"):
    """Generate enhanced comparison plots"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Extract data
    methods = []
    pearson_scaled = []
    pearson_raw = []
    roc_auc = []
    f1 = []
    
    for metrics in all_metrics:
        if metrics is None:
            continue
        methods.append(metrics['method'])
        pearson_scaled.append(metrics['regression_scaled']['pearson_r'])
        pearson_raw.append(metrics['regression_raw']['pearson_r'])
        roc_auc.append(metrics['classification']['roc_auc'] if metrics['classification']['roc_auc'] is not None else 0)
        f1.append(metrics['classification']['f1'])
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    
    # Pearson (scaled)
    ax = axes[0, 0]
    bars = ax.barh(methods, pearson_scaled, color='steelblue')
    ax.set_xlabel('Pearson Correlation (with linear scaling)', fontsize=12)
    ax.set_title('Pearson Correlation with Like Scores (SCALED)', fontsize=14, fontweight='bold')
    ax.set_xlim([-0.3, 1.0])
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + 0.02 if width >= 0 else width - 0.02, bar.get_y() + bar.get_height()/2, 
               f'{pearson_scaled[i]:.3f}', ha='left' if width >= 0 else 'right', va='center', fontsize=9)
    
    # Pearson (raw)
    ax = axes[0, 1]
    bars = ax.barh(methods, pearson_raw, color='darkorange')
    ax.set_xlabel('Pearson Correlation (raw predictions)', fontsize=12)
    ax.set_title('Pearson Correlation with Like Scores (RAW)', fontsize=14, fontweight='bold')
    ax.set_xlim([-0.3, 1.0])
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + 0.02 if width >= 0 else width - 0.02, bar.get_y() + bar.get_height()/2,
               f'{pearson_raw[i]:.3f}', ha='left' if width >= 0 else 'right', va='center', fontsize=9)
    
    # ROC AUC
    ax = axes[1, 0]
    bars = ax.barh(methods, roc_auc, color='forestgreen')
    ax.set_xlabel('ROC AUC', fontsize=12)
    ax.set_title('ROC AUC (Binary: both_like≥5 → Match)', fontsize=14, fontweight='bold')
    ax.set_xlim([0, 1.0])
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + 0.02, bar.get_y() + bar.get_height()/2, f'{roc_auc[i]:.3f}',
               ha='left', va='center', fontsize=9)
    
    # F1 Score
    ax = axes[1, 1]
    bars = ax.barh(methods, f1, color='mediumvioletred')
    ax.set_xlabel('F1 Score', fontsize=12)
    ax.set_title('F1 Score (Binary: both_like≥5 → Match)', fontsize=14, fontweight='bold')
    ax.set_xlim([0, 1.0])
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + 0.02, bar.get_y() + bar.get_height()/2, f'{f1[i]:.3f}',
               ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    plot_path = output_path / "like_score_comparison_improved.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved comparison plot to: {plot_path}")
    plt.close()


def main():
    """Main evaluation function"""
    print("🎯 IMPROVED Evaluation Against Ground Truth 'Like' Scores")
    print("=" * 70)
    print("Improvements:")
    print("1. Train linear scaling (k*pred + b) for each method")
    print("2. Include ALL baseline methods")
    print("3. Clear binary classification explanation")
    print("=" * 70)
    
    # Paths
    personas_path = "results/personas.json"
    llm_path = "results/llm_score_evaluation.json"
    ensemble_path = "results/ensemble_evaluation.json"
    baseline_path = "results/baseline_comparison_v2.json"
    output_path = "results/like_score_evaluation_improved.json"
    
    # Load data
    print("\n📂 Loading data...")
    like_scores = load_like_scores(personas_path)
    all_predictions = load_all_predictions(llm_path, ensemble_path, baseline_path)
    
    # Evaluate all methods
    print("\n📊 Evaluating all methods with linear scaling...")
    all_metrics = []
    
    for method_name, predictions in all_predictions.items():
        print(f"\n   Processing: {method_name}...")
        metrics = evaluate_method_with_scaling(
            predictions,
            like_scores,
            method_name,
            like_threshold=5.0
        )
        if metrics:
            all_metrics.append(metrics)
    
    # Print results
    print_results_table(all_metrics)
    
    # Generate plots
    print("\n📊 Generating comparison plots...")
    plot_comparison(all_metrics, output_dir="results")
    
    # Save results
    print(f"\n💾 Saving results to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'evaluation_summary': 'Improved evaluation with linear scaling and all methods',
            'like_threshold': 5.0,
            'explanation': {
                'linear_scaling': 'Each method trains: like_combined = k*prediction + b',
                'binary_classification': 'Like scores converted to binary: (like1≥5.0 AND like2≥5.0) → Match=1',
                'like_combined': 'Combined like score = (like1 * like2) / 100, normalized to [0,1]'
            },
            'metrics': all_metrics
        }, f, indent=2)
    
    # Find best methods
    print("\n" + "=" * 70)
    print("🏆 BEST METHODS BY METRIC")
    print("=" * 70)
    
    best_pearson_scaled = max(all_metrics, key=lambda x: x['regression_scaled']['pearson_r'])
    print(f"\n📈 Best Pearson (Scaled): {best_pearson_scaled['method']}")
    print(f"   Pearson r = {best_pearson_scaled['regression_scaled']['pearson_r']:.4f}")
    print(f"   Scaling: {best_pearson_scaled['linear_scaling']['formula']}")
    
    best_pearson_raw = max(all_metrics, key=lambda x: x['regression_raw']['pearson_r'])
    print(f"\n📈 Best Pearson (Raw): {best_pearson_raw['method']}")
    print(f"   Pearson r = {best_pearson_raw['regression_raw']['pearson_r']:.4f}")
    
    valid_roc = [m for m in all_metrics if m['classification']['roc_auc'] is not None]
    if valid_roc:
        best_roc = max(valid_roc, key=lambda x: x['classification']['roc_auc'])
        print(f"\n📈 Best ROC AUC: {best_roc['method']}")
        print(f"   ROC AUC = {best_roc['classification']['roc_auc']:.4f}")
    
    best_f1 = max(all_metrics, key=lambda x: x['classification']['f1'])
    print(f"\n📈 Best F1 Score: {best_f1['method']}")
    print(f"   F1 = {best_f1['classification']['f1']:.4f}")
    
    print("\n✅ Improved evaluation completed!")


if __name__ == "__main__":
    main()
