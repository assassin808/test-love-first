"""
Ensemble Model for Speed Dating Prediction

Combines 3 LLM scoring methods using machine learning:
1. Participant scores (combined)
2. Observer scores
3. Advanced Observer scores (with ICL)

Trains Linear Regression and Logistic Regression models to optimally
combine these scores for better prediction accuracy.
"""

import json
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve, confusion_matrix
)
import matplotlib.pyplot as plt


def load_evaluation_results(results_path: str):
    """Load evaluation results from JSON"""
    with open(results_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def prepare_features(results: dict):
    """
    Extract features and labels from evaluation results
    
    Returns:
        X: Feature matrix (N x 3) containing [participant_score, observer_score, advanced_observer_score]
        y: Ground truth labels (N,)
        pair_ids: List of pair IDs for tracking
    """
    # Extract data from all three methods - new structure
    participant_data = {
        entry['pair_id']: entry['combined_score']
        for entry in results.get('participant_method', {}).get('scores', [])
    }
    
    observer_data = {
        entry['pair_id']: entry['normalized_score']
        for entry in results.get('observer_method', {}).get('scores', [])
    }
    
    advanced_observer_data = {
        entry['pair_id']: entry['normalized_score']
        for entry in results.get('advanced_observer_method', {}).get('scores', [])
    }
    
    # Find common pairs across all methods
    common_pairs = set(participant_data.keys()) & set(observer_data.keys()) & set(advanced_observer_data.keys())
    
    if not common_pairs:
        raise ValueError("No common pairs found across all three methods!")
    
    print(f"Found {len(common_pairs)} pairs with all three scores")
    
    # Build feature matrix
    X = []
    y = []
    pair_ids = []
    
    ground_truth = results.get('ground_truth', [])
    pair_id_list = results.get('pair_ids', [])
    
    for idx, (pair_id, gt) in enumerate(zip(pair_id_list, ground_truth)):
        if pair_id not in common_pairs:
            continue
        
        # Features: [participant_combined, observer_normalized, advanced_observer_normalized]
        features = [
            participant_data[pair_id],
            observer_data[pair_id],
            advanced_observer_data[pair_id]
        ]
        
        X.append(features)
        y.append(int(gt))
        pair_ids.append(pair_id)
    
    return np.array(X), np.array(y), pair_ids


def train_ensemble_models(X, y):
    """
    Train ensemble models using cross-validation
    
    Returns:
        models: Dictionary of trained models
        cv_scores: Cross-validation scores for each model
    """
    print("\n" + "=" * 70)
    print("TRAINING ENSEMBLE MODELS")
    print("=" * 70)
    
    models = {}
    cv_scores = {}
    
    # 1. Linear Regression (for continuous prediction)
    print("\n📊 Training Linear Regression...")
    lr = LinearRegression()
    lr_cv = cross_val_score(lr, X, y, cv=5, scoring='roc_auc')
    lr.fit(X, y)
    models['linear_regression'] = lr
    cv_scores['linear_regression'] = lr_cv
    print(f"   Cross-val ROC AUC: {lr_cv.mean():.4f} (+/- {lr_cv.std():.4f})")
    print(f"   Coefficients: Participant={lr.coef_[0]:.4f}, Observer={lr.coef_[1]:.4f}, Advanced={lr.coef_[2]:.4f}")
    print(f"   Intercept: {lr.intercept_:.4f}")
    
    # 2. Logistic Regression (for binary classification)
    print("\n📊 Training Logistic Regression...")
    logreg = LogisticRegression(random_state=42, max_iter=1000)
    logreg_cv = cross_val_score(logreg, X, y, cv=5, scoring='roc_auc')
    logreg.fit(X, y)
    models['logistic_regression'] = logreg
    cv_scores['logistic_regression'] = logreg_cv
    print(f"   Cross-val ROC AUC: {logreg_cv.mean():.4f} (+/- {logreg_cv.std():.4f})")
    print(f"   Coefficients: Participant={logreg.coef_[0][0]:.4f}, Observer={logreg.coef_[0][1]:.4f}, Advanced={logreg.coef_[0][2]:.4f}")
    print(f"   Intercept: {logreg.intercept_[0]:.4f}")
    
    return models, cv_scores


def evaluate_ensemble(models, X, y, pair_ids, threshold=0.5):
    """
    Evaluate ensemble models and compare with individual methods
    
    Returns:
        results: Dictionary with evaluation metrics for each model
    """
    print("\n" + "=" * 70)
    print("EVALUATING ENSEMBLE MODELS")
    print("=" * 70)
    
    results = {}
    
    # Evaluate each ensemble model
    for model_name, model in models.items():
        print(f"\n📈 {model_name.upper().replace('_', ' ')}")
        
        if model_name == 'linear_regression':
            # Predict continuous values, then threshold
            y_pred_continuous = model.predict(X)
            y_pred = (y_pred_continuous >= threshold).astype(int)
            y_scores = y_pred_continuous
        else:  # logistic_regression
            # Predict probabilities
            y_scores = model.predict_proba(X)[:, 1]
            y_pred = (y_scores >= threshold).astype(int)
        
        # Compute metrics
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, zero_division=0)
        recall = recall_score(y, y_pred, zero_division=0)
        f1 = f1_score(y, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y, y_scores)
        pr_auc = average_precision_score(y, y_scores)
        
        print(f"   Accuracy:  {accuracy:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall:    {recall:.4f}")
        print(f"   F1:        {f1:.4f}")
        print(f"   ROC AUC:   {roc_auc:.4f}")
        print(f"   PR-AUC:    {pr_auc:.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(y, y_pred)
        print(f"\n   Confusion Matrix:")
        print(f"   TN={cm[0,0]:3d}  FP={cm[0,1]:3d}")
        print(f"   FN={cm[1,0]:3d}  TP={cm[1,1]:3d}")
        
        results[model_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'y_pred': y_pred.tolist(),
            'y_scores': y_scores.tolist(),
            'confusion_matrix': cm.tolist()
        }
    
    # Evaluate individual methods for comparison
    print("\n" + "=" * 70)
    print("INDIVIDUAL METHODS (for comparison)")
    print("=" * 70)
    
    for idx, method_name in enumerate(['Participant', 'Observer', 'Advanced Observer']):
        print(f"\n📊 {method_name.upper()}")
        
        X_single = X[:, idx]
        y_pred_single = (X_single >= threshold).astype(int)
        
        accuracy = accuracy_score(y, y_pred_single)
        precision = precision_score(y, y_pred_single, zero_division=0)
        recall = recall_score(y, y_pred_single, zero_division=0)
        f1 = f1_score(y, y_pred_single, zero_division=0)
        roc_auc = roc_auc_score(y, X_single)
        pr_auc = average_precision_score(y, X_single)
        
        print(f"   Accuracy:  {accuracy:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall:    {recall:.4f}")
        print(f"   F1:        {f1:.4f}")
        print(f"   ROC AUC:   {roc_auc:.4f}")
        print(f"   PR-AUC:    {pr_auc:.4f}")
        
        results[f'individual_{method_name.lower().replace(" ", "_")}'] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc
        }
    
    return results


def plot_comparison(results, output_dir="results"):
    """Plot comparison of all methods"""
    print("\n" + "=" * 70)
    print("GENERATING COMPARISON PLOTS")
    print("=" * 70)
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Prepare data
    methods = ['Participant', 'Observer', 'Advanced\nObserver', 'Linear\nRegression', 'Logistic\nRegression']
    method_keys = [
        'individual_participant',
        'individual_observer',
        'individual_advanced_observer',
        'linear_regression',
        'logistic_regression'
    ]
    
    metrics_to_plot = ['roc_auc', 'pr_auc', 'f1', 'accuracy']
    metric_names = ['ROC AUC', 'PR-AUC', 'F1 Score', 'Accuracy']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, (metric, metric_name) in enumerate(zip(metrics_to_plot, metric_names)):
        ax = axes[idx]
        values = [results[key][metric] for key in method_keys]
        
        bars = ax.bar(methods, values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8'])
        ax.set_ylabel(metric_name, fontsize=12)
        ax.set_ylim([0, 1])
        ax.set_title(f'{metric_name} Comparison', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plot_path = output_path / "ensemble_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved comparison plot to: {plot_path}")
    plt.close()


def main():
    """Main function"""
    print("🎯 Ensemble Model Training for Speed Dating Prediction")
    print("=" * 70)
    
    # Load evaluation results
    results_path = "results/llm_score_evaluation.json"
    print(f"\n📂 Loading evaluation results from: {results_path}")
    results = load_evaluation_results(results_path)
    
    # Prepare features
    X, y, pair_ids = prepare_features(results)
    print(f"\n✅ Prepared dataset:")
    print(f"   Features: {X.shape}")
    print(f"   Labels: {y.shape}")
    print(f"   Positive samples: {y.sum()} ({y.mean()*100:.1f}%)")
    print(f"   Negative samples: {len(y)-y.sum()} ({(1-y.mean())*100:.1f}%)")
    
    # Train ensemble models
    models, cv_scores = train_ensemble_models(X, y)
    
    # Evaluate ensemble models
    eval_results = evaluate_ensemble(models, X, y, pair_ids, threshold=0.5)
    
    # Generate comparison plots
    plot_comparison(eval_results, output_dir="results")
    
    # Save results
    output_path = Path("results/ensemble_evaluation.json")
    output_data = {
        'feature_names': ['participant_combined', 'observer_normalized', 'advanced_observer_normalized'],
        'num_samples': len(X),
        'num_positive': int(y.sum()),
        'num_negative': int(len(y) - y.sum()),
        'models': {
            'linear_regression': {
                'coefficients': models['linear_regression'].coef_.tolist(),
                'intercept': float(models['linear_regression'].intercept_),
                'cv_roc_auc_mean': float(cv_scores['linear_regression'].mean()),
                'cv_roc_auc_std': float(cv_scores['linear_regression'].std())
            },
            'logistic_regression': {
                'coefficients': models['logistic_regression'].coef_[0].tolist(),
                'intercept': float(models['logistic_regression'].intercept_[0]),
                'cv_roc_auc_mean': float(cv_scores['logistic_regression'].mean()),
                'cv_roc_auc_std': float(cv_scores['logistic_regression'].std())
            }
        },
        'evaluation_results': eval_results,
        'pair_ids': pair_ids
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n💾 Saved ensemble evaluation results to: {output_path}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    best_model = max(eval_results.items(), key=lambda x: x[1]['roc_auc'] if 'roc_auc' in x[1] else 0)
    print(f"\n🏆 Best Model: {best_model[0].upper().replace('_', ' ')}")
    print(f"   ROC AUC: {best_model[1]['roc_auc']:.4f}")
    print(f"   PR-AUC:  {best_model[1]['pr_auc']:.4f}")
    print(f"   F1:      {best_model[1]['f1']:.4f}")
    
    print("\n✅ Ensemble training and evaluation completed!")


if __name__ == "__main__":
    main()
