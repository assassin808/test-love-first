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


def calibrate_threshold(y_true, scores, metric='f1', grid_size=100):
    """Find the threshold in [0,1] that maximizes the chosen metric on y_true.
    Default metric is F1. Returns best_threshold, best_metric.
    """
    # If scores are outside [0,1], rescale min-max to [0,1] for threshold search
    s = np.array(scores, dtype=float)
    s_min, s_max = float(np.min(s)), float(np.max(s))
    if s_max > s_min:
        s_norm = (s - s_min) / (s_max - s_min)
    else:
        s_norm = np.zeros_like(s)
    best_t, best_val = 0.5, -1.0
    for t in np.linspace(0.0, 1.0, grid_size + 1):
        y_pred = (s_norm >= t).astype(int)
        if metric == 'f1':
            val = f1_score(y_true, y_pred, zero_division=0)
        elif metric == 'accuracy':
            val = accuracy_score(y_true, y_pred)
        else:
            # Fallback to F1
            val = f1_score(y_true, y_pred, zero_division=0)
        if val > best_val:
            best_val = val
            best_t = t
    # Convert threshold back to score scale
    if s_max > s_min:
        threshold_raw = best_t * (s_max - s_min) + s_min
    else:
        threshold_raw = s_min
    return float(threshold_raw), float(best_val)


def train_mul_add(X_train, y_train):
    """Train k1, k2 and threshold for score = k1*(r1*r2) + k2*r3.
    Uses simple grid search on k1,k2 in [-2, 2]. Returns dict with params.
    Requires X with 3 columns. If only 2 columns, returns None.
    """
    if X_train.shape[1] < 3:
        return None
    r1 = X_train[:, 0]
    r2 = X_train[:, 1]
    r3 = X_train[:, 2]
    z1 = r1 * r2
    z2 = r3
    best = {
        'k1': 0.0,
        'k2': 1.0,
        'threshold': 0.5,
        'best_f1': -1.0
    }
    grid = np.linspace(-2.0, 2.0, 21)
    for k1 in grid:
        for k2 in grid:
            scores = k1 * z1 + k2 * z2
            th, f1_val = calibrate_threshold(y_train, scores, metric='f1', grid_size=200)
            if f1_val > best['best_f1']:
                best.update({'k1': float(k1), 'k2': float(k2), 'threshold': th, 'best_f1': float(f1_val)})
    return best


def load_evaluation_results(results_path: str):
    """Load evaluation results from JSON"""
    with open(results_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def prepare_features(results: dict):
    """
    Extract features and labels from evaluation results
    
    Returns:
        X: Feature matrix (N x F) containing available features in order
           [participant_score, observer_score, advanced_observer_score?]
        y: Ground truth labels (N,)
        pair_ids: List of pair IDs for tracking
        feature_names: Names corresponding to feature columns
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
    
    # Determine available methods (observer required, advanced optional)
    have_participant = len(participant_data) > 0
    have_observer = len(observer_data) > 0
    have_advanced = len(advanced_observer_data) > 0

    if not (have_participant and have_observer):
        raise ValueError("Participant and Observer scores are required to train the ensemble.")

    # Find common pairs across available methods
    common_pairs = set(participant_data.keys()) & set(observer_data.keys())
    if have_advanced:
        common_pairs = common_pairs & set(advanced_observer_data.keys())
    
    print(f"Found {len(common_pairs)} pairs with all three scores")
    
    # Build feature matrix
    X = []
    y = []
    pair_ids = []
    feature_names = ['participant_combined', 'observer_normalized'] + (['advanced_observer_normalized'] if have_advanced else [])
    
    ground_truth = results.get('ground_truth', [])
    pair_id_list = results.get('pair_ids', [])
    
    for idx, (pair_id, gt) in enumerate(zip(pair_id_list, ground_truth)):
        if pair_id not in common_pairs:
            continue
        
        # Features in order: participant, observer, (optional) advanced
        features = [participant_data[pair_id], observer_data[pair_id]]
        if have_advanced:
            features.append(advanced_observer_data[pair_id])
        
        X.append(features)
        y.append(int(gt))
        pair_ids.append(pair_id)
    
    return np.array(X), np.array(y), pair_ids, feature_names


def train_ensemble_models(X, y, feature_names):
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
    coef_str = ", ".join([f"{name}={coef:.4f}" for name, coef in zip(feature_names, lr.coef_)])
    print(f"   Coefficients: {coef_str}")
    print(f"   Intercept: {lr.intercept_:.4f}")
    
    # 2. Logistic Regression (for binary classification)
    print("\n📊 Training Logistic Regression...")
    logreg = LogisticRegression(random_state=42, max_iter=1000)
    logreg_cv = cross_val_score(logreg, X, y, cv=5, scoring='roc_auc')
    logreg.fit(X, y)
    models['logistic_regression'] = logreg
    cv_scores['logistic_regression'] = logreg_cv
    print(f"   Cross-val ROC AUC: {logreg_cv.mean():.4f} (+/- {logreg_cv.std():.4f})")
    coef_str = ", ".join([f"{name}={coef:.4f}" for name, coef in zip(feature_names, logreg.coef_[0])])
    print(f"   Coefficients: {coef_str}")
    print(f"   Intercept: {logreg.intercept_[0]:.4f}")
    
    return models, cv_scores


def evaluate_ensemble(models, X, y, pair_ids, feature_names, threshold=0.5, method_thresholds=None, learned_params=None):
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
    
    for idx, method_name in enumerate([n.replace('_', ' ').title() for n in feature_names]):
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
    
    # Add scaled-mean variant: k * mean([r1, r2, r3]) with k=1 (monotonic scaling)
    if X.shape[1] >= 2:  # require at least participant and observer
        print("\n📊 SCALED MEAN (k * mean(r1,r2,r3) with k=1)")
        mean_scores = X.mean(axis=1)
        t_mean = method_thresholds.get('scaled_mean', threshold) if method_thresholds else threshold
        y_pred_mean = (mean_scores >= t_mean).astype(int)

        accuracy = accuracy_score(y, y_pred_mean)
        precision = precision_score(y, y_pred_mean, zero_division=0)
        recall = recall_score(y, y_pred_mean, zero_division=0)
        f1 = f1_score(y, y_pred_mean, zero_division=0)
        roc_auc = roc_auc_score(y, mean_scores)
        pr_auc = average_precision_score(y, mean_scores)

        print(f"   Accuracy:  {accuracy:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall:    {recall:.4f}")
        print(f"   F1:        {f1:.4f}")
        print(f"   ROC AUC:   {roc_auc:.4f}")
        print(f"   PR-AUC:    {pr_auc:.4f}")

        results['scaled_mean'] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'threshold': float(t_mean),
            'y_pred': y_pred_mean.tolist(),
            'y_scores': mean_scores.tolist(),
            'confusion_matrix': confusion_matrix(y, y_pred_mean).tolist()
        }

    # Add multiplicative-additive variant: score = k1*(r1*r2) + k2*r3
    if X.shape[1] >= 3 and learned_params and learned_params.get('mul_add'):
        print("\n📊 MUL_ADD (score = k1*r1*r2 + k2*r3)")
        r1 = X[:, 0]
        r2 = X[:, 1]
        r3 = X[:, 2]
        k1 = learned_params['mul_add']['k1']
        k2 = learned_params['mul_add']['k2']
        t_mul = learned_params['mul_add']['threshold']
        scores = k1 * (r1 * r2) + k2 * r3
        y_pred_mul = (scores >= t_mul).astype(int)

        accuracy = accuracy_score(y, y_pred_mul)
        precision = precision_score(y, y_pred_mul, zero_division=0)
        recall = recall_score(y, y_pred_mul, zero_division=0)
        f1 = f1_score(y, y_pred_mul, zero_division=0)
        roc_auc = roc_auc_score(y, scores)
        pr_auc = average_precision_score(y, scores)

        print(f"   k1={k1:.3f}  k2={k2:.3f}  thr={t_mul:.3f}")
        print(f"   Accuracy:  {accuracy:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall:    {recall:.4f}")
        print(f"   F1:        {f1:.4f}")
        print(f"   ROC AUC:   {roc_auc:.4f}")
        print(f"   PR-AUC:    {pr_auc:.4f}")

        results['mul_add'] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'k1': float(k1),
            'k2': float(k2),
            'threshold': float(t_mul),
            'y_pred': y_pred_mul.tolist(),
            'y_scores': scores.tolist(),
            'confusion_matrix': confusion_matrix(y, y_pred_mul).tolist()
        }

    return results


def plot_comparison(results, feature_names, output_dir="results"):
    """Plot comparison of all methods"""
    print("\n" + "=" * 70)
    print("GENERATING COMPARISON PLOTS")
    print("=" * 70)
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Prepare data
    base_methods = [n.replace('_', ' ').title() for n in feature_names]
    methods = base_methods + ['Scaled\nMean', 'Linear\nRegression', 'Logistic\nRegression']
    method_keys = [f"individual_{n}" for n in feature_names] + ['scaled_mean', 'linear_regression', 'logistic_regression']
    # Optionally include mul_add if available
    if 'mul_add' in results:
        methods.append('Mul+Add')
        method_keys.append('mul_add')
    
    metrics_to_plot = ['roc_auc', 'pr_auc', 'f1', 'accuracy']
    metric_names = ['ROC AUC', 'PR-AUC', 'F1 Score', 'Accuracy']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, (metric, metric_name) in enumerate(zip(metrics_to_plot, metric_names)):
        ax = axes[idx]
        values = [results[key][metric] for key in method_keys]
        
        # Generate colors dynamically to match number of methods
        base_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#A29BFE', '#55EFC4']
        colors = (base_colors * ((len(methods) // len(base_colors)) + 1))[:len(methods)]
        bars = ax.bar(methods, values, color=colors)
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
    import argparse
    parser = argparse.ArgumentParser(description='Train ensemble models on LLM results')
    parser.add_argument('--llm-results', default='results/llm_score_evaluation.json',
                       help='Path to LLM evaluation results JSON')
    parser.add_argument('--observer-results', default=None,
                       help='Optional path to an alternate results JSON to override observer (and advanced observer) sections, e.g., stage-specific observer rerun')
    parser.add_argument('--participant-results', default=None,
                       help='Optional path to an alternate results JSON to override participant section')
    parser.add_argument('--output', default='results/ensemble_evaluation.json',
                       help='Output path for ensemble results')
    args = parser.parse_args()
    
    print("🎯 Ensemble Model Training for Speed Dating Prediction")
    print("=" * 70)
    
    # Load evaluation results
    results_path = args.llm_results
    print(f"\n📂 Loading evaluation results from: {results_path}")
    results = load_evaluation_results(results_path)
    
    # Optionally override sections from alternate files for fresh observer/participant runs
    if args.observer_results:
        print(f"📌 Overriding observer sections from: {args.observer_results}")
        obs_res = load_evaluation_results(args.observer_results)
        if 'observer_method' in obs_res:
            results['observer_method'] = obs_res['observer_method']
        if 'advanced_observer_method' in obs_res:
            results['advanced_observer_method'] = obs_res['advanced_observer_method']
    if args.participant_results:
        print(f"📌 Overriding participant section from: {args.participant_results}")
        part_res = load_evaluation_results(args.participant_results)
        if 'participant_method' in part_res:
            results['participant_method'] = part_res['participant_method']
    
    # Prepare features
    X, y, pair_ids, feature_names = prepare_features(results)
    print(f"\n✅ Prepared dataset:")
    print(f"   Features: {X.shape}")
    print(f"   Labels: {y.shape}")
    print(f"   Positive samples: {y.sum()} ({y.mean()*100:.1f}%)")
    print(f"   Negative samples: {len(y)-y.sum()} ({(1-y.mean())*100:.1f}%)")
    # Split into 30% train and 70% test
    X_train, X_test, y_train, y_test, pair_train, pair_test = train_test_split(
        X, y, pair_ids, test_size=0.7, random_state=42, stratify=y
    )
    print(f"\n🔀 Train/Test split → Train: {len(y_train)} ({len(y_train)/len(y)*100:.1f}%), Test: {len(y_test)} ({len(y_test)/len(y)*100:.1f}%)")

    # Train ensemble models on 30%
    models, cv_scores = train_ensemble_models(X_train, y_train, feature_names)

    # Calibrate thresholds for simple score-based ensembles on train
    method_thresholds = {}
    if X_train.shape[1] >= 2:
        mean_scores_train = X_train.mean(axis=1)
        thr_mean, _ = calibrate_threshold(y_train, mean_scores_train, metric='f1', grid_size=200)
        method_thresholds['scaled_mean'] = thr_mean

    # Learn k1,k2,threshold for mul_add on train
    learned_params = {}
    mul_add_params = train_mul_add(X_train, y_train)
    if mul_add_params:
        learned_params['mul_add'] = mul_add_params
    
    # Evaluate ensemble models on 70% test set
    eval_results = evaluate_ensemble(
        models, X_test, y_test, pair_test, feature_names,
        threshold=0.5,
        method_thresholds=method_thresholds,
        learned_params=learned_params
    )
    
    # Generate comparison plots
    plot_comparison(eval_results, feature_names, output_dir="results")
    
    # Save results
    output_path = Path(args.output) if args.output else Path("results/ensemble_evaluation.json")
    output_data = {
    'feature_names': feature_names,
    'num_samples': len(X_test),
    'num_positive': int(y_test.sum()),
    'num_negative': int(len(y_test) - y_test.sum()),
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
            },
            # Scaled-mean records the calibrated threshold (k=1 fixed)
            'scaled_mean': {
                'k': 1.0,
                'threshold': float(method_thresholds.get('scaled_mean', 0.5)),
                'description': 'k * mean([participant, observer, advanced]); threshold calibrated on train'
            },
            # Mul-add model parameters if available
            **({'mul_add': {
                'k1': float(learned_params['mul_add']['k1']),
                'k2': float(learned_params['mul_add']['k2']),
                'threshold': float(learned_params['mul_add']['threshold']),
                'description': 'score = k1*(r1*r2) + k2*r3; params learned on train'
            }} if 'mul_add' in learned_params else {})
        },
        'evaluation_results': eval_results,
        'pair_ids_test': pair_test
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
