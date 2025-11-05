"""
Comprehensive Comparison: LLM Predictions vs Traditional ML Baselines

Compares:
1. LLM Self-Evaluation (both participants say YES)
2. LLM Observer Evaluation (恋爱观察员 prediction)
3. Similarity Baseline (cosine similarity)
4. Logistic Regression
5. Random Forest
6. XGBoost

Generates comparison table, statistical analysis, and visualizations.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import os

def load_llm_results(conversations_path: str = 'results/conversations.json') -> Dict:
    """Load LLM evaluation results."""
    with open(conversations_path, 'r') as f:
        conversations = json.load(f)
    
    return conversations

def load_baseline_results(baseline_path: str = 'results/baseline_comparison.json') -> Dict:
    """Load baseline model results."""
    with open(baseline_path, 'r') as f:
        baseline_results = json.load(f)
    
    return baseline_results

def compare_all_methods(
    conversations_path: str = 'results/conversations.json',
    baseline_path: str = 'results/baseline_comparison.json',
    output_dir: str = 'results'
):
    """
    Generate comprehensive comparison report.
    """
    print("=" * 70)
    print("COMPREHENSIVE COMPARISON: LLM vs Traditional ML Baselines")
    print("=" * 70)
    
    # Load LLM results
    if not os.path.exists(conversations_path):
        print(f"\nERROR: LLM results not found at {conversations_path}")
        print("Please run the speed dating simulator first:")
        print("  cd experiments && echo '3' | python speed_dating_simulator.py")
        return
    
    conversations = load_llm_results(conversations_path)
    print(f"\nLoaded {len(conversations)} LLM conversations")
    
    # Calculate LLM metrics
    llm_self_predictions = []
    llm_observer_predictions = []
    ground_truth = []
    
    for conv in conversations:
        # Ground truth
        gt = conv['ground_truth']['match']
        ground_truth.append(gt)
        
        # LLM Self-evaluation (both must say YES)
        evals = conv.get('evaluations', {})
        p1_decision = evals.get('person1_self_evaluation', {}).get('decision', '').upper()
        p2_decision = evals.get('person2_self_evaluation', {}).get('decision', '').upper()
        
        self_pred = (p1_decision == 'YES' and p2_decision == 'YES')
        llm_self_predictions.append(self_pred)
        
        # LLM Observer evaluation
        observer_pred = evals.get('observer_evaluation', {}).get('prediction', '').upper()
        llm_observer_predictions.append(observer_pred == 'MATCH')
    
    # Calculate LLM metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    llm_self_metrics = {
        'model': 'LLM Self-Evaluation',
        'accuracy': accuracy_score(ground_truth, llm_self_predictions),
        'precision': precision_score(ground_truth, llm_self_predictions, zero_division=0),
        'recall': recall_score(ground_truth, llm_self_predictions, zero_division=0),
        'f1': f1_score(ground_truth, llm_self_predictions, zero_division=0),
    }
    
    llm_observer_metrics = {
        'model': 'LLM Observer',
        'accuracy': accuracy_score(ground_truth, llm_observer_predictions),
        'precision': precision_score(ground_truth, llm_observer_predictions, zero_division=0),
        'recall': recall_score(ground_truth, llm_observer_predictions, zero_division=0),
        'f1': f1_score(ground_truth, llm_observer_predictions, zero_division=0),
    }
    
    print("\n--- LLM EVALUATION RESULTS ---")
    print(f"Self-Evaluation  - Accuracy: {llm_self_metrics['accuracy']:.3f}, "
          f"Precision: {llm_self_metrics['precision']:.3f}, "
          f"Recall: {llm_self_metrics['recall']:.3f}, "
          f"F1: {llm_self_metrics['f1']:.3f}")
    
    print(f"Observer         - Accuracy: {llm_observer_metrics['accuracy']:.3f}, "
          f"Precision: {llm_observer_metrics['precision']:.3f}, "
          f"Recall: {llm_observer_metrics['recall']:.3f}, "
          f"F1: {llm_observer_metrics['f1']:.3f}")
    
    # Load baseline results
    if not os.path.exists(baseline_path):
        print(f"\nWARNING: Baseline results not found at {baseline_path}")
        print("Skipping baseline comparison. Run baseline evaluation first:")
        print("  python experiments/baseline_models.py")
        
        # Create comparison with just LLM results
        comparison_df = pd.DataFrame([llm_self_metrics, llm_observer_metrics])
    else:
        baseline_results = load_baseline_results(baseline_path)
        
        print("\n--- BASELINE MODEL RESULTS ---")
        for key, result in baseline_results.items():
            if key != 'summary':
                print(f"{result['model']:20s} - Accuracy: {result['accuracy']:.3f}, "
                      f"Precision: {result['precision']:.3f}, "
                      f"Recall: {result['recall']:.3f}, "
                      f"F1: {result['f1']:.3f}")
        
        # Create comparison dataframe
        all_results = [llm_self_metrics, llm_observer_metrics]
        
        for key in ['similarity', 'logistic_regression', 'random_forest', 'xgboost']:
            if key in baseline_results:
                all_results.append(baseline_results[key])
        
        comparison_df = pd.DataFrame(all_results)
    
    # Print comparison table
    print("\n" + "=" * 70)
    print("COMPREHENSIVE COMPARISON TABLE")
    print("=" * 70)
    
    comparison_df_display = comparison_df.copy()
    for col in ['accuracy', 'precision', 'recall', 'f1']:
        if col in comparison_df_display.columns:
            comparison_df_display[col] = comparison_df_display[col].apply(lambda x: f"{x:.3f}")
    
    # Add AUC if available
    if 'auc' in comparison_df.columns:
        comparison_df_display['auc'] = comparison_df['auc'].apply(
            lambda x: f"{x:.3f}" if isinstance(x, (int, float)) else 'N/A'
        )
    
    print(comparison_df_display[['model', 'accuracy', 'precision', 'recall', 'f1']].to_string(index=False))
    
    # Find best methods
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)
    
    best_accuracy = comparison_df.loc[comparison_df['accuracy'].idxmax()]
    best_f1 = comparison_df.loc[comparison_df['f1'].idxmax()]
    
    print(f"\n✓ Best Accuracy:  {best_accuracy['model']} ({best_accuracy['accuracy']:.3f})")
    print(f"✓ Best F1 Score:  {best_f1['model']} ({best_f1['f1']:.3f})")
    
    # Compare LLM vs Best ML
    llm_best_f1 = max(llm_self_metrics['f1'], llm_observer_metrics['f1'])
    llm_best_model = 'LLM Self-Evaluation' if llm_self_metrics['f1'] > llm_observer_metrics['f1'] else 'LLM Observer'
    
    if os.path.exists(baseline_path):
        ml_best_f1 = comparison_df[~comparison_df['model'].str.contains('LLM')]['f1'].max()
        ml_best_model = comparison_df.loc[
            comparison_df[~comparison_df['model'].str.contains('LLM')]['f1'].idxmax(),
            'model'
        ]
        
        print(f"\n--- LLM vs Traditional ML ---")
        print(f"Best LLM:  {llm_best_model} (F1 = {llm_best_f1:.3f})")
        print(f"Best ML:   {ml_best_model} (F1 = {ml_best_f1:.3f})")
        
        if llm_best_f1 > ml_best_f1:
            improvement = (llm_best_f1 - ml_best_f1) / ml_best_f1 * 100
            print(f"\n🎉 LLM outperforms traditional ML by {improvement:.1f}%!")
        elif ml_best_f1 > llm_best_f1:
            improvement = (ml_best_f1 - llm_best_f1) / llm_best_f1 * 100
            print(f"\n📊 Traditional ML outperforms LLM by {improvement:.1f}%")
        else:
            print(f"\n⚖️  LLM and traditional ML perform equally well")
    
    # Analyze prediction patterns
    print("\n--- PREDICTION PATTERNS ---")
    
    # LLM prediction distribution
    llm_self_yes = sum(llm_self_predictions)
    llm_observer_yes = sum(llm_observer_predictions)
    ground_truth_yes = sum(ground_truth)
    
    print(f"Ground Truth Matches:     {ground_truth_yes}/{len(ground_truth)} ({ground_truth_yes/len(ground_truth)*100:.1f}%)")
    print(f"LLM Self-Eval Predicts:   {llm_self_yes}/{len(llm_self_predictions)} ({llm_self_yes/len(llm_self_predictions)*100:.1f}%)")
    print(f"LLM Observer Predicts:    {llm_observer_yes}/{len(llm_observer_predictions)} ({llm_observer_yes/len(llm_observer_predictions)*100:.1f}%)")
    
    # Bias analysis
    self_bias = llm_self_yes / len(llm_self_predictions) - ground_truth_yes / len(ground_truth)
    observer_bias = llm_observer_yes / len(llm_observer_predictions) - ground_truth_yes / len(ground_truth)
    
    print(f"\nPrediction Bias:")
    print(f"  Self-Eval:  {self_bias:+.1%} ({'pessimistic' if self_bias < 0 else 'optimistic'})")
    print(f"  Observer:   {observer_bias:+.1%} ({'pessimistic' if observer_bias < 0 else 'optimistic'})")
    
    # Save comparison report
    report = {
        'llm_results': {
            'self_evaluation': llm_self_metrics,
            'observer': llm_observer_metrics
        },
        'comparison_table': comparison_df.to_dict('records'),
        'key_findings': {
            'best_accuracy_model': best_accuracy['model'],
            'best_accuracy_score': float(best_accuracy['accuracy']),
            'best_f1_model': best_f1['model'],
            'best_f1_score': float(best_f1['f1']),
        },
        'prediction_patterns': {
            'ground_truth_match_rate': ground_truth_yes / len(ground_truth),
            'llm_self_match_rate': llm_self_yes / len(llm_self_predictions),
            'llm_observer_match_rate': llm_observer_yes / len(llm_observer_predictions),
            'self_bias': float(self_bias),
            'observer_bias': float(observer_bias)
        }
    }
    
    if os.path.exists(baseline_path):
        report['baseline_results'] = {
            k: v for k, v in baseline_results.items() if k != 'summary'
        }
    
    output_path = os.path.join(output_dir, 'comprehensive_comparison.json')
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✓ Comparison report saved to: {output_path}")
    
    # Generate visualizations
    try:
        create_comparison_plots(comparison_df, output_dir)
        print(f"✓ Comparison plots saved to: {output_dir}/comparison_*.png")
    except Exception as e:
        print(f"\nNote: Could not generate plots: {e}")
    
    print("\n" + "=" * 70)
    print("COMPARISON COMPLETE")
    print("=" * 70)
    
    return report


def create_comparison_plots(comparison_df: pd.DataFrame, output_dir: str):
    """Generate comparison visualizations."""
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 6)
    
    # 1. Metrics comparison bar chart
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    titles = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx // 2, idx % 2]
        
        data = comparison_df[['model', metric]].sort_values(metric, ascending=False)
        
        # Color LLM bars differently
        colors = ['#FF6B6B' if 'LLM' in model else '#4ECDC4' 
                  for model in data['model']]
        
        ax.barh(data['model'], data[metric], color=colors)
        ax.set_xlabel(title)
        ax.set_xlim(0, 1)
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(data[metric]):
            ax.text(v + 0.02, i, f'{v:.3f}', va='center')
    
    plt.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. F1 Score comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    data = comparison_df[['model', 'f1']].sort_values('f1', ascending=True)
    colors = ['#FF6B6B' if 'LLM' in model else '#4ECDC4' 
              for model in data['model']]
    
    bars = ax.barh(data['model'], data['f1'], color=colors)
    ax.set_xlabel('F1 Score', fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_title('F1 Score Comparison: LLM vs Traditional ML', 
                 fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.02, bar.get_y() + bar.get_height()/2,
                f'{width:.3f}', ha='left', va='center')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#FF6B6B', label='LLM-based'),
        Patch(facecolor='#4ECDC4', label='Traditional ML')
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison_f1.png'), dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    import sys
    
    # Change to project root
    if os.path.basename(os.getcwd()) == 'experiments':
        os.chdir('..')
    
    report = compare_all_methods()
