"""
Comprehensive Evaluation Runner: All Methods Comparison

Runs and compares:
1. LLM Stage 1 (immediate evaluation) - Participant & Observer
2. LLM Stage 2 (with reflection) - Participant & Observer  
3. All Baseline Models (Similarity, Logistic, Random Forest, XGBoost)

Displays results for:
- Like Score Prediction (Pearson r, MAE, RMSE)
- Match Prediction (Accuracy, F1, AUC-ROC, PR-AUC)
"""

import subprocess
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
import sys
import os

# Add experiments directory to path
sys.path.insert(0, str(Path(__file__).parent))

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, mean_absolute_error,
    mean_squared_error
)
from scipy.stats import pearsonr, spearmanr


def run_command(cmd: List[str], description: str):
    """Run a shell command and report status."""
    print(f"\n{'='*70}")
    print(f"Running: {description}")
    print(f"{'='*70}")
    print(f"Command: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        print(f"\n✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {description} failed with error code {e.returncode}")
        return False


def load_results(stage1_path: str, stage2_path: str, baseline_path: str) -> Dict:
    """Load all evaluation results."""
    results = {}
    
    # Load LLM Stage 1
    if Path(stage1_path).exists():
        with open(stage1_path, 'r') as f:
            results['stage1'] = json.load(f)
        print(f"✅ Loaded Stage 1 results from {stage1_path}")
    else:
        print(f"⚠️  Stage 1 results not found: {stage1_path}")
        results['stage1'] = None
    
    # Load LLM Stage 2
    if Path(stage2_path).exists():
        with open(stage2_path, 'r') as f:
            results['stage2'] = json.load(f)
        print(f"✅ Loaded Stage 2 results from {stage2_path}")
    else:
        print(f"⚠️  Stage 2 results not found: {stage2_path}")
        results['stage2'] = None
    
    # Load Baselines
    if Path(baseline_path).exists():
        with open(baseline_path, 'r') as f:
            results['baselines'] = json.load(f)
        print(f"✅ Loaded baseline results from {baseline_path}")
    else:
        print(f"⚠️  Baseline results not found: {baseline_path}")
        results['baselines'] = None
    
    return results


def extract_like_scores_llm(llm_results: Dict, method: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract like scores from LLM results.
    
    Returns:
        pred_person1: Predicted person1 like scores
        pred_person2: Predicted person2 like scores
        ground_truth_person1: Ground truth person1 like scores
        ground_truth_person2: Ground truth person2 like scores
    """
    if method == 'participant':
        scores = llm_results.get('participant_scores', [])
        gt_p1 = np.array(llm_results.get('ground_truth_person1_like', []))
        gt_p2 = np.array(llm_results.get('ground_truth_person2_like', []))
        
        pred_p1 = np.array([s['person1_score'] for s in scores])
        pred_p2 = np.array([s['person2_score'] for s in scores])
        
        return pred_p1, pred_p2, gt_p1, gt_p2
    
    elif method in ['observer', 'advanced_observer']:
        key = 'observer_scores' if method == 'observer' else 'advanced_observer_scores'
        scores = llm_results.get(key, [])
        gt_p1 = np.array(llm_results.get('ground_truth_person1_like', []))
        gt_p2 = np.array(llm_results.get('ground_truth_person2_like', []))
        
        # Observer predicts a single score for the pair
        # We compare it to the average of ground truth likes
        pred_scores = np.array([s['score'] for s in scores])
        
        # Return observer score for both (it's a pair-level prediction)
        return pred_scores, pred_scores, gt_p1, gt_p2
    
    return None, None, None, None


def calculate_like_score_metrics(pred_p1, pred_p2, gt_p1, gt_p2) -> Dict:
    """Calculate like score prediction metrics."""
    # Remove None values
    valid_p1 = ~pd.isna(gt_p1)
    valid_p2 = ~pd.isna(gt_p2)
    
    metrics = {}
    
    if valid_p1.sum() > 0:
        pred_p1_valid = pred_p1[valid_p1]
        gt_p1_valid = gt_p1[valid_p1]
        
        r_p1, p_val_p1 = pearsonr(pred_p1_valid, gt_p1_valid)
        mae_p1 = mean_absolute_error(gt_p1_valid, pred_p1_valid)
        rmse_p1 = np.sqrt(mean_squared_error(gt_p1_valid, pred_p1_valid))
        
        metrics['person1'] = {
            'pearson_r': r_p1,
            'p_value': p_val_p1,
            'mae': mae_p1,
            'rmse': rmse_p1,
            'n': len(gt_p1_valid)
        }
    
    if valid_p2.sum() > 0:
        pred_p2_valid = pred_p2[valid_p2]
        gt_p2_valid = gt_p2[valid_p2]
        
        r_p2, p_val_p2 = pearsonr(pred_p2_valid, gt_p2_valid)
        mae_p2 = mean_absolute_error(gt_p2_valid, pred_p2_valid)
        rmse_p2 = np.sqrt(mean_squared_error(gt_p2_valid, pred_p2_valid))
        
        metrics['person2'] = {
            'pearson_r': r_p2,
            'p_value': p_val_p2,
            'mae': mae_p2,
            'rmse': rmse_p2,
            'n': len(gt_p2_valid)
        }
    
    # Combined metrics
    if valid_p1.sum() > 0 and valid_p2.sum() > 0:
        all_pred = np.concatenate([pred_p1[valid_p1], pred_p2[valid_p2]])
        all_gt = np.concatenate([gt_p1[valid_p1], gt_p2[valid_p2]])
        
        r_combined, p_val_combined = pearsonr(all_pred, all_gt)
        mae_combined = mean_absolute_error(all_gt, all_pred)
        rmse_combined = np.sqrt(mean_squared_error(all_gt, all_pred))
        
        metrics['combined'] = {
            'pearson_r': r_combined,
            'p_value': p_val_combined,
            'mae': mae_combined,
            'rmse': rmse_combined,
            'n': len(all_gt)
        }
    
    return metrics


def extract_match_predictions_llm(llm_results: Dict, method: str, threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract match predictions from LLM results.
    
    Returns:
        predictions: Binary match predictions
        probabilities: Match probability scores
        ground_truth: Ground truth match labels
    """
    ground_truth = np.array(llm_results.get('ground_truth', []))
    
    if method == 'participant':
        scores = llm_results.get('participant_scores', [])
        probs = np.array([s['combined_score'] for s in scores])
        preds = (probs >= threshold).astype(int)
        return preds, probs, ground_truth
    
    elif method == 'observer':
        scores = llm_results.get('observer_scores', [])
        probs = np.array([s['normalized_score'] for s in scores])
        preds = (probs >= threshold).astype(int)
        return preds, probs, ground_truth
    
    elif method == 'advanced_observer':
        scores = llm_results.get('advanced_observer_scores', [])
        probs = np.array([s['normalized_score'] for s in scores])
        preds = (probs >= threshold).astype(int)
        return preds, probs, ground_truth
    
    return None, None, ground_truth


def calculate_match_metrics(preds, probs, ground_truth) -> Dict:
    """Calculate match prediction metrics."""
    metrics = {
        'accuracy': accuracy_score(ground_truth, preds),
        'precision': precision_score(ground_truth, preds, zero_division=0),
        'recall': recall_score(ground_truth, preds, zero_division=0),
        'f1': f1_score(ground_truth, preds, zero_division=0),
    }
    
    # AUC metrics (only if we have both classes)
    if len(np.unique(ground_truth)) > 1:
        try:
            metrics['roc_auc'] = roc_auc_score(ground_truth, probs)
            metrics['pr_auc'] = average_precision_score(ground_truth, probs)
        except:
            metrics['roc_auc'] = None
            metrics['pr_auc'] = None
    else:
        metrics['roc_auc'] = None
        metrics['pr_auc'] = None
    
    return metrics


def print_results_table(all_metrics: Dict):
    """Print comprehensive results table."""
    
    print("\n" + "="*100)
    print("COMPREHENSIVE EVALUATION RESULTS - ALL METHODS")
    print("="*100)
    
    # Like Score Prediction Results
    print("\n" + "="*100)
    print("1. LIKE SCORE PREDICTION (Pearson r, MAE, RMSE)")
    print("="*100)
    
    like_data = []
    for method_name, metrics in all_metrics.items():
        if 'like_scores' in metrics and metrics['like_scores']:
            like_metrics = metrics['like_scores']
            if 'combined' in like_metrics:
                m = like_metrics['combined']
                like_data.append({
                    'Method': method_name,
                    'Pearson r': f"{m['pearson_r']:.3f}",
                    'p-value': f"{m['p_value']:.4f}",
                    'MAE': f"{m['mae']:.3f}",
                    'RMSE': f"{m['rmse']:.3f}",
                    'N': m['n']
                })
    
    if like_data:
        df_like = pd.DataFrame(like_data)
        print("\n" + df_like.to_string(index=False))
    else:
        print("\n⚠️  No like score prediction results available")
    
    # Match Prediction Results
    print("\n" + "="*100)
    print("2. MATCH PREDICTION (Accuracy, Precision, Recall, F1, AUC)")
    print("="*100)
    
    match_data = []
    for method_name, metrics in all_metrics.items():
        if 'match_prediction' in metrics and metrics['match_prediction']:
            m = metrics['match_prediction']
            match_data.append({
                'Method': method_name,
                'Accuracy': f"{m['accuracy']:.3f}",
                'Precision': f"{m['precision']:.3f}",
                'Recall': f"{m['recall']:.3f}",
                'F1': f"{m['f1']:.3f}",
                'ROC-AUC': f"{m['roc_auc']:.3f}" if m['roc_auc'] is not None else 'N/A',
                'PR-AUC': f"{m['pr_auc']:.3f}" if m['pr_auc'] is not None else 'N/A'
            })
    
    if match_data:
        df_match = pd.DataFrame(match_data)
        print("\n" + df_match.to_string(index=False))
    else:
        print("\n⚠️  No match prediction results available")
    
    # Best Methods Summary
    print("\n" + "="*100)
    print("3. BEST METHODS SUMMARY")
    print("="*100)
    
    if like_data:
        df_like_numeric = pd.DataFrame(like_data)
        df_like_numeric['Pearson r'] = df_like_numeric['Pearson r'].astype(float)
        df_like_numeric['MAE'] = df_like_numeric['MAE'].astype(float)
        
        best_pearson = df_like_numeric.loc[df_like_numeric['Pearson r'].idxmax()]
        best_mae = df_like_numeric.loc[df_like_numeric['MAE'].idxmin()]
        
        print(f"\n📊 LIKE SCORE PREDICTION:")
        print(f"   Best Pearson r: {best_pearson['Method']} (r={best_pearson['Pearson r']:.3f})")
        print(f"   Best MAE: {best_mae['Method']} (MAE={best_mae['MAE']:.3f})")
    
    if match_data:
        df_match_numeric = pd.DataFrame(match_data)
        df_match_numeric['F1'] = df_match_numeric['F1'].astype(float)
        df_match_numeric['ROC-AUC'] = pd.to_numeric(df_match_numeric['ROC-AUC'], errors='coerce')
        
        best_f1 = df_match_numeric.loc[df_match_numeric['F1'].idxmax()]
        
        # Best AUC (excluding None values)
        valid_auc = df_match_numeric[df_match_numeric['ROC-AUC'].notna()]
        if len(valid_auc) > 0:
            best_auc = valid_auc.loc[valid_auc['ROC-AUC'].idxmax()]
            print(f"\n🎯 MATCH PREDICTION:")
            print(f"   Best F1: {best_f1['Method']} (F1={best_f1['F1']:.3f})")
            print(f"   Best ROC-AUC: {best_auc['Method']} (AUC={best_auc['ROC-AUC']:.3f})")
        else:
            print(f"\n🎯 MATCH PREDICTION:")
            print(f"   Best F1: {best_f1['Method']} (F1={best_f1['F1']:.3f})")


def main():
    """Main execution function."""
    print("="*100)
    print("COMPREHENSIVE EVALUATION: ALL METHODS")
    print("="*100)
    print("\nThis script will run and compare:")
    print("1. LLM Stage 1 (Participant & Observer)")
    print("2. LLM Stage 2 (Participant & Observer with Reflection)")
    print("3. Baseline Models (Similarity, Logistic Regression, Random Forest, XGBoost)")
    print("\nMetrics:")
    print("- Like Score Prediction: Pearson r, MAE, RMSE")
    print("- Match Prediction: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC")
    
    # Define paths
    base_dir = Path(__file__).parent.parent
    venv_python = base_dir.parent / '.venv' / 'bin' / 'python'
    test_dir = base_dir
    
    conversations_path = test_dir / 'results' / 'conversations.json'
    personas_path = test_dir / 'results' / 'personas.json'
    icl_path = test_dir / 'results' / 'icl_examples.json'
    
    stage1_output = test_dir / 'results' / 'llm_score_evaluation_stage1.json'
    stage2_output = test_dir / 'results' / 'llm_score_evaluation_stage2.json'
    baseline_output = test_dir / 'results' / 'baseline_comparison_v2.json'
    
    # Check if required files exist
    if not conversations_path.exists():
        print(f"\n❌ ERROR: conversations.json not found at {conversations_path}")
        print("Please run the speed dating simulator first!")
        return
    
    if not personas_path.exists():
        print(f"\n❌ ERROR: personas.json not found at {personas_path}")
        print("Please generate personas first!")
        return
    
    # Ask user what to run
    print("\n" + "="*100)
    print("EXECUTION OPTIONS")
    print("="*100)
    print("\n1. Run Stage 1 LLM evaluation")
    print("2. Run Stage 2 LLM evaluation")
    print("3. Run baseline models")
    print("4. Run ALL evaluations (1+2+3)")
    print("5. Skip execution and just display existing results")
    
    choice = input("\nYour choice (1-5): ").strip()
    
    # Execute based on choice
    if choice == '1' or choice == '4':
        cmd = [
            str(venv_python),
            str(test_dir / 'experiments' / 'llm_score_evaluator.py'),
            '--conversations', str(conversations_path),
            '--output-dir', str(test_dir / 'results'),
            '--stage', '1',
            '--icl-examples', str(icl_path),
            '--max-pair-workers', '10',
            '--method', 'both'
        ]
        run_command(cmd, "LLM Stage 1 Evaluation")
    
    if choice == '2' or choice == '4':
        cmd = [
            str(venv_python),
            str(test_dir / 'experiments' / 'llm_score_evaluator.py'),
            '--conversations', str(conversations_path),
            '--output-dir', str(test_dir / 'results'),
            '--stage', '2',
            '--icl-examples', str(icl_path),
            '--max-pair-workers', '10',
            '--method', 'both'
        ]
        run_command(cmd, "LLM Stage 2 Evaluation")
    
    if choice == '3' or choice == '4':
        cmd = [
            str(venv_python),
            str(test_dir / 'experiments' / 'baseline_models_v2.py'),
            '--personas', str(personas_path),
            '--output-dir', str(test_dir / 'results')
        ]
        run_command(cmd, "Baseline Models Evaluation")
    
    # Load all results
    print("\n" + "="*100)
    print("LOADING RESULTS")
    print("="*100)
    
    results = load_results(
        str(stage1_output),
        str(stage2_output),
        str(baseline_output)
    )
    
    # Calculate metrics for all methods
    all_metrics = {}
    
    # LLM Stage 1
    if results['stage1']:
        print("\nProcessing Stage 1 results...")
        
        for method in ['participant', 'observer', 'advanced_observer']:
            pred_p1, pred_p2, gt_p1, gt_p2 = extract_like_scores_llm(results['stage1'], method)
            
            if pred_p1 is not None:
                like_metrics = calculate_like_score_metrics(pred_p1, pred_p2, gt_p1, gt_p2)
                preds, probs, gt_match = extract_match_predictions_llm(results['stage1'], method)
                match_metrics = calculate_match_metrics(preds, probs, gt_match)
                
                method_name = f"LLM Stage 1 - {method.replace('_', ' ').title()}"
                all_metrics[method_name] = {
                    'like_scores': like_metrics,
                    'match_prediction': match_metrics
                }
    
    # LLM Stage 2
    if results['stage2']:
        print("Processing Stage 2 results...")
        
        for method in ['participant', 'observer', 'advanced_observer']:
            pred_p1, pred_p2, gt_p1, gt_p2 = extract_like_scores_llm(results['stage2'], method)
            
            if pred_p1 is not None:
                like_metrics = calculate_like_score_metrics(pred_p1, pred_p2, gt_p1, gt_p2)
                preds, probs, gt_match = extract_match_predictions_llm(results['stage2'], method)
                match_metrics = calculate_match_metrics(preds, probs, gt_match)
                
                method_name = f"LLM Stage 2 - {method.replace('_', ' ').title()}"
                all_metrics[method_name] = {
                    'like_scores': like_metrics,
                    'match_prediction': match_metrics
                }
    
    # Baselines
    if results['baselines']:
        print("Processing baseline results...")
        
        baseline_methods = results['baselines'].get('methods', {})
        for method_name, method_results in baseline_methods.items():
            # Baseline models typically only do match prediction
            # Extract metrics if available
            if 'test_metrics' in method_results:
                test_metrics = method_results['test_metrics']
                all_metrics[f"Baseline - {method_name}"] = {
                    'like_scores': None,  # Baselines don't predict like scores
                    'match_prediction': {
                        'accuracy': test_metrics.get('accuracy', 0),
                        'precision': test_metrics.get('precision', 0),
                        'recall': test_metrics.get('recall', 0),
                        'f1': test_metrics.get('f1', 0),
                        'roc_auc': test_metrics.get('auc_roc', None),
                        'pr_auc': test_metrics.get('auc_pr', None)
                    }
                }
    
    # Display results
    print_results_table(all_metrics)
    
    # Save summary
    summary_path = test_dir / 'results' / 'comprehensive_evaluation_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n💾 Saved comprehensive summary to: {summary_path}")
    
    print("\n" + "="*100)
    print("EVALUATION COMPLETE!")
    print("="*100)


if __name__ == '__main__':
    main()
