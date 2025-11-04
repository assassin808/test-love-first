"""
Analysis Script - Calculate Accuracy Metrics

Compares:
1. Self-evaluation (both participants say YES) vs Ground Truth
2. Observer prediction (MATCH/NO_MATCH) vs Ground Truth
3. Individual decision accuracy (Person 1 & Person 2)
"""

import json
from pathlib import Path
from typing import Dict, List


def load_conversations(filepath: str = "results/conversations.json") -> List[Dict]:
    """Load conversation data"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def calculate_metrics(conversations: List[Dict]) -> Dict:
    """Calculate accuracy metrics"""
    
    total = len(conversations)
    
    # Initialize counters
    metrics = {
        'total_pairs': total,
        'ground_truth_matches': 0,
        'ground_truth_no_matches': 0,
        
        # Self-evaluation metrics
        'self_eval_correct': 0,
        'self_eval_incorrect': 0,
        'self_eval_true_positive': 0,
        'self_eval_true_negative': 0,
        'self_eval_false_positive': 0,
        'self_eval_false_negative': 0,
        
        # Observer metrics
        'observer_correct': 0,
        'observer_incorrect': 0,
        'observer_true_positive': 0,
        'observer_true_negative': 0,
        'observer_false_positive': 0,
        'observer_false_negative': 0,
        
        # Individual person accuracy
        'person1_correct': 0,
        'person2_correct': 0,
        
        # Decision patterns
        'both_yes_ground_yes': 0,
        'both_yes_ground_no': 0,
        'both_no_ground_yes': 0,
        'both_no_ground_no': 0,
        'mixed_decisions': 0,
    }
    
    for conv in conversations:
        # Ground truth
        gt_match = conv['ground_truth']['match'] == 1
        if gt_match:
            metrics['ground_truth_matches'] += 1
        else:
            metrics['ground_truth_no_matches'] += 1
        
        # Get evaluations
        evals = conv.get('evaluations', {})
        
        # Self-evaluation (both must say YES for predicted match)
        p1_decision = evals.get('person1_self_evaluation', {}).get('decision', 'UNKNOWN')
        p2_decision = evals.get('person2_self_evaluation', {}).get('decision', 'UNKNOWN')
        
        self_eval_match = (p1_decision == 'YES' and p2_decision == 'YES')
        
        # Check self-evaluation accuracy
        if self_eval_match == gt_match:
            metrics['self_eval_correct'] += 1
            if gt_match:
                metrics['self_eval_true_positive'] += 1
            else:
                metrics['self_eval_true_negative'] += 1
        else:
            metrics['self_eval_incorrect'] += 1
            if self_eval_match and not gt_match:
                metrics['self_eval_false_positive'] += 1
            elif not self_eval_match and gt_match:
                metrics['self_eval_false_negative'] += 1
        
        # Observer evaluation
        observer_pred = evals.get('observer_evaluation', {}).get('prediction', 'UNKNOWN')
        observer_match = (observer_pred == 'MATCH')
        
        if observer_match == gt_match:
            metrics['observer_correct'] += 1
            if gt_match:
                metrics['observer_true_positive'] += 1
            else:
                metrics['observer_true_negative'] += 1
        else:
            metrics['observer_incorrect'] += 1
            if observer_match and not gt_match:
                metrics['observer_false_positive'] += 1
            elif not observer_match and gt_match:
                metrics['observer_false_negative'] += 1
        
        # Individual person accuracy
        p1_gt_decision = conv['ground_truth']['person1_dec'] == 1
        p2_gt_decision = conv['ground_truth']['person2_dec'] == 1
        
        if (p1_decision == 'YES') == p1_gt_decision:
            metrics['person1_correct'] += 1
        if (p2_decision == 'YES') == p2_gt_decision:
            metrics['person2_correct'] += 1
        
        # Decision patterns
        if p1_decision == 'YES' and p2_decision == 'YES':
            if gt_match:
                metrics['both_yes_ground_yes'] += 1
            else:
                metrics['both_yes_ground_no'] += 1
        elif p1_decision == 'NO' and p2_decision == 'NO':
            if gt_match:
                metrics['both_no_ground_yes'] += 1
            else:
                metrics['both_no_ground_no'] += 1
        else:
            metrics['mixed_decisions'] += 1
    
    # Calculate percentages
    metrics['self_eval_accuracy'] = (metrics['self_eval_correct'] / total * 100) if total > 0 else 0
    metrics['observer_accuracy'] = (metrics['observer_correct'] / total * 100) if total > 0 else 0
    metrics['person1_accuracy'] = (metrics['person1_correct'] / total * 100) if total > 0 else 0
    metrics['person2_accuracy'] = (metrics['person2_correct'] / total * 100) if total > 0 else 0
    
    # Precision and Recall for self-evaluation
    if (metrics['self_eval_true_positive'] + metrics['self_eval_false_positive']) > 0:
        metrics['self_eval_precision'] = metrics['self_eval_true_positive'] / (
            metrics['self_eval_true_positive'] + metrics['self_eval_false_positive']
        ) * 100
    else:
        metrics['self_eval_precision'] = 0
    
    if (metrics['self_eval_true_positive'] + metrics['self_eval_false_negative']) > 0:
        metrics['self_eval_recall'] = metrics['self_eval_true_positive'] / (
            metrics['self_eval_true_positive'] + metrics['self_eval_false_negative']
        ) * 100
    else:
        metrics['self_eval_recall'] = 0
    
    if (metrics['self_eval_precision'] + metrics['self_eval_recall']) > 0:
        metrics['self_eval_f1'] = 2 * (metrics['self_eval_precision'] * metrics['self_eval_recall']) / (
            metrics['self_eval_precision'] + metrics['self_eval_recall']
        )
    else:
        metrics['self_eval_f1'] = 0
    
    # Precision and Recall for observer
    if (metrics['observer_true_positive'] + metrics['observer_false_positive']) > 0:
        metrics['observer_precision'] = metrics['observer_true_positive'] / (
            metrics['observer_true_positive'] + metrics['observer_false_positive']
        ) * 100
    else:
        metrics['observer_precision'] = 0
    
    if (metrics['observer_true_positive'] + metrics['observer_false_negative']) > 0:
        metrics['observer_recall'] = metrics['observer_true_positive'] / (
            metrics['observer_true_positive'] + metrics['observer_false_negative']
        ) * 100
    else:
        metrics['observer_recall'] = 0
    
    if (metrics['observer_precision'] + metrics['observer_recall']) > 0:
        metrics['observer_f1'] = 2 * (metrics['observer_precision'] * metrics['observer_recall']) / (
            metrics['observer_precision'] + metrics['observer_recall']
        )
    else:
        metrics['observer_f1'] = 0
    
    return metrics


def print_report(metrics: Dict):
    """Print comprehensive analysis report"""
    
    print("="*80)
    print("🎯 SPEED DATING SIMULATION - ACCURACY ANALYSIS")
    print("="*80)
    
    print(f"\n📊 DATASET SUMMARY:")
    print(f"   Total pairs analyzed: {metrics['total_pairs']}")
    print(f"   Ground truth MATCHES: {metrics['ground_truth_matches']} ({metrics['ground_truth_matches']/metrics['total_pairs']*100:.1f}%)")
    print(f"   Ground truth NO MATCHES: {metrics['ground_truth_no_matches']} ({metrics['ground_truth_no_matches']/metrics['total_pairs']*100:.1f}%)")
    
    print(f"\n{'='*80}")
    print(f"🤖 SELF-EVALUATION ACCURACY (Both Participants Must Say YES)")
    print(f"{'='*80}")
    print(f"   Overall Accuracy: {metrics['self_eval_accuracy']:.2f}%")
    print(f"   Correct predictions: {metrics['self_eval_correct']}/{metrics['total_pairs']}")
    print(f"   Incorrect predictions: {metrics['self_eval_incorrect']}/{metrics['total_pairs']}")
    print(f"\n   Confusion Matrix:")
    print(f"      True Positives (predicted MATCH, actual MATCH): {metrics['self_eval_true_positive']}")
    print(f"      True Negatives (predicted NO, actual NO): {metrics['self_eval_true_negative']}")
    print(f"      False Positives (predicted MATCH, actual NO): {metrics['self_eval_false_positive']}")
    print(f"      False Negatives (predicted NO, actual MATCH): {metrics['self_eval_false_negative']}")
    print(f"\n   Performance Metrics:")
    print(f"      Precision: {metrics['self_eval_precision']:.2f}%")
    print(f"      Recall: {metrics['self_eval_recall']:.2f}%")
    print(f"      F1 Score: {metrics['self_eval_f1']:.2f}")
    
    print(f"\n{'='*80}")
    print(f"👁️ OBSERVER (恋爱观察员) ACCURACY")
    print(f"{'='*80}")
    print(f"   Overall Accuracy: {metrics['observer_accuracy']:.2f}%")
    print(f"   Correct predictions: {metrics['observer_correct']}/{metrics['total_pairs']}")
    print(f"   Incorrect predictions: {metrics['observer_incorrect']}/{metrics['total_pairs']}")
    print(f"\n   Confusion Matrix:")
    print(f"      True Positives (predicted MATCH, actual MATCH): {metrics['observer_true_positive']}")
    print(f"      True Negatives (predicted NO_MATCH, actual NO): {metrics['observer_true_negative']}")
    print(f"      False Positives (predicted MATCH, actual NO): {metrics['observer_false_positive']}")
    print(f"      False Negatives (predicted NO_MATCH, actual MATCH): {metrics['observer_false_negative']}")
    print(f"\n   Performance Metrics:")
    print(f"      Precision: {metrics['observer_precision']:.2f}%")
    print(f"      Recall: {metrics['observer_recall']:.2f}%")
    print(f"      F1 Score: {metrics['observer_f1']:.2f}")
    
    print(f"\n{'='*80}")
    print(f"👤 INDIVIDUAL PARTICIPANT ACCURACY")
    print(f"{'='*80}")
    print(f"   Person 1 accuracy: {metrics['person1_accuracy']:.2f}%")
    print(f"   Person 2 accuracy: {metrics['person2_accuracy']:.2f}%")
    print(f"   Average individual accuracy: {(metrics['person1_accuracy'] + metrics['person2_accuracy'])/2:.2f}%")
    
    print(f"\n{'='*80}")
    print(f"🔍 DECISION PATTERNS")
    print(f"{'='*80}")
    print(f"   Both YES + Ground Truth MATCH: {metrics['both_yes_ground_yes']}")
    print(f"   Both YES + Ground Truth NO: {metrics['both_yes_ground_no']}")
    print(f"   Both NO + Ground Truth MATCH: {metrics['both_no_ground_yes']}")
    print(f"   Both NO + Ground Truth NO: {metrics['both_no_ground_no']}")
    print(f"   Mixed decisions (one YES, one NO): {metrics['mixed_decisions']}")
    
    print(f"\n{'='*80}")
    print(f"📈 KEY FINDINGS")
    print(f"{'='*80}")
    
    # Compare self-eval vs observer
    if metrics['self_eval_accuracy'] > metrics['observer_accuracy']:
        diff = metrics['self_eval_accuracy'] - metrics['observer_accuracy']
        print(f"   ✅ Self-evaluation is MORE accurate than Observer by {diff:.2f}%")
    elif metrics['observer_accuracy'] > metrics['self_eval_accuracy']:
        diff = metrics['observer_accuracy'] - metrics['self_eval_accuracy']
        print(f"   ✅ Observer is MORE accurate than Self-evaluation by {diff:.2f}%")
    else:
        print(f"   ⚖️ Self-evaluation and Observer have EQUAL accuracy")
    
    # Bias analysis
    total_predictions = metrics['total_pairs']
    self_yes_rate = (metrics['both_yes_ground_yes'] + metrics['both_yes_ground_no']) / total_predictions * 100
    observer_yes_rate = (metrics['observer_true_positive'] + metrics['observer_false_positive']) / total_predictions * 100
    actual_yes_rate = metrics['ground_truth_matches'] / total_predictions * 100
    
    print(f"\n   Prediction Bias:")
    print(f"      Self-eval YES rate: {self_yes_rate:.1f}% (actual: {actual_yes_rate:.1f}%)")
    print(f"      Observer MATCH rate: {observer_yes_rate:.1f}% (actual: {actual_yes_rate:.1f}%)")
    
    if self_yes_rate > actual_yes_rate + 10:
        print(f"      ⚠️ Self-evaluation is OPTIMISTIC (over-predicting matches)")
    elif self_yes_rate < actual_yes_rate - 10:
        print(f"      ⚠️ Self-evaluation is PESSIMISTIC (under-predicting matches)")
    
    if observer_yes_rate > actual_yes_rate + 10:
        print(f"      ⚠️ Observer is OPTIMISTIC (over-predicting matches)")
    elif observer_yes_rate < actual_yes_rate - 10:
        print(f"      ⚠️ Observer is PESSIMISTIC (under-predicting matches)")
    
    print(f"\n{'='*80}")


def save_report(metrics: Dict, output_path: str = "results/accuracy_report.json"):
    """Save metrics to JSON file"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    print(f"💾 Detailed metrics saved to {output_path}")


def main():
    """Main analysis function"""
    print("\n🔬 Loading conversation data...")
    
    conversations = load_conversations()
    
    print(f"✅ Loaded {len(conversations)} conversations")
    
    print("\n📊 Calculating accuracy metrics...")
    metrics = calculate_metrics(conversations)
    
    print_report(metrics)
    
    save_report(metrics)
    
    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()
