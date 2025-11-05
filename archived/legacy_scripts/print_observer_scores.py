import json

# Load Stage 1 results
with open('results/llm_score_evaluation_stage1.json', 'r') as f:
    results = json.load(f)

print("=" * 80)
print("OBSERVER STAGE 1 SCORES - ALL 100 PAIRS")
print("=" * 80)
print()

# Extract data
observer_scores = results['observer_method']['scores']
ground_truth = results['ground_truth']
pair_ids = results['pair_ids']

print(f"Total pairs: {len(observer_scores)}")
print()

# Build lookup dict
score_dict = {s['pair_id']: s['score'] for s in observer_scores}
gt_dict = {pair_ids[i]: ground_truth[i] for i in range(len(pair_ids))}

# Separate by ground truth
match_data = []
non_match_data = []

for pair_id in pair_ids:
    score = score_dict.get(pair_id)
    is_match = gt_dict.get(pair_id)
    
    if score is not None:
        if is_match:
            match_data.append((pair_id, score))
        else:
            non_match_data.append((pair_id, score))

print(f"Matches: {len(match_data)}")
print(f"Non-matches: {len(non_match_data)}")
print()

# Print match pairs
print("-" * 80)
print("MATCH PAIRS (Ground Truth = Match)")
print("-" * 80)
for i, (pair_id, score) in enumerate(sorted(match_data, key=lambda x: x[1], reverse=True), 1):
    print(f"{i:2d}. {pair_id:15s} | Observer: {score:.1f}/10")

print()

# Print non-match pairs
print("-" * 80)
print("NON-MATCH PAIRS (Ground Truth = No Match)")
print("-" * 80)
for i, (pair_id, score) in enumerate(sorted(non_match_data, key=lambda x: x[1], reverse=True), 1):
    print(f"{i:2d}. {pair_id:15s} | Observer: {score:.1f}/10")

print()
print("=" * 80)
print("SCORE DISTRIBUTION SUMMARY")
print("=" * 80)

# Calculate statistics
all_scores = [s['score'] for s in observer_scores]
match_scores = [score for _, score in match_data]
non_match_scores = [score for _, score in non_match_data]

print(f"\nAll pairs:")
print(f"  Min: {min(all_scores):.1f}, Max: {max(all_scores):.1f}, Mean: {sum(all_scores)/len(all_scores):.2f}")

print(f"\nMatch pairs (should be HIGH):")
print(f"  Min: {min(match_scores):.1f}, Max: {max(match_scores):.1f}, Mean: {sum(match_scores)/len(match_scores):.2f}")

print(f"\nNon-match pairs (should be LOW):")
print(f"  Min: {min(non_match_scores):.1f}, Max: {max(non_match_scores):.1f}, Mean: {sum(non_match_scores)/len(non_match_scores):.2f}")

# Check discrimination
mean_diff = sum(match_scores)/len(match_scores) - sum(non_match_scores)/len(non_match_scores)
print(f"\n⚠️  Mean difference (Match - NonMatch): {mean_diff:.2f}")
if mean_diff > 0.5:
    print("✅ GOOD: Match pairs scored higher than non-match pairs")
elif mean_diff < -0.5:
    print("❌ BAD: Inverse correlation - match pairs scored LOWER than non-match!")
else:
    print("⚠️  WARNING: Very small difference - poor discrimination")

# Check metrics
print()
print("=" * 80)
print("MODEL PERFORMANCE METRICS")
print("=" * 80)
metrics = results['observer_method']['metrics']
print(f"Accuracy:  {metrics['accuracy']:.3f}")
print(f"Precision: {metrics['precision']:.3f}")
print(f"Recall:    {metrics['recall']:.3f}")
print(f"F1 Score:  {metrics['f1']:.3f}")
print(f"AUC-ROC:   {metrics['auc_roc']:.3f}")
print(f"PR-AUC:    {metrics['pr_auc']:.3f}")

print()
