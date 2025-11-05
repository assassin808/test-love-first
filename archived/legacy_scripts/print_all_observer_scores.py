import json
import os

print("=" * 80)
print("SEARCHING FOR OBSERVER STAGE 1 RESULTS")
print("=" * 80)
print()

# Check for Stage 1 results file
stage1_file = 'results/llm_score_evaluation_stage1.json'

if not os.path.exists(stage1_file):
    print(f"❌ File not found: {stage1_file}")
    print()
    print("Available result files:")
    for f in os.listdir('results'):
        if 'llm_score' in f and f.endswith('.json'):
            print(f"  - results/{f}")
    exit(1)

# Load Stage 1 results
with open(stage1_file, 'r') as f:
    results = json.load(f)

print(f"✅ Loaded: {stage1_file}")
print()

# Extract data
observer_scores = results.get('observer_method', {}).get('scores', [])
ground_truth = results.get('ground_truth', [])
pair_ids = results.get('pair_ids', [])

if not observer_scores:
    print("❌ No observer scores found in the file!")
    print(f"Available keys: {list(results.keys())}")
    exit(1)

print("=" * 80)
print(f"OBSERVER STAGE 1 SCORES - ALL {len(observer_scores)} PAIRS")
print("=" * 80)
print()

# Build lookup dict
score_dict = {s['pair_id']: s for s in observer_scores}
gt_dict = {pair_ids[i]: ground_truth[i] for i in range(len(pair_ids))}

# Separate by ground truth
match_data = []
non_match_data = []

for pair_id in pair_ids:
    score_data = score_dict.get(pair_id)
    is_match = gt_dict.get(pair_id)
    
    if score_data:
        score = score_data.get('score')
        response = score_data.get('response', '')
        response_len = len(response)
        
        if is_match:
            match_data.append((pair_id, score, response_len, response))
        else:
            non_match_data.append((pair_id, score, response_len, response))

print(f"Total pairs: {len(observer_scores)}")
print(f"Matches: {len(match_data)}")
print(f"Non-matches: {len(non_match_data)}")
print()

# Print match pairs
print("-" * 80)
print("MATCH PAIRS (Ground Truth = Match) - SORTED BY SCORE")
print("-" * 80)
for i, (pair_id, score, resp_len, resp) in enumerate(sorted(match_data, key=lambda x: x[1], reverse=True), 1):
    print(f"{i:2d}. {pair_id:15s} | Score: {score:.1f}/10 | Response: {resp_len:4d} chars")
    if i <= 3:  # Show first 3 full responses
        print(f"    First 200 chars: {resp[:200]}...")
print()

# Print non-match pairs
print("-" * 80)
print("NON-MATCH PAIRS (Ground Truth = No Match) - SORTED BY SCORE")
print("-" * 80)
for i, (pair_id, score, resp_len, resp) in enumerate(sorted(non_match_data, key=lambda x: x[1], reverse=True), 1):
    print(f"{i:2d}. {pair_id:15s} | Score: {score:.1f}/10 | Response: {resp_len:4d} chars")
    if i <= 3:  # Show first 3 full responses
        print(f"    First 200 chars: {resp[:200]}...")
print()

# Statistics
print("=" * 80)
print("SCORE DISTRIBUTION SUMMARY")
print("=" * 80)

all_scores = [s['score'] for s in observer_scores]
match_scores = [score for _, score, _, _ in match_data]
non_match_scores = [score for _, score, _, _ in non_match_data]

all_response_lens = [len(s.get('response', '')) for s in observer_scores]
match_response_lens = [resp_len for _, _, resp_len, _ in match_data]
non_match_response_lens = [resp_len for _, _, resp_len, _ in non_match_data]

print(f"\nAll pairs:")
print(f"  Scores:    Min={min(all_scores):.1f}, Max={max(all_scores):.1f}, Mean={sum(all_scores)/len(all_scores):.2f}")
print(f"  Responses: Min={min(all_response_lens)} chars, Max={max(all_response_lens)} chars, Mean={sum(all_response_lens)/len(all_response_lens):.0f} chars")

print(f"\nMatch pairs (should be HIGH):")
print(f"  Scores:    Min={min(match_scores):.1f}, Max={max(match_scores):.1f}, Mean={sum(match_scores)/len(match_scores):.2f}")
print(f"  Responses: Mean={sum(match_response_lens)/len(match_response_lens):.0f} chars")

print(f"\nNon-match pairs (should be LOW):")
print(f"  Scores:    Min={min(non_match_scores):.1f}, Max={max(non_match_scores):.1f}, Mean={sum(non_match_scores)/len(non_match_scores):.2f}")
print(f"  Responses: Mean={sum(non_match_response_lens)/len(non_match_response_lens):.0f} chars")

# Check discrimination
mean_diff = sum(match_scores)/len(match_scores) - sum(non_match_scores)/len(non_match_scores)
print(f"\n⚠️  Mean difference (Match - NonMatch): {mean_diff:.2f}")
if mean_diff > 0.5:
    print("✅ GOOD: Match pairs scored higher than non-match pairs")
elif mean_diff < -0.5:
    print("❌ BAD: Inverse correlation - match pairs scored LOWER than non-match!")
else:
    print("⚠️  WARNING: Very small difference - poor discrimination")

# Check if responses are complete
print()
print("=" * 80)
print("RESPONSE COMPLETENESS CHECK")
print("=" * 80)
short_responses = [s for s in observer_scores if len(s.get('response', '')) < 50]
print(f"Responses < 50 chars (likely truncated): {len(short_responses)}/{len(observer_scores)}")
if short_responses:
    print("\nExamples of short responses:")
    for s in short_responses[:5]:
        print(f"  {s['pair_id']}: '{s.get('response', '')}'")

# Model metrics
print()
print("=" * 80)
print("MODEL PERFORMANCE METRICS")
print("=" * 80)
metrics = results.get('observer_method', {}).get('metrics', {})
if metrics:
    print(f"Accuracy:  {metrics.get('accuracy', 0):.3f}")
    print(f"Precision: {metrics.get('precision', 0):.3f}")
    print(f"Recall:    {metrics.get('recall', 0):.3f}")
    print(f"F1 Score:  {metrics.get('f1', 0):.3f}")
    print(f"AUC-ROC:   {metrics.get('auc_roc', 0):.3f}")
    print(f"PR-AUC:    {metrics.get('pr_auc', 0):.3f}")
else:
    print("No metrics found")

print()
print("=" * 80)
print("SAMPLE FULL RESPONSES")
print("=" * 80)
print("\nShowing 2 sample responses (1 match, 1 non-match):\n")

if match_data:
    pair_id, score, resp_len, resp = match_data[0]
    print(f"MATCH EXAMPLE: {pair_id} (Score: {score:.1f}/10)")
    print("-" * 80)
    print(resp)
    print()

if non_match_data:
    pair_id, score, resp_len, resp = non_match_data[0]
    print(f"NON-MATCH EXAMPLE: {pair_id} (Score: {score:.1f}/10)")
    print("-" * 80)
    print(resp)
    print()

print("=" * 80)
print("✅ COMPLETE OBSERVER SCORES PRINTED")
print("=" * 80)
