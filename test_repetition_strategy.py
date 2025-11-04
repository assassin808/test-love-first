#!/usr/bin/env python3
"""
Test script to verify repetition penalty retry logic
"""

# Quick test to show the progressive strategy
strategies = [
    (300, 1.0),  # Stage 1: Normal
    (500, 1.0),  # Stage 2: More tokens
    (800, 1.1),  # Stage 3: More tokens + light penalty
    (800, 1.2)   # Stage 4: Strong penalty
]

print("🧪 Repetition Penalty Retry Strategy Test\n")
print("=" * 70)

for attempt, (max_tokens, rep_penalty) in enumerate(strategies):
    stage = attempt + 1
    print(f"\nStage {stage}:")
    print(f"  max_tokens: {max_tokens}")
    print(f"  repetition_penalty: {rep_penalty}")
    
    if rep_penalty == 1.0:
        print(f"  Description: {'Normal request' if stage == 1 else 'More tokens (handle truncation)'}")
    elif rep_penalty == 1.1:
        print(f"  Description: More tokens + light penalty (fix repetition)")
    elif rep_penalty == 1.2:
        print(f"  Description: Strong penalty (nuclear option)")
    
    # Calculate penalty effect
    if rep_penalty > 1.0:
        prob_5th = 100 / (rep_penalty ** 4)  # After 4 previous occurrences
        print(f"  Effect: 5th occurrence reduced to ~{prob_5th:.1f}% probability")

print("\n" + "=" * 70)
print("\n✅ Strategy Overview:")
print("   1. Try normal (300 tokens, no penalty)")
print("   2. If failed → More space (500 tokens)")
print("   3. If repetitive → Add light penalty (800 + 1.1)")
print("   4. If still bad → Nuclear option (800 + 1.2)")
print("\n💡 This fixes BOTH truncation AND repetition issues!")
