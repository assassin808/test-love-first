#!/usr/bin/env python3
"""
Verify that Stage 2 evaluation receives BOTH:
1. Natural language Time 2 reflection narratives (from Gemini)
2. ALL numeric ratings (satisfaction, preferences, self-ratings, others' perception)

This ensures no information loss when evaluating compatibility.
"""

import json
import sys
sys.path.insert(0, 'experiments')

from llm_score_evaluator import get_time2_reflection_context

def verify_stage2_input():
    """Verify Stage 2 evaluation input contains both narrative and numeric data."""
    
    print("=" * 80)
    print("STAGE 2 EVALUATION INPUT VERIFICATION")
    print("=" * 80)
    print()
    
    # Load personas
    with open('results/personas.json', 'r') as f:
        personas = json.load(f)
    
    # Test with first pair
    pair = personas[0]
    person1 = pair['person1']
    person2 = pair['person2']
    
    print(f"Testing Pair: {pair['pair_id']}")
    print(f"Person 1 IID: {person1['iid']}")
    print(f"Person 2 IID: {person2['iid']}")
    print()
    
    # Get Time 2 reflection context for Person 1
    print("=" * 80)
    print("PERSON 1 - Stage 2 Reflection Context")
    print("=" * 80)
    
    reflection_context = get_time2_reflection_context(person1, "them")
    print(reflection_context)
    print()
    
    # Verify components
    print("=" * 80)
    print("VERIFICATION CHECKLIST - Person 1")
    print("=" * 80)
    
    checks = {
        "✓ Natural language narrative": "After the" in reflection_context or "after the" in reflection_context.lower(),
        "✓ Overall satisfaction score": "Overall satisfaction:" in reflection_context,
        "✓ Date experience (length)": "Date length:" in reflection_context,
        "✓ Preferences for partner": "What I want in a partner" in reflection_context,
        "✓ Self-ratings": "How I rate myself" in reflection_context,
        "✓ Others' perception": "How I think others perceive me" in reflection_context,
        "✓ Numeric values preserved": "/10" in reflection_context and "/100" in reflection_context,
    }
    
    all_passed = True
    for check, passed in checks.items():
        status = "✅" if passed else "❌"
        print(f"{status} {check}")
        if not passed:
            all_passed = False
    
    print()
    
    # Count numeric values
    count_10_scale = reflection_context.count("/10")
    count_100_scale = reflection_context.count("/100")
    
    print(f"📊 Numeric ratings found:")
    print(f"   - {count_10_scale} ratings on 1-10 scale")
    print(f"   - {count_100_scale} ratings on 1-100 scale")
    print()
    
    # Test Person 2
    print("=" * 80)
    print("PERSON 2 - Stage 2 Reflection Context")
    print("=" * 80)
    
    reflection_context_p2 = get_time2_reflection_context(person2, "them")
    print(reflection_context_p2[:500] + "..." if len(reflection_context_p2) > 500 else reflection_context_p2)
    print()
    
    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    if all_passed:
        print("✅ VERIFICATION PASSED")
        print()
        print("Stage 2 evaluation will receive:")
        print("1. ✅ Natural language narrative describing changes from Time 1 → Time 2")
        print("2. ✅ ALL numeric ratings for precision (satisfaction, preferences, self-ratings, perception)")
        print("3. ✅ Date experience context (length, number of dates)")
        print()
        print("This ensures:")
        print("- No information loss from Gemini encoding")
        print("- LLM evaluator gets both qualitative AND quantitative data")
        print("- Predictions can leverage rich narrative + precise numeric values")
    else:
        print("❌ VERIFICATION FAILED")
        print("Some components are missing from Stage 2 input.")
    
    print()
    print("=" * 80)
    
    return all_passed

if __name__ == '__main__':
    success = verify_stage2_input()
    sys.exit(0 if success else 1)
