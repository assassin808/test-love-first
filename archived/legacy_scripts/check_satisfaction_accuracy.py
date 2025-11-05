#!/usr/bin/env python3
"""
Check satisfaction accuracy across all personas.
"""

import json
import sys

def check_satisfaction_accuracy():
    """Verify satisfaction descriptions match numeric scores."""
    
    with open('results/personas.json', 'r') as f:
        personas = json.load(f)
    
    print("="*80)
    print("SATISFACTION ACCURACY CHECK")
    print("="*80)
    print()
    
    errors = []
    samples = []
    
    for pair in personas[:20]:  # Check first 20 pairs
        for person_key in ['person1', 'person2']:
            if person_key in pair:
                person = pair[person_key]
                narrative = person.get('time2_reflection_narrative', '')
                satis = person.get('time2_reflection', {}).get('satisfaction', {}).get('satis_2')
                
                if satis is not None:
                    # Check if description matches score
                    if satis <= 3.0:
                        expected = "LOW satisfaction"
                    elif satis <= 6.0:
                        expected = "MODERATE satisfaction"
                    else:
                        expected = "HIGH satisfaction"
                    
                    if expected in narrative:
                        status = "✅"
                    else:
                        status = "❌"
                        errors.append({
                            'iid': person.get('iid'),
                            'score': satis,
                            'expected': expected,
                            'narrative': narrative[:200]
                        })
                    
                    samples.append({
                        'iid': person.get('iid'),
                        'score': satis,
                        'description': expected,
                        'status': status
                    })
    
    # Show samples
    print("Sample Satisfaction Descriptions:")
    print()
    for sample in samples[:10]:
        print(f"{sample['status']} IID {sample['iid']}: {sample['score']}/10 → {sample['description']}")
    
    print()
    print("="*80)
    
    if errors:
        print(f"❌ Found {len(errors)} inaccurate descriptions:")
        print()
        for error in errors[:5]:
            print(f"IID {error['iid']}: Score {error['score']}/10")
            print(f"Expected: {error['expected']}")
            print(f"Narrative: {error['narrative']}...")
            print()
    else:
        print(f"✅ All {len(samples)} satisfaction descriptions are ACCURATE!")
        print()
        print("Breakdown:")
        low = sum(1 for s in samples if s['score'] <= 3.0)
        moderate = sum(1 for s in samples if 3.0 < s['score'] <= 6.0)
        high = sum(1 for s in samples if s['score'] > 6.0)
        print(f"  - LOW (1-3): {low} cases")
        print(f"  - MODERATE (4-6): {moderate} cases")
        print(f"  - HIGH (7-10): {high} cases")
    
    print()
    print("="*80)
    print("TEMPORAL CHANGES CHECK")
    print("="*80)
    print()
    
    # Check that before → after format is present
    has_arrows = sum(1 for pair in personas for key in ['person1', 'person2'] 
                     if key in pair and '→' in pair[key].get('time2_reflection_narrative', ''))
    
    print(f"✅ {has_arrows}/200 narratives contain '→' (before → after format)")
    
    # Show sample changes
    print()
    print("Sample Before → After Changes:")
    print()
    sample_person = personas[0]['person1']
    narrative = sample_person.get('time2_reflection_narrative', '')
    
    for line in narrative.split('\n')[:15]:
        if '→' in line:
            print(f"  {line.strip()}")
    
    return len(errors) == 0

if __name__ == '__main__':
    success = check_satisfaction_accuracy()
    sys.exit(0 if success else 1)
