import json

# Load personas
with open('results/personas.json', 'r') as f:
    personas = json.load(f)

print('='*80)
print('ANALYZING LOW SATISFACTION NARRATIVES (satis_2 <= 4)')
print('='*80)

low_satis_examples = []
for pair in personas[:100]:
    for person_key in ['person1', 'person2']:
        if person_key in pair:
            person = pair[person_key]
            satis = person.get('time2_reflection', {}).get('satisfaction', {}).get('satis_2')
            if satis and satis <= 4:
                narrative = person.get('time2_reflection_narrative', '')
                first_sent = narrative.split('.')[0] if narrative else ''
                low_satis_examples.append({
                    'score': satis,
                    'narrative': first_sent,
                    'iid': person.get('iid')
                })

# Show examples
print(f"\nFound {len(low_satis_examples)} low satisfaction cases\n")
for ex in sorted(low_satis_examples, key=lambda x: x['score'])[:15]:
    print(f"Score: {ex['score']}/10 (IID {ex['iid']})")
    print(f"Narrative: {ex['narrative']}...")
    
    # Check if narrative properly reflects low satisfaction
    narrative_lower = ex['narrative'].lower()
    if any(word in narrative_lower for word in ['quite satisfied', 'very satisfied', 'high', 'great', 'excellent']):
        print("❌ PROBLEM: Narrative sounds TOO POSITIVE for low score!")
    elif 'moderate' in narrative_lower and ex['score'] < 4:
        print("⚠️  WARNING: 'Moderate' seems too neutral for score < 4")
    else:
        print("✅ OK: Narrative matches low satisfaction")
    print()
