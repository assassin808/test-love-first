import json
import sys
sys.path.insert(0, '/Users/assassin808/Desktop/research_2025_xuan/yan/test/experiments')
from llm_score_evaluator import get_time2_reflection_context

# Load personas
with open('results/personas.json', 'r') as f:
    personas = json.load(f)

print('='*80)
print('TESTING COMBINED TIME 2 REFLECTION CONTEXT')
print('='*80)

# Test first 3 examples with different satisfaction levels
for i in range(min(3, len(personas))):
    pair = personas[i]
    person1 = pair['person1']
    
    print(f'\n\n📊 EXAMPLE {i+1}: Pair {pair["pair_id"]}, Person 1 (IID {person1["iid"]})')
    
    # Get raw satisfaction score
    satis = person1.get('time2_reflection', {}).get('satisfaction', {}).get('satis_2')
    print(f'\n🔢 Raw satisfaction score: {satis}/10')
    
    # Get combined context
    context = get_time2_reflection_context(person1, "them")
    
    print(f'\n📝 Combined context (narrative + numeric):')
    print('-' * 80)
    print(context)
    print('-' * 80)
    
    # Check if numeric rating is present
    if f'satisfaction rating: {satis}' in context.lower():
        print('✅ Numeric satisfaction rating preserved!')
    else:
        print('⚠️  Numeric rating not found in context')

print('\n' + '='*80)
print('Context now includes BOTH natural language AND precise numeric ratings!')
print('='*80)
