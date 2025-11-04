"""
Create In-Context Learning Examples for Advanced Observer

This script selects 5 pairs OUTSIDE the 100 test samples to use as
in-context learning examples for the advanced observer method.
"""

import json
import random
from pathlib import Path

def create_icl_examples(
    conversations_path: str = "results/conversations.json",
    personas_path: str = "results/personas.json",
    output_path: str = "results/icl_examples.json",
    num_examples: int = 5
):
    """
    Create ICL examples by selecting pairs outside the test set.
    
    For now, this will select the first 5 pairs from the conversations file
    and format them for ICL. In production, you should select from a separate
    held-out set.
    """
    
    print("=" * 70)
    print("CREATING IN-CONTEXT LEARNING EXAMPLES")
    print("=" * 70)
    
    # Load conversations and personas
    with open(conversations_path, 'r', encoding='utf-8') as f:
        conversations = json.load(f)
    
    with open(personas_path, 'r', encoding='utf-8') as f:
        personas = json.load(f)
    
    # Create person lookup
    person_lookup = {}
    for persona in personas:
        p1 = persona['person1']
        p2 = persona['person2']
        pair_key = (p1['iid'], p2['iid'])
        person_lookup[pair_key] = {
            'person1': p1,
            'person2': p2
        }
    
    print(f"Total conversations available: {len(conversations)}")
    print(f"Selecting {num_examples} examples for ICL...\n")
    
    # Select diverse examples (mix of matches and non-matches)
    matches = [c for c in conversations if c.get('ground_truth', {}).get('match', False)]
    non_matches = [c for c in conversations if not c.get('ground_truth', {}).get('match', True)]
    
    # Try to get balanced examples
    selected = []
    if len(matches) >= num_examples // 2 and len(non_matches) >= num_examples // 2:
        # Balanced selection
        selected = random.sample(matches, num_examples // 2) + random.sample(non_matches, num_examples - num_examples // 2)
    else:
        # Just take first N
        selected = conversations[:num_examples]
    
    icl_examples = []
    
    for conv_data in selected:
        person1_iid = conv_data['person1_iid']
        person2_iid = conv_data['person2_iid']
        pair_key = (person1_iid, person2_iid)
        
        if pair_key not in person_lookup:
            print(f"Warning: No persona data for pair {person1_iid}_{person2_iid}, skipping...")
            continue
        
        person1_data = person_lookup[pair_key]['person1']
        person2_data = person_lookup[pair_key]['person2']
        
        ground_truth_match = conv_data.get('ground_truth', {}).get('match', None)
        
        if ground_truth_match is None:
            continue
        
        # For ICL: Use FULL background, NO chat history
        # This gives the observer context about the personas without showing the conversation
        example = {
            'person1_background': person1_data.get('persona_narrative', ''),
            'person2_background': person2_data.get('persona_narrative', ''),
            'person1_age': person1_data.get('age'),
            'person1_gender': person1_data.get('gender'),
            'person2_age': person2_data.get('age'),
            'person2_gender': person2_data.get('gender'),
            'match': ground_truth_match
        }
        
        icl_examples.append(example)
        print(f"Added example: pair_{person1_iid}_{person2_iid} - {'Match' if ground_truth_match else 'No Match'}")
    
    print(f"\nCreated {len(icl_examples)} ICL examples")
    print(f"Saving to: {output_path}")
    
    # Save to file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(icl_examples, f, indent=2, ensure_ascii=False)
    
    print("✅ Done!")
    
    return icl_examples


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create ICL examples for advanced observer")
    parser.add_argument(
        "--conversations",
        type=str,
        default="results/conversations.json",
        help="Path to conversations.json"
    )
    parser.add_argument(
        "--personas",
        type=str,
        default="results/personas.json",
        help="Path to personas.json"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/icl_examples.json",
        help="Output path for ICL examples"
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=5,
        help="Number of ICL examples to create"
    )
    
    args = parser.parse_args()
    
    create_icl_examples(
        conversations_path=args.conversations,
        personas_path=args.personas,
        output_path=args.output,
        num_examples=args.num_examples
    )
