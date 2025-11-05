"""
Encode Time 2 Reflection Data with Gemini (Enhanced with Time 1 comparison)

This script reads personas.json and adds Gemini-encoded natural language narratives
for each person's Time 2 reflection data, INCLUDING changes from Time 1.
"""

import json
import asyncio
import argparse
from pathlib import Path
from typing import Dict, Optional
from tqdm.asyncio import tqdm_asyncio
from openai import AsyncOpenAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenRouter client
client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
)


def calculate_changes(time1_data: Dict, time2_data: Dict) -> Dict:
    """
    Calculate changes between Time 1 (before date) and Time 2 (after date).
    """
    changes = {}
    
    # Compare self-ratings
    if 'self_ratings' in time1_data and 'updated_self_ratings' in time2_data:
        changes['self_ratings_change'] = {}
        for trait in time1_data['self_ratings']:
            val1 = time1_data['self_ratings'].get(trait)
            val2 = time2_data['updated_self_ratings'].get(trait)
            if val1 is not None and val2 is not None:
                change = float(val2) - float(val1)
                changes['self_ratings_change'][trait] = {
                    'before': float(val1),
                    'after': float(val2),
                    'change': change
                }
    
    # Compare others' perception
    if 'others_perception' in time1_data and 'updated_others_perception' in time2_data:
        changes['others_perception_change'] = {}
        for trait in time1_data['others_perception']:
            val1 = time1_data['others_perception'].get(trait)
            val2 = time2_data['updated_others_perception'].get(trait)
            if val1 is not None and val2 is not None:
                change = float(val2) - float(val1)
                changes['others_perception_change'][trait] = {
                    'before': float(val1),
                    'after': float(val2),
                    'change': change
                }
    
    # Compare preferences for partner
    if 'preferences_self' in time1_data and 'updated_preferences_self' in time2_data:
        changes['preferences_change'] = {}
        for trait in time1_data['preferences_self']:
            val1 = time1_data['preferences_self'].get(trait)
            val2 = time2_data['updated_preferences_self'].get(trait)
            if val1 is not None and val2 is not None:
                change = float(val2) - float(val1)
                changes['preferences_change'][trait] = {
                    'before': float(val1),
                    'after': float(val2),
                    'change': change
                }
    
    return changes


def create_enhanced_time2_prompt(person_data: Dict, partner_name: str = "them") -> str:
    """
    Create enhanced prompt that includes Time 1 → Time 2 changes.
    """
    # Convert numpy types
    def convert_types(obj):
        if isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(v) for v in obj]
        elif hasattr(obj, 'item'):
            return obj.item()
        else:
            return obj
    
    # Get Time 1 and Time 2 data
    time1_data = person_data.get('pre_event_data', {})
    time2_data = person_data.get('time2_reflection', {})
    
    # Calculate changes
    changes = calculate_changes(time1_data, time2_data)
    
    # Build comprehensive data package
    reflection_package = {
        'time2_reflection': convert_types(time2_data),
        'changes_from_before_date': convert_types(changes)
    }
    
    prompt = f"""You are helping someone describe their thoughts after a speed date. Convert the following post-date reflection data into a natural, flowing narrative paragraph.

CRITICAL RULES:
1. Keep ALL information including CHANGES from before the date
2. Remove ALL numbers - use qualitative descriptions only
3. Write as a coherent paragraph describing:
   - How they feel after meeting {partner_name}
   - How their perceptions CHANGED after the date (very important!)
   - Their updated feelings and preferences
4. Natural, reflective tone (not bullet points)
5. Emphasize what CHANGED vs what stayed the same
6. Be specific about satisfaction levels

IMPORTANT: Pay special attention to the "changes_from_before_date" section - this shows how the date affected their views!

Post-Date Reflection Data (including changes):
{json.dumps(reflection_package, indent=2)}

Write a comprehensive reflection paragraph that captures:
1. Overall satisfaction with the date
2. How their self-perception changed (if at all)
3. How they think others perceive them changed (if at all)
4. How their preferences for a partner changed (if at all)
5. What aspects changed most dramatically

Paragraph:"""
    
    return prompt


async def encode_with_gemini(prompt: str, model_name: str = "google/gemini-2.5-flash") -> str:
    """
    Use Gemini (via OpenRouter) to encode structured data into natural language.
    """
    try:
        response = await client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=3072,  # Increased for longer narratives with changes
        )
        
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        print(f"Error encoding with Gemini: {e}")
        return None


async def encode_time2_with_changes(person_data: Dict, partner_name: str, 
                                     semaphore: asyncio.Semaphore) -> Optional[str]:
    """
    Encode Time 2 reflection data with Time 1 changes to natural language.
    """
    async with semaphore:
        prompt = create_enhanced_time2_prompt(person_data, partner_name)
        reflection = await encode_with_gemini(prompt)
        await asyncio.sleep(0.5)  # Rate limiting
        return reflection


async def process_personas(personas_path: str, output_path: str, max_concurrent: int = 10):
    """
    Process personas.json and add Gemini-encoded Time 2 reflection narratives with changes.
    """
    print("🔄 Loading personas...")
    with open(personas_path, 'r', encoding='utf-8') as f:
        personas_list = json.load(f)
    
    print(f"✅ Loaded {len(personas_list)} persona pairs")
    
    # Create semaphore for rate limiting
    semaphore = asyncio.Semaphore(max_concurrent)
    
    # Collect all Time 2 reflection encoding tasks
    print("\n🤖 Encoding Time 2 reflections with changes (Gemini 2.5 Flash)...")
    tasks = []
    task_metadata = []
    
    for pair_idx, pair in enumerate(personas_list):
        for person_key in ['person1', 'person2']:
            person = pair[person_key]
            time2_data = person.get('time2_reflection', {})
            
            if time2_data:
                # Get partner gender for pronoun
                partner_key = 'person2' if person_key == 'person1' else 'person1'
                partner_gender = pair[partner_key].get('gender', 1)
                partner_name = "her" if partner_gender == 0 else "him"
                
                tasks.append(encode_time2_with_changes(person, partner_name, semaphore))
                task_metadata.append((pair_idx, person_key))
    
    print(f"   Total reflections to encode: {len(tasks)}")
    
    # Run all encoding tasks with progress bar
    results = await tqdm_asyncio.gather(*tasks, desc="Encoding Time 2 reflections with changes")
    
    # Update personas with encoded reflections
    success_count = 0
    for (pair_idx, person_key), narrative in zip(task_metadata, results):
        if narrative:
            personas_list[pair_idx][person_key]['time2_reflection_narrative'] = narrative
            success_count += 1
        else:
            print(f"⚠️ Failed to encode Time 2 reflection for pair {pair_idx}, {person_key}")
    
    print(f"\n✅ Successfully encoded {success_count}/{len(tasks)} Time 2 reflections")
    
    # Save updated personas
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(personas_list, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Saved updated personas to: {output_path}")
    
    # Show sample
    print("\n📖 Sample Time 2 reflection narrative (with changes):")
    for pair in personas_list[:1]:
        person1 = pair['person1']
        narrative = person1.get('time2_reflection_narrative')
        if narrative:
            print(f"\nPair: {pair['pair_id']}, person1 (IID {person1['iid']}):")
            print("-" * 80)
            print(narrative)
            print("-" * 80)
            
            # Show what changed
            time1 = person1.get('pre_event_data', {})
            time2 = person1.get('time2_reflection', {})
            changes = calculate_changes(time1, time2)
            
            print("\n🔄 Actual numeric changes:")
            if 'self_ratings_change' in changes:
                print("\nSelf-ratings changes:")
                for trait, data in changes['self_ratings_change'].items():
                    if abs(data['change']) > 0:
                        direction = "↑" if data['change'] > 0 else "↓"
                        print(f"  {trait}: {data['before']} → {data['after']} ({direction}{abs(data['change'])})")
            break
    
    return personas_list


def main():
    parser = argparse.ArgumentParser(
        description="Encode Time 2 reflection data with changes from Time 1 using Gemini"
    )
    parser.add_argument(
        '--personas',
        type=str,
        default='results/personas.json',
        help='Path to input personas.json file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results/personas.json',
        help='Path to output personas.json file (can be same as input)'
    )
    parser.add_argument(
        '--max-concurrent',
        type=int,
        default=10,
        help='Maximum concurrent API calls'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("TIME 2 REFLECTION ENCODER WITH CHANGES")
    print("Structured Data + Changes → Natural Language")
    print("=" * 70)
    print(f"Model: Gemini 2.5 Flash (via OpenRouter)")
    print(f"Input: {args.personas}")
    print(f"Output: {args.output}")
    print(f"Max concurrent: {args.max_concurrent}")
    print(f"Includes: Time 1 → Time 2 changes analysis")
    print("=" * 70)
    
    # Run encoding
    asyncio.run(process_personas(args.personas, args.output, args.max_concurrent))
    
    print("\n✅ Time 2 reflection encoding with changes complete!")


if __name__ == '__main__':
    main()
