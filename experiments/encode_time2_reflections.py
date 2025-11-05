"""
Encode Time 2 Reflection Data with Gemini

This script reads personas.json and adds Gemini-encoded natural language narratives
for each person's Time 2 reflection data.
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


def create_time2_encoding_prompt(time2_data: Dict, partner_name: str = "them") -> str:
    """
    Create prompt for encoding Time 2 reflection data (post-meeting perception).
    """
    # Convert numpy types to native Python types for JSON serialization
    def convert_types(obj):
        if isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(v) for v in obj]
        elif hasattr(obj, 'item'):  # numpy types have .item() method
            return obj.item()
        else:
            return obj
    
    serializable_data = convert_types(time2_data)
    
    prompt = f"""You are helping someone describe their thoughts after a speed date. Convert the following post-date reflection data into a natural, flowing paragraph.

CRITICAL RULES:
1. Keep ALL information
2. Remove ALL numbers - use qualitative descriptions only
3. Write as a coherent paragraph describing how they feel after meeting {partner_name}
4. Natural, reflective tone (not bullet points)
5. Describe perception changes and current feelings
6. Be specific about satisfaction levels and updated perceptions

Post-Date Reflection Data:
{json.dumps(serializable_data, indent=2)}

Write the reflection paragraph (describing feelings AFTER the date):"""
    
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
            max_tokens=2048,
        )
        
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        print(f"Error encoding with Gemini: {e}")
        return None


async def encode_time2_reflection(time2_data: Dict, partner_name: str, 
                                   semaphore: asyncio.Semaphore) -> Optional[str]:
    """
    Encode Time 2 reflection data to natural language.
    """
    async with semaphore:
        prompt = create_time2_encoding_prompt(time2_data, partner_name)
        reflection = await encode_with_gemini(prompt)
        await asyncio.sleep(0.5)  # Rate limiting
        return reflection


async def process_personas(personas_path: str, output_path: str, max_concurrent: int = 10):
    """
    Process personas.json and add Gemini-encoded Time 2 reflection narratives.
    
    Args:
        personas_path: Path to personas.json
        output_path: Path to save updated personas.json
        max_concurrent: Maximum concurrent API calls
    """
    print("🔄 Loading personas...")
    with open(personas_path, 'r', encoding='utf-8') as f:
        personas_list = json.load(f)
    
    print(f"✅ Loaded {len(personas_list)} persona pairs")
    
    # Create semaphore for rate limiting
    semaphore = asyncio.Semaphore(max_concurrent)
    
    # Collect all Time 2 reflection encoding tasks
    print("\n🤖 Encoding Time 2 reflections with Gemini...")
    tasks = []
    task_metadata = []  # Store (pair_idx, person_key) for each task
    
    for pair_idx, pair in enumerate(personas_list):
        for person_key in ['person1', 'person2']:
            person = pair[person_key]
            time2_data = person.get('time2_reflection', {})
            
            if time2_data:
                # Get partner gender for pronoun
                partner_key = 'person2' if person_key == 'person1' else 'person1'
                partner_gender = pair[partner_key].get('gender', 1)
                partner_name = "her" if partner_gender == 0 else "him"
                
                tasks.append(encode_time2_reflection(time2_data, partner_name, semaphore))
                task_metadata.append((pair_idx, person_key))
    
    print(f"   Total reflections to encode: {len(tasks)}")
    
    # Run all encoding tasks with progress bar
    results = await tqdm_asyncio.gather(*tasks, desc="Encoding Time 2 reflections")
    
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
    print("\n📖 Sample Time 2 reflection narrative:")
    for pair in personas_list[:2]:
        for person_key in ['person1', 'person2']:
            narrative = pair[person_key].get('time2_reflection_narrative')
            if narrative:
                print(f"\nPair: {pair['pair_id']}, {person_key} (IID {pair[person_key]['iid']}):")
                print(narrative)
                break
        break
    
    return personas_list


def main():
    parser = argparse.ArgumentParser(
        description="Encode Time 2 reflection data with Gemini for all personas"
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
    print("TIME 2 REFLECTION ENCODER - Structured Data → Natural Language")
    print("=" * 70)
    print(f"Model: Gemini 2.5 Flash (via OpenRouter)")
    print(f"Input: {args.personas}")
    print(f"Output: {args.output}")
    print(f"Max concurrent: {args.max_concurrent}")
    print("=" * 70)
    
    # Run encoding
    asyncio.run(process_personas(args.personas, args.output, args.max_concurrent))
    
    print("\n✅ Time 2 reflection encoding complete!")


if __name__ == '__main__':
    main()
