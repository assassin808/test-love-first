"""
Feature Encoder - Convert Structured Data to Natural Language
Using Gemini 2.5 Flash for better rate limits and performance

Purpose:
- Convert numeric/categorical dating profile data → coherent natural language
- Remove ALL numbers, convert to qualitative descriptions
- Preserve ALL information from original data
- Create flowing narrative paragraphs (not bullet points)
"""

import json
import asyncio
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
from tqdm.asyncio import tqdm_asyncio
from openai import AsyncOpenAI
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize OpenRouter client
client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
)


def create_encoding_prompt(structured_data: Dict) -> str:
    """
    Create prompt for encoding structured persona data into natural language.
    Uses qualitative descriptors instead of numbers.
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
    
    serializable_data = convert_types(structured_data)
    
    prompt = f"""You are helping someone describe their personality and preferences in a natural, authentic way. Convert the following structured profile data into a flowing narrative paragraph.

CRITICAL RULES:
1. NEVER include ANY numbers (no "7/10", "5 out of 10", etc.)
2. Convert numeric scales to qualitative words:
   - 1-2 → "minimal" / "slight" / "barely"
   - 3-4 → "somewhat" / "moderate" / "mild"
   - 5-6 → "fairly" / "reasonably" / "average"
   - 7-8 → "quite" / "strong" / "considerable"
   - 9-10 → "very" / "extremely" / "highly"
3. Write as a flowing paragraph (NO bullet points)
4. Use natural, conversational language
5. Make it sound like a person describing themselves authentically

Structured Profile Data:
{json.dumps(serializable_data, indent=2)}

Write the natural language narrative (one coherent paragraph):"""
    
    return prompt


def create_time2_encoding_prompt(time2_data: Dict, partner_name: str) -> str:
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
2. Remove ALL numbers - use qualitative descriptions
3. Write as a coherent paragraph describing how they feel after meeting {partner_name}
4. Natural, reflective tone (not bullet points)
5. Describe perception changes and current feelings

Post-Date Reflection Data:
{json.dumps(serializable_data, indent=2)}

Write the reflection paragraph (describing feelings AFTER the date):"""
    
    return prompt


async def encode_with_gemini(prompt: str, model_name: str = "google/gemini-2.5-flash") -> str:
    """
    Use Gemini (via OpenRouter) to encode structured data into natural language.
    
    Args:
        prompt: The encoding prompt
        model_name: Gemini model to use (default: google/gemini-2.5-flash for better rate limits)
    
    Returns:
        Natural language narrative
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


def extract_person_data(row: pd.Series) -> Dict:
    """
    Extract all relevant features for a person from the dataset.
    """
    # Interest fields
    interest_fields = ['sports', 'tvsports', 'exercise', 'dining', 'museums', 'art', 
                      'hiking', 'gaming', 'clubbing', 'reading', 'tv', 'theater', 
                      'movies', 'concerts', 'music', 'shopping', 'yoga']
    
    # Self-rating fields
    self_rating_fields = ['attractive', 'sincere', 'intelligent', 'fun', 'ambitious']
    
    # Preference fields
    preference_fields = ['attr1_1', 'sinc1_1', 'intel1_1', 'fun1_1', 'amb1_1', 'shar1_1']
    
    person_data = {
        'demographics': {
            'age': int(row['age']) if pd.notna(row['age']) else None,
            'gender': row['gender'],
            'race': row['race'] if pd.notna(row['race']) else 'Not specified',
            'field_of_study': row.get('field', 'Not specified'),
            'career': row.get('career_c', 'Not specified'),
        },
        'interests': {},
        'self_ratings': {},
        'partner_preferences': {}
    }
    
    # Add interests
    for field in interest_fields:
        if field in row and pd.notna(row[field]):
            person_data['interests'][field] = float(row[field])
    
    # Add self-ratings
    for field in self_rating_fields:
        if field in row and pd.notna(row[field]):
            person_data['self_ratings'][field] = float(row[field])
    
    # Add preferences
    for field in preference_fields:
        if field in row and pd.notna(row[field]):
            # Map field names to readable names
            readable_name = field.replace('1_1', '').replace('_', ' ')
            person_data['partner_preferences'][readable_name] = float(row[field])
    
    return person_data


def extract_time2_data(row: pd.Series) -> Dict:
    """
    Extract Time 2 reflection data (post-meeting perceptions).
    """
    time2_fields = {
        'satisfaction': 'satis_2',
        'attractive_rating': 'attr2_1',
        'sincere_rating': 'sinc2_1',
        'intelligent_rating': 'intel2_1',
        'fun_rating': 'fun2_1',
        'ambitious_rating': 'amb2_1',
        'shared_interests': 'shar2_1'
    }
    
    time2_data = {}
    for readable_name, field in time2_fields.items():
        if field in row and pd.notna(row[field]):
            time2_data[readable_name] = float(row[field])
    
    return time2_data


async def encode_persona(person_data: Dict, semaphore: asyncio.Semaphore) -> Optional[str]:
    """
    Encode a single persona to natural language.
    """
    async with semaphore:
        prompt = create_encoding_prompt(person_data)
        narrative = await encode_with_gemini(prompt)
        await asyncio.sleep(0.5)  # Rate limiting
        return narrative


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


async def process_dataset(csv_path: str, output_path: str, max_concurrent: int = 5):
    """
    Process entire dataset and encode all personas.
    
    Args:
        csv_path: Path to Speed Dating Data.csv
        output_path: Path to save encoded narratives
        max_concurrent: Maximum concurrent API calls
    """
    print("🔄 Loading dataset...")
    df = pd.read_csv(csv_path, encoding='latin-1')
    
    # Get unique participants
    unique_iids = df['iid'].unique()
    print(f"✅ Found {len(unique_iids)} unique participants")
    
    # Create semaphore for rate limiting
    semaphore = asyncio.Semaphore(max_concurrent)
    
    # Encode all personas
    print("\n🤖 Encoding personas with Gemini 2.5 Flash...")
    encoded_data = {}
    
    tasks = []
    iid_list = []
    
    for iid in unique_iids:
        # Get first row for this participant (all rows have same demographics)
        person_row = df[df['iid'] == iid].iloc[0]
        person_data = extract_person_data(person_row)
        
        tasks.append(encode_persona(person_data, semaphore))
        iid_list.append(iid)
    
    # Run all encoding tasks with progress bar
    results = await tqdm_asyncio.gather(*tasks, desc="Encoding personas")
    
    # Store results
    for iid, narrative in zip(iid_list, results):
        if narrative:
            encoded_data[str(iid)] = {
                'iid': int(iid),
                'persona_narrative': narrative
            }
    
    print(f"\n✅ Successfully encoded {len(encoded_data)} personas")
    
    # Save to JSON
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(encoded_data, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Saved encoded narratives to: {output_path}")
    
    # Show sample
    print("\n📖 Sample encoded narrative:")
    sample_iid = list(encoded_data.keys())[0]
    print(f"\nIID {sample_iid}:")
    print(encoded_data[sample_iid]['persona_narrative'])
    
    return encoded_data


def main():
    parser = argparse.ArgumentParser(
        description="Encode structured dating profiles to natural language using Gemini"
    )
    parser.add_argument(
        '--input',
        type=str,
        default='Speed Dating Data.csv',
        help='Path to input CSV file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results/encoded_narratives.json',
        help='Path to output JSON file'
    )
    parser.add_argument(
        '--max-concurrent',
        type=int,
        default=5,
        help='Maximum concurrent API calls'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("FEATURE ENCODER - Structured Data → Natural Language")
    print("=" * 70)
    print(f"Model: Gemini 2.0 Flash (via OpenRouter)")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Max concurrent: {args.max_concurrent}")
    print("=" * 70)
    
    # Run encoding
    asyncio.run(process_dataset(args.input, args.output, args.max_concurrent))
    
    print("\n✅ Feature encoding complete!")


if __name__ == '__main__':
    main()
