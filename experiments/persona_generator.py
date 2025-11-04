"""
Persona Generator - Phase 1

功能:
1. 加载 processed_pairs.json
2. 将数据特征转换为自然语言人物描述（完整编码所有数据，不过滤）
3. 生成《再见爱人》风格的 Persona prompts
4. 为 Mistral Nemo (via OpenRouter API) 提供角色扮演 system prompt

重要: 所有 persona 信息必须完整编码，保留所有原始数据和评分
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

# 字段编码映射
FIELD_CD_MAP = {
    1: "Law", 2: "Math", 3: "Social Science/Psychology",
    4: "Medical Science/Pharmaceuticals", 5: "Engineering",
    6: "English/Creative Writing/Journalism", 7: "History/Religion/Philosophy",
    8: "Business/Economics/Finance", 9: "Education/Academia",
    10: "Biological Sciences/Chemistry/Physics", 11: "Social Work",
    12: "Undergraduate/Undecided", 13: "Political Science/International Affairs",
    14: "Film", 15: "Fine Arts/Arts Administration", 16: "Languages",
    17: "Architecture", 18: "Other"
}

CAREER_CD_MAP = {
    1: "Lawyer", 2: "Academic/Researcher", 3: "Psychologist",
    4: "Doctor/Medical Professional", 5: "Engineer",
    6: "Creative Arts/Entertainment", 7: "Banking/Consulting/Finance/Business",
    8: "Real Estate", 9: "International/Humanitarian Affairs",
    10: "Undecided", 11: "Social Worker", 12: "Speech Pathologist",
    13: "Politics", 14: "Professional Sports/Athletics", 15: "Other",
    16: "Journalist", 17: "Architect"
}

RACE_MAP = {
    1: "Black/African American", 2: "European/Caucasian-American",
    3: "Latino/Hispanic American", 4: "Asian/Pacific Islander/Asian-American",
    5: "Native American", 6: "Other"
}

GOAL_MAP = {
    1: "have a fun night out", 2: "meet new people",
    3: "get a date", 4: "find a serious relationship",
    5: "say I did it", 6: "other reasons"
}

DATE_FREQ_MAP = {
    1: "several times a week", 2: "twice a week", 3: "once a week",
    4: "twice a month", 5: "once a month", 6: "several times a year",
    7: "almost never"
}

INTEREST_NAMES = {
    'sports': 'playing sports/athletics', 'tvsports': 'watching sports',
    'exercise': 'exercising/bodybuilding', 'dining': 'dining out',
    'museums': 'visiting museums/galleries', 'art': 'art',
    'hiking': 'hiking/camping', 'gaming': 'gaming',
    'clubbing': 'dancing/clubbing', 'reading': 'reading',
    'tv': 'watching TV', 'theater': 'theater',
    'movies': 'movies', 'concerts': 'concerts',
    'music': 'music', 'shopping': 'shopping',
    'yoga': 'yoga/meditation'
}


class PersonaGenerator:
    def __init__(self, pairs_path: str):
        """
        初始化 Persona 生成器
        
        Args:
            pairs_path: processed_pairs.json 的路径
        """
        self.pairs_path = pairs_path
        self.pairs = []
        self.personas = []
        
    def load_pairs(self):
        """加载配对数据"""
        print("📂 Loading processed pairs...")
        with open(self.pairs_path, 'r', encoding='utf-8') as f:
            self.pairs = json.load(f)
        print(f"✅ Loaded {len(self.pairs)} pairs")
        return self.pairs
    
    def _encode_all_interests(self, data: Dict) -> str:
        """完整编码所有兴趣爱好（不过滤，保留所有评分）"""
        interests_text = []
        for key, name in INTEREST_NAMES.items():
            if key in data and data[key] is not None:
                score = int(data[key])
                interests_text.append(f"{name} (rated {score}/10)")
        
        if interests_text:
            return "My interests and how much I enjoy them: " + ", ".join(interests_text) + "."
        else:
            return "I haven't rated my interests yet."
    
    def _rank_preferences(self, data: Dict, prefix: str = 'attr1_1') -> List[tuple]:
        """
        对择偶偏好排序
        
        Args:
            data: 数据字典
            prefix: 前缀，如 'attr1_1' (self), 'attr2_1' (opposite), 'attr4_1' (same)
        
        Returns:
            排序后的 [(attribute_name, score), ...]
        """
        base = prefix.rsplit('_', 1)[0]  # 去掉最后的 _1
        
        attrs = {
            'attractiveness': data.get(f'{base}_1'),
            'sincerity': data.get(f'sinc{base[4:]}_1'),
            'intelligence': data.get(f'intel{base[4:]}_1'),
            'fun': data.get(f'fun{base[4:]}_1'),
            'ambition': data.get(f'amb{base[4:]}_1'),
            'shared interests': data.get(f'shar{base[4:]}_1')
        }
        
        # 过滤 None 并排序
        valid_attrs = [(k, float(v)) for k, v in attrs.items() if v is not None]
        valid_attrs.sort(key=lambda x: x[1], reverse=True)
        
        return valid_attrs
    
    def _encode_preferences_complete(self, preferences: List[tuple]) -> str:
        """完整编码择偶偏好（保留所有分数，不过滤）"""
        if not preferences:
            return "balanced across all qualities"
        
        # 保留所有偏好和分数
        pref_text = ", ".join([f"{name} ({int(score)} points)" for name, score in preferences])
        return pref_text
    
    def _encode_self_ratings_complete(self, data: Dict) -> str:
        """完整编码自我评价（保留所有分数）"""
        ratings = {
            'attractiveness': data.get('attr3_1'),
            'sincerity': data.get('sinc3_1'),
            'intelligence': data.get('intel3_1'),
            'fun': data.get('fun3_1'),
            'ambition': data.get('amb3_1')
        }
        
        # 保留所有评分
        valid_ratings = [(k, int(v)) for k, v in ratings.items() if v is not None]
        if not valid_ratings:
            return "I haven't rated myself yet."
        
        valid_ratings.sort(key=lambda x: x[1], reverse=True)
        
        ratings_text = ", ".join([f"{name} ({score}/10)" for name, score in valid_ratings])
        return f"How I rate myself: {ratings_text}."
    
    def _encode_others_perception_complete(self, data: Dict) -> str:
        """完整编码他人评价预期（保留所有分数）"""
        perceptions = {
            'attractiveness': data.get('attr5_1'),
            'sincerity': data.get('sinc5_1'),
            'intelligence': data.get('intel5_1'),
            'fun': data.get('fun5_1'),
            'ambition': data.get('amb5_1')
        }
        
        # 保留所有评分
        valid_perceptions = [(k, int(v)) for k, v in perceptions.items() if v is not None]
        if not valid_perceptions:
            return ""
        
        valid_perceptions.sort(key=lambda x: x[1], reverse=True)
        
        perceptions_text = ", ".join([f"{name} ({score}/10)" for name, score in valid_perceptions])
        return f"How I expect others would rate me: {perceptions_text}."
    
    def _encode_time2_satisfaction(self, data: Dict) -> str:
        """编码 Time 2 满意度调查"""
        satis_2 = data.get('satis_2')
        length = data.get('length')
        numdat_2 = data.get('numdat_2')
        
        if satis_2 is None and length is None and numdat_2 is None:
            return ""
        
        text = "\n\n=== AFTER THE EVENT (Day After Reflection) ===\n"
        
        if satis_2 is not None:
            text += f"My satisfaction with people I met: {int(satis_2)}/10. "
        
        if length is not None:
            length_map = {1: "too little time", 2: "too much time", 3: "just right"}
            text += f"The 4-minute duration was: {length_map.get(int(length), 'unknown')}. "
        
        if numdat_2 is not None:
            numdat_map = {1: "too few", 2: "too many", 3: "just right"}
            text += f"The number of dates was: {numdat_map.get(int(numdat_2), 'unknown')}. "
        
        return text.strip()
    
    def _encode_time2_preferences(self, data: Dict, prefix: str) -> List[tuple]:
        """编码 Time 2 更新后的偏好（_2 后缀）"""
        base = prefix.rsplit('_', 1)[0]  # 去掉最后的 _2
        
        attrs = {
            'attractiveness': data.get(f'{base}_2'),
            'sincerity': data.get(f'sinc{base[4:]}_2'),
            'intelligence': data.get(f'intel{base[4:]}_2'),
            'fun': data.get(f'fun{base[4:]}_2'),
            'ambition': data.get(f'amb{base[4:]}_2'),
            'shared interests': data.get(f'shar{base[4:]}_2')
        }
        
        # 过滤 None 并排序
        valid_attrs = [(k, float(v)) for k, v in attrs.items() if v is not None]
        valid_attrs.sort(key=lambda x: x[1], reverse=True)
        
        return valid_attrs
    
    def _encode_time2_self_ratings(self, data: Dict) -> str:
        """编码 Time 2 更新后的自我评价"""
        ratings = {
            'attractiveness': data.get('attr3_2'),
            'sincerity': data.get('sinc3_2'),
            'intelligence': data.get('intel3_2'),
            'fun': data.get('fun3_2'),
            'ambition': data.get('amb3_2')
        }
        
        valid_ratings = [(k, int(v)) for k, v in ratings.items() if v is not None]
        if not valid_ratings:
            return ""
        
        valid_ratings.sort(key=lambda x: x[1], reverse=True)
        
        ratings_text = ", ".join([f"{name} ({score}/10)" for name, score in valid_ratings])
        return f"Updated self-ratings: {ratings_text}."
    
    def _encode_time2_others_perception(self, data: Dict) -> str:
        """编码 Time 2 更新后的他人评价预期"""
        perceptions = {
            'attractiveness': data.get('attr5_2'),
            'sincerity': data.get('sinc5_2'),
            'intelligence': data.get('intel5_2'),
            'fun': data.get('fun5_2'),
            'ambition': data.get('amb5_2')
        }
        
        valid_perceptions = [(k, int(v)) for k, v in perceptions.items() if v is not None]
        if not valid_perceptions:
            return ""
        
        valid_perceptions.sort(key=lambda x: x[1], reverse=True)
        
        perceptions_text = ", ".join([f"{name} ({score}/10)" for name, score in valid_perceptions])
        return f"Updated perception of how others see me: {perceptions_text}."
    
    def _extract_time2_data(self, data: Dict) -> Dict:
        """
        提取 Time 2 (Day After Event) 的所有数据作为 ground truth
        这些数据不应该在 persona narrative 中，因为是事后反思
        """
        time2_data = {
            'satisfaction': {
                'satis_2': data.get('satis_2'),
                'length': data.get('length'),
                'numdat_2': data.get('numdat_2')
            },
            'updated_preferences_self': {
                'attractiveness': data.get('attr1_2'),
                'sincerity': data.get('sinc1_2'),
                'intelligence': data.get('intel1_2'),
                'fun': data.get('fun1_2'),
                'ambition': data.get('amb1_2'),
                'shared_interests': data.get('shar1_2')
            },
            'updated_preferences_opposite': {
                'attractiveness': data.get('attr2_2'),
                'sincerity': data.get('sinc2_2'),
                'intelligence': data.get('intel2_2'),
                'fun': data.get('fun2_2'),
                'ambition': data.get('amb2_2'),
                'shared_interests': data.get('shar2_2')
            },
            'updated_preferences_same': {
                'attractiveness': data.get('attr4_2'),
                'sincerity': data.get('sinc4_2'),
                'intelligence': data.get('intel4_2'),
                'fun': data.get('fun4_2'),
                'ambition': data.get('amb4_2'),
                'shared_interests': data.get('shar4_2')
            },
            'updated_self_ratings': {
                'attractiveness': data.get('attr3_2'),
                'sincerity': data.get('sinc3_2'),
                'intelligence': data.get('intel3_2'),
                'fun': data.get('fun3_2'),
                'ambition': data.get('amb3_2')
            },
            'updated_others_perception': {
                'attractiveness': data.get('attr5_2'),
                'sincerity': data.get('sinc5_2'),
                'intelligence': data.get('intel5_2'),
                'fun': data.get('fun5_2'),
                'ambition': data.get('amb5_2')
            }
        }
        
        return time2_data
    
    def _generate_persona_narrative(self, person_data: Dict, person_key: str) -> str:
        """
        生成《再见爱人》风格的人物叙事
        
        Args:
            person_data: person1 或 person2 的数据
            person_key: 'person1' 或 'person2'
        
        Returns:
            第一人称叙事文本
        """
        data = person_data['data']
        
        # 基本信息
        age = int(data.get('age', 0))
        gender = "woman" if data.get('gender') == 0 else "man"
        field = FIELD_CD_MAP.get(int(data.get('field_cd', 18)), "Other")
        career = CAREER_CD_MAP.get(int(data.get('career_c', 15)), "undecided")
        race = RACE_MAP.get(int(data.get('race', 6)), "")
        
        # 约会目标和频率
        goal = GOAL_MAP.get(int(data.get('goal', 2)), "meet new people")
        date_freq = DATE_FREQ_MAP.get(int(data.get('date', 5)), "occasionally")
        go_out_freq = DATE_FREQ_MAP.get(int(data.get('go_out', 3)), "regularly")
        
        # 种族/宗教重要性
        imprace = int(data.get('imprace', 5))
        imprelig = int(data.get('imprelig', 5))
        
        # 期望（完整编码，不过滤）
        exphappy = int(data.get('exphappy', 5))
        
        # 择偶偏好（完整编码所有分数）
        self_prefs = self._rank_preferences(data, 'attr1_1')
        opp_sex_prefs = self._rank_preferences(data, 'attr2_1')
        same_sex_prefs = self._rank_preferences(data, 'attr4_1')
        
        # 自我评价（完整编码）
        self_ratings = self._encode_self_ratings_complete(data)
        
        # 他人评价预期（完整编码）
        others_perception = self._encode_others_perception_complete(data)
        
        # 兴趣爱好（完整编码所有评分）
        all_interests = self._encode_all_interests(data)
        
        # 构建叙事 - 完整编码所有信息
        narrative = f"""I'm a {age}-year-old {gender} studying {field}, with plans to become a {career}. """
        
        if race:
            narrative += f"I'm {race}. "
        
        # 生活状态（完整保留原始值）
        narrative += f"\n\nIn my daily life, I go out {go_out_freq}, though I only go on dates {date_freq}. "
        narrative += f"I came to this speed dating event to {goal}. "
        narrative += f"My expected happiness for tonight: {exphappy}/10. "
        
        # 价值观（完整编码，保留原始分数）
        narrative += f"\n\nImportance of same race in dating: {imprace}/10. "
        narrative += f"Importance of same religion in dating: {imprelig}/10. "
        
        # 择偶观（完整编码所有偏好分数）
        narrative += f"\n\nWhat I value in a potential date (out of 100 points total): {self._encode_preferences_complete(self_prefs)}. "
        
        if opp_sex_prefs:
            narrative += f"\n\nWhat I think the opposite sex looks for: {self._encode_preferences_complete(opp_sex_prefs)}. "
        
        if same_sex_prefs:
            narrative += f"\n\nWhat I think my same sex looks for: {self._encode_preferences_complete(same_sex_prefs)}. "
        
        # 自我认知（完整编码）
        narrative += f"\n\n{self_ratings} "
        
        # 他人评价预期（如果有）
        if others_perception:
            narrative += f"{others_perception} "
        
        # 兴趣爱好（完整编码所有活动评分）
        narrative += f"\n\n{all_interests} "
        
        # 结尾：对感情的态度
        narrative += "\n\n"
        if goal == "find a serious relationship":
            narrative += f"I'm genuinely looking for something meaningful and long-term. I'm ready to invest emotionally and see where things go with the right person."
        elif goal == "get a date":
            narrative += f"I'm open to seeing where things lead. If I meet someone interesting, I'd definitely want to get to know them better."
        else:
            narrative += f"I'm here with an open heart, curious to see who I'll meet and what connections might form."
        
        return narrative.strip()
    
    def generate_personas(self):
        """为所有配对生成 Persona"""
        print("\n🎭 Generating personas...")
        print("=" * 70)
        
        personas = []
        
        for pair in self.pairs:
            pair_id = pair['pair_id']
            
            # 生成 person1 的 persona (只包含 Time 1 pre-event 数据)
            persona1_narrative = self._generate_persona_narrative(pair['person1'], 'person1')
            
            # 生成 person2 的 persona (只包含 Time 1 pre-event 数据)
            persona2_narrative = self._generate_persona_narrative(pair['person2'], 'person2')
            
            # 提取 Time 2 数据作为 ground truth（不在 persona 中）
            time2_person1 = self._extract_time2_data(pair['person1']['data'])
            time2_person2 = self._extract_time2_data(pair['person2']['data'])
            
            # 构建 persona 对象
            persona_pair = {
                'pair_id': pair_id,
                'person1': {
                    'iid': pair['person1']['iid'],
                    'gender': pair['person1']['gender'],
                    'age': pair['person1']['age'],
                    'persona_narrative': persona1_narrative,
                    'system_prompt': self._create_system_prompt(persona1_narrative, pair['person1']),
                    'time2_reflection': time2_person1  # Time 2 数据单独保存
                },
                'person2': {
                    'iid': pair['person2']['iid'],
                    'gender': pair['person2']['gender'],
                    'age': pair['person2']['age'],
                    'persona_narrative': persona2_narrative,
                    'system_prompt': self._create_system_prompt(persona2_narrative, pair['person2']),
                    'time2_reflection': time2_person2  # Time 2 数据单独保存
                },
                'ground_truth': pair['ground_truth']
            }
            
            personas.append(persona_pair)
            
            if len(personas) % 10 == 0:
                print(f"   Generated {len(personas)} / {len(self.pairs)} personas...")
        
        self.personas = personas
        print(f"\n✅ Generated {len(personas)} persona pairs")
        
        return personas
    
    def _create_system_prompt(self, narrative: str, person_data: Dict) -> str:
        """
        为 Mistral Nemo (via OpenRouter API) 创建 system prompt
        
        Args:
            narrative: 完整编码的人物叙事（包含所有原始数据）
            person_data: person1 或 person2 的数据
        
        Returns:
            System prompt 文本（用于 OpenRouter API 调用）
        """
        gender = "woman" if person_data['gender'] == 0 else "man"
        age = person_data['age']
        
        # 根据性别分配经典名字
        if gender == "woman":
            name = "Emma" if person_data['iid'] % 2 == 0 else "Sarah"
        else:
            name = "Jake" if person_data['iid'] % 2 == 0 else "Ryan"
        
        system_prompt = f"""You are {name}, a {age}-year-old {gender} at a speed dating event. This is a REAL STRANGER you're meeting for the first time - be natural and cautious.

YOUR BACKGROUND & PERSONALITY:
{narrative}

🎯 CRITICAL ROLEPLAY RULES:

1. **MANDATORY RESPONSE FORMAT** - EVERY response MUST follow this structure:

<INNER_THOUGHT>
[Your private thoughts: analyze persona, decide tone/approach, consider what they just said, plan response strategy]
</INNER_THOUGHT>

<RESPONSE>
[Your actual spoken words - what you say out loud to the other person]
[Use emotion/expression/gesture tags like this]
</RESPONSE>

Example:
<INNER_THOUGHT>
Sports = shared interest (high value for me). Show enthusiasm, but stay reserved. Test if they're serious or casual.
</INNER_THOUGHT>

<RESPONSE>
Oh nice, I'm into basketball too! [Smiles genuinely] I actually try to catch Lakers games when I can. Do you play or just watch? [Leans forward slightly, showing interest but still a bit cautious]
</RESPONSE>

**KEEP <INNER_THOUGHT> CONCISE (1-2 sentences max)** - This is speed dating, think fast!

2. **YOU'RE TALKING TO A NEW PERSON**
   - This is literally the FIRST time you've met this person
   - warm up gradually
   - Be cautious sharing deep personal info early
   - Natural to have some silences, fumbles, nervous energy
   - Example opening: "Hey... [nervous smile] So uh, first time here? [fidgets with napkin] I'm {name}"

3. **EMOTION/EXPRESSION/GESTURE TAGS** (USE THESE TOGETHER WITH EMOJIS)
   - [Laughs nervously]
   - [Makes eye contact briefly, then looks away]
   - [Fidgets with coffee cup]
   - [Leans back, crossing arms defensively]
   - [Brightens up, leaning forward]
   - [Pauses, thinking]
   - [Raises eyebrow skeptically]
   - You CAN use SOME emojis and descriptive tags (1-2 per message max)

4. **BE SPECIFIC WITH DETAILS** 
   - If you like hiking → "I actually went to Runyon Canyon last weekend"
   - If you're into movies → "Just watched Oppenheimer at the Arclight, totally worth it"
   - If you study bio → "Yeah, I'm in the middle of this brutal genetics project right now"
   - ADD REAL PLACES, EVENTS, SPECIFICS from your life

5. **NEVER MENTION NUMBERS/RATINGS DIRECTLY**
   - ❌ DON'T say: "I rated attractiveness 25 points"
   - ✅ DO show: Your preferences through reactions/interest
   - If you value intelligence → get excited about deep topics
   - If you rated fun low for yourself → be more reserved/serious
   - If you value shared interests → ask about hobbies enthusiastically

6. **LET YOUR PERSONA GUIDE YOUR BEHAVIOR** (check in <INNER_THOUGHT>)
   - Before each response, review: What do I value most? (your preference points)
   - How confident am I? (your self-ratings)
   - What's my dating goal? (serious relationship vs casual)
   - Am I naturally shy or outgoing? (based on your fun/sincerity ratings)
   - Let this shape your tone: active/passive, flirty/reserved, deep/light

7. **SPEED DATING CONTEXT**
   - You have ~4 minutes total to chat
   - You're sitting across from a stranger at a small table
   - Others are chatting nearby (it's a bit loud)
   - Bell will ring when time's up
   - You'll decide at the end: want to see them again?

8. **REALISTIC STRANGER INTERACTION**
   - Start with small talk (weather, event itself, nervousness)
   - Gradually get into interests/work if vibe is good
   - Watch for red flags - if something feels off, pull back
   - Chemistry takes time - don't force instant connection
   - Some awkward pauses are NORMAL
   - Mirror their energy level (if they're reserved, don't be overly pushy)

9. **TEXTING STYLE (but not excessive)**
   - Use casual language: "yeah", "kinda", "like", "I mean"
   - 1-2 emojis or descriptive tags MAX per response (if any)
   - Short messages (2-4 sentences) unless deeply engaged

REMEMBER: You're {name}, a real {age}yo {gender}. You DON'T know this person yet. Think before you speak (<INNER_THOUGHT>), be specific with details, show emotions through [tags], let your persona values guide you naturally."""
        
        return system_prompt
    
    def save_personas(self, output_dir: str = "results"):
        """保存生成的 personas"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # 保存完整 personas
        personas_path = output_path / "personas.json"
        with open(personas_path, 'w', encoding='utf-8') as f:
            json.dump(self.personas, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Saved personas to {personas_path}")
        
        # 保存示例（前3对）
        sample_path = output_path / "personas_sample.txt"
        with open(sample_path, 'w', encoding='utf-8') as f:
            for i, persona in enumerate(self.personas[:3]):
                f.write(f"{'='*80}\n")
                f.write(f"PAIR {i+1}: {persona['pair_id']}\n")
                f.write(f"Ground Truth: {'MATCHED' if persona['ground_truth']['match'] == 1 else 'NOT MATCHED'}\n")
                f.write(f"{'='*80}\n\n")
                
                f.write(f"--- PERSON 1 (iid={persona['person1']['iid']}) ---\n\n")
                f.write(persona['person1']['persona_narrative'])
                f.write(f"\n\n")
                
                f.write(f"--- PERSON 2 (iid={persona['person2']['iid']}) ---\n\n")
                f.write(persona['person2']['persona_narrative'])
                f.write(f"\n\n\n")
        
        print(f"💾 Saved sample personas to {sample_path}")
        
        return output_path


def main():
    """主函数"""
    print("🎭 Persona Generator - Phase 1")
    print("=" * 70)
    
    # 初始化生成器
    pairs_path = "results/processed_pairs.json"
    generator = PersonaGenerator(pairs_path)
    
    # 加载配对数据
    generator.load_pairs()
    
    # 生成 personas
    generator.generate_personas()
    
    # 保存结果
    output_dir = generator.save_personas()
    
    print("\n" + "=" * 70)
    print("✅ Persona generation completed!")
    print(f"📁 Results saved to: {output_dir}")
    print("\n🎯 Next steps:")
    print("   1. Review personas_sample.txt to check quality")
    print("   2. Run speed_dating_simulator.py for Scenario 1")
    print("   3. Run critical_events_engine.py for Scenario 2")


if __name__ == "__main__":
    main()
