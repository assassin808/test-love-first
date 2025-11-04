"""
Persona Generator - Phase 1

功能:
1. 加载 processed_pairs.json
2. 将数据特征转换为自然语言人物描述
3. 生成《再见爱人》风格的 Persona prompts
4. 为 Mistral Nemo 提供角色扮演 system prompt
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
    
    def _get_top_interests(self, data: Dict, top_n: int = 5) -> List[str]:
        """获取最感兴趣的活动"""
        interests = []
        for key, name in INTEREST_NAMES.items():
            if key in data and data[key] is not None:
                interests.append((name, float(data[key])))
        
        # 按评分排序
        interests.sort(key=lambda x: x[1], reverse=True)
        return [name for name, _ in interests[:top_n]]
    
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
    
    def _generate_preference_text(self, preferences: List[tuple]) -> str:
        """生成择偶偏好文本"""
        if not preferences:
            return "balanced across all qualities"
        
        top3 = preferences[:3]
        pref_text = ", ".join([f"{name}" for name, _ in top3])
        return pref_text
    
    def _generate_self_perception(self, data: Dict) -> str:
        """生成自我认知描述"""
        ratings = {
            'attractiveness': data.get('attr3_1'),
            'sincerity': data.get('sinc3_1'),
            'intelligence': data.get('intel3_1'),
            'fun': data.get('fun3_1'),
            'ambition': data.get('amb3_1')
        }
        
        # 找到最高和最低的评分
        valid_ratings = [(k, float(v)) for k, v in ratings.items() if v is not None]
        if not valid_ratings:
            return "I'm still figuring out who I am."
        
        valid_ratings.sort(key=lambda x: x[1], reverse=True)
        
        strengths = [name for name, score in valid_ratings[:2] if score >= 7]
        
        if strengths:
            return f"I see myself as {' and '.join(strengths)}."
        else:
            return "I'm working on growing in all areas of life."
    
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
        
        # 期望
        exphappy = int(data.get('exphappy', 5))
        
        # 择偶偏好
        self_prefs = self._rank_preferences(data, 'attr1_1')
        opp_sex_prefs = self._rank_preferences(data, 'attr2_1')
        same_sex_prefs = self._rank_preferences(data, 'attr4_1')
        
        # 自我认知
        self_perception = self._generate_self_perception(data)
        
        # 兴趣爱好
        top_interests = self._get_top_interests(data, top_n=5)
        
        # 构建叙事
        narrative = f"""I'm a {age}-year-old {gender} studying {field}, with plans to become a {career}. """
        
        if race:
            narrative += f"I'm {race}. "
        
        # 生活状态
        narrative += f"\n\nIn my daily life, I go out {go_out_freq}, though I only go on dates {date_freq}. "
        narrative += f"I came to this speed dating event to {goal}. "
        
        if exphappy >= 7:
            narrative += f"I'm quite optimistic and expect to have a good time tonight. "
        elif exphappy <= 4:
            narrative += f"I'm not sure what to expect, but I'm keeping an open mind. "
        
        # 择偶观
        narrative += f"\n\nWhen it comes to dating, what I value most is {self._generate_preference_text(self_prefs)}. "
        
        if opp_sex_prefs:
            opp_pref_text = self._generate_preference_text(opp_sex_prefs)
            narrative += f"I think the opposite sex usually looks for {opp_pref_text}. "
        
        # 价值观
        if imprace >= 7:
            narrative += f"It's quite important to me that my partner shares my racial/ethnic background. "
        if imprelig >= 7:
            narrative += f"Religious compatibility is also important to me. "
        
        # 自我认知
        narrative += f"\n\n{self_perception} "
        
        # 兴趣爱好
        if top_interests:
            interests_text = ", ".join(top_interests[:3])
            narrative += f"In my free time, I really enjoy {interests_text}"
            if len(top_interests) > 3:
                narrative += f", among other things"
            narrative += ". "
        
        # 结尾：对感情的态度
        if goal == "find a serious relationship":
            narrative += f"\n\nI'm genuinely looking for something meaningful and long-term. I'm ready to invest emotionally and see where things go with the right person."
        elif goal == "get a date":
            narrative += f"\n\nI'm open to seeing where things lead. If I meet someone interesting, I'd definitely want to get to know them better."
        else:
            narrative += f"\n\nI'm here with an open heart, curious to see who I'll meet and what connections might form."
        
        return narrative.strip()
    
    def generate_personas(self):
        """为所有配对生成 Persona"""
        print("\n🎭 Generating personas...")
        print("=" * 70)
        
        personas = []
        
        for pair in self.pairs:
            pair_id = pair['pair_id']
            
            # 生成 person1 的 persona
            persona1_narrative = self._generate_persona_narrative(pair['person1'], 'person1')
            
            # 生成 person2 的 persona
            persona2_narrative = self._generate_persona_narrative(pair['person2'], 'person2')
            
            # 构建 persona 对象
            persona_pair = {
                'pair_id': pair_id,
                'person1': {
                    'iid': pair['person1']['iid'],
                    'gender': pair['person1']['gender'],
                    'age': pair['person1']['age'],
                    'persona_narrative': persona1_narrative,
                    'system_prompt': self._create_system_prompt(persona1_narrative, pair['person1'])
                },
                'person2': {
                    'iid': pair['person2']['iid'],
                    'gender': pair['person2']['gender'],
                    'age': pair['person2']['age'],
                    'persona_narrative': persona2_narrative,
                    'system_prompt': self._create_system_prompt(persona2_narrative, pair['person2'])
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
        为 Mistral Nemo 创建 system prompt
        
        Args:
            narrative: 人物叙事
            person_data: person1 或 person2 的数据
        
        Returns:
            System prompt 文本
        """
        gender = "woman" if person_data['gender'] == 0 else "man"
        age = person_data['age']
        
        system_prompt = f"""You are roleplaying as a real person in a speed dating scenario. Here is your character:

{narrative}

IMPORTANT INSTRUCTIONS:
1. Stay completely in character - respond as this person would, using first person ("I", "me", "my")
2. Be natural and conversational, as if you're really on a 4-minute speed date
3. Show genuine emotions and reactions based on your personality and values
4. Ask questions about your date partner to show interest
5. Share personal stories and experiences that reflect your character
6. React authentically - if something resonates with you, show excitement; if not, be honest but polite
7. Keep track of what you learn about your partner throughout the conversation
8. Your responses should be 2-4 sentences unless asked for more detail

Remember: You are a {age}-year-old {gender} on a real speed date. Be yourself, be genuine, and see if there's a connection!"""
        
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
