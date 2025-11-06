"""
Divorce Persona Generator

功能:
1. 从 divorce_clean.csv 加载 Gottman DPS 特征（54个问题，0-4分）
2. 将数值特征转换为自然语言人物描述
3. 为每对夫妻生成 Husband 和 Wife 的 persona
4. 输出格式与 Speed Dating 的 persona_generator.py 一致

参考: test/experiments/persona_generator.py
"""

import pandas as pd
import json
import argparse
from pathlib import Path
from typing import Dict, List

# Gottman DPS 问题映射（从 doc.txt）
GOTTMAN_QUESTIONS = {
    # Conflict Resolution (Q1-4)
    1: "When one of us apologizes when our discussions go in a bad direction, the issue does not extend",
    2: "I know we can ignore our differences, even if things get hard sometimes",
    3: "When we need it, we can take our discussions with my spouse from the beginning and correct it",
    4: "When I argue with my spouse, it will eventually work for me to contact them",
    
    # Time & Goals (Q5-14)
    5: "The time I spent with my spouse is special for us",
    6: "We don't have time at home as partners",
    7: "We are like two strangers who share the same environment at home rather than family",
    8: "I enjoy our holidays with my spouse",
    9: "I enjoy traveling with my spouse",
    10: "My spouse and most of our goals are common",
    11: "I think that one day in the future, when I look back, I see that my spouse and I are in harmony with each other",
    12: "My spouse and I have similar values in terms of personal freedom",
    13: "My spouse and I have similar entertainment",
    14: "Most of our goals for people (children, friends, etc.) are the same",
    
    # Values & Compatibility (Q15-20)
    15: "Our dreams of living with my spouse are similar and harmonious",
    16: "We're compatible with my spouse about what love should be",
    17: "We share the same views with my spouse about being happy in your life",
    18: "My spouse and I have similar ideas about how marriage should be",
    19: "My spouse and I have similar ideas about how roles should be in marriage",
    20: "My spouse and I have similar values in trust",
    
    # Mutual Understanding (Q21-30)
    21: "I know exactly what my spouse likes",
    22: "I know how my spouse wants to be taken care of when they're sick",
    23: "I know my spouse's favorite food",
    24: "I can tell you what kind of stress my spouse is facing in their life",
    25: "I have knowledge of my spouse's inner world",
    26: "I know my spouse's basic concerns",
    27: "I know what my spouse's current sources of stress are",
    28: "I know my spouse's hopes and wishes",
    29: "I know my spouse very well",
    30: "I know my spouse's friends and their social relationships",
    
    # Aggression in Conflict (Q31-40)
    31: "I feel aggressive when I argue with my spouse",
    32: "When discussing with my spouse, I usually use expressions such as 'you always' or 'you never'",
    33: "I can use negative statements about my spouse's personality during our discussions",
    34: "I can use offensive expressions during our discussions",
    35: "I can insult during our discussions",
    36: "I can be humiliating when we argue",
    37: "My argument with my spouse is not calm",
    38: "I hate my spouse's way of bringing issues up",
    39: "Fights often occur suddenly",
    40: "We're just starting a fight before I know what's going on",
    
    # Withdrawal & Avoidance (Q41-47)
    41: "When I talk to my spouse about something, my calm suddenly breaks",
    42: "When I argue with my spouse, it only snaps in and I don't say a word",
    43: "I'm mostly trying to calm the environment a little bit",
    44: "Sometimes I think it's good for me to leave home for a while",
    45: "I'd rather stay silent than argue with my spouse",
    46: "Even if I'm right in the argument, I'm trying not to upset the other side",
    47: "When I argue with my spouse, I remain silent because I am afraid of not being able to control my anger",
    
    # Defensiveness (Q48-54)
    48: "I feel right in our discussions",
    49: "I have nothing to do with what I've been accused of",
    50: "I'm not actually the one who's guilty about what I'm accused of",
    51: "I'm not the one who's wrong about problems at home",
    52: "I wouldn't hesitate to tell my spouse about their inadequacy",
    53: "When I discuss, I remind my spouse of their inadequate issues",
    54: "I'm not afraid to tell my spouse about their incompetence",
}

# Scale interpretation (0=Never, 1=Seldom, 2=Averagely, 3=Frequently, 4=Always)
SCALE_MAP = {
    0: "never",
    1: "seldom",
    2: "sometimes",
    3: "frequently",
    4: "always"
}


class DivorcePersonaGenerator:
    """生成离婚预测实验的 persona（基于 Gottman DPS）"""
    
    def __init__(self, clean_data_path: str = "divorce_clean.csv"):
        # 自动识别分隔符：divorce.csv 通常是分号；divorce_clean.csv 通常是逗号
        sep = self._infer_sep(clean_data_path)
        self.df = pd.read_csv(clean_data_path, sep=sep)
        print(f"✅ Loaded {len(self.df)} couples")
        self.personas = []

    def _infer_sep(self, path: str) -> str:
        """简易分隔符推断：首行包含分号则用 ';'，否则默认 ','"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                header = f.readline()
            return ';' if ';' in header and header.count(';') >= header.count(',') else ','
        except Exception:
            return ','
    
    def _score_to_text(self, score: int) -> str:
        """将 0-4 分数转为文字描述"""
        return SCALE_MAP.get(int(score), "sometimes")
    
    def _analyze_conflict_style(self, features: Dict) -> str:
        """分析冲突处理风格 (Q1-4, Q31-47)"""
        # 检查攻击性
        aggression_scores = [
            features.get('Atr31', 2), features.get('Atr32', 2),
            features.get('Atr33', 2), features.get('Atr34', 2),
            features.get('Atr35', 2), features.get('Atr36', 2)
        ]
        avg_aggression = sum(aggression_scores) / len(aggression_scores)
        
        # 检查回避倾向
        avoidance_scores = [
            features.get('Atr42', 2), features.get('Atr44', 2),
            features.get('Atr45', 2), features.get('Atr47', 2)
        ]
        avg_avoidance = sum(avoidance_scores) / len(avoidance_scores)
        
        # 检查修复能力
        repair_scores = [
            features.get('Atr1', 2), features.get('Atr2', 2),
            features.get('Atr3', 2), features.get('Atr4', 2)
        ]
        avg_repair = sum(repair_scores) / len(repair_scores)
        
        # 生成描述
        style_parts = []
        
        if avg_aggression >= 3:
            style_parts.append("I tend to become aggressive and use harsh words during arguments")
        elif avg_aggression >= 2:
            style_parts.append("I sometimes raise my voice or use negative expressions when we fight")
        else:
            style_parts.append("I try to stay calm during conflicts")
        
        if avg_avoidance >= 3:
            style_parts.append("When things get heated, I prefer to shut down or leave the room")
        elif avg_avoidance >= 2:
            style_parts.append("I sometimes go silent or need space when we're arguing")
        
        if avg_repair < 2:
            style_parts.append("It's hard for me to apologize or restart a conversation after a fight")
        elif avg_repair >= 3:
            style_parts.append("I'm usually able to apologize and work things out after we argue")
        
        return ". ".join(style_parts) + "."
    
    def _analyze_values_compatibility(self, features: Dict) -> str:
        """分析价值观契合度 (Q10-20)"""
        values_scores = [
            ('shared goals', features.get('Atr10', 2)),
            ('personal freedom values', features.get('Atr12', 2)),
            ('entertainment preferences', features.get('Atr13', 2)),
            ('trust values', features.get('Atr20', 2)),
            ('views on happiness', features.get('Atr17', 2)),
        ]
        
        aligned = [name for name, score in values_scores if score >= 3]
        misaligned = [name for name, score in values_scores if score <= 1]
        
        parts = []
        if aligned:
            parts.append(f"My spouse and I are aligned on {', '.join(aligned)}")
        if misaligned:
            parts.append(f"We differ significantly on {', '.join(misaligned)}")
        
        return ". ".join(parts) + "." if parts else "We have mixed compatibility on core values."
    
    def _analyze_emotional_connection(self, features: Dict) -> str:
        """分析情感连接 (Q5-9, Q21-30)"""
        connection_scores = [
            features.get('Atr5', 2),  # time together is special
            features.get('Atr8', 2),  # enjoy holidays
            features.get('Atr9', 2),  # enjoy traveling
        ]
        avg_connection = sum(connection_scores) / len(connection_scores)
        
        understanding_scores = [
            features.get('Atr25', 2),  # know inner world
            features.get('Atr28', 2),  # know hopes/wishes
            features.get('Atr29', 2),  # know very well
        ]
        avg_understanding = sum(understanding_scores) / len(understanding_scores)
        
        parts = []
        if avg_connection >= 3:
            parts.append("I genuinely enjoy spending time with my spouse")
        elif avg_connection <= 1:
            parts.append("We feel like strangers sharing a home rather than true partners")
        
        if avg_understanding >= 3:
            parts.append("I feel I deeply understand their inner world and needs")
        elif avg_understanding <= 1:
            parts.append("I often don't know what they're thinking or feeling")
        
        return ". ".join(parts) + "." if parts else "We have a moderate emotional connection."
    
    def _analyze_communication_pattern(self, features: Dict) -> str:
        """分析沟通模式 (Q31-47)"""
        defensiveness = [
            features.get('Atr48', 2),  # feel right
            features.get('Atr49', 2),  # nothing to do with accusations
            features.get('Atr51', 2),  # not the one who's wrong
        ]
        avg_defensive = sum(defensiveness) / len(defensiveness)
        
        criticism = [
            features.get('Atr52', 2),  # point out inadequacy
            features.get('Atr53', 2),  # remind of issues
            features.get('Atr54', 2),  # not afraid to point out incompetence
        ]
        avg_criticism = sum(criticism) / len(criticism)
        
        parts = []
        if avg_defensive >= 3:
            parts.append("In conflicts, I tend to feel I'm right and defend my position strongly")
        elif avg_defensive >= 2:
            parts.append("I sometimes get defensive when criticized")
        
        if avg_criticism >= 3:
            parts.append("I don't hesitate to point out my spouse's flaws and mistakes")
        elif avg_criticism >= 2:
            parts.append("I sometimes bring up their past failures during arguments")
        
        return ". ".join(parts) + "." if parts else "We have typical communication patterns."
    
    def generate_persona_narrative(self, couple_id: int, role: str) -> str:
        """
        为一个人生成 persona 叙述
        
        Args:
            couple_id: 夫妻 ID
            role: 'husband' 或 'wife'
        
        Returns:
            自然语言 persona 段落
        """
        row = self.df.iloc[couple_id]
        features = row.to_dict()
        
        # 生成各维度描述
        conflict_style = self._analyze_conflict_style(features)
        values = self._analyze_values_compatibility(features)
        connection = self._analyze_emotional_connection(features)
        communication = self._analyze_communication_pattern(features)
        
        # 组装 persona
        persona = f"""I am {role.capitalize()} in a marriage. Here's how I experience our relationship:

**How I Handle Conflict:**
{conflict_style}

**Our Values & Compatibility:**
{values}

**Our Emotional Connection:**
{connection}

**Communication Patterns:**
{communication}

**My Inner Truth:**
When facing major life challenges, I tend to follow my gut instincts rather than just what I "should" do. I know my limits and what I can tolerate in this relationship.
"""
        return persona.strip()
    
    def generate_all_personas(self, output_path: str = "divorce_personas.json"):
        """为所有夫妻生成 persona"""
        print(f"\n🎭 Generating personas for {len(self.df)} couples...")
        
        personas = []
        for couple_id in range(len(self.df)):
            row = self.df.iloc[couple_id]
            
            # 为夫妻双方生成 persona
            husband_persona = self.generate_persona_narrative(couple_id, 'husband')
            wife_persona = self.generate_persona_narrative(couple_id, 'wife')
            
            personas.append({
                'couple_id': couple_id,
                'ground_truth_divorced': int(row['Class'] == 0),
                'husband': {
                    'role': 'husband',
                    'persona_narrative': husband_persona,
                },
                'wife': {
                    'role': 'wife',
                    'persona_narrative': wife_persona,
                }
            })
            
            if (couple_id + 1) % 50 == 0:
                print(f"   Generated {couple_id + 1}/{len(self.df)} couples...")
        
        # 保存
        output_path = Path(output_path)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(personas, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Saved {len(personas)} couple personas to: {output_path}")
        
        # 保存示例
        sample_path = output_path.parent / "divorce_personas_sample.txt"
        with open(sample_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("SAMPLE PERSONAS (First 3 Couples)\n")
            f.write("=" * 80 + "\n\n")
            for i in range(min(3, len(personas))):
                couple = personas[i]
                f.write(f"\n{'='*80}\n")
                f.write(f"Couple {i} (Divorced: {couple['ground_truth_divorced']})\n")
                f.write(f"{'='*80}\n\n")
                f.write("--- HUSBAND ---\n")
                f.write(couple['husband']['persona_narrative'])
                f.write("\n\n--- WIFE ---\n")
                f.write(couple['wife']['persona_narrative'])
                f.write("\n\n")
        
        print(f"✅ Saved sample to: {sample_path}")
        return personas


def main():
    parser = argparse.ArgumentParser(description="Generate personas for divorce prediction")
    parser.add_argument('--data', type=str, default='divorce_clean.csv',
                       help='Path to clean divorce dataset')
    parser.add_argument('--output', type=str, default='divorce_personas.json',
                       help='Output path for personas JSON')
    args = parser.parse_args()
    
    print("=" * 70)
    print("DIVORCE PERSONA GENERATOR")
    print("=" * 70)
    
    generator = DivorcePersonaGenerator(args.data)
    personas = generator.generate_all_personas(args.output)
    
    print("\n🎯 Next steps:")
    print("   1. Review divorce_personas_sample.txt")
    print("   2. Run 03_critical_events_simulator.py")


if __name__ == "__main__":
    main()
