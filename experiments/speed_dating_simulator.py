"""
Speed Dating Simulator - Phase 2

功能:
1. 加载 personas.json
2. 使用 OpenRouter API (Mistral Nemo) 模拟 10 轮 speed dating 对话
3. 记录每轮对话和情感状态
4. 保存完整对话日志

OpenRouter API: https://openrouter.ai/
Model: mistralai/mistral-nemo
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional
import requests
import time
from datetime import datetime
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()


class SpeedDatingSimulator:
    def __init__(self, personas_path: str, api_key: Optional[str] = None):
        """
        初始化 Speed Dating 模拟器
        
        Args:
            personas_path: personas.json 的路径
            api_key: OpenRouter API key (如果为 None，从环境变量读取)
        """
        self.personas_path = personas_path
        self.api_key = api_key or os.getenv('OPENROUTER_API_KEY')
        
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not found. Please set it in environment or pass as parameter.")
        
        self.personas = []
        self.conversations = []
        
        # OpenRouter API 配置
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.model = "mistralai/mistral-nemo"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://github.com/your-repo",  # 可选，用于跟踪
            "X-Title": "Speed Dating Experiment"  # 可选
        }
    
    def load_personas(self):
        """加载 persona 数据"""
        print("📂 Loading personas...")
        with open(self.personas_path, 'r', encoding='utf-8') as f:
            self.personas = json.load(f)
        print(f"✅ Loaded {len(self.personas)} persona pairs")
        return self.personas
    
    def call_openrouter_api(self, messages: List[Dict], temperature: float = 0.7) -> str:
        """
        调用 OpenRouter API
        
        Args:
            messages: 消息列表 [{"role": "system/user/assistant", "content": "..."}]
            temperature: 温度参数 (0-1)
        
        Returns:
            模型返回的文本
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 200  # Speed dating 回复不应该太长
        }
        
        try:
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            return result['choices'][0]['message']['content']
            
        except requests.exceptions.RequestException as e:
            print(f"❌ API call failed: {e}")
            return f"[API Error: {str(e)}]"
    
    def simulate_conversation(self, persona_pair: Dict, num_rounds: int = 10) -> Dict:
        """
        模拟一次 speed dating 对话
        
        Args:
            persona_pair: 包含 person1 和 person2 的 persona 数据
            num_rounds: 对话轮数（默认 10 轮）
        
        Returns:
            完整对话记录
        """
        pair_id = persona_pair['pair_id']
        person1 = persona_pair['person1']
        person2 = persona_pair['person2']
        
        print(f"\n{'='*70}")
        print(f"🎭 Simulating Speed Date: {pair_id}")
        print(f"   Person 1 (iid={person1['iid']}): {person1['age']}yo, gender={person1['gender']}")
        print(f"   Person 2 (iid={person2['iid']}): {person2['age']}yo, gender={person2['gender']}")
        print(f"   Ground Truth: {'MATCHED' if persona_pair['ground_truth']['match'] == 1 else 'NOT MATCHED'}")
        print(f"{'='*70}")
        
        # 初始化对话历史
        conversation = {
            'pair_id': pair_id,
            'person1_iid': person1['iid'],
            'person2_iid': person2['iid'],
            'ground_truth': persona_pair['ground_truth'],
            'timestamp': datetime.now().isoformat(),
            'rounds': []
        }
        
        # Person 1 和 Person 2 的对话历史（用于 API 调用）
        person1_history = [{"role": "system", "content": person1['system_prompt']}]
        person2_history = [{"role": "system", "content": person2['system_prompt']}]
        
        # 开场白：Person 1 先说话
        print(f"\n🎬 Round 0: Opening (Person 1 speaks first)")
        
        opening_prompt = "You just sat down at the speed dating table. Introduce yourself and start the conversation naturally. Remember to be yourself!"
        person1_history.append({"role": "user", "content": opening_prompt})
        
        person1_opening = self.call_openrouter_api(person1_history, temperature=0.8)
        person1_history.append({"role": "assistant", "content": person1_opening})
        
        print(f"   Person 1: {person1_opening}")
        
        conversation['rounds'].append({
            'round': 0,
            'speaker': 'person1',
            'message': person1_opening,
            'type': 'opening'
        })
        
        time.sleep(1)  # 避免 API rate limit
        
        # 10 轮对话
        for round_num in range(1, num_rounds + 1):
            print(f"\n🔄 Round {round_num}:")
            
            # Person 2 回应
            person2_history.append({"role": "user", "content": person1_history[-1]['content']})
            person2_response = self.call_openrouter_api(person2_history, temperature=0.7)
            person2_history.append({"role": "assistant", "content": person2_response})
            
            print(f"   Person 2: {person2_response}")
            
            conversation['rounds'].append({
                'round': round_num,
                'speaker': 'person2',
                'message': person2_response
            })
            
            time.sleep(1)
            
            # Person 1 回应
            person1_history.append({"role": "user", "content": person2_response})
            person1_response = self.call_openrouter_api(person1_history, temperature=0.7)
            person1_history.append({"role": "assistant", "content": person1_response})
            
            print(f"   Person 1: {person1_response}")
            
            conversation['rounds'].append({
                'round': round_num,
                'speaker': 'person1',
                'message': person1_response
            })
            
            time.sleep(1)
        
        print(f"\n✅ Conversation completed: {len(conversation['rounds'])} exchanges")
        
        return conversation
    
    def simulate_all_pairs(self, num_pairs: Optional[int] = None, start_from: int = 0):
        """
        模拟所有配对的对话
        
        Args:
            num_pairs: 要模拟的配对数量（None = 全部）
            start_from: 从第几对开始（用于断点续传）
        """
        pairs_to_simulate = self.personas[start_from:start_from + num_pairs] if num_pairs else self.personas[start_from:]
        
        print(f"\n🚀 Starting simulation for {len(pairs_to_simulate)} pairs...")
        print(f"   Model: {self.model}")
        print(f"   API: OpenRouter")
        
        for i, persona_pair in enumerate(pairs_to_simulate):
            actual_index = start_from + i
            print(f"\n📍 Progress: {actual_index + 1}/{len(self.personas)}")
            
            try:
                conversation = self.simulate_conversation(persona_pair, num_rounds=10)
                self.conversations.append(conversation)
                
                # 每完成 5 对就保存一次（防止丢失）
                if (i + 1) % 5 == 0:
                    self.save_conversations(output_dir="results", checkpoint=True)
                    print(f"💾 Checkpoint saved: {i + 1} conversations")
                
            except Exception as e:
                print(f"❌ Error simulating pair {persona_pair['pair_id']}: {e}")
                continue
        
        print(f"\n🎉 All simulations completed!")
        print(f"   Total conversations: {len(self.conversations)}")
    
    def save_conversations(self, output_dir: str = "results", checkpoint: bool = False):
        """
        保存对话日志
        
        Args:
            output_dir: 输出目录
            checkpoint: 是否为 checkpoint（中间保存）
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # 保存完整对话 JSON
        if checkpoint:
            filename = f"conversations_checkpoint_{len(self.conversations)}.json"
        else:
            filename = "conversations.json"
        
        conversations_path = output_path / filename
        with open(conversations_path, 'w', encoding='utf-8') as f:
            json.dump(self.conversations, f, indent=2, ensure_ascii=False)
        
        if not checkpoint:
            print(f"\n💾 Saved conversations to {conversations_path}")
            
            # 保存可读文本版本（前 3 对示例）
            sample_path = output_path / "conversations_sample.txt"
            with open(sample_path, 'w', encoding='utf-8') as f:
                for conv in self.conversations[:3]:
                    f.write(f"{'='*80}\n")
                    f.write(f"PAIR: {conv['pair_id']}\n")
                    f.write(f"Person 1 (iid={conv['person1_iid']}) vs Person 2 (iid={conv['person2_iid']})\n")
                    f.write(f"Ground Truth: {'MATCHED' if conv['ground_truth']['match'] == 1 else 'NOT MATCHED'}\n")
                    f.write(f"Timestamp: {conv['timestamp']}\n")
                    f.write(f"{'='*80}\n\n")
                    
                    for round_data in conv['rounds']:
                        speaker_label = f"Person {round_data['speaker'][-1]}"
                        f.write(f"[Round {round_data['round']}] {speaker_label}:\n")
                        f.write(f"{round_data['message']}\n\n")
                    
                    f.write(f"\n\n")
            
            print(f"💾 Saved sample conversations to {sample_path}")
        
        return output_path


def main():
    """主函数"""
    print("🎭 Speed Dating Simulator - Phase 2")
    print("=" * 70)
    
    # 检查 API key
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        print("❌ Error: OPENROUTER_API_KEY not found in environment variables")
        print("Please set it using: export OPENROUTER_API_KEY='your-key-here'")
        return
    
    # 初始化模拟器
    personas_path = "results/personas.json"
    simulator = SpeedDatingSimulator(personas_path, api_key=api_key)
    
    # 加载 personas
    simulator.load_personas()
    
    # 选择模拟模式
    print("\n📋 Simulation Mode:")
    print("   1. Test mode (first 2 pairs)")
    print("   2. Small batch (first 10 pairs)")
    print("   3. Full simulation (all 100 pairs)")
    
    mode = input("Select mode (1/2/3): ").strip()
    
    if mode == "1":
        num_pairs = 2
    elif mode == "2":
        num_pairs = 10
    elif mode == "3":
        num_pairs = None  # All pairs
    else:
        print("Invalid mode, using test mode (2 pairs)")
        num_pairs = 2
    
    # 开始模拟
    simulator.simulate_all_pairs(num_pairs=num_pairs)
    
    # 保存结果
    output_dir = simulator.save_conversations()
    
    print("\n" + "=" * 70)
    print("✅ Speed dating simulation completed!")
    print(f"📁 Results saved to: {output_dir}")
    print("\n🎯 Next steps:")
    print("   1. Review conversations_sample.txt")
    print("   2. Run evaluation_system.py to analyze compatibility")
    print("   3. Compare with ground truth")


if __name__ == "__main__":
    main()
