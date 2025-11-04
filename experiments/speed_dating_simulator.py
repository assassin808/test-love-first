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
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# 加载 .env 文件
load_dotenv()

# Thread-safe lock for saving checkpoints
save_lock = threading.Lock()


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
    
    def call_openrouter_api(self, messages: List[Dict], temperature: float = 0.7, max_retries: int = 3) -> Dict[str, str]:
        """
        调用 OpenRouter API with retry logic
        
        Args:
            messages: 消息列表 [{"role": "system/user/assistant", "content": "..."}]
            temperature: 温度参数 (0-1)
            max_retries: 最大重试次数
        
        Returns:
            字典包含 'inner_thought' 和 'response'
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 300  # 增加 token 以容纳 inner thought + response
        }
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.api_url,
                    headers=self.headers,
                    json=payload,
                    timeout=30
                )
                response.raise_for_status()
                
                result = response.json()
                full_response = result['choices'][0]['message']['content']
                
                # 解析 <INNER_THOUGHT> 和 <RESPONSE>
                inner_thought = ""
                response_text = full_response
                
                if "<INNER_THOUGHT>" in full_response and "</INNER_THOUGHT>" in full_response:
                    thought_start = full_response.find("<INNER_THOUGHT>") + len("<INNER_THOUGHT>")
                    thought_end = full_response.find("</INNER_THOUGHT>")
                    inner_thought = full_response[thought_start:thought_end].strip()
                
                if "<RESPONSE>" in full_response and "</RESPONSE>" in full_response:
                    response_start = full_response.find("<RESPONSE>") + len("<RESPONSE>")
                    response_end = full_response.find("</RESPONSE>")
                    response_text = full_response[response_start:response_end].strip()
                elif "<RESPONSE>" in full_response:
                    # 如果只有开头标签，取到末尾
                    response_start = full_response.find("<RESPONSE>") + len("<RESPONSE>")
                    response_text = full_response[response_start:].strip()
                
                return {
                    'inner_thought': inner_thought,
                    'response': response_text,
                    'full_text': full_response
                }
                
            except requests.exceptions.RequestException as e:
                wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                
                if attempt < max_retries - 1:
                    print(f"⚠️ API call failed (attempt {attempt + 1}/{max_retries}): {e}")
                    print(f"   Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"❌ API call failed after {max_retries} attempts: {e}")
                    return {
                        'inner_thought': '',
                        'response': f"[API Error after {max_retries} retries: {str(e)}]",
                        'full_text': f"[API Error after {max_retries} retries: {str(e)}]"
                    }
            
            except Exception as e:
                print(f"❌ Unexpected error in API call: {e}")
                return {
                    'inner_thought': '',
                    'response': f"[Unexpected Error: {str(e)}]",
                    'full_text': f"[Unexpected Error: {str(e)}]"
                }
    
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
        
        opening_prompt = """The bell just rang! You're sitting across from a COMPLETE STRANGER at the speed dating table. This is your first impression. 

Start with your <INNER_THOUGHT> (what's your strategy? what tone will you take based on YOUR persona?), then give your <RESPONSE> (introduce yourself naturally - remember you DON'T know this person at all!)"""
        person1_history.append({"role": "user", "content": opening_prompt})
        
        person1_result = self.call_openrouter_api(person1_history, temperature=0.8)
        person1_history.append({"role": "assistant", "content": person1_result['full_text']})
        
        print(f"   Person 1 [THINKING]: {person1_result['inner_thought'][:80]}...")
        print(f"   Person 1 [SAYS]: {person1_result['response']}")
        
        conversation['rounds'].append({
            'round': 0,
            'speaker': 'person1',
            'inner_thought': person1_result['inner_thought'],
            'message': person1_result['response'],
            'type': 'opening'
        })
        
        time.sleep(1)  # 避免 API rate limit
        
        # 10 轮对话
        for round_num in range(1, num_rounds + 1):
            print(f"\n🔄 Round {round_num}:")
            
            # Person 2 回应（看到 Person 1 的 response 部分）
            person2_history.append({"role": "user", "content": person1_result['response']})
            person2_result = self.call_openrouter_api(person2_history, temperature=0.7)
            person2_history.append({"role": "assistant", "content": person2_result['full_text']})
            
            print(f"   Person 2 [THINKING]: {person2_result['inner_thought'][:80]}...")
            print(f"   Person 2 [SAYS]: {person2_result['response']}")
            
            conversation['rounds'].append({
                'round': round_num,
                'speaker': 'person2',
                'inner_thought': person2_result['inner_thought'],
                'message': person2_result['response']
            })
            
            time.sleep(1)
            
            # Person 1 回应（看到 Person 2 的 response 部分）
            person1_history.append({"role": "user", "content": person2_result['response']})
            person1_result = self.call_openrouter_api(person1_history, temperature=0.7)
            person1_history.append({"role": "assistant", "content": person1_result['full_text']})
            
            print(f"   Person 1 [THINKING]: {person1_result['inner_thought'][:80]}...")
            print(f"   Person 1 [SAYS]: {person1_result['response']}")
            
            conversation['rounds'].append({
                'round': round_num,
                'speaker': 'person1',
                'inner_thought': person1_result['inner_thought'],
                'message': person1_result['response']
            })
            
            time.sleep(1)
        
        print(f"\n✅ Conversation completed: {len(conversation['rounds'])} exchanges")
        
        # === POST-DATE EVALUATION ===
        print(f"\n📋 Collecting evaluations...")
        
        # 1. Person 1 自己判断要不要再一起
        person1_eval_prompt = """The bell just rang - the speed date is over. 

Now, privately evaluate this person and decide if you want to see them again.

Use this format:
<INNER_THOUGHT>
[Reflect on the conversation: what did you like/dislike? Did they align with YOUR values (check your persona)? Any red flags? Chemistry?]
</INNER_THOUGHT>

<DECISION>
YES or NO - Do you want to see them again?
</DECISION>

<REASONING>
[1-2 sentences: Why yes or no?]
</REASONING>"""
        
        person1_history.append({"role": "user", "content": person1_eval_prompt})
        person1_eval = self.call_openrouter_api(person1_history, temperature=0.6)
        
        # Debug: Print raw response
        print(f"\n🔍 DEBUG - Person 1 Eval Response:")
        print(f"   Full text: {person1_eval.get('full_text', '')[:200]}...")
        
        # 2. Person 2 自己判断要不要再一起
        person2_eval_prompt = """The bell just rang - the speed date is over. 

Now, privately evaluate this person and decide if you want to see them again.

Use this format:
<INNER_THOUGHT>
[Reflect on the conversation: what did you like/dislike? Did they align with YOUR values (check your persona)? Any red flags? Chemistry?]
</INNER_THOUGHT>

<DECISION>
YES or NO - Do you want to see them again?
</DECISION>

<REASONING>
[1-2 sentences: Why yes or no?]
</REASONING>"""
        
        person2_history.append({"role": "user", "content": person2_eval_prompt})
        person2_eval = self.call_openrouter_api(person2_history, temperature=0.6)
        
        # Debug: Print raw response
        print(f"\n🔍 DEBUG - Person 2 Eval Response:")
        print(f"   Full text: {person2_eval.get('full_text', '')[:200]}...")
        
        # 3. 恋爱观察员 Observer Evaluation
        observer_prompt = f"""You are a professional dating coach observing this speed date conversation. Analyze the compatibility between these two people.

PERSON 1 BACKGROUND:
{person1['persona_narrative'][:500]}...

PERSON 2 BACKGROUND:
{person2['persona_narrative'][:500]}...

CONVERSATION SUMMARY:
{len(conversation['rounds'])} exchanges covering topics like: {self._extract_topics(conversation)}

Provide your expert evaluation:

<COMPATIBILITY_SCORE>
[0-10, where 0=terrible match, 10=perfect match]
</COMPATIBILITY_SCORE>

<ANALYSIS>
[3-4 sentences analyzing:
1. Shared values/interests alignment
2. Conversation chemistry and engagement
3. Long-term compatibility potential
4. Any red flags or concerns]
</ANALYSIS>

<PREDICTION>
MATCH or NO_MATCH - Will they choose to see each other again?
</PREDICTION>"""
        
        observer_history = [{"role": "user", "content": observer_prompt}]
        observer_eval = self.call_openrouter_api(observer_history, temperature=0.5)
        
        # Debug: Print raw response
        print(f"\n🔍 DEBUG - Observer Eval Response:")
        print(f"   Full text: {observer_eval.get('full_text', '')[:300]}...")
        
        # Add evaluations to conversation
        conversation['evaluations'] = {
            'person1_self_evaluation': {
                'inner_thought': person1_eval.get('inner_thought', ''),
                'decision': self._extract_decision(person1_eval.get('full_text', '')),
                'reasoning': self._extract_reasoning(person1_eval.get('full_text', ''))
            },
            'person2_self_evaluation': {
                'inner_thought': person2_eval.get('inner_thought', ''),
                'decision': self._extract_decision(person2_eval.get('full_text', '')),
                'reasoning': self._extract_reasoning(person2_eval.get('full_text', ''))
            },
            'observer_evaluation': {
                'compatibility_score': self._extract_score(observer_eval.get('full_text', '')),
                'analysis': self._extract_analysis(observer_eval.get('full_text', '')),
                'prediction': self._extract_prediction(observer_eval.get('full_text', ''))
            }
        }
        
        print(f"   Person 1 Decision: {conversation['evaluations']['person1_self_evaluation']['decision']}")
        print(f"   Person 2 Decision: {conversation['evaluations']['person2_self_evaluation']['decision']}")
        print(f"   Observer Score: {conversation['evaluations']['observer_evaluation']['compatibility_score']}/10")
        print(f"   Observer Prediction: {conversation['evaluations']['observer_evaluation']['prediction']}")
        
        return conversation
    
    def _extract_topics(self, conversation: Dict) -> str:
        """Extract main topics from conversation"""
        # Simple keyword extraction from messages
        all_text = " ".join([r['message'] for r in conversation['rounds'][:5]])
        topics = []
        keywords = ['sports', 'basketball', 'music', 'book', 'work', 'travel', 'art', 'science', 'movie', 'game']
        for kw in keywords:
            if kw in all_text.lower():
                topics.append(kw)
        return ", ".join(topics[:3]) if topics else "general interests"
    
    def _extract_decision(self, text: str) -> str:
        """Extract YES/NO decision from evaluation"""
        # Try exact tag match first
        if '<DECISION>' in text and '</DECISION>' in text:
            start = text.find('<DECISION>') + len('<DECISION>')
            end = text.find('</DECISION>')
            decision = text[start:end].strip().upper()
            return 'YES' if 'YES' in decision else 'NO'
        
        # Fallback: search for YES/NO patterns in text
        text_upper = text.upper()
        if 'DECISION' in text_upper or 'WANT TO SEE' in text_upper:
            # Look for explicit YES or NO after "DECISION" keyword
            decision_idx = text_upper.find('DECISION')
            if decision_idx != -1:
                snippet = text_upper[decision_idx:decision_idx+100]
                if 'YES' in snippet and 'NO' not in snippet:
                    return 'YES'
                elif 'NO' in snippet:
                    return 'NO'
        
        return 'UNKNOWN'
    
    def _extract_reasoning(self, text: str) -> str:
        """Extract reasoning from evaluation"""
        if '<REASONING>' in text and '</REASONING>' in text:
            start = text.find('<REASONING>') + len('<REASONING>')
            end = text.find('</REASONING>')
            return text[start:end].strip()
        elif '<REASONING>' in text:
            start = text.find('<REASONING>') + len('<REASONING>')
            return text[start:].strip()
        return ""
    
    def _extract_score(self, text: str) -> str:
        """Extract compatibility score"""
        import re
        
        # Try exact tag match first
        if '<COMPATIBILITY_SCORE>' in text and '</COMPATIBILITY_SCORE>' in text:
            start = text.find('<COMPATIBILITY_SCORE>') + len('<COMPATIBILITY_SCORE>')
            end = text.find('</COMPATIBILITY_SCORE>')
            score = text[start:end].strip()
            match = re.search(r'\d+', score)
            return match.group() if match else "N/A"
        
        # Fallback: look for patterns like "score: 7" or "7/10" or "7 out of 10"
        patterns = [
            r'(?:score|rating)[:\s]+(\d+)',
            r'(\d+)\s*(?:/|out of)\s*10',
            r'compatibility[:\s]+(\d+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                score = int(match.group(1))
                if 0 <= score <= 10:
                    return str(score)
        
        return "N/A"
    
    def _extract_analysis(self, text: str) -> str:
        """Extract analysis from observer evaluation"""
        if '<ANALYSIS>' in text and '</ANALYSIS>' in text:
            start = text.find('<ANALYSIS>') + len('<ANALYSIS>')
            end = text.find('</ANALYSIS>')
            return text[start:end].strip()
        elif '<ANALYSIS>' in text:
            start = text.find('<ANALYSIS>') + len('<ANALYSIS>')
            return text[start:].strip()
        return ""
    
    def _extract_prediction(self, text: str) -> str:
        """Extract match prediction from observer"""
        # Try exact tag match first
        if '<PREDICTION>' in text and '</PREDICTION>' in text:
            start = text.find('<PREDICTION>') + len('<PREDICTION>')
            end = text.find('</PREDICTION>')
            pred = text[start:end].strip().upper()
            return 'MATCH' if 'MATCH' in pred and 'NO_MATCH' not in pred else 'NO_MATCH'
        
        # Fallback: search for match prediction keywords
        text_upper = text.upper()
        if 'PREDICTION' in text_upper:
            pred_idx = text_upper.find('PREDICTION')
            snippet = text_upper[pred_idx:pred_idx+150]
            
            # Check for NO_MATCH or "NO MATCH" first (more specific)
            if 'NO_MATCH' in snippet or 'NO MATCH' in snippet or "WON'T MATCH" in snippet:
                return 'NO_MATCH'
            elif 'MATCH' in snippet:
                return 'MATCH'
        
        # Last resort: check entire text
        if 'NO_MATCH' in text_upper or 'NO MATCH' in text_upper:
            return 'NO_MATCH'
        elif 'MATCH' in text_upper and 'WILL' in text_upper:
            return 'MATCH'
        
        return 'UNKNOWN'
    
    def simulate_all_pairs(self, num_pairs: Optional[int] = None, start_from: int = 0, max_workers: int = 10):
        """
        模拟所有配对的对话（支持多线程）
        
        Args:
            num_pairs: 要模拟的配对数量（None = 全部）
            start_from: 从第几对开始（用于断点续传）
            max_workers: 并行线程数（默认3，避免 API rate limit）
        """
        pairs_to_simulate = self.personas[start_from:start_from + num_pairs] if num_pairs else self.personas[start_from:]
        
        print(f"\n🚀 Starting simulation for {len(pairs_to_simulate)} pairs...")
        print(f"   Model: {self.model}")
        print(f"   API: OpenRouter")
        print(f"   Parallel threads: {max_workers}")
        
        completed_count = 0
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_pair = {
                executor.submit(self.simulate_conversation, persona_pair, 10): (i, persona_pair)
                for i, persona_pair in enumerate(pairs_to_simulate)
            }
            
            # Process completed tasks
            for future in as_completed(future_to_pair):
                i, persona_pair = future_to_pair[future]
                actual_index = start_from + i
                
                try:
                    conversation = future.result()
                    
                    # Thread-safe append
                    with save_lock:
                        self.conversations.append(conversation)
                        completed_count += 1
                    
                    print(f"\n✅ Completed: {actual_index + 1}/{len(self.personas)} - {persona_pair['pair_id']}")
                    
                    # 每完成 5 对就保存一次（防止丢失）
                    if completed_count % 5 == 0:
                        with save_lock:
                            self.save_conversations(output_dir="results", checkpoint=True)
                            print(f"💾 Checkpoint saved: {completed_count} conversations")
                    
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
                    
                    # Conversation rounds
                    for round_data in conv['rounds']:
                        speaker_label = f"Person {round_data['speaker'][-1]}"
                        f.write(f"[Round {round_data['round']}] {speaker_label}:\n")
                        
                        # 显示内心想法
                        if 'inner_thought' in round_data and round_data['inner_thought']:
                            f.write(f"💭 {round_data['inner_thought']}\n")
                        
                        # 显示实际说的话
                        f.write(f"💬 {round_data['message']}\n\n")
                    
                    # Post-date evaluations
                    if 'evaluations' in conv:
                        f.write(f"\n{'='*80}\n")
                        f.write(f"POST-DATE EVALUATIONS\n")
                        f.write(f"{'='*80}\n\n")
                        
                        eval_data = conv['evaluations']
                        
                        # Person 1 self-evaluation
                        f.write(f"👤 PERSON 1 DECISION: {eval_data['person1_self_evaluation']['decision']}\n")
                        f.write(f"   Reasoning: {eval_data['person1_self_evaluation']['reasoning']}\n\n")
                        
                        # Person 2 self-evaluation
                        f.write(f"👤 PERSON 2 DECISION: {eval_data['person2_self_evaluation']['decision']}\n")
                        f.write(f"   Reasoning: {eval_data['person2_self_evaluation']['reasoning']}\n\n")
                        
                        # Observer evaluation
                        f.write(f"👁️ OBSERVER EVALUATION:\n")
                        f.write(f"   Compatibility Score: {eval_data['observer_evaluation']['compatibility_score']}/10\n")
                        f.write(f"   Prediction: {eval_data['observer_evaluation']['prediction']}\n")
                        f.write(f"   Analysis: {eval_data['observer_evaluation']['analysis']}\n\n")
                        
                        # Compare with ground truth
                        p1_match = eval_data['person1_self_evaluation']['decision'] == 'YES'
                        p2_match = eval_data['person2_self_evaluation']['decision'] == 'YES'
                        llm_mutual = p1_match and p2_match
                        actual_match = conv['ground_truth']['match'] == 1
                        
                        f.write(f"📊 COMPARISON:\n")
                        f.write(f"   LLM Match: {'YES' if llm_mutual else 'NO'} (P1: {eval_data['person1_self_evaluation']['decision']}, P2: {eval_data['person2_self_evaluation']['decision']})\n")
                        f.write(f"   Real Match: {'YES' if actual_match else 'NO'}\n")
                        f.write(f"   Prediction Correct: {'✅' if llm_mutual == actual_match else '❌'}\n")
                    
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
    print("   1. Test mode (1 pair, 2 rounds only - FAST)")
    print("   2. Small batch (first 10 pairs, 10 rounds, 3 threads)")
    print("   3. Full simulation (all 100 pairs, 10 rounds, 5 threads)")
    
    mode = input("Select mode (1/2/3): ").strip()
    
    if mode == "1":
        num_pairs = 1
        max_workers = 1  # Single thread for testing
        num_rounds = 2   # Quick test
    elif mode == "2":
        num_pairs = 10
        max_workers = 3
        num_rounds = 10
    elif mode == "3":
        num_pairs = None  # All pairs
        max_workers = 10  # More threads for full run
        num_rounds = 10
    else:
        print("Invalid mode, using test mode (1 pair, 2 rounds)")
        num_pairs = 1
        max_workers = 1
        num_rounds = 2
    
    # 开始模拟
    print(f"\n⚙️ Settings: {num_pairs or 'all'} pairs, {num_rounds} rounds per pair, {max_workers} threads")
    
    # Need to modify simulate_all_pairs to pass num_rounds
    if mode == "1":
        # For test mode, simulate single pair directly
        print(f"\n🚀 Starting test simulation...")
        print(f"   Model: {simulator.model}")
        print(f"   API: OpenRouter")
        
        conversation = simulator.simulate_conversation(simulator.personas[0], num_rounds=num_rounds)
        simulator.conversations.append(conversation)
        print(f"\n✅ Test simulation completed!")
    else:
        simulator.simulate_all_pairs(num_pairs=num_pairs, max_workers=max_workers)
    
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
