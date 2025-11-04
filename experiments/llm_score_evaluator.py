"""
LLM Score-based Evaluator

重新评估LLM方法，使用0-10分评分而非二元决策：
1. 参与者评分：每个参与者给出0-10分的见面意愿
2. 观察员评分：观察员给出0-10分的匹配度
3. 计算概率分数用于AUC计算：
   - 参与者方法：两个参与者分数的乘积 (normalized to 0-1)
   - 观察员方法：观察员分数 (normalized to 0-1)
"""

import json
import re
import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    precision_recall_curve, roc_curve
)

# 导入环境变量和API客户端
import requests
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    # If python-dotenv isn't installed, we'll try a manual fallback below
    pass

def _load_env_fallback():
    """Minimal .env loader to ensure OPENROUTER_API_KEY is available without python-dotenv.
    Searches common locations relative to CWD and this file.
    """
    possible_paths = [
        Path.cwd() / '.env',
        Path(__file__).resolve().parent / '.env',
        Path(__file__).resolve().parent.parent / '.env',
    ]
    for p in possible_paths:
        try:
            if p.exists():
                with p.open('r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith('#') or '=' not in line:
                            continue
                        k, v = line.split('=', 1)
                        k = k.strip()
                        v = v.strip().strip('"').strip("'")
                        if k and v and os.getenv(k) is None:
                            os.environ[k] = v
        except Exception:
            continue

# OpenRouter API 配置
if os.getenv('OPENROUTER_API_KEY') is None:
    _load_env_fallback()
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "mistralai/mistral-nemo"

if not OPENROUTER_API_KEY:
    print("❌ Error: OPENROUTER_API_KEY not found in environment variables")
    print("Please set it in .env file or environment")
    sys.exit(1)

def call_openrouter_api(messages: List[Dict], temperature: float = 0.3, max_retries: int = 10, max_tokens: int = 300, repetition_penalty: float = 1.0) -> str:
    """
    调用 OpenRouter API with robust retry and exponential backoff
    
    Args:
        messages: 消息列表 [{"role": "system/user/assistant", "content": "..."}]
        temperature: 温度参数 (0-1)
        max_retries: 最大重试次数 (default 10)
        max_tokens: 最大输出token数 (default 300)
        repetition_penalty: 重复惩罚 (1.0 = no penalty, >1.0 = penalize repetition)
    
    Returns:
        API响应内容
    """
    import time
    import random
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "https://github.com/your-repo",
        "X-Title": "Speed Dating Score Evaluation"
    }
    
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    
    # Add repetition_penalty if not default
    if repetition_penalty != 1.0:
        payload["repetition_penalty"] = repetition_penalty
    
    # Debug: Print API parameters on first attempt
    if max_tokens == 300 and repetition_penalty == 1.0:
        print(f"      [API] model={MODEL}, temp={temperature}, max_tokens={max_tokens}")
    elif repetition_penalty != 1.0:
        print(f"      [API] model={MODEL}, temp={temperature}, max_tokens={max_tokens}, rep_penalty={repetition_penalty}")
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                API_URL,
                headers=headers,
                json=payload,
                timeout=45  # Increased timeout for high concurrency
            )
            response.raise_for_status()
            
            result = response.json()
            return result['choices'][0]['message']['content']
            
        except requests.exceptions.RequestException as e:
            # Exponential backoff with jitter to avoid thundering herd
            base_wait = 2 ** attempt
            jitter = random.uniform(0, min(base_wait * 0.3, 3))  # Up to 30% jitter, max 3s
            wait_time = base_wait + jitter
            
            if attempt < max_retries - 1:
                print(f"⚠️ API call failed (attempt {attempt + 1}/{max_retries}): {e}")
                print(f"   Retrying in {wait_time:.1f}s...")
                time.sleep(wait_time)
            else:
                print(f"❌ API call failed after {max_retries} attempts: {e}")
                return f"[API Error: {str(e)}]"
        
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return f"[Error: {str(e)}]"

def extract_score_from_response(response: str, score_type: str = "participant") -> float:
    """
    从LLM响应中提取0-10分的评分
    
    Args:
        response: LLM的响应文本
        score_type: "participant" 或 "observer"
    
    Returns:
        0-10之间的分数，如果提取失败返回5.0（中间值）
    """
    # 寻找评分模式（支持中英文）
    patterns = [
        # Chinese formats
        r'评分[：:]\s*(\d+(?:\.\d+)?)',
        r'分数[：:]\s*(\d+(?:\.\d+)?)',
        r'得分[：:]\s*(\d+(?:\.\d+)?)',
        r'(\d+(?:\.\d+)?)\s*分',
        # English formats
        r'score\s*[:=]\s*(\d+(?:\.\d+)?)(?:\s*/\s*10)?',
        r'rating\s*[:=]\s*(\d+(?:\.\d+)?)(?:\s*/\s*10)?',
        r'match\s*score\s*[:=]\s*(\d+(?:\.\d+)?)(?:\s*/\s*10)?',
        # Generic X/10 at any place
        r'(\d+(?:\.\d+)?)(?:\s*/\s*10)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            try:
                score = float(match.group(1))
                if 0 <= score <= 10:
                    return score
            except ValueError:
                continue
    
    # 如果找不到评分，返回None表示需要重试
    return None


def call_api_with_score_extraction(messages: List[Dict], temperature: float = 0.3, score_type: str = "participant") -> float:
    """
    调用API并提取评分，如果失败则用更长的token和更高的repetition penalty重试
    
    Args:
        messages: 消息列表
        temperature: 温度参数
        score_type: 评分类型 ("participant" 或 "observer")
    
    Returns:
        0-10之间的分数
    
    Raises:
        ValueError: 如果所有尝试都无法提取有效评分
    """
    # Progressive strategies: (max_tokens, repetition_penalty)
    # 1st: Normal (300 tokens, no penalty)
    # 2nd: More tokens (500 tokens, no penalty)
    # 3rd: More tokens + penalty (800 tokens, 1.1 penalty)
    # 4th: Strong penalty (800 tokens, 1.2 penalty)
    strategies = [
        (300, 1.0),
        (500, 1.0),
        (800, 1.1),
        (800, 1.2)
    ]
    
    for attempt, (max_tokens, rep_penalty) in enumerate(strategies):
        if rep_penalty == 1.0:
            print(f"   Attempt {attempt + 1}/{len(strategies)} with max_tokens={max_tokens}...")
        else:
            print(f"   Attempt {attempt + 1}/{len(strategies)} with max_tokens={max_tokens}, repetition_penalty={rep_penalty}...")
        
        response = call_openrouter_api(messages, temperature=temperature, max_tokens=max_tokens, repetition_penalty=rep_penalty)
        
        # Check for API errors
        if response.startswith("[API Error:") or response.startswith("[Error:"):
            if attempt < len(strategies) - 1:
                print(f"   API error, retrying with different strategy...")
                continue
            else:
                raise ValueError(f"API call failed after all attempts: {response}")
        
        # Check for repetitive token pattern (sign of truncation/repetition issue)
        # If we see the same word repeated 5+ times, it's likely truncated or needs repetition penalty
        is_repetitive = False
        words = response.split()
        if len(words) > 10:
            last_words = words[-10:]
            for word in set(last_words):
                if last_words.count(word) >= 5:
                    print(f"   ⚠️ Detected repetitive pattern ('{word}' x{last_words.count(word)})")
                    is_repetitive = True
                    break
        
        # Try to extract score
        score = extract_score_from_response(response, score_type)
        
        # If repetitive pattern detected and no valid score, force retry with stronger penalty
        if is_repetitive and score is None:
            if attempt < len(strategies) - 1:
                print(f"   Retrying with higher repetition penalty to fix repetition issue...")
                continue
        
        if score is not None:
            if attempt > 0:
                print(f"   ✅ Successfully extracted score on attempt {attempt + 1}")
            return score
        else:
            if attempt < len(strategies) - 1:
                print(f"   ⚠️ Could not extract score from response, trying next strategy...")
                print(f"   Response preview: {response[:200]}...")
            else:
                print(f"   ❌ Failed to extract score after all attempts")
                print(f"   Final response: {response}")
                raise ValueError(f"Could not extract valid score from response after {len(strategies)} attempts with progressive strategies")
    
    # This should never be reached, but just in case
    raise ValueError("Unexpected error in score extraction retry logic")


def build_role_messages_from_conversation(conversation: List[Dict], perspective: str) -> List[Dict]:
    """
    Build OpenRouter chat messages from a conversation list.
    - perspective: 'person1' or 'person2' to decide which side becomes 'user'.
    - If speaker labels exist, map accordingly; otherwise alternate roles.
    """
    messages: List[Dict] = []
    has_speaker = any(isinstance(m, dict) and m.get('speaker') for m in conversation)
    if has_speaker:
        for m in conversation:
            if not isinstance(m, dict):
                text = str(m).strip()
                if text:
                    # default alternation if unknown
                    messages.append({"role": "user", "content": text})
                continue
            text = (m.get('content') or m.get('message') or '').strip()
            if not text:
                continue
            spk = m.get('speaker')
            if spk in ("person1", "p1"):  # map to role based on perspective
                role = "user" if perspective == "person1" else "assistant"
            elif spk in ("person2", "p2"):
                role = "user" if perspective == "person2" else "assistant"
            else:
                role = "user"
            messages.append({"role": role, "content": text})
    else:
        # Alternate roles, assume person1 starts
        next_role_for_p1 = "user" if perspective == "person1" else "assistant"
        next_role_for_p2 = "assistant" if next_role_for_p1 == "user" else "user"
        toggle = 0
        for m in conversation:
            text = m.get('content') if isinstance(m, dict) else str(m)
            text = (text or '').strip()
            if not text:
                continue
            role = next_role_for_p1 if toggle % 2 == 0 else next_role_for_p2
            messages.append({"role": role, "content": text})
            toggle += 1
    return messages


def build_participant_messages(person_data: Dict, partner_data: Dict, conversation: List[Dict], perspective: str) -> List[Dict]:
    """
    Build messages for participant scoring using structured chat history + final instruction.
    Output order: reasoning first, then a separate line with 'Score: X.X/10'.
    perspective: 'person1' or 'person2'
    """
    persona_block = (person_data.get('persona_narrative') or '').strip()
    msgs = [
        {"role": "system", "content": (
            "You are the participant (Person A) in a speed dating session. Reply in English. "
            "Output format and order:\n"
            "1) Start with a private inner monologue wrapped in <INNER_THOUGHT>...</INNER_THOUGHT> reflecting your feelings, chemistry, alignment, and any red flags based on YOUR persona.\n"
            "2) Then provide 1-2 short sentences of outward reasoning as Person A (visible explanation).\n"
            "3) Then include one line specifying the rubric: 'Rubric: 0 = absolutely unwilling to meet again (no chemistry, clear misalignment, or red flags); 5 = indifferent — could meet or not (could take it or leave it); 10 = perfect match, nothing can break us! (strong chemistry, aligned values/interests, clear mutual interest)'.\n"
            "4) On a new separate last line, write exactly: 'Score: <number>/10' where <number> can be any decimal (e.g., 4.3).\n"
            f"Your background: {persona_block}"
        )}
    ]
    msgs.extend(build_role_messages_from_conversation(conversation, perspective=perspective))
    msgs.append({
        "role": "user",
        "content": (
            "Now, based on the conversation, rate your willingness to meet the partner again on a 0-10 scale. "
            "Follow the required order: <INNER_THOUGHT>...</INNER_THOUGHT> (private), then 1-2 sentences of reasoning, then the rubric line, and finally the last line 'Score: <number>/10' (decimals allowed, e.g., 4.3)."
        )
    })
    return msgs


def build_observer_messages(person1_data: Dict, person2_data: Dict, conversation: List[Dict], icl_examples: List[Dict] = None) -> List[Dict]:
    """Build messages for observer scoring with conversation embedded in prompt text.
    Output order: reasoning first, then rubric line, then 'Score: X.X/10'.
    
    Args:
        person1_data: Person 1 data
        person2_data: Person 2 data
        conversation: Conversation messages
        icl_examples: Optional list of in-context learning examples with structure:
                     [{'person1': Dict, 'person2': Dict, 'conversation': List, 'match': bool}, ...]
    """
    p1 = (person1_data.get('persona_narrative') or '').strip()
    p2 = (person2_data.get('persona_narrative') or '').strip()
    
    # Build conversation transcript as plain text
    conversation_lines = []
    for idx, msg in enumerate(conversation, start=1):
        if isinstance(msg, dict):
            text = msg.get('content') or msg.get('message') or ''
            speaker = msg.get('speaker', f'Message {idx}')
        else:
            text = str(msg)
            speaker = f'Message {idx}'
        if text:
            conversation_lines.append(f"{speaker}: {text}")
    conversation_text = "\n".join(conversation_lines)
    
    # Build system prompt
    system_content = (
        "You are an experienced relationship observer. Reply in English. "
        "Output format and order:\n"
        "1) Provide 2-3 concise sentences explaining your assessment considering values/interests alignment, conversational flow, mutual attraction/chemistry, and long-term potential.\n"
        "2) Then include one line specifying the rubric: 'Rubric: 0 = absolutely unwilling to meet again (no chemistry, clear misalignment, or red flags); 5 = indifferent — could meet or not (could take it or leave it); 10 = perfect match, nothing can break them! (strong chemistry, aligned values/interests, clear mutual interest)'.\n"
        "3) On a new separate line at the very end, write exactly: 'Score: X.X/10' where X.X can be any decimal (e.g., 4.3)."
    )
    
    # Add in-context learning examples if provided
    if icl_examples:
        system_content += "\n\nHere are some example evaluations to calibrate your scoring:"
        for idx, ex in enumerate(icl_examples, start=1):
            # Use FULL background, NO conversation history (new ICL format)
            ex_p1_bg = ex.get('person1_background', '').strip()
            ex_p2_bg = ex.get('person2_background', '').strip()
            ex_p1_age = ex.get('person1_age', 'N/A')
            ex_p1_gender = ex.get('person1_gender', 'N/A')
            ex_p2_age = ex.get('person2_age', 'N/A')
            ex_p2_gender = ex.get('person2_gender', 'N/A')
            
            # Convert ground truth to score (match=10, no match=0)
            gt_score = 10.0 if ex['match'] else 0.0
            
            system_content += (
                f"\n\nExample {idx}:\n"
                f"Person 1 ({ex_p1_gender}, age {ex_p1_age}):\n{ex_p1_bg}\n\n"
                f"Person 2 ({ex_p2_gender}, age {ex_p2_age}):\n{ex_p2_bg}\n\n"
                f"Ground Truth: {'Match' if ex['match'] else 'No Match'} → Score: {gt_score}/10"
            )
    
    msgs = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": (
            f"Person 1 background: {p1}\n\n"
            f"Person 2 background: {p2}\n\n"
            f"Conversation transcript:\n{conversation_text}\n\n"
            "Now, rate the overall match quality on a 0-10 scale. "
            "Follow the required order: reasoning (2-3 sentences), then the rubric line, and finally the last line 'Score: X.X/10' (decimals allowed, e.g., 4.3)."
        )}
    ]
    return msgs


def get_participant_score_prompt(person_data: Dict, partner_data: Dict, conversation: List[Dict]) -> str:
    """Generate participant scoring prompt (0-10) in English with full conversation context."""
    # Build a simple transcript block
    lines = []
    for idx, msg in enumerate(conversation, start=1):
        text = msg.get('content') or msg.get('message') or ''
        if text:
            lines.append(f"Message {idx}: {text}")
    conversation_text = "\n".join(lines)

    persona_block = person_data.get('persona_narrative', '').strip()

    prompt = (
        "You are Person A in a speed dating session. Based on your own perspective and preferences, "
        "rate your willingness to meet the partner again after reading the entire conversation transcript.\n\n"
        f"Your background (for context): {persona_block}\n\n"
        "Conversation transcript:\n"
        f"{conversation_text}\n\n"
        "Please reply in English. Start your answer with a single line in the exact format:\n"
        "Score: X.X/10\n"
        "Then provide 1-2 short sentences explaining your reasoning."
    )
    return prompt


def get_observer_score_prompt(person1_data: Dict, person2_data: Dict, conversation: List[Dict]) -> str:
    """Generate observer scoring prompt (0-10) in English with full conversation context."""
    lines = []
    for idx, msg in enumerate(conversation, start=1):
        text = msg.get('content') or msg.get('message') or ''
        if text:
            lines.append(f"Message {idx}: {text}")
    conversation_text = "\n".join(lines)

    p1 = person1_data.get('persona_narrative', '').strip()
    p2 = person2_data.get('persona_narrative', '').strip()

    prompt = (
        "You are an experienced relationship observer reviewing a speed dating conversation. "
        "Evaluate the overall match quality between the two participants.\n\n"
        f"Person 1 background: {p1}\n"
        f"Person 2 background: {p2}\n\n"
        "Conversation transcript:\n"
        f"{conversation_text}\n\n"
        "Please reply in English. Start your answer with a single line in the exact format:\n"
        "Score: X.X/10\n"
        "Then provide 2-3 concise sentences explaining your assessment, considering: values/interests alignment, conversational flow, mutual attraction/chemistry, and long-term potential."
    )
    return prompt


def evaluate_with_scores(
    conversations_path: str,
    output_dir: str = "results",
    max_pair_workers: int = 3,
    method: str = "both",
    threshold: float = 0.5,
    report_curves: bool = False,
    icl_examples_path: str = None,
):
    """
    使用0-10分评分系统重新评估LLM方法
    
    Args:
        conversations_path: conversations.json路径
        output_dir: 输出目录
        icl_examples_path: Optional path to in-context learning examples JSON for advanced observer
    """
    
    print("=" * 70)
    print("LLM SCORE-BASED EVALUATION")
    print("=" * 70)
    print(f"Loading conversations from: {conversations_path}\n")
    print(f"Method: {method} | Decision threshold: {threshold} | Report curves: {report_curves}")
    if icl_examples_path:
        print(f"Using in-context learning examples from: {icl_examples_path}")
    
    # 加载对话数据
    with open(conversations_path, 'r', encoding='utf-8') as f:
        conversations = json.load(f)
    
    # Load in-context learning examples if provided
    icl_examples = None
    if icl_examples_path:
        print(f"Loading ICL examples from: {icl_examples_path}")
        with open(icl_examples_path, 'r', encoding='utf-8') as f:
            icl_data = json.load(f)
            # icl_data should be a list of examples with person1, person2, conversation, match
            icl_examples = icl_data  # Use all examples (10 total: 5 matches + 5 non-matches)
            print(f"Loaded {len(icl_examples)} in-context learning examples for advanced observer (balanced dataset)\n")
    
    results = {
        'participant_scores': [],  # 存储参与者评分
        'observer_scores': [],     # 存储观察员评分
        'advanced_observer_scores': [],  # 存储高级观察员评分（带ICL）
        'ground_truth': [],        # 真实标签
        'pair_ids': []             # pair ID
    }
    
    print(f"Evaluating {len(conversations)} pairs...\n")
    
    # 加载personas.json获取person_data
    personas_path = str(Path(conversations_path).with_name('personas.json'))
    with open(personas_path, 'r', encoding='utf-8') as f:
        personas = json.load(f)
    
    # 创建person查找字典
    person_lookup = {}
    for persona in personas:
        p1 = persona['person1']
        p2 = persona['person2']
        pair_key = (p1['iid'], p2['iid'])
        person_lookup[pair_key] = {
            'person1': p1,
            'person2': p2
        }
    
    # 使用线程池并行处理多个配对
    import concurrent.futures as _cf

    def _process_single_pair(conv_data: Dict, index: int, total: int):
        person1_iid = conv_data['person1_iid']
        person2_iid = conv_data['person2_iid']
        pair_id = f"pair_{person1_iid}_{person2_iid}"

        # 获取person数据
        pair_key = (person1_iid, person2_iid)
        if pair_key not in person_lookup:
            print(f"  Warning: No persona data found for {pair_id}, skipping...")
            return None

        person1_data = person_lookup[pair_key]['person1']
        person2_data = person_lookup[pair_key]['person2']

        rounds = conv_data.get('rounds', [])
        ground_truth = conv_data.get('ground_truth', {}).get('match', None)
        if ground_truth is None:
            print(f"  Warning: Missing ground truth 'match' for {pair_id}, skipping...")
            return None

        print(f"[{index}/{total}] Evaluating {pair_id}...")

        # 提取所有对话内容，支持两种结构
        all_messages: List[Dict] = []
        for round_data in rounds:
            if isinstance(round_data, dict) and isinstance(round_data.get('conversation'), list):
                for m in round_data.get('conversation', []):
                    if isinstance(m, dict):
                        text = m.get('message') or m.get('content') or ''
                        spk = m.get('speaker')
                        if text:
                            if spk:
                                all_messages.append({'speaker': spk, 'content': text})
                            else:
                                all_messages.append({'content': text})
            else:
                text = None
                spk = None
                if isinstance(round_data, dict):
                    text = round_data.get('message') or round_data.get('content')
                    spk = round_data.get('speaker')
                elif isinstance(round_data, str):
                    text = round_data
                if text:
                    if spk:
                        all_messages.append({'speaker': spk, 'content': text})
                    else:
                        all_messages.append({'content': text})

        if not all_messages:
            print(f"  Warning: No conversation found for {pair_id}, skipping...")
            return None

        try:
            # 根据 method 并行发起所需的调用（使用自动重试的评分提取）
            with _cf.ThreadPoolExecutor(max_workers=4) as ex_pair:
                future_map = {}
                if method in ("participant", "both"):
                    msg_p1 = build_participant_messages(person1_data, person2_data, all_messages, perspective="person1")
                    msg_p2 = build_participant_messages(person2_data, person1_data, all_messages, perspective="person2")
                    future_map['p1'] = ex_pair.submit(call_api_with_score_extraction, msg_p1, 0.3, "participant")
                    future_map['p2'] = ex_pair.submit(call_api_with_score_extraction, msg_p2, 0.3, "participant")
                if method in ("observer", "both"):
                    # Regular observer
                    msg_obs = build_observer_messages(person1_data, person2_data, all_messages)
                    future_map['obs'] = ex_pair.submit(call_api_with_score_extraction, msg_obs, 0.3, "observer")
                    # Advanced observer with ICL (if examples provided)
                    if icl_examples:
                        msg_obs_adv = build_observer_messages(person1_data, person2_data, all_messages, icl_examples=icl_examples)
                        future_map['obs_adv'] = ex_pair.submit(call_api_with_score_extraction, msg_obs_adv, 0.3, "observer")

                scores = {k: fut.result() for k, fut in future_map.items()}

            participant_entry = None
            observer_entry = None
            advanced_observer_entry = None

            if 'p1' in scores and 'p2' in scores:
                s1 = scores['p1']
                print(f"  Person 1 score: {s1:.1f}/10")
                s2 = scores['p2']
                print(f"  Person 2 score: {s2:.1f}/10")
                combined = (s1 * s2) / 100.0
                print(f"  Combined score: {combined:.3f}")
                participant_entry = {
                    'pair_id': pair_id,
                    'person1_score': s1,
                    'person2_score': s2,
                    'combined_score': combined,
                    'person1_response': f"Score: {s1}",  # Store score instead of full response
                    'person2_response': f"Score: {s2}"
                }

            if 'obs' in scores:
                so = scores['obs']
                print(f"  Observer score: {so:.1f}/10")
                norm_o = so / 10.0
                print(f"  Observer normalized: {norm_o:.3f}")
                observer_entry = {
                    'pair_id': pair_id,
                    'score': so,
                    'normalized_score': norm_o,
                    'response': f"Score: {so}"
                }

            if 'obs_adv' in scores:
                so_adv = scores['obs_adv']
                print(f"  Advanced Observer score: {so_adv:.1f}/10")
                norm_o_adv = so_adv / 10.0
                print(f"  Advanced Observer normalized: {norm_o_adv:.3f}")
                advanced_observer_entry = {
                    'pair_id': pair_id,
                    'score': so_adv,
                    'normalized_score': norm_o_adv,
                    'response': f"Score: {so_adv}"
                }

            print(f"  Ground truth: {'Match' if ground_truth else 'No match'}\n")

            return {
                'pair_id': pair_id,
                'participant_entry': participant_entry,
                'observer_entry': observer_entry,
                'advanced_observer_entry': advanced_observer_entry,
                'ground_truth': ground_truth
            }
        except Exception as e:
            print(f"  Error evaluating {pair_id}: {e}\n")
            return None

    total = len(conversations)
    with _cf.ThreadPoolExecutor(max_workers=max_pair_workers) as ex:
        futures = [ex.submit(_process_single_pair, conv, idx + 1, total) for idx, conv in enumerate(conversations)]
        for fut in _cf.as_completed(futures):
            res = fut.result()
            if not res:
                continue
            if res.get('participant_entry') is not None:
                results['participant_scores'].append(res['participant_entry'])
            if res.get('observer_entry') is not None:
                results['observer_scores'].append(res['observer_entry'])
            if res.get('advanced_observer_entry') is not None:
                results['advanced_observer_scores'].append(res['advanced_observer_entry'])
            results['ground_truth'].append(res['ground_truth'])
            results['pair_ids'].append(res['pair_id'])
    
    # 计算评估指标
    print("\n" + "=" * 70)
    print("COMPUTING METRICS")
    print("=" * 70)
    
    y_true = np.array(results['ground_truth'])
    if len(y_true) == 0:
        print("No valid pairs with ground truth. Skipping metrics.")
        return {}
    
    participant_probs = np.array([s['combined_score'] for s in results['participant_scores']]) if results['participant_scores'] else np.array([])
    observer_probs = np.array([s['normalized_score'] for s in results['observer_scores']]) if results['observer_scores'] else np.array([])
    advanced_observer_probs = np.array([s['normalized_score'] for s in results['advanced_observer_scores']]) if results['advanced_observer_scores'] else np.array([])
    
    # 使用可配置阈值进行二分类（AUC指标本身阈值无关，仅用于摘要指标）
    participant_preds = (participant_probs >= threshold).astype(int) if participant_probs.size else np.array([])
    observer_preds = (observer_probs >= threshold).astype(int) if observer_probs.size else np.array([])
    advanced_observer_preds = (advanced_observer_probs >= threshold).astype(int) if advanced_observer_probs.size else np.array([])
    
    # 参与者方法指标
    fpr = tpr = roc_th = prec = rec = pr_th = None
    if participant_probs.size:
        print("\n1. PARTICIPANT METHOD (Product of two scores):")
        print(f"   Score range: [{participant_probs.min():.3f}, {participant_probs.max():.3f}]")
        print(f"   Mean score: {participant_probs.mean():.3f}")
        print(f"   Accuracy:   {accuracy_score(y_true, participant_preds):.3f}")
        print(f"   Precision:  {precision_score(y_true, participant_preds, zero_division=0):.3f}")
        print(f"   Recall:     {recall_score(y_true, participant_preds, zero_division=0):.3f}")
        print(f"   F1 Score:   {f1_score(y_true, participant_preds, zero_division=0):.3f}")
        try:
            print(f"   AUC-ROC:    {roc_auc_score(y_true, participant_probs):.3f}")
        except Exception as e:
            print(f"   AUC-ROC:    N/A ({e})")
        try:
            print(f"   PR-AUC:     {average_precision_score(y_true, participant_probs):.3f}")
        except Exception as e:
            print(f"   PR-AUC:     N/A ({e})")
        cm_participant = confusion_matrix(y_true, participant_preds, labels=[0, 1])
        print(f"   Confusion Matrix: TN={cm_participant[0,0]}, FP={cm_participant[0,1]}, FN={cm_participant[1,0]}, TP={cm_participant[1,1]}")
        if report_curves and len(set(y_true)) > 1:
            fpr, tpr, roc_th = roc_curve(y_true, participant_probs)
            prec, rec, pr_th = precision_recall_curve(y_true, participant_probs)
    else:
        cm_participant = np.array([[0,0],[0,0]])
        print("\n1. PARTICIPANT METHOD: No scores available.")
    
    # 观察员方法指标
    fpr_o = tpr_o = roc_th_o = prec_o = rec_o = pr_th_o = None
    if observer_probs.size:
        print("\n2. OBSERVER METHOD (Single observer score):")
        print(f"   Score range: [{observer_probs.min():.3f}, {observer_probs.max():.3f}]")
        print(f"   Mean score: {observer_probs.mean():.3f}")
        print(f"   Accuracy:   {accuracy_score(y_true, observer_preds):.3f}")
        print(f"   Precision:  {precision_score(y_true, observer_preds, zero_division=0):.3f}")
        print(f"   Recall:     {recall_score(y_true, observer_preds, zero_division=0):.3f}")
        print(f"   F1 Score:   {f1_score(y_true, observer_preds, zero_division=0):.3f}")
        try:
            print(f"   AUC-ROC:    {roc_auc_score(y_true, observer_probs):.3f}")
        except Exception as e:
            print(f"   AUC-ROC:    N/A ({e})")
        try:
            print(f"   PR-AUC:     {average_precision_score(y_true, observer_probs):.3f}")
        except Exception as e:
            print(f"   PR-AUC:     N/A ({e})")
        cm_observer = confusion_matrix(y_true, observer_preds, labels=[0, 1])
        print(f"   Confusion Matrix: TN={cm_observer[0,0]}, FP={cm_observer[0,1]}, FN={cm_observer[1,0]}, TP={cm_observer[1,1]}")
        if report_curves and len(set(y_true)) > 1:
            fpr_o, tpr_o, roc_th_o = roc_curve(y_true, observer_probs)
            prec_o, rec_o, pr_th_o = precision_recall_curve(y_true, observer_probs)
    else:
        cm_observer = np.array([[0,0],[0,0]])
        print("\n2. OBSERVER METHOD: No scores available.")
    
    # 高级观察员方法指标（带ICL）
    fpr_oa = tpr_oa = roc_th_oa = prec_oa = rec_oa = pr_th_oa = None
    if advanced_observer_probs.size:
        print("\n3. ADVANCED OBSERVER METHOD (With in-context learning):")
        print(f"   Score range: [{advanced_observer_probs.min():.3f}, {advanced_observer_probs.max():.3f}]")
        print(f"   Mean score: {advanced_observer_probs.mean():.3f}")
        print(f"   Accuracy:   {accuracy_score(y_true, advanced_observer_preds):.3f}")
        print(f"   Precision:  {precision_score(y_true, advanced_observer_preds, zero_division=0):.3f}")
        print(f"   Recall:     {recall_score(y_true, advanced_observer_preds, zero_division=0):.3f}")
        print(f"   F1 Score:   {f1_score(y_true, advanced_observer_preds, zero_division=0):.3f}")
        try:
            print(f"   AUC-ROC:    {roc_auc_score(y_true, advanced_observer_probs):.3f}")
        except Exception as e:
            print(f"   AUC-ROC:    N/A ({e})")
        try:
            print(f"   PR-AUC:     {average_precision_score(y_true, advanced_observer_probs):.3f}")
        except Exception as e:
            print(f"   PR-AUC:     N/A ({e})")
        cm_advanced_observer = confusion_matrix(y_true, advanced_observer_preds, labels=[0, 1])
        print(f"   Confusion Matrix: TN={cm_advanced_observer[0,0]}, FP={cm_advanced_observer[0,1]}, FN={cm_advanced_observer[1,0]}, TP={cm_advanced_observer[1,1]}")
        if report_curves and len(set(y_true)) > 1:
            fpr_oa, tpr_oa, roc_th_oa = roc_curve(y_true, advanced_observer_probs)
            prec_oa, rec_oa, pr_th_oa = precision_recall_curve(y_true, advanced_observer_probs)
    else:
        cm_advanced_observer = np.array([[0,0],[0,0]])
        print("\n3. ADVANCED OBSERVER METHOD: No scores available.")
    
    # 保存详细结果
    output_path = f"{output_dir}/llm_score_evaluation.json"
    output_data = {
        'participant_method': {
            'scores': results['participant_scores'],
            'metrics': {
                'accuracy': float(accuracy_score(y_true, participant_preds)) if participant_probs.size else None,
                'precision': float(precision_score(y_true, participant_preds, zero_division=0)) if participant_probs.size else None,
                'recall': float(recall_score(y_true, participant_preds, zero_division=0)) if participant_probs.size else None,
                'f1': float(f1_score(y_true, participant_preds, zero_division=0)) if participant_probs.size else None,
                'auc_roc': (float(roc_auc_score(y_true, participant_probs)) if participant_probs.size and len(set(y_true)) > 1 else None),
                'pr_auc': (float(average_precision_score(y_true, participant_probs)) if participant_probs.size else None),
                'confusion_matrix': cm_participant.tolist() if participant_probs.size else None
            },
            'curves': (
                {
                    'roc': {
                        'fpr': fpr.tolist() if isinstance(fpr, np.ndarray) else (list(fpr) if fpr is not None else None),
                        'tpr': tpr.tolist() if isinstance(tpr, np.ndarray) else (list(tpr) if tpr is not None else None),
                        'thresholds': roc_th.tolist() if isinstance(roc_th, np.ndarray) else (list(roc_th) if roc_th is not None else None),
                    },
                    'pr': {
                        'precision': prec.tolist() if isinstance(prec, np.ndarray) else (list(prec) if prec is not None else None),
                        'recall': rec.tolist() if isinstance(rec, np.ndarray) else (list(rec) if rec is not None else None),
                        'thresholds': pr_th.tolist() if isinstance(pr_th, np.ndarray) else (list(pr_th) if pr_th is not None else None),
                    },
                } if report_curves else None
            )
        },
        'observer_method': {
            'scores': results['observer_scores'],
            'metrics': {
                'accuracy': float(accuracy_score(y_true, observer_preds)) if observer_probs.size else None,
                'precision': float(precision_score(y_true, observer_preds, zero_division=0)) if observer_probs.size else None,
                'recall': float(recall_score(y_true, observer_preds, zero_division=0)) if observer_probs.size else None,
                'f1': float(f1_score(y_true, observer_preds, zero_division=0)) if observer_probs.size else None,
                'auc_roc': (float(roc_auc_score(y_true, observer_probs)) if observer_probs.size and len(set(y_true)) > 1 else None),
                'pr_auc': (float(average_precision_score(y_true, observer_probs)) if observer_probs.size else None),
                'confusion_matrix': cm_observer.tolist() if observer_probs.size else None
            },
            'curves': (
                {
                    'roc': {
                        'fpr': fpr_o.tolist() if isinstance(fpr_o, np.ndarray) else (list(fpr_o) if fpr_o is not None else None),
                        'tpr': tpr_o.tolist() if isinstance(tpr_o, np.ndarray) else (list(tpr_o) if tpr_o is not None else None),
                        'thresholds': roc_th_o.tolist() if isinstance(roc_th_o, np.ndarray) else (list(roc_th_o) if roc_th_o is not None else None),
                    },
                    'pr': {
                        'precision': prec_o.tolist() if isinstance(prec_o, np.ndarray) else (list(prec_o) if prec_o is not None else None),
                        'recall': rec_o.tolist() if isinstance(rec_o, np.ndarray) else (list(rec_o) if rec_o is not None else None),
                        'thresholds': pr_th_o.tolist() if isinstance(pr_th_o, np.ndarray) else (list(pr_th_o) if pr_th_o is not None else None),
                    },
                } if report_curves else None
            )
        },
        'advanced_observer_method': {
            'scores': results['advanced_observer_scores'],
            'metrics': {
                'accuracy': float(accuracy_score(y_true, advanced_observer_preds)) if advanced_observer_probs.size else None,
                'precision': float(precision_score(y_true, advanced_observer_preds, zero_division=0)) if advanced_observer_probs.size else None,
                'recall': float(recall_score(y_true, advanced_observer_preds, zero_division=0)) if advanced_observer_probs.size else None,
                'f1': float(f1_score(y_true, advanced_observer_preds, zero_division=0)) if advanced_observer_probs.size else None,
                'auc_roc': (float(roc_auc_score(y_true, advanced_observer_probs)) if advanced_observer_probs.size and len(set(y_true)) > 1 else None),
                'pr_auc': (float(average_precision_score(y_true, advanced_observer_probs)) if advanced_observer_probs.size else None),
                'confusion_matrix': cm_advanced_observer.tolist() if advanced_observer_probs.size else None
            },
            'curves': (
                {
                    'roc': {
                        'fpr': fpr_oa.tolist() if isinstance(fpr_oa, np.ndarray) else (list(fpr_oa) if fpr_oa is not None else None),
                        'tpr': tpr_oa.tolist() if isinstance(tpr_oa, np.ndarray) else (list(tpr_oa) if tpr_oa is not None else None),
                        'thresholds': roc_th_oa.tolist() if isinstance(roc_th_oa, np.ndarray) else (list(roc_th_oa) if roc_th_oa is not None else None),
                    },
                    'pr': {
                        'precision': prec_oa.tolist() if isinstance(prec_oa, np.ndarray) else (list(prec_oa) if prec_oa is not None else None),
                        'recall': rec_oa.tolist() if isinstance(rec_oa, np.ndarray) else (list(rec_oa) if rec_oa is not None else None),
                        'thresholds': pr_th_oa.tolist() if isinstance(pr_th_oa, np.ndarray) else (list(pr_th_oa) if pr_th_oa is not None else None),
                    },
                } if report_curves else None
            )
        },
        'ground_truth': results['ground_truth'],
        'pair_ids': results['pair_ids']
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Results saved to: {output_path}")
    
    return output_data


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate LLM with 0-10 score system")
    parser.add_argument(
        "--conversations",
        type=str,
        default="results/conversations.json",
        help="Path to conversations.json"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory"
    )
    
    parser.add_argument(
        "--max-pair-workers",
        type=int,
        default=3,
        help="Max number of pairs to evaluate concurrently"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["participant", "observer", "both"],
        default="both",
        help="Which method(s) to evaluate"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold for binary summaries (accuracy, precision, recall, F1)"
    )
    parser.add_argument(
        "--report-curves",
        action="store_true",
        help="If set, compute and save ROC and PR curve points"
    )
    parser.add_argument(
        "--icl-examples",
        type=str,
        default=None,
        help="Path to in-context learning examples JSON for advanced observer (5 pairs outside test set)"
    )

    args = parser.parse_args()

    evaluate_with_scores(
        args.conversations,
        args.output_dir,
        max_pair_workers=args.max_pair_workers,
        method=args.method,
        threshold=args.threshold,
        report_curves=args.report_curves,
        icl_examples_path=args.icl_examples,
    )
