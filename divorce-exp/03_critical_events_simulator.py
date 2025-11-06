"""
Critical Events Simulator for Divorce Prediction

基于 Speed Dating Simulator 的架构，模拟夫妻在压力事件下的交互。

关键设计：
1. World Engine（环境）：生成事件、描述环境变化（不替agent思考）
2. Husband Agent：基于 persona 响应（有独立 history）
3. Wife Agent：基于 persona 响应（有独立 history）
4. History 分离：共享信息（事件描述）+ 私人信息（inner thoughts）

参考：test/experiments/speed_dating_simulator.py
"""

import os
import json
import argparse
import threading
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import random

# Import question mappings for ICL
from divorce_questions import format_couple_features

# 加载环境变量
load_dotenv()

# Thread-safe lock
save_lock = threading.Lock()

# Global API concurrency & rate limiting
_API_SEM = threading.Semaphore(int(os.getenv('OPENROUTER_MAX_CONCURRENCY', '2')))
_RATE_LOCK = threading.Lock()
_LAST_CALL_TS = 0.0
_RPS = float(os.getenv('OPENROUTER_RPS', '0.8'))  # requests per second (global)
_SCENE_REACT_EVERY = int(os.getenv('SCENE_REACT_EVERY', '2'))  # environment reacts every N turns


class CriticalEventsSimulator:
    """模拟夫妻在关键事件下的交互"""
    
    def __init__(self, personas_path: str, events_path: str, use_icl_scenarios: bool = False, clean_data_path: Optional[str] = None, enable_agent_icl: bool = False, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('OPENROUTER_API_KEY')
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not found")
        
        base_url = os.getenv('OPENROUTER_BASE_URL', 'https://openrouter.ai/api/v1')
        self.api_url = f"{base_url.rstrip('/')}/chat/completions"
        self.model = os.getenv('OPENROUTER_MODEL', 'mistralai/mistral-nemo')

        # HTTP session reuse
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        self._session = requests.Session()
        retry = Retry(
            total=3,
            connect=3,
            read=3,
            status=3,
            backoff_factor=0.5,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=False,  # retry on any
        )
        adapter = HTTPAdapter(max_retries=retry, pool_connections=20, pool_maxsize=20)
        self._session.mount('https://', adapter)
        self._session.mount('http://', adapter)
        
        # 加载数据
        with open(personas_path, 'r', encoding='utf-8') as f:
            self.personas = json.load(f)
        
        with open(events_path, 'r', encoding='utf-8') as f:
            self.events_data = json.load(f)
        
        print(f"✅ Loaded {len(self.personas)} couples")
        print(f"✅ Loaded events for {len(self.events_data)} couples")
        self.use_icl_scenarios = use_icl_scenarios
        self.enable_agent_icl = enable_agent_icl
        
        # Load clean dataset for ICL examples
        self.df_clean = None
        if enable_agent_icl and clean_data_path:
            self.df_clean = pd.read_csv(clean_data_path)
            print(f"✅ Loaded {len(self.df_clean)} couples from clean dataset for ICL")
        
        self.simulations = []

        # Build ICL example cache (for scenario generation and agent prompts)
        self._icl_examples: List[Tuple[int, int]] = []
        self._icl_divorced_ids: List[int] = []
        self._icl_married_ids: List[int] = []
        if use_icl_scenarios or enable_agent_icl:
            self._build_icl_examples()
    
    def call_openrouter_api(self, messages: List[Dict], temperature: float = 0.7, max_retries: int = 5) -> Dict[str, str]:
        """调用 OpenRouter API，带并发限制、全局 RPS 节流、指数退避 + 抖动，并处理 429 Retry-After。"""
        import requests

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 500
        }

        def _respect_rps():
            if _RPS <= 0:
                return
            min_interval = 1.0 / _RPS
            global _LAST_CALL_TS
            with _RATE_LOCK:
                now = time.time()
                wait = _LAST_CALL_TS + min_interval - now
                if wait > 0:
                    time.sleep(wait)
                    now = time.time()
                _LAST_CALL_TS = now

        with _API_SEM:
            backoff_base = 0.8
            for attempt in range(1, max_retries + 1):
                try:
                    _respect_rps()
                    resp = self._session.post(self.api_url, headers=headers, json=payload, timeout=(10, 120))
                    # 429 handling
                    if resp.status_code == 429:
                        retry_after = resp.headers.get('Retry-After')
                        delay = float(retry_after) if retry_after else (backoff_base * (2 ** (attempt - 1)))
                        delay += random.uniform(0, 0.5)
                        print(f"   ⚠️  Rate limited (429). Waiting {delay:.1f}s before retry {attempt}/{max_retries}...")
                        time.sleep(delay)
                        continue
                    resp.raise_for_status()
                    data = resp.json()
                    content = data['choices'][0]['message']['content']
                    # small pacing after success to avoid bursts
                    time.sleep(random.uniform(0.05, 0.15))
                    return {"success": True, "content": content}
                except requests.exceptions.RequestException as e:
                    if attempt >= max_retries:
                        return {"success": False, "error": str(e)}
                    delay = backoff_base * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
                    print(f"   ⚠️  API error (attempt {attempt}/{max_retries}): {e}")
                    print(f"   Retrying in {delay:.1f}s...")
                    time.sleep(delay)
            return {"success": False, "error": "Max retries exceeded"}
    
    def _build_icl_examples(self, k_per_class: int = 5, seed: int = 42):
        """Build ICL examples (5 divorced + 5 married couple indices) for scenario generation and agent prompts."""
        rng = random.Random(seed)
        divorced = [i for i, p in enumerate(self.personas) if p.get('ground_truth_divorced') in (1, True)]
        married = [i for i, p in enumerate(self.personas) if p.get('ground_truth_divorced') in (0, False)]

        rng.shuffle(divorced)
        rng.shuffle(married)

        take_div = divorced[:k_per_class]
        take_mar = married[:k_per_class]
        
        self._icl_divorced_ids = take_div
        self._icl_married_ids = take_mar
        
        examples = [(i, 1) for i in take_div] + [(i, 0) for i in take_mar]
        rng.shuffle(examples)
        self._icl_examples = examples
        
        if self.use_icl_scenarios:
            print(f"✅ Prepared {len(self._icl_examples)} ICL examples for scenario generation")
        if self.enable_agent_icl:
            print(f"✅ Prepared {len(self._icl_examples)} ICL examples for agent prompts")
    
    def _build_agent_icl_block(self, exclude_id: int) -> str:
        """Build ICL examples for agent prompts (show survey data + outcome patterns)."""
        if not self.enable_agent_icl or self.df_clean is None:
            return ""
        
        icl_parts = []
        
        # Select examples (exclude current couple)
        divorced_examples = [cid for cid in self._icl_divorced_ids if cid != exclude_id][:3]
        married_examples = [cid for cid in self._icl_married_ids if cid != exclude_id][:3]
        
        all_examples = [(cid, 1) for cid in divorced_examples] + [(cid, 0) for cid in married_examples]
        random.shuffle(all_examples)
        
        for couple_id, label in all_examples:
            # Get survey responses
            try:
                row = self.df_clean.iloc[couple_id]
                features = row.drop('Class').values.tolist()
                features_str = format_couple_features(features)
            except Exception:
                features_str = "[features unavailable]"
            
            # Get persona narratives
            try:
                h_persona = self.personas[couple_id]['husband']['persona_narrative']
                w_persona = self.personas[couple_id]['wife']['persona_narrative']
            except Exception:
                h_persona = "[unavailable]"
                w_persona = "[unavailable]"
            
            outcome = "Divorced" if label == 1 else "Stayed Married"
            
            icl_parts.append(
                f"<EXAMPLE_COUPLE outcome=\"{outcome}\">\n"
                f"<SURVEY_RESPONSES>\n{features_str}\n</SURVEY_RESPONSES>\n"
                f"<HUSBAND_PERSONA>{h_persona}</HUSBAND_PERSONA>\n"
                f"<WIFE_PERSONA>{w_persona}</WIFE_PERSONA>\n"
                f"</EXAMPLE_COUPLE>\n"
            )
        
        if icl_parts:
            header = f"""
=== EXAMPLE COUPLES (FOR REFERENCE) ===
Below are {len(icl_parts)} example couples with their survey responses and personas.
Pay attention to patterns in communication, trust, and conflict resolution that led to their outcomes.

"""
            return header + "\n".join(icl_parts)
        return ""

    def _generate_scenario_with_icl(self, couple_id: int) -> Dict:
        """Generate a couple-specific critical scenario using ICL.

        Returns a dict with 'description' and 'type' keys (compatible with existing event_data format).
        """
        husband_persona = self.personas[couple_id]['husband']['persona_narrative']
        wife_persona = self.personas[couple_id]['wife']['persona_narrative']

        # Build ICL examples block
        examples_block = ""
        for example_cid, label in self._icl_examples:
            hp = self.personas[example_cid]['husband']['persona_narrative']
            wp = self.personas[example_cid]['wife']['persona_narrative']
            status = "Divorced" if label == 1 else "Married"
            examples_block += (
                f"<EXAMPLE>\n"
                f"<BACKGROUND>\nHusband: {hp}\n\nWife: {wp}\n</BACKGROUND>\n"
                f"<COUPLE_STATUS>{status}</COUPLE_STATUS>\n"
                f"</EXAMPLE>\n\n"
            )

        prompt = f"""You are a relationship scenario designer. Your task: generate a tailored critical life event scenario for a couple based on their profiles.

=== EXAMPLES: How different couple types face critical scenarios ===
{examples_block}

=== TARGET COUPLE (STATUS: unknown) ===
<BACKGROUND>
Husband: {husband_persona}

Wife: {wife_persona}
</BACKGROUND>

=== YOUR TASK ===
Generate ONE critical scenario that would specifically test THIS couple's compatibility and resilience. Consider their personalities, values, and likely pain points based on their background.

Output format (ONLY):

<SCENARIO_TYPE>
[E.g., "Financial Crisis", "Infidelity Discovery", "Career Conflict", "Health Crisis", "Betrayal of Trust", etc. - choose what is most relevant for this couple]
</SCENARIO_TYPE>

<SCENARIO_DESCRIPTION>
[Detailed description of the critical event: 3-4 sentences. Describe the specific situation, stakes, and immediate pressure facing the couple. Make it vivid and emotionally charged. Include realistic consequences and time pressure.]
</SCENARIO_DESCRIPTION>
"""

        response = self.call_openrouter_api([
            {"role": "system", "content": "You are a creative scenario designer for relationship counseling research. Generate realistic and tailored critical scenarios."},
            {"role": "user", "content": prompt}
        ], temperature=0.7)

        if not response['success']:
            # Fallback to a generic scenario if generation fails
            return {
                "type": "generic_crisis",
                "description": "A sudden major life event tests the couple's communication and mutual support."
            }

        # Parse scenario type and description
        import re
        type_match = re.search(r'<SCENARIO_TYPE>\s*(.+?)\s*</SCENARIO_TYPE>', response['content'], re.DOTALL | re.IGNORECASE)
        desc_match = re.search(r'<SCENARIO_DESCRIPTION>\s*(.+?)\s*</SCENARIO_DESCRIPTION>', response['content'], re.DOTALL | re.IGNORECASE)

        scenario_type = type_match.group(1).strip() if type_match else "unknown_scenario"
        scenario_desc = desc_match.group(1).strip() if desc_match else "A critical test of the couple's bond."

        return {
            "type": scenario_type.lower().replace(' ', '_'),
            "description": scenario_desc
        }
    
    def simulate_event(self, couple_id: int, event_data: Dict, num_rounds: int = 6) -> Dict:
        """
        模拟一个关键事件（类似 speed_dating 的 conversation rounds）
        
        Args:
            couple_id: 夫妻 ID
            event_data: 事件数据（从 critical_events.json）
            num_rounds: 交互轮次（默认 6 轮）
        
        Returns:
            交互记录 + 评估结果
        """
        couple_personas = self.personas[couple_id]
        husband_persona = couple_personas['husband']['persona_narrative']
        wife_persona = couple_personas['wife']['persona_narrative']
        
        # 初始化 history（每个 agent 有独立的 history）
        husband_history = []  # Husband 看到的历史
        wife_history = []     # Wife 看到的历史
        shared_history = []   # World Engine 看到的历史（双方公开行为）
        
        interactions = []

        # === Round 1: World Engine 描述初始事件 ===
        event_description = event_data['description']

        # Build optional ICL reference block (reuse agent ICL builder)
        world_icl_block = ""
        if self.enable_agent_icl:
            world_icl_block = self._build_agent_icl_block(exclude_id=couple_id)

        # World Engine 的初始提示
        world_engine_prompt = f"""You are the LIFE CIRCUMSTANCES narrator for a married couple facing a critical event.

Your role:
- Describe realistic scenarios and environmental changes
- Never speak FOR the husband or wife - only describe what happens around them
- After they respond, describe how the environment/situation evolves
- Make the situation genuinely critical: add realistic stakes, time pressure, resource constraints, and potential irreversible consequences (job at risk, medical decisions, trust erosion). No numbers or scores.

Use the couple background (personas) and reference examples to keep the world consistent with their tendencies. Do NOT put words in their mouths.

=== COUPLE BACKGROUND (PERSONAS) ===
Husband Persona:
{husband_persona}

Wife Persona:
{wife_persona}

{world_icl_block}

Critical Event:
{event_description}

Set the scene. Describe where they are, the atmosphere, and the initial moment of this event with concrete, sensory details. Keep it vivid and realistic (2-3 sentences)."""
        
        # 调用 World Engine
        world_response = self.call_openrouter_api([
            {"role": "system", "content": "You are a relationship scenario narrator. Describe scenes vividly but never speak for the participants."},
            {"role": "user", "content": world_engine_prompt}
        ], temperature=0.7)
        
        if not world_response['success']:
            return {"error": "World Engine failed to initialize"}
        
        scene_setting = world_response['content']
        
        # 双方都看到场景设置
        husband_history.append({"role": "environment", "content": scene_setting})
        wife_history.append({"role": "environment", "content": scene_setting})
        shared_history.append({"role": "environment", "content": scene_setting})
        
        interactions.append({
            "round": 1,
            "type": "scene_setting",
            "content": scene_setting
        })
        
        # === Round 2-N: 交互循环 ===
        for round_num in range(2, num_rounds + 1):
            # --- Husband's turn ---
            husband_prompt = self._build_agent_prompt(
                persona=husband_persona,
                role="husband",
                history=husband_history,
                round_num=round_num,
                couple_id=couple_id
            )
            
            husband_response = self.call_openrouter_api(
                [{"role": "user", "content": husband_prompt}],
                temperature=0.7
            )
            
            if not husband_response['success']:
                break
            
            husband_text = husband_response['content']
            
            # 提取 inner thought 和 public response
            husband_inner, husband_public = self._parse_agent_response(husband_text)
            
            # 更新 history
            husband_history.append({"role": "self", "content": husband_text})
            wife_history.append({"role": "husband", "content": husband_public})  # Wife 只看到公开部分
            shared_history.append({"role": "husband", "content": husband_public})
            
            interactions.append({
                "round": round_num,
                "speaker": "husband",
                "inner_thought": husband_inner,
                "public_response": husband_public,
                "full_text": husband_text
            })
            
            # pacing between turns to avoid burst
            time.sleep(random.uniform(0.05, 0.15))

            # --- Wife's turn ---
            wife_prompt = self._build_agent_prompt(
                persona=wife_persona,
                role="wife",
                history=wife_history,
                round_num=round_num,
                couple_id=couple_id
            )
            
            wife_response = self.call_openrouter_api(
                [{"role": "user", "content": wife_prompt}],
                temperature=0.7
            )
            
            if not wife_response['success']:
                break
            
            wife_text = wife_response['content']
            
            # 提取 inner thought 和 public response
            wife_inner, wife_public = self._parse_agent_response(wife_text)
            
            # 更新 history
            wife_history.append({"role": "self", "content": wife_text})
            husband_history.append({"role": "wife", "content": wife_public})  # Husband 只看到公开部分
            shared_history.append({"role": "wife", "content": wife_public})
            
            interactions.append({
                "round": round_num,
                "speaker": "wife",
                "inner_thought": wife_inner,
                "public_response": wife_public,
                "full_text": wife_text
            })
            
            # --- World Engine 反应 ---
            if round_num < num_rounds and (_SCENE_REACT_EVERY > 0 and (round_num % _SCENE_REACT_EVERY == 0)):
                # Provide personas and (optional) ICL references so the environment can evolve consistently
                reaction_icl_block = world_icl_block  # same selection is fine; cheap reuse
                world_reaction_prompt = f"""You are the LIFE CIRCUMSTANCES narrator.

Guidelines:
- Never speak for the husband or wife.
- Evolve the environment and situational pressures only.
- Keep evolution coherent with the couple's personas and patterns from the reference examples.

=== COUPLE BACKGROUND (PERSONAS) ===
Husband Persona:
{husband_persona}

Wife Persona:
{wife_persona}

{reaction_icl_block}

=== Latest public actions ===
- Husband: {husband_public}
- Wife: {wife_public}

Describe how the environment/situation evolves. What changes? How does the atmosphere shift? (2-3 sentences only, no numbers or percentages)"""
                
                world_reaction = self.call_openrouter_api([
                    {"role": "system", "content": "You describe environmental changes in relationship scenarios."},
                    {"role": "user", "content": world_reaction_prompt}
                ], temperature=0.7)
                
                if world_reaction['success']:
                    env_change = world_reaction['content']
                    
                    # 双方都看到环境变化
                    husband_history.append({"role": "environment", "content": env_change})
                    wife_history.append({"role": "environment", "content": env_change})
                    shared_history.append({"role": "environment", "content": env_change})
                    
                    interactions.append({
                        "round": round_num,
                        "type": "environment_reaction",
                        "content": env_change
                    })
                # small pacing after environment
                time.sleep(random.uniform(0.05, 0.15))
        
        return {
            "couple_id": couple_id,
            "event_type": event_data['type'],
            "interactions": interactions,
            "husband_final_history": husband_history,
            "wife_final_history": wife_history
        }
    
    def _build_agent_prompt(self, persona: str, role: str, history: List[Dict], round_num: int, couple_id: int) -> str:
        """构建 agent 的 prompt（包含 persona + history + optional ICL）"""
        # Get ICL block if enabled
        icl_block = ""
        if self.enable_agent_icl:
            icl_block = self._build_agent_icl_block(exclude_id=couple_id)
        
        # 格式化 history
        history_text = "\n\n".join([
            f"[{h['role'].upper()}]: {h['content']}" for h in history
        ])
        
        prompt = f"""You are the {role} in this marriage. Here is your persona (you MUST follow it strictly):

{persona}

=== YOUR PERSONA RULES ===
- Your INNER_THOUGHT must explicitly check alignment with your persona (values, conflict style, boundaries).
- If there is any conflict between social desirability and your persona, choose your persona.
- Keep tone, priorities, and coping style consistent across rounds.

{icl_block}

=== WHAT HAS HAPPENED SO FAR ===
{history_text}

=== YOUR RESPONSE (Round {round_num}) ===

Respond authentically based on your persona. Format your response as:

<INNER_THOUGHT>
[Self-check: How does your reaction align with your persona? Be explicit about which traits guide you.]
[Your private true feelings about this situation - what you REALLY think, not what you "should" think]
</INNER_THOUGHT>

<RESPONSE>
[What you actually say or do - your outward behavior that your spouse can observe]
</RESPONSE>

Remember: Follow your inner heart, not social expectations. Be honest about your limits and what you can tolerate."""
        
        return prompt
    
    def _parse_agent_response(self, text: str) -> tuple[str, str]:
        """提取 inner thought 和 public response"""
        import re
        
        inner_match = re.search(r'<INNER_THOUGHT>(.*?)</INNER_THOUGHT>', text, re.DOTALL | re.IGNORECASE)
        response_match = re.search(r'<RESPONSE>(.*?)</RESPONSE>', text, re.DOTALL | re.IGNORECASE)
        
        inner = inner_match.group(1).strip() if inner_match else "[No inner thought captured]"
        public = response_match.group(1).strip() if response_match else text.strip()
        
        return inner, public
    
    def simulate_couple(self, couple_id: int, num_rounds: int = 6) -> Dict:
        """为一对夫妻模拟关键事件（使用 ICL 生成或预定义）"""
        print(f"\n{'='*70}")
        print(f"Simulating Couple {couple_id}")
        print(f"{'='*70}")
        
        if self.use_icl_scenarios:
            # Generate scenario dynamically for this couple
            print(f"   🎭 Generating scenario with ICL for couple {couple_id}...")
            generated_scenario = self._generate_scenario_with_icl(couple_id)
            couple_events = {'event1': generated_scenario}
        else:
            # Use pre-defined events
            couple_events = self.events_data[couple_id]['events']
        
        # 为评估提供背景
        h_p = self.personas[couple_id]['husband']['persona_narrative']
        w_p = self.personas[couple_id]['wife']['persona_narrative']

        results = {
            "couple_id": couple_id,
            "ground_truth_divorced": self.personas[couple_id]['ground_truth_divorced'],
            "events_simulations": [],
            "husband_persona": h_p,
            "wife_persona": w_p
        }
        
        # 模拟 1 个（ICL 生成的）或 3 个（预定义的）事件
        if self.use_icl_scenarios:
            event_keys = ['event1']
        else:
            event_keys = ['event1_marriage_milestone', 'event2_trust_breach', 'event3_illness_burden']

        for event_key in event_keys:
            event_data = couple_events[event_key]
            event_type_str = event_data.get('type', 'unknown').replace('_', ' ').title()
            print(f"\n   🎭 Simulating: {event_type_str}")
            
            simulation = self.simulate_event(couple_id, event_data, num_rounds)
            results['events_simulations'].append(simulation)
        
        return results
    
    def simulate_all_couples(self, num_couples: Optional[int] = None, max_workers: int = 5, indices: Optional[List[int]] = None):
        """并行模拟所有夫妻

        Args:
            num_couples: 模拟的夫妻数量（若提供 indices 则忽略该数量，按 indices 长度为准）
            max_workers: 最大并发线程数
            indices: 指定要模拟的 couple 索引列表（支持分层抽样的结果）
        """
        if indices is not None:
            couple_ids = list(indices)
        else:
            total = num_couples or len(self.personas)
            couple_ids = list(range(total))

        total = len(couple_ids)

        # Clamp workers to global concurrency to prevent bursts
        try:
            max_workers = max(1, min(max_workers, int(os.getenv('OPENROUTER_MAX_CONCURRENCY', '2'))))
        except Exception:
            max_workers = max(1, max_workers)
        print(f"\n🎭 Starting simulation for {total} couples (max {max_workers} threads)...")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.simulate_couple, cid): cid
                for cid in couple_ids
            }

            for future in as_completed(futures):
                couple_id = futures[future]
                try:
                    result = future.result()
                    self.simulations.append(result)
                    print(f"✅ Completed couple {couple_id}/{total}")
                except Exception as e:
                    print(f"❌ Error on couple {couple_id}: {e}")

        print(f"\n✅ Simulation complete: {len(self.simulations)}/{total} couples")

    def stratified_subset_indices(self, n: int, seed: Optional[int] = None) -> List[int]:
        """按 ground_truth_divorced 分层抽样，返回 n 个 couple 的索引。

        若 n 为奇数，正负类尽量均衡，余数从样本较多一类补齐。
        """
        rng = random.Random(seed)
        divorced = [i for i, p in enumerate(self.personas) if p.get('ground_truth_divorced') == 1 or p.get('ground_truth_divorced') is True]
        married = [i for i, p in enumerate(self.personas) if p.get('ground_truth_divorced') == 0 or p.get('ground_truth_divorced') is False]

        if not divorced or not married:
            # 无法分层，退化为前 n 个
            return list(range(min(n, len(self.personas))))

        half = n // 2
        rem = n - 2 * half

        rng.shuffle(divorced)
        rng.shuffle(married)

        take_div = min(half + (1 if rem > 0 and len(divorced) >= len(married) else 0), len(divorced))
        take_mar = min(n - take_div, len(married))

        selected = divorced[:take_div] + married[:take_mar]
        # 如果还不够 n，则从剩余中补齐
        if len(selected) < n:
            remaining = divorced[take_div:] + married[take_mar:]
            rng.shuffle(remaining)
            selected += remaining[: max(0, n - len(selected))]

        return selected[:n]
    
    def save_simulations(self, output_path: str = "divorce_simulations.json"):
        """保存模拟结果"""
        output_path = Path(output_path)
        
        with save_lock:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.simulations, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Saved simulations to: {output_path}")
        
        # 保存示例
        sample_path = output_path.parent / "divorce_simulations_sample.txt"
        with open(sample_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("SAMPLE SIMULATION (First Couple, First Event)\n")
            f.write("=" * 80 + "\n\n")
            
            if self.simulations:
                first_couple = self.simulations[0]
                first_event = first_couple['events_simulations'][0]
                
                f.write(f"Couple {first_couple['couple_id']} (Divorced: {first_couple['ground_truth_divorced']})\n")
                f.write(f"Event: {first_event['event_type']}\n\n")
                
                for interaction in first_event['interactions'][:10]:  # 前 10 轮
                    f.write(f"\n--- Round {interaction.get('round', '?')} ---\n")
                    if 'speaker' in interaction:
                        f.write(f"Speaker: {interaction['speaker'].upper()}\n")
                        f.write(f"Inner Thought: {interaction['inner_thought']}\n")
                        f.write(f"Public Response: {interaction['public_response']}\n")
                    elif 'type' in interaction:
                        f.write(f"Type: {interaction['type']}\n")
                        f.write(f"Content: {interaction['content']}\n")
        
        print(f"✅ Saved sample to: {sample_path}")


def main():
    parser = argparse.ArgumentParser(description="Simulate critical events for divorce prediction")
    parser.add_argument('--personas', type=str, default='divorce_personas.json',
                       help='Path to personas JSON')
    parser.add_argument('--events', type=str, default='critical_events.json',
                       help='Path to events JSON')
    parser.add_argument('--clean-data', type=str, default='divorce_clean.csv',
                       help='Path to clean dataset (for agent ICL examples)')
    parser.add_argument('--use-icl-scenarios', action='store_true',
                       help='If set, generate scenarios dynamically using ICL instead of pre-defined events')
    parser.add_argument('--enable-agent-icl', action='store_true',
                       help='If set, inject ICL examples into agent prompts during simulation')
    parser.add_argument('--output', type=str, default='divorce_simulations.json',
                       help='Output path for simulations')
    parser.add_argument('--num-couples', type=int, default=None,
                       help='Number of couples to simulate (default: all)')
    parser.add_argument('--rounds', type=int, default=6,
                       help='Number of interaction rounds per event')
    parser.add_argument('--max-workers', type=int, default=5,
                       help='Max parallel threads')
    parser.add_argument('--subset-stratified', type=int, default=None,
                       help='If set, perform stratified sampling by ground truth to pick this many couples')
    parser.add_argument('--subset-seed', type=int, default=None,
                       help='Random seed for stratified sampling')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("CRITICAL EVENTS SIMULATOR")
    print("=" * 70)
    
    simulator = CriticalEventsSimulator(
        args.personas, 
        args.events, 
        use_icl_scenarios=args.use_icl_scenarios,
        clean_data_path=args.clean_data if args.enable_agent_icl else None,
        enable_agent_icl=args.enable_agent_icl
    )

    indices = None
    if args.subset_stratified:
        n = max(1, args.subset_stratified)
        indices = simulator.stratified_subset_indices(n=n, seed=args.subset_seed)
        print(f"\n📐 Using stratified subset of {len(indices)} couples: {indices}")

    simulator.simulate_all_couples(args.num_couples, args.max_workers, indices=indices)
    simulator.save_simulations(args.output)
    
    print("\n🎯 Next step: Evaluate predictions with 04_evaluate_predictions.py")


if __name__ == "__main__":
    main()
