"""
Divorce Prediction Evaluation

评估方法：
1. Baseline: Logistic Regression (使用 clean features，移除泄漏特征)
2. Observer Method: LLM 观察员根据交互判断是否离婚
3. Participant Method: Husband/Wife 自己判断婚姻前景

参考：test/experiments/llm_score_evaluator.py (Speed Dating)
"""

import os
import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import threading
import random
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

# Import question mappings
from divorce_questions import format_couple_features

# 加载环境变量
load_dotenv()

# Thread-safe lock
save_lock = threading.Lock()

# Global API concurrency & rate limiting (for evaluator as well)
_API_SEM = threading.Semaphore(int(os.getenv('OPENROUTER_MAX_CONCURRENCY', '2')))
_RATE_LOCK = threading.Lock()
_LAST_CALL_TS = 0.0
_RPS = float(os.getenv('OPENROUTER_RPS', '0.8'))


class DivorceEvaluator:
    """评估离婚预测（Baseline + LLM Methods）"""
    
    def __init__(self, simulations_path: str, clean_data_path: str, personas_path: Optional[str] = None, threshold: float = 5.0, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('OPENROUTER_API_KEY')
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not found")
        
        base_url = os.getenv('OPENROUTER_BASE_URL', 'https://openrouter.ai/api/v1')
        self.api_url = f"{base_url.rstrip('/')}/chat/completions"
        self.model = os.getenv('OPENROUTER_MODEL', 'mistralai/mistral-nemo')
        
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
            allowed_methods=False,
        )
        adapter = HTTPAdapter(max_retries=retry, pool_connections=20, pool_maxsize=20)
        self._session.mount('https://', adapter)
        self._session.mount('http://', adapter)
        self.threshold = float(threshold)
        
        # 加载数据
        with open(simulations_path, 'r', encoding='utf-8') as f:
            self.simulations = json.load(f)
        
        # 读取 clean data（移除泄漏特征后的数据）
        # divorce_clean.csv 使用逗号分隔
        self.df_clean = pd.read_csv(clean_data_path)

        # 可选：加载 personas（用于背景）
        self.personas = None
        if personas_path and Path(personas_path).exists():
            with open(personas_path, 'r', encoding='utf-8') as f:
                self.personas = json.load(f)
        
        
        print(f"✅ Loaded {len(self.simulations)} simulations")
        print(f"✅ Loaded {len(self.df_clean)} couples from clean dataset")
        
        self.results = []

        # Cache lists for ICL sampling
        self._icl_divorced_ids: List[int] = []
        self._icl_married_ids: List[int] = []
        if self.personas:
            for i, p in enumerate(self.personas):
                try:
                    gt = 1 if p.get('ground_truth_divorced') in (1, True) else 0
                    if gt == 1:
                        self._icl_divorced_ids.append(i)
                    else:
                        self._icl_married_ids.append(i)
                except Exception:
                    continue
    
    def call_openrouter_api(self, messages: List[Dict], temperature: float = 0.3, max_retries: int = 5) -> Dict[str, str]:
        """调用 OpenRouter API：并发限制、RPS 节流、指数退避 + 抖动、处理 429。"""
        import requests

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 300
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
                    time.sleep(random.uniform(0.05, 0.15))
                    return {"success": True, "content": content}
                except requests.exceptions.RequestException as e:
                    if attempt >= max_retries:
                        return {"success": False, "error": str(e)}
                    delay = backoff_base * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
                    print(f"   ⚠️  API error (attempt {attempt}/{max_retries}): {e}")
                    time.sleep(delay)
            return {"success": False, "error": "Max retries exceeded"}
    
    # ========== BASELINE: Logistic Regression ==========
    
    def train_baseline(self, test_size: float = 0.2, random_state: int = 42):
        """训练 Logistic Regression baseline（使用 clean features）"""
        print("\n" + "=" * 70)
        print("BASELINE: LOGISTIC REGRESSION")
        print("=" * 70)
        
        # 准备特征（移除 Class 列）
        X = self.df_clean.drop('Class', axis=1)
        y = self.df_clean['Class']
        
        # 转换标签（0=divorced, 1=married -> 0=married, 1=divorced）
        y = (y == 0).astype(int)
        
        print(f"\n📊 Dataset: {len(X)} samples, {X.shape[1]} features")
        print(f"   Divorced: {y.sum()} ({y.sum()/len(y)*100:.1f}%)")
        print(f"   Married: {(1-y).sum()} ({(1-y).sum()/len(y)*100:.1f}%)")
        
        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"\n📊 Train: {len(X_train)} samples | Test: {len(X_test)} samples")
        
        # 训练模型
        lr = LogisticRegression(max_iter=1000, random_state=random_state)
        lr.fit(X_train, y_train)
        
        # 评估
        y_pred = lr.predict(X_test)
        y_pred_proba = lr.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        print(f"\n✅ Baseline Results:")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   AUC-ROC: {auc:.4f}")
        print(f"\n{classification_report(y_test, y_pred, target_names=['Married', 'Divorced'])}")
        
        # 保存结果
        baseline_results = {
            "method": "Logistic Regression (Baseline)",
            "accuracy": float(accuracy),
            "auc": float(auc),
            "n_features": X.shape[1],
            "test_size": len(X_test),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "classification_report": classification_report(y_test, y_pred, target_names=['Married', 'Divorced'], output_dict=True)
        }
        
        return baseline_results, lr, (X_test, y_test, y_pred_proba)
    
    # ========== OBSERVER METHOD ==========
    
    def evaluate_observer(self, couple_data: Dict) -> Dict:
        """观察员评估：LLM 根据交互判断是否会离婚（强化 ICL 意识和评分标准）"""
        couple_id = couple_data['couple_id']
        
        # 构建观察摘要（所有事件）
        observation_text = self._build_observation_summary(couple_data)

        # 背景：两位参与者的 persona 概要
        husband_bg = couple_data.get('husband_persona')
        wife_bg = couple_data.get('wife_persona')
        if (husband_bg is None or wife_bg is None) and self.personas is not None and couple_id < len(self.personas):
            try:
                husband_bg = self.personas[couple_id]['husband']['persona_narrative']
                wife_bg = self.personas[couple_id]['wife']['persona_narrative']
            except Exception:
                pass
        husband_bg = husband_bg or "[background unavailable]"
        wife_bg = wife_bg or "[background unavailable]"

        # 采样 5 个离婚 + 5 个未离婚的参考（使用问卷格式）
        icl_examples = self._select_icl_examples(k_per_class=5, exclude_id=couple_id, seed=42)
        reference_patterns = ""
        for cid, lbl in icl_examples:
            try:
                row = self.df_clean.iloc[cid]
                features = row.drop('Class').values.tolist()
                features_str = format_couple_features(features)
                
                # Generate score (same logic as _build_icl_block)
                rng = random.Random(cid)
                if lbl == 1:  # Divorced
                    score = round(rng.uniform(6.0, 9.0), 1)
                else:  # Married
                    score = round(rng.uniform(1.0, 4.5), 1)
                
                status = "Divorced" if lbl == 1 else "Stayed Married"
                reference_patterns += (
                    f"\n<REFERENCE_COUPLE outcome=\"{status}\" score=\"{score}\">\n"
                    f"{features_str}\n"
                    f"</REFERENCE_COUPLE>\n\n"
                )
            except Exception:
                continue

        # Get target couple's survey responses
        try:
            target_row = self.df_clean.iloc[couple_id]
            target_features = target_row.drop('Class').values.tolist()
            target_features_str = format_couple_features(target_features)
        except Exception:
            target_features_str = "[features unavailable]"        # Observer prompt（使用问卷 + 数值格式）
        prompt = f"""You are an expert relationship psychologist. You have 10 reference couples' survey responses (27 questions, 0-4 scale: 0=Never, 1=Seldom, 2=Sometimes, 3=Often, 4=Always).

=== REFERENCE COUPLES (for calibration) ===
{reference_patterns}

**CALIBRATION PATTERN:**
- Divorced couples (score 6.0-9.0): Low scores on communication, trust, shared values → incompatibility
- Married couples (score 1.0-4.5): High scores on communication, trust, shared values → compatibility
- **Decision rule: Score > 5.0 = Divorced; Score ≤ 5.0 = Married**

=== TARGET COUPLE ===
<SURVEY_RESPONSES>
{target_features_str}
</SURVEY_RESPONSES>

<BEHAVIORAL_OBSERVATIONS>
{observation_text}
</BEHAVIORAL_OBSERVATIONS>

=== YOUR TASK ===
1. **Compare** target's survey responses with the 10 reference couples
2. **Identify** which references have similar response patterns (especially on key questions like trust, communication, shared values)
3. **Assign a divorce likelihood score** matching the similar references' range

Output ONLY:

<ANALYSIS>
[Which references are most similar? Which questions show divorce risk vs. stability? - 2-3 sentences]
</ANALYSIS>

<SCORE>x.x</SCORE>

One decimal place. Base score on survey similarity."""
        
        # 调用 LLM
        response = self.call_openrouter_api([
            {"role": "system", "content": "You are an expert relationship psychologist specializing in divorce prediction."},
            {"role": "user", "content": prompt}
        ], temperature=0.3)

        print(f"prompt detailed: {prompt} ")
        
        if not response['success']:
            return {"error": "API call failed"}
        
        # 解析评分
        score = self._parse_score(response['content'])
        label = 1 if score is not None and score >= self.threshold else 0  # 1=divorced, 0=married
        
        return {
            "couple_id": couple_id,
            "method": "observer",
            "score": score,
            "pred_label": label,
            "raw_response": response['content'],
            "ground_truth": couple_data['ground_truth_divorced']
        }
    
    def _build_observation_summary(self, couple_data: Dict) -> str:
        """构建观察员看到的交互摘要"""
        events_simulations = couple_data['events_simulations']
        
        summary_parts = []
        
        for idx, event_sim in enumerate(events_simulations, 1):
            event_type = event_sim['event_type']
            interactions = event_sim['interactions']
            
            summary_parts.append(f"\n--- Event {idx}: {event_type.replace('_', ' ').title()} ---")
            
            # 提取关键交互（前 6 轮公开行为）
            for interaction in interactions[:8]:
                if 'speaker' in interaction:
                    speaker = interaction['speaker'].capitalize()
                    public_response = interaction['public_response']
                    summary_parts.append(f"{speaker}: {public_response}")
                elif 'type' in interaction and interaction['type'] == 'scene_setting':
                    summary_parts.append(f"[Scene]: {interaction['content']}")
        
        return "\n".join(summary_parts)

    # ========== OBSERVER METHOD (ICL) ==========

    def _select_icl_examples(self, k_per_class: int = 5, exclude_id: Optional[int] = None, seed: int = 42) -> List[Tuple[int, int]]:
        """Select k_per_class married and divorced couple indices for ICL examples.

        Returns a list of tuples (couple_id, label) where label is 0=married, 1=divorced.
        """
        rng = random.Random(seed)
        div = [i for i in self._icl_divorced_ids if i != exclude_id]
        mar = [i for i in self._icl_married_ids if i != exclude_id]
        rng.shuffle(div)
        rng.shuffle(mar)
        take_d = div[:k_per_class]
        take_m = mar[:k_per_class]
        examples = [(i, 1) for i in take_d] + [(i, 0) for i in take_m]
        rng.shuffle(examples)
        return examples

    def _build_icl_block(self, couple_id: int, label: int) -> str:
        """Build one ICL example with QUESTION + NUMERIC ANSWER format."""
        # Get numeric features from clean dataset
        try:
            row = self.df_clean.iloc[couple_id]
            features = row.drop('Class').values.tolist()
            # Format as question-answer pairs
            features_str = format_couple_features(features)
        except Exception:
            features_str = "[features unavailable]"
        
        # Generate a realistic score based on label
        # Divorced: 6.0-9.0 range, Married: 1.0-4.5 range
        rng = random.Random(couple_id)
        if label == 1:  # Divorced
            score = round(rng.uniform(6.0, 9.0), 1)
        else:  # Married
            score = round(rng.uniform(1.0, 4.5), 1)
        
        lbl_txt = 'Divorced' if label == 1 else 'Married'
        return (
            f"<EXAMPLE>\n"
            f"<SURVEY_RESPONSES>\n{features_str}\n</SURVEY_RESPONSES>\n"
            f"<DIVORCE_LIKELIHOOD_SCORE>{score}</DIVORCE_LIKELIHOOD_SCORE>\n"
            f"<LABEL>{lbl_txt}</LABEL>\n"
            f"</EXAMPLE>\n\n"
        )

    def evaluate_observer_icl(self, couple_data: Dict, k_per_class: int = 5, seed: int = 42) -> Dict:
        """Observer with ICL using QUESTION + ANSWER format with semantic meaning."""
        couple_id = couple_data['couple_id']

        # Build ICL examples with survey Q&A format
        examples = self._select_icl_examples(k_per_class=k_per_class, exclude_id=couple_id, seed=seed)
        icl_text = "".join([self._build_icl_block(cid, lbl) for cid, lbl in examples])

        # Get target couple's survey responses
        try:
            target_row = self.df_clean.iloc[couple_id]
            target_features = target_row.drop('Class').values.tolist()
            target_features_str = format_couple_features(target_features)
        except Exception:
            target_features_str = "[features unavailable]"

        # Also include observation summary for additional signal
        observation_text = self._build_observation_summary(couple_data)

        prompt = f"""You are an expert relationship psychologist. You have 10 labeled couples' survey responses to learn from.

=== TRAINING EXAMPLES (10 labeled couples) ===
Each couple answered 27 questions about their relationship (0=Never, 1=Seldom, 2=Sometimes, 3=Often, 4=Always).

{icl_text}

**LEARN THESE PATTERNS:**
- **Divorced couples (score 6.0-9.0)**: Low scores on trust, communication, shared values → incompatibility signals
- **Married couples (score 1.0-4.5)**: High scores on trust, communication, shared values → compatibility signals
- **Decision rule: Score > 5.0 = Divorced; Score ≤ 5.0 = Married**

=== TARGET COUPLE (unknown status) ===
<SURVEY_RESPONSES>
{target_features_str}
</SURVEY_RESPONSES>

<BEHAVIORAL_OBSERVATIONS>
{observation_text}
</BEHAVIORAL_OBSERVATIONS>

=== YOUR TASK ===
1. **Compare** target's survey responses with the 10 labeled examples
2. **Identify** which examples show similar patterns (especially key questions: trust, communication, shared goals)
3. **Assign a score** in the range matching similar examples

Output ONLY:

<ANALYSIS>
[Which examples are most similar? What survey patterns indicate divorce risk vs. stability? - 3-4 sentences]
</ANALYSIS>

<SCORE>x.x</SCORE>

One decimal place. Base score on survey response similarity."""

        response = self.call_openrouter_api([
            {"role": "system", "content": "You are an expert relationship psychologist. Be decisive and numeric in outputs."},
            {"role": "user", "content": prompt}
        ], temperature=0.3)

        if not response['success']:
            return {"error": "API call failed"}

        score = self._parse_score(response['content'])
        label = 1 if score is not None and score >= self.threshold else 0

        return {
            "couple_id": couple_id,
            "method": "observer_icl",
            "score": score,
            "pred_label": label,
            "raw_response": response['content'],
            "ground_truth": couple_data['ground_truth_divorced']
        }
    
    # ========== PARTICIPANT METHOD ==========
    
    def evaluate_participant_icl(self, couple_data: Dict, k_per_class: int = 5, seed: int = 42) -> Dict:
        """Participant method with ICL: Show examples of self-assessment → actual divorce outcome."""
        couple_id = couple_data['couple_id']
        
        # Build ICL examples (same data, but framed as "self-assessment")
        examples = self._select_icl_examples(k_per_class=k_per_class, exclude_id=couple_id, seed=seed)
        icl_text = self._build_participant_icl_block(examples)
        
        # Get participant's own history
        results = {"husband": None, "wife": None}
        
        for role in ['husband', 'wife']:
            history_text = self._build_participant_history(couple_data, role)
            bg_text = couple_data.get('husband_persona') if role == 'husband' else couple_data.get('wife_persona')
            if (bg_text is None) and self.personas is not None and couple_id < len(self.personas):
                try:
                    bg_text = self.personas[couple_id][role]['persona_narrative']
                except Exception:
                    pass
            bg_text = bg_text or "[background unavailable]"
            
            prompt = f"""You are the {role} in this marriage. Based on your experiences and feelings, assess your divorce likelihood.

=== TRAINING EXAMPLES (10 labeled couples' self-perceived divorce risk) ===
{icl_text}

**LEARN THESE PATTERNS:**
- **Divorced couples (score 6.0-9.0)**: High perceived divorce risk → actually divorced
- **Married couples (score 1.0-4.5)**: Low perceived divorce risk → stayed married
- **Key insight**: Self-perceived divorce likelihood aligns with actual outcome

=== YOUR SITUATION ===
<YOUR_BACKGROUND>
{bg_text}
</YOUR_BACKGROUND>

<WHAT_YOU_EXPERIENCED>
{history_text}
</WHAT_YOU_EXPERIENCED>

=== YOUR TASK ===
1. **Compare** your feelings with the 10 labeled examples
2. **Identify** which examples match your current emotional state
3. **Rate your divorce likelihood** on a 0.0-10.0 scale

Output ONLY:

<REFLECTION>
[Comparing with examples: which couples' situations are similar to yours? How likely is divorce in your view? - 2-3 sentences]
</REFLECTION>

<SCORE>x.x</SCORE>

One decimal place (0.0 = no divorce risk, 10.0 = definitely heading to divorce)."""
            
            response = self.call_openrouter_api([
                {"role": "system", "content": "You are role-playing as a married person reflecting on your relationship."},
                {"role": "user", "content": prompt}
            ], temperature=0.5)
            
            if response['success']:
                score = self._parse_score(response['content'])
                results[role] = {
                    "score": score,
                    "raw_response": response['content']
                }
        
        # Now scores are direct divorce likelihood (no mapping needed)
        h_score = results['husband']['score'] if results['husband'] else None
        w_score = results['wife']['score'] if results['wife'] else None
        scores = []
        if h_score is not None:
            scores.append(h_score)
        if w_score is not None:
            scores.append(w_score)
        final_score = float(np.mean(scores)) if scores else None
        final_label = 1 if (final_score is not None and final_score >= self.threshold) else 0
        
        return {
            "couple_id": couple_id,
            "method": "participant_icl",
            "score": final_score,
            "pred_label": final_label,
            "husband": results['husband'],
            "wife": results['wife'],
            "ground_truth": couple_data['ground_truth_divorced']
        }
    
    def _build_participant_icl_block(self, examples: List[Tuple[int, int]]) -> str:
        """Build ICL examples for participant method (self-assessment perspective)."""
        icl_parts = []
        
        for couple_id, label in examples:
            # Get survey responses
            try:
                row = self.df_clean.iloc[couple_id]
                features = row.drop('Class').values.tolist()
                features_str = format_couple_features(features)
            except Exception:
                features_str = "[features unavailable]"
            
            # Generate realistic DIVORCE LIKELIHOOD scores directly (same as observer-icl)
            # Divorced: 6.0-9.0 (high divorce risk), Married: 1.0-4.5 (low divorce risk)
            rng = random.Random(couple_id)
            if label == 1:  # Divorced
                divorce_likelihood = round(rng.uniform(6.0, 9.0), 1)
            else:  # Married
                divorce_likelihood = round(rng.uniform(1.0, 4.5), 1)
            
            lbl_txt = 'Divorced' if label == 1 else 'Married'
            
            icl_parts.append(
                f"<EXAMPLE>\n"
                f"<COUPLE_SELF_ASSESSMENT>\n{features_str}\n</COUPLE_SELF_ASSESSMENT>\n"
                f"<PERCEIVED_DIVORCE_LIKELIHOOD>{divorce_likelihood}</PERCEIVED_DIVORCE_LIKELIHOOD>\n"
                f"<ACTUAL_OUTCOME>{lbl_txt}</ACTUAL_OUTCOME>\n"
                f"</EXAMPLE>\n"
            )
        
        return "\n".join(icl_parts)
    
    def evaluate_participant(self, couple_data: Dict) -> Dict:
        """参与者评估：Husband/Wife 自己判断婚姻前景"""
        couple_id = couple_data['couple_id']
        
        results = {"husband": None, "wife": None}
        
        # 分别询问 Husband 和 Wife
        for role in ['husband', 'wife']:
            # 构建参与者视角的历史
            history_text = self._build_participant_history(couple_data, role)
            # 背景（persona 概述）
            bg_text = couple_data.get('husband_persona') if role == 'husband' else couple_data.get('wife_persona')
            if (bg_text is None) and self.personas is not None and couple_id < len(self.personas):
                try:
                    bg_text = self.personas[couple_id][role]['persona_narrative']
                except Exception:
                    pass
            bg_text = bg_text or "[background unavailable]"
            
            prompt = f"""You are the {role} in this marriage. You just experienced three major life events with your spouse.

=== YOUR BACKGROUND (PERSONA) ===
{bg_text}

=== WHAT YOU EXPERIENCED ===
{history_text}

=== REFLECTION QUESTION ===
How willing are you to continue being with your spouse? Answer on a 0.0 to 10.0 scale (one decimal place), where 0.0 means "not willing at all" and 10.0 means "absolutely willing".

Output your assessment in this EXACT format (no extra text):

<REFLECTION>
[Your honest feelings about the marriage - 2-3 sentences]
</REFLECTION>

<SCORE>x.x</SCORE>"""
            
            # 调用 LLM
            response = self.call_openrouter_api([
                {"role": "user", "content": prompt}
            ], temperature=0.5)
            
            if response['success']:
                score = self._parse_score(response['content'])
                results[role] = {
                    "score": score,
                    "raw_response": response['content']
                }
        
        # 综合判断（使用 10 - willingness 作为离婚倾向评分再与阈值比较）
        # 映射：willingness score (0-10) -> divorce likelihood = 10 - willingness
        h_score = results['husband']['score'] if results['husband'] else None
        w_score = results['wife']['score'] if results['wife'] else None
        # 若任一缺失，则仅用另一个；若两者都有，则取平均
        mapped_scores = []
        if h_score is not None:
            mapped_scores.append(10.0 - h_score)
        if w_score is not None:
            mapped_scores.append(10.0 - w_score)
        final_score = float(np.mean(mapped_scores)) if mapped_scores else None
        final_label = 1 if (final_score is not None and final_score >= self.threshold) else 0
        
        return {
            "couple_id": couple_id,
            "method": "participant",
            "score": final_score,
            "pred_label": final_label,
            "husband": results['husband'],
            "wife": results['wife'],
            "ground_truth": couple_data['ground_truth_divorced']
        }
    
    def _build_participant_history(self, couple_data: Dict, role: str) -> str:
        """构建参与者视角的历史（包含内心想法）"""
        events_simulations = couple_data['events_simulations']
        
        history_parts = []
        
        for idx, event_sim in enumerate(events_simulations, 1):
            event_type = event_sim['event_type']
            interactions = event_sim['interactions']
            
            history_parts.append(f"\n--- Event {idx}: {event_type.replace('_', ' ').title()} ---")
            
            # 提取该角色的交互（包含 inner thoughts）
            for interaction in interactions[:8]:
                if 'speaker' in interaction:
                    if interaction['speaker'] == role:
                        # 自己的想法和行为
                        history_parts.append(f"[Your inner thought]: {interaction['inner_thought']}")
                        history_parts.append(f"[What you said/did]: {interaction['public_response']}")
                    else:
                        # 对方的行为（看不到对方内心）
                        other = "wife" if role == "husband" else "husband"
                        history_parts.append(f"[{other.capitalize()}'s response]: {interaction['public_response']}")
                elif 'type' in interaction and interaction['type'] == 'scene_setting':
                    history_parts.append(f"[Situation]: {interaction['content']}")
        
        return "\n".join(history_parts)
    
    def _parse_score(self, text: str) -> Optional[float]:
        """从 LLM 输出中提取 0.0-10.0 的分数"""
        import re
        
        # 优先解析 <SCORE>x.x</SCORE>
        match = re.search(r'<SCORE>\s*([0-9]+(?:\.[0-9])?)\s*</SCORE>', text, re.IGNORECASE)
        if match:
            try:
                val = float(match.group(1))
                # 裁剪到 0-10 范围
                return max(0.0, min(10.0, val))
            except ValueError:
                pass
        
        # Fallback: 文本中的第一个 0-10 范围的数字
        match2 = re.search(r'\b([0-9]+(?:\.[0-9])?)\b', text)
        if match2:
            try:
                val = float(match2.group(1))
                if 0.0 <= val <= 10.0:
                    return val
            except ValueError:
                pass
        return None
    
    # ========== 并行评估 ==========
    
    def evaluate_all(self, method: str = "both", max_workers: int = 5, icl_k: int = 5, icl_seed: int = 42):
        """并行评估所有夫妻"""
        print(f"\n{'='*70}")
        print(f"LLM EVALUATION: {method.upper()}")
        print(f"{'='*70}")
        
        methods_to_run: List[str] = []
        if method == "both":
            methods_to_run = ["observer", "participant"]
        elif method in ["observer", "participant", "observer-icl", "participant-icl", "all"]:
            if method == "all":
                methods_to_run = ["observer", "observer-icl", "participant", "participant-icl"]
            else:
                methods_to_run = [method]
        else:
            raise ValueError(f"Invalid method: {method}")
        
        for eval_method in methods_to_run:
            print(f"\n🎯 Running {eval_method} method...")
            
            if eval_method == "observer":
                eval_func = self.evaluate_observer
            elif eval_method == "observer-icl":
                eval_func = lambda sim: self.evaluate_observer_icl(sim, k_per_class=icl_k, seed=icl_seed)
            elif eval_method == "participant-icl":
                eval_func = lambda sim: self.evaluate_participant_icl(sim, k_per_class=icl_k, seed=icl_seed)
            else:
                eval_func = self.evaluate_participant
            
            # Clamp workers to global concurrency to prevent bursts
            try:
                max_workers = max(1, min(max_workers, int(os.getenv('OPENROUTER_MAX_CONCURRENCY', '2'))))
            except Exception:
                max_workers = max(1, max_workers)

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(eval_func, sim): sim['couple_id']
                    for sim in self.simulations
                }
                
                for future in as_completed(futures):
                    couple_id = futures[future]
                    try:
                        result = future.result()
                        self.results.append(result)
                        
                        # 计算当前准确率（基于分数与阈值）
                        if 'error' not in result:
                            pred_label = result.get('pred_label')
                            if pred_label is None and result.get('score') is not None:
                                pred_label = 1 if result['score'] >= self.threshold else 0
                            truth = 1 if result['ground_truth'] == 1 else 0
                            if pred_label is not None:
                                correct = (pred_label == truth)
                                label_txt = 'divorced' if pred_label == 1 else 'married'
                                print(f"   ✅ Couple {couple_id}: {label_txt} (Truth: {'divorced' if truth else 'married'}) - {'✓' if correct else '✗'}")
                    except Exception as e:
                        print(f"   ❌ Error on couple {couple_id}: {e}")
        
        print(f"\n✅ Evaluation complete: {len(self.results)} predictions")
    
    # ========== 结果分析 ==========
    
    def compute_metrics(self):
        """计算各方法的准确率和 AUC"""
        print(f"\n{'='*70}")
        print("RESULTS SUMMARY")
        print(f"{'='*70}")
        
        # 按方法分组
        observer_results = [r for r in self.results if r.get('method') == 'observer' and 'error' not in r]
        observer_icl_results = [r for r in self.results if r.get('method') == 'observer_icl' and 'error' not in r]
        participant_results = [r for r in self.results if r.get('method') == 'participant' and 'error' not in r]
        participant_icl_results = [r for r in self.results if r.get('method') == 'participant_icl' and 'error' not in r]
        
        summary = {}
        
        for method_name, method_results in [
            ("Observer", observer_results), 
            ("Observer-ICL", observer_icl_results), 
            ("Participant", participant_results),
            ("Participant-ICL", participant_icl_results)
        ]:
            if not method_results:
                continue
            
            # 计算准确率（通过阈值把分数映射到标签：1=divorced, 0=married）
            y_true = [r['ground_truth'] for r in method_results]
            y_pred = [1 if (r.get('score') is not None and r['score'] >= self.threshold) else 0 for r in method_results]
            y_scores = [r.get('score') if r.get('score') is not None else 0.0 for r in method_results]
            
            correct = sum(int(p == t) for p, t in zip(y_pred, y_true))
            total = len(method_results)
            accuracy = correct / total if total > 0 else 0
            
            # 计算 AUC（使用连续分数）
            try:
                auc = roc_auc_score(y_true, y_scores)
            except Exception as e:
                print(f"   ⚠️  Could not compute AUC: {e}")
                auc = None
            
            print(f"\n🎯 {method_name} Method:")
            print(f"   Accuracy: {accuracy:.4f} ({correct}/{total})")
            if auc is not None:
                print(f"   AUC-ROC:  {auc:.4f}")
            # 使用固定标签顺序，避免单类别时出错
            print(f"\n{classification_report(y_true, y_pred, labels=[0,1], target_names=['Married', 'Divorced'], zero_division=0)}")
            
            summary[method_name.lower()] = {
                "accuracy": float(accuracy),
                "auc": float(auc) if auc is not None else None,
                "n_samples": total,
                "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0,1]).tolist(),
                "threshold": self.threshold
            }
        
        return summary
    
    def save_results(self, output_path: str = "divorce_evaluation_results.json", baseline_results: Optional[Dict] = None):
        """保存评估结果"""
        output_path = Path(output_path)
        
        # 计算 LLM 方法的 metrics
        llm_metrics = self.compute_metrics()
        
        # 综合结果
        full_results = {
            "timestamp": datetime.now().isoformat(),
            "n_couples_evaluated": len(self.simulations),
            "baseline": baseline_results,
            "llm_methods": llm_metrics,
            "detailed_predictions": self.results
        }
        
        with save_lock:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(full_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Saved evaluation results to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate divorce predictions")
    parser.add_argument('--simulations', type=str, default='divorce_simulations.json',
                       help='Path to simulations JSON')
    parser.add_argument('--clean-data', type=str, default='divorce_clean.csv',
                       help='Path to clean dataset (removed leakage features)')
    parser.add_argument('--personas', type=str, default=None,
                       help='Optional path to personas JSON (for background injection)')
    parser.add_argument('--output', type=str, default='divorce_evaluation_results.json',
                       help='Output path for evaluation results')
    parser.add_argument('--method', type=str, default='both', choices=['observer', 'participant', 'observer-icl', 'participant-icl', 'both', 'all'],
                       help='LLM evaluation method')
    parser.add_argument('--max-workers', type=int, default=5,
                       help='Max parallel threads for LLM evaluation')
    parser.add_argument('--icl-k', type=int, default=5, help='ICL: examples per class (married/divorced)')
    parser.add_argument('--icl-seed', type=int, default=42, help='ICL: random seed')
    parser.add_argument('--skip-baseline', action='store_true',
                       help='Skip baseline Logistic Regression')
    parser.add_argument('--threshold', type=float, default=5.0,
                       help='Decision threshold on 0-10 score for divorce classification (>= means divorced)')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("DIVORCE PREDICTION EVALUATION")
    print("=" * 70)
    
    evaluator = DivorceEvaluator(args.simulations, args.clean_data, args.personas, args.threshold)
    
    # 1. Baseline
    baseline_results = None
    if not args.skip_baseline:
        baseline_results, lr_model, _ = evaluator.train_baseline()
    
    # 2. LLM Evaluation
    evaluator.evaluate_all(args.method, args.max_workers, icl_k=args.icl_k, icl_seed=args.icl_seed)
    
    # 3. Save Results
    evaluator.save_results(args.output, baseline_results)
    
    print("\n✅ Evaluation pipeline complete!")


if __name__ == "__main__":
    main()
