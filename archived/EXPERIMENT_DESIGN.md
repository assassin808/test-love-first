# 🎭 Speed Dating LLM Matching 实验设计

## 📊 实验目标
使用 Speed Dating 数据集测试 LLM（Mistral Nemo）在约会匹配中的表现，对比真实人类匹配结果。

---

## 🏗️ 实验架构

```
真实数据 (Speed Dating Dataset)
    ↓
提取特征 → 生成 Persona Prompt
    ↓
┌─────────────────────────────────────────┐
│  场景1: Speed Dating 纯聊天 (4分钟)      │
│  - 10轮对话交互                          │
│  - 无 Dating Engine                     │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  场景2: Critical Events 模拟 (3个场景)   │
│  - Dating Engine 生成关键事件            │
│  - Agent 做重大决策                      │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  评估系统                                │
│  1. 恋爱观察员 LLM (0-10分)              │
│  2. 当事人自评 (约会意愿 + 是否在一起)    │
└─────────────────────────────────────────┘
    ↓
对比真实 match 结果 (Ground Truth)
```

---

## 📋 Phase 0: 数据预处理

### 0.1 数据加载与清洗

**关键字段**:
```python
demographics = [
    'age', 'gender', 'race', 'field_cd', 'career_c',
    'income', 'goal', 'date', 'go_out'
]

preferences_time1 = [
    'attr1_1', 'sinc1_1', 'intel1_1', 'fun1_1', 'amb1_1', 'shar1_1'
]

self_ratings = [
    'attr3_1', 'sinc3_1', 'intel3_1', 'fun3_1', 'amb3_1'
]

interests = [
    'sports', 'tvsports', 'exercise', 'dining', 'museums', 'art',
    'hiking', 'gaming', 'clubbing', 'reading', 'tv', 'theater',
    'movies', 'concerts', 'music', 'shopping', 'yoga'
]

scorecard_ratings = [
    'attr', 'sinc', 'intel', 'fun', 'amb', 'shar', 'like', 'prob'
]

ground_truth = ['dec', 'match']  # 决策和是否匹配
```

### 0.2 数据过滤策略

```python
# 高质量样本标准
quality_criteria = {
    'demographics': ['age', 'gender', 'field_cd', 'career_c'],  # 必填
    'preferences': ['attr1_1', 'sinc1_1', 'intel1_1', 'fun1_1', 'amb1_1', 'shar1_1'],
    'self_ratings': ['attr3_1', 'sinc3_1', 'intel3_1', 'fun3_1', 'amb3_1'],
    'interests': 至少10个有效值,
    'ground_truth': ['dec', 'match']  # 必须有明确结果
}

# 过滤条件
1. 关键字段无空值
2. Preference 总和 = 100 (100点分配制)
3. 参与了完整的 speed dating (有完整 scorecard)
4. 有明确的 match 结果
```

### 0.3 样本选择

```python
# 目标: 选择 50-100 对高质量配对
selection_strategy = {
    'matched_pairs': 25-50对,      # match=1 的配对
    'unmatched_pairs': 25-50对,    # match=0 的配对
    'balance': '性别平衡，年龄分布均匀'
}
```

---

## 📝 Phase 1: Persona Prompt 生成

### 1.1 《再见爱人》风格的 Persona 模板

参考《再见爱人》嘉宾介绍风格，生成真实、立体的人物描述。

**模板结构**:

```python
persona_template = """
我是{nickname}，今年{age}岁，{gender_desc}。

【基本信息】
我目前在{field}领域工作/学习，未来的职业目标是成为{career}。
我来自{location_desc}，成长环境{income_desc}。

【性格特质】（基于自我评价）
- 外貌吸引力: {attr3_1}/10 {attractiveness_desc}
- 真诚度: {sinc3_1}/10 {sincerity_desc}
- 智力水平: {intel3_1}/10 {intelligence_desc}
- 有趣程度: {fun3_1}/10 {fun_desc}
- 事业心: {amb3_1}/10 {ambition_desc}

【生活方式】
我的约会频率是{date_freq_desc}，平时{go_out_desc}。
参加这次活动的主要目的是{goal_desc}。

【兴趣爱好】
{top_3_interests}是我最热爱的活动。
我{interest_level}运动，{interest_level}艺术文化活动，{interest_level}社交娱乐。

【理想型标准】（100分分配）
在选择伴侣时，我最看重的品质排序是：
1. {top_preference_1}: {attr1_1}分
2. {top_preference_2}: {sinc1_1}分
3. {top_preference_3}: {intel1_1}分
（其他依次为: Fun {fun1_1}分, Ambitious {amb1_1}分, Shared Interests {shar1_1}分）

【对异性的理解】
我认为异性最看重的是{attr2_1_desc}，其次是{sinc2_1_desc}。

【内心独白】
{inner_monologue}  # 基于 goal, exphappy, expnum 生成
"""
```

### 1.2 生成函数示例

```python
def generate_persona_prompt(row):
    """
    将 Speed Dating 数据转化为《再见爱人》风格的角色描述
    """
    # 详见代码实现
    return persona_prompt
```

---

## 🎬 Phase 2: 场景1 - Speed Dating 纯聊天

### 2.1 场景设计

**参考**: 《再见爱人》第一次见面、《心动的信号》速配环节

**场景描述**:
```
你们在一个温馨的咖啡厅进行4分钟的速配约会。
桌上有两杯咖啡，灯光柔和，背景音乐轻柔。
这是你们第一次见面，彼此都有点紧张又充满期待。
你有4分钟的时间了解对方，决定是否愿意进一步交往。
```

### 2.2 对话轮次: 10轮

```python
conversation_structure = [
    # 轮次1-2: 破冰
    {"round": 1, "topic": "自我介绍"},
    {"round": 2, "topic": "回应+提问"},
    
    # 轮次3-6: 深入了解
    {"round": 3, "topic": "兴趣爱好"},
    {"round": 4, "topic": "工作/学业"},
    {"round": 5, "topic": "生活方式"},
    {"round": 6, "topic": "价值观/梦想"},
    
    # 轮次7-8: 测试兼容性
    {"round": 7, "topic": "周末计划/约会偏好"},
    {"round": 8, "topic": "对关系的期待"},
    
    # 轮次9-10: 收尾
    {"round": 9, "topic": "互相询问或表达好感"},
    {"round": 10, "topic": "结束语"}
]
```

---

## 🎲 Phase 3: 场景2 - Critical Events 模拟

### 3.1 理论假设

基于强化学习视角：
1. **稀疏奖励**: 关系成功取决于少数关键事件
2. **明显偏好**: 在重大决策面前，偏好明显（不是0.5概率）
3. **采样假设**: One sample is enough (单次采样足够反映偏好)

### 3.2 Critical Events 设计

参考《再见爱人》中的关键时刻：

```python
critical_event_categories = [
    "价值观冲突",      # 例如：对未来规划的分歧
    "生活方式差异",    # 例如：社交需求不同
    "情感表达方式",    # 例如：沟通风格冲突
    "家庭观念",        # 例如：对婚姻/孩子的看法
    "金钱观",          # 例如：消费观念差异
    "亲密度测试"       # 例如：是否愿意妥协/付出
]
```

### 3.3 流程

```
1. Dating Engine 基于两人persona生成3个关键场景
2. Agent1 和 Agent2 分别独立做决策
3. 揭晓选择，观察反应
4. 重复3次
```

---

## 📊 Phase 4: 评估系统

### 4.1 恋爱观察员 LLM 评分

参考《再见爱人》《心动的信号》的观察员点评风格

**评估维度**:
- overall_score: 0-10分（整体匹配度）
- chemistry_score: 0-10分（化学反应）
- value_alignment_score: 0-10分（价值观一致性）
- conflict_resolution_score: 0-10分（处理冲突的能力）
- long_term_potential: 0-10分（长期潜力）

### 4.2 当事人自评

**问题**:
1. 约会意愿 (1-10分)
2. 关系期待 (A.试试看 / B.开始约会 / C.认真交往 / D.无兴趣)
3. 兼容性自评 (1-10分)
4. 最吸引的点
5. 最大的顾虑
6. 是否愿意在一起 (是/否/需要更多时间)

---

## 🎯 Phase 5: 对比真实数据

### 5.1 Ground Truth 提取

```python
ground_truth = {
    'person1_decision': dec (1=yes, 0=no),
    'person2_decision': dec (1=yes, 0=no),
    'match': match (1=match, 0=no match),
    'person1_ratings': {attr, sinc, intel, fun, amb, like},
    'person2_ratings': {attr, sinc, intel, fun, amb, like}
}
```

### 5.2 评估指标

```python
metrics = {
    'match_prediction_accuracy': '预测 match 的准确率',
    'decision_prediction_accuracy': '预测个人决策的准确率',
    'rating_correlation': 'LLM评分与真实评分的相关性',
    'observer_score_correlation': '观察员评分与真实like评分的相关性'
}
```

---

## 🔧 Phase 6: Mistral Nemo 配置

### 6.1 模型选择

```python
model = "mistralai/mistral-nemo"

# Via OpenRouter
OPENROUTER_API_KEY=sk-or-v1-...
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
```

### 6.2 参数设置

```python
generation_params = {
    'temperature': 0.7,  # Speed Dating 和 Critical Events
    'temperature': 0.3,  # 评估系统（更客观）
    'response_format': {"type": "json_object"}  # 强制JSON输出
}
```

---

## 📁 文件结构

```
test/
├── Speed Dating Data.csv              # 原始数据
├── Speed Dating Data Key.txt          # 数据说明
├── EXPERIMENT_DESIGN.md               # 本文件
│
├── experiments/
│   ├── data_preprocessing.py          # 数据预处理
│   ├── persona_generator.py           # Persona生成器
│   ├── speed_dating_simulator.py      # 场景1模拟器
│   ├── critical_events_engine.py      # 场景2引擎
│   ├── evaluation_system.py           # 评估系统
│   ├── main_experiment.py             # 主实验脚本
│   └── analysis.py                    # 结果分析
│
├── results/
│   ├── processed_samples.json         # 处理后的样本
│   ├── personas.json                  # 生成的personas
│   ├── experiments/
│   │   ├── pair_001/                  # 每对的详细记录
│   │   └── ...
│   ├── summary_statistics.json        # 总体统计
│   └── analysis_report.md             # 分析报告
│
└── .env                               # 配置文件
```

---

## 📊 实验流程总结

```python
def run_full_experiment(pair_id, person1_data, person2_data):
    """
    完整实验流程
    """
    
    # Phase 1: 生成 Personas
    persona1 = generate_persona_prompt(person1_data)
    persona2 = generate_persona_prompt(person2_data)
    
    # Phase 2: Speed Dating
    speed_dating_log = run_speed_dating(agent1, agent2)
    
    # Phase 3: Critical Events
    critical_events_log = run_critical_events(agent1, agent2, speed_dating_log)
    
    # Phase 4: 评估
    observer_eval = LoveObserver().evaluate_couple(speed_dating_log, critical_events_log)
    agent1_eval = evaluate_self(agent1, all_logs)
    agent2_eval = evaluate_self(agent2, all_logs)
    
    # Phase 5: 对比真实数据
    ground_truth = extract_ground_truth(data, person1_data['iid'], person2_data['iid'])
    comparison = compare_with_ground_truth(llm_results, ground_truth)
    
    return comparison
```

---

## 🎯 预期成果

1. **定量结果**
   - Match预测准确率: 目标 > 65%
   - 决策预测准确率: 目标 > 60%
   - 评分相关性: 目标 r > 0.5

2. **定性发现**
   - LLM 在哪些类型的配对上表现更好？
   - 哪些特征对匹配预测最重要？
   - Critical Events 是否比 Speed Dating 更能预测长期兼容性？

3. **Workshop 贡献**
   - PersonaLLM 在约会场景的应用验证
   - "Love First, Know Later" 理念的实证支持
   - LLM作为关系模拟器的潜力

---

**准备开始实现！** 🚀
