# Speed Dating LLM Experiment

## Phase 2: Speed Dating Simulator (使用 OpenRouter API)

### 🔑 设置 API Key

1. **获取 OpenRouter API Key**:
   - 访问 https://openrouter.ai/
   - 注册账号并获取 API key

2. **设置环境变量**:

   方法 1: 创建 `.env` 文件（推荐）
   ```bash
   echo "OPENROUTER_API_KEY=your-key-here" > .env
   ```

   方法 2: 导出环境变量
   ```bash
   export OPENROUTER_API_KEY='your-key-here'
   ```

3. **验证 API Key**:
   ```bash
   python3 test_api_key.py
   ```

### 📦 安装依赖

```bash
pip3 install -r requirements.txt
```

或使用 virtual environment:
```bash
source ../.venv/bin/activate
pip install -r requirements.txt
```

### 🚀 运行模拟器

```bash
python3 experiments/speed_dating_simulator.py
```

模式选择：
- **Test mode (1)**: 测试前 2 对（快速验证）
- **Small batch (2)**: 前 10 对（小规模测试）
- **Full simulation (3)**: 所有 100 对（完整实验）

### 📁 输出文件

- `results/conversations.json` - 完整对话 JSON
- `results/conversations_sample.txt` - 前 3 对对话示例（可读格式）
- `results/conversations_checkpoint_*.json` - 每 5 对自动保存的检查点

### 💰 API 成本估算

**Mistral Nemo 定价** (via OpenRouter):
- Input: $0.13 / 1M tokens
- Output: $0.13 / 1M tokens

**每对对话估算**:
- System prompt: ~600 tokens
- 10 轮对话: ~2000 tokens (input + output)
- 总计: ~2600 tokens/pair

**100 对完整实验**:
- 总 tokens: ~260,000 tokens
- **预估成本: ~$0.034 (约 $0.03-0.05)**

非常便宜！✅

### 🎯 实验流程

```
Phase 0: Data Preprocessing ✅
         ↓
Phase 1: Persona Generation ✅
         ↓
Phase 2: Speed Dating Simulator ← 当前阶段
         ↓
Phase 3: Evaluation System
         ↓
Phase 4: Analysis & Comparison
```

### 📝 对话格式示例

```json
{
  "pair_id": "pair_001",
  "person1_iid": 467,
  "person2_iid": 492,
  "ground_truth": {
    "match": 1,
    "person1_dec": 1,
    "person2_dec": 1
  },
  "rounds": [
    {
      "round": 0,
      "speaker": "person1",
      "message": "Hi! I'm...",
      "type": "opening"
    },
    {
      "round": 1,
      "speaker": "person2",
      "message": "Nice to meet you!..."
    }
  ]
}
```

### 🔧 故障排除

**问题: API Key not found**
```bash
# 检查环境变量
python3 test_api_key.py

# 如果失败，手动设置
export OPENROUTER_API_KEY='sk-or-v1-...'
```

**问题: Rate limit**
- 代码自动在每次 API 调用后等待 1 秒
- 如果遇到 rate limit，增加 `time.sleep()` 的时间

**问题: API 调用失败**
- 检查网络连接
- 确认 API key 有效
- 查看 OpenRouter 账户余额

### 📊 预期结果

- 100 对完整对话
- 每对 10 轮交流（~21 次发言）
- 总时长: ~30-40 分钟（取决于 API 速度）
- 自动检查点保存（每 5 对）

### 🎭 下一步

完成 Phase 2 后：
1. 审查对话质量（`conversations_sample.txt`）
2. 运行 `evaluation_system.py` 分析兼容性
3. 与 ground truth 对比准确率
