"""
测试 OpenRouter API 连接
"""

import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

api_key = os.getenv('OPENROUTER_API_KEY')

if api_key:
    print("✅ OPENROUTER_API_KEY found")
    print(f"   Key preview: {api_key[:10]}...{api_key[-4:]}")
else:
    print("❌ OPENROUTER_API_KEY not found")
    print("\n💡 To set it:")
    print("   1. Create a .env file in the project root")
    print("   2. Add: OPENROUTER_API_KEY=your-key-here")
    print("   OR")
    print("   export OPENROUTER_API_KEY='your-key-here'")
