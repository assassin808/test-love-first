#!/bin/bash

# Run full 100-pair experiment
echo "🚀 Starting Full Speed Dating Simulation"
echo "========================================"
echo ""
echo "Settings:"
echo "  - 100 pairs"
echo "  - 10 rounds per conversation"
echo "  - 5 parallel threads"
echo "  - Estimated time: ~30-40 minutes"
echo "  - Estimated cost: ~$0.034"
echo ""
echo "Output will be saved to:"
echo "  - results/conversations.json"
echo "  - results/conversations_checkpoint_*.json (every 5 pairs)"
echo ""
echo "Press Ctrl+C to stop (checkpoints will be saved)"
echo ""
echo "========================================"
echo ""

cd /Users/assassin808/Desktop/research_2025_xuan/yan/test

# Run simulation (mode 3 = full)
echo "3" | /Users/assassin808/Desktop/research_2025_xuan/yan/.venv/bin/python experiments/speed_dating_simulator.py

# Check if successful
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Simulation completed successfully!"
    echo ""
    echo "🔬 Running accuracy analysis..."
    /Users/assassin808/Desktop/research_2025_xuan/yan/.venv/bin/python analyze_results.py
    
    echo ""
    echo "📊 Results available in:"
    echo "  - results/conversations.json"
    echo "  - results/conversations_sample.txt"
    echo "  - results/accuracy_report.json"
else
    echo ""
    echo "❌ Simulation failed. Check checkpoint files in results/"
fi
