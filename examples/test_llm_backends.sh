#!/bin/bash
# =============================================================================
# RAGIX LLM Backend Test Script
# =============================================================================
#
# This script tests the LLM backends with real calls to Ollama.
# It compares response quality and speed between different models.
#
# Prerequisites:
#   - Ollama running: ollama serve
#   - Models pulled: ollama pull mistral && ollama pull granite3.1-moe:3b
#
# Usage:
#   ./examples/test_llm_backends.sh
#
# Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-11-25
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}======================================================================${NC}"
echo -e "${CYAN}  RAGIX v0.7 - LLM Backend Real Integration Test${NC}"
echo -e "${CYAN}======================================================================${NC}"
echo ""

# Check if Ollama is running
echo -e "${BLUE}[1/4] Checking Ollama status...${NC}"
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo -e "${RED}ERROR: Ollama is not running!${NC}"
    echo "Start it with: ollama serve"
    exit 1
fi
echo -e "${GREEN}✓ Ollama is running${NC}"

# List available models
echo ""
echo -e "${BLUE}[2/4] Available models:${NC}"
curl -s http://localhost:11434/api/tags | python3 -c "
import json, sys
data = json.load(sys.stdin)
for m in data.get('models', []):
    size_gb = m.get('size', 0) / 1e9
    print(f\"  - {m['name']} ({size_gb:.1f} GB)\")
"

# Run the Python comparison test
echo ""
echo -e "${BLUE}[3/4] Running backend comparison test...${NC}"
echo ""

python3 << 'PYTHON_SCRIPT'
#!/usr/bin/env python3
"""
Real LLM Backend Comparison Test

This script makes actual calls to Ollama to compare:
- Response time
- Response quality
- Sovereignty status
"""

import time
import sys
from pathlib import Path

# Add RAGIX to path
sys.path.insert(0, str(Path(__file__).parent.parent if '__file__' in dir() else Path.cwd()))

from ragix_core import OllamaLLM, create_llm_backend, SovereigntyStatus

# =============================================================================
# Test Configuration
# =============================================================================

MODELS_TO_TEST = [
    ("mistral:latest", "Mistral 7B - General purpose, good quality"),
    ("granite3.1-moe:3b", "Granite 3B MoE - IBM model, super fast"),
]

TEST_PROMPTS = [
    {
        "name": "Simple greeting",
        "system": "You are a helpful assistant. Be concise.",
        "user": "Say hello in exactly 5 words.",
    },
    {
        "name": "Code explanation",
        "system": "You are a Python expert. Be concise.",
        "user": "What does `lambda x: x*2` do? One sentence only.",
    },
    {
        "name": "Code generation",
        "system": "You are a Python expert. Only output code, no explanation.",
        "user": "Write a function to check if a number is prime.",
    },
]

# =============================================================================
# Test Runner
# =============================================================================

def test_model(model_name: str, description: str) -> dict:
    """Test a single model with all prompts."""

    print(f"\n{'='*60}")
    print(f"🧪 Testing: {model_name}")
    print(f"   {description}")
    print(f"{'='*60}")

    try:
        llm = OllamaLLM(model_name)

        # Check sovereignty
        sovereignty_icon = "🟢" if llm.sovereignty == SovereigntyStatus.SOVEREIGN else "🔴"
        print(f"\n{sovereignty_icon} Sovereignty: {llm.sovereignty.value.upper()}")

        if not llm.is_available():
            print(f"⚠️  Model {model_name} not available. Skipping.")
            return None

        results = []
        total_time = 0

        for prompt in TEST_PROMPTS:
            print(f"\n📝 Test: {prompt['name']}")
            print(f"   Prompt: \"{prompt['user'][:50]}...\"" if len(prompt['user']) > 50 else f"   Prompt: \"{prompt['user']}\"")

            # Time the generation
            start = time.perf_counter()
            response = llm.generate(
                system_prompt=prompt["system"],
                history=[{"role": "user", "content": prompt["user"]}]
            )
            elapsed = time.perf_counter() - start
            total_time += elapsed

            # Truncate response for display
            response_preview = response[:200] + "..." if len(response) > 200 else response
            response_preview = response_preview.replace('\n', '\n      ')

            print(f"   ⏱️  Time: {elapsed:.2f}s")
            print(f"   📤 Response ({len(response)} chars):")
            print(f"      {response_preview}")

            results.append({
                "test": prompt["name"],
                "time": elapsed,
                "response_length": len(response),
            })

        avg_time = total_time / len(TEST_PROMPTS)
        print(f"\n📊 Summary for {model_name}:")
        print(f"   Total time: {total_time:.2f}s")
        print(f"   Average per prompt: {avg_time:.2f}s")

        return {
            "model": model_name,
            "total_time": total_time,
            "avg_time": avg_time,
            "results": results,
        }

    except Exception as e:
        print(f"❌ Error testing {model_name}: {e}")
        return None


def main():
    """Run comparison test."""

    print("\n" + "="*60)
    print("  🚀 RAGIX LLM Backend Comparison")
    print("  Testing with REAL Ollama calls")
    print("="*60)

    all_results = []

    for model_name, description in MODELS_TO_TEST:
        result = test_model(model_name, description)
        if result:
            all_results.append(result)

    # Final comparison
    if len(all_results) >= 2:
        print("\n" + "="*60)
        print("  📊 FINAL COMPARISON")
        print("="*60)

        # Sort by speed
        sorted_results = sorted(all_results, key=lambda x: x["avg_time"])

        print("\n🏆 Speed Ranking (fastest first):")
        for i, r in enumerate(sorted_results, 1):
            medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
            print(f"   {medal} {r['model']}: {r['avg_time']:.2f}s avg")

        # Speed comparison
        if len(sorted_results) >= 2:
            fastest = sorted_results[0]
            slowest = sorted_results[-1]
            speedup = slowest["avg_time"] / fastest["avg_time"]
            print(f"\n⚡ {fastest['model']} is {speedup:.1f}x faster than {slowest['model']}")

    print("\n" + "="*60)
    print("  ✅ Test complete!")
    print("  🟢 All tests used SOVEREIGN (local) backends")
    print("  🔒 No data was sent to any cloud service")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
PYTHON_SCRIPT

# Summary
echo ""
echo -e "${BLUE}[4/4] Test complete!${NC}"
echo ""
echo -e "${GREEN}Key takeaways:${NC}"
echo "  🟢 Both models run 100% locally (SOVEREIGN)"
echo "  ⚡ granite3.1-moe:3b is optimized for speed"
echo "  🎯 mistral:latest offers better quality for complex tasks"
echo ""
echo -e "${YELLOW}To use in RAGIX:${NC}"
echo "  export UNIX_RAG_MODEL=\"mistral:latest\"     # Quality"
echo "  export UNIX_RAG_MODEL=\"granite3.1-moe:3b\"  # Speed"
echo ""
