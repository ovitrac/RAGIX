#!/usr/bin/env python3
"""
Smoke Test — Quick validation that benchmarks run correctly.

Runs B01 (Find Needle) and B02 (Count Lines) with granite3.1-moe:3b
to verify the system works. Takes ~30 seconds.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
"""

import subprocess
import sys
import time
from pathlib import Path

# Configuration
FAST_MODEL = "granite3.1-moe:3b"
SMOKE_BENCHMARKS = ["01", "02"]  # Quick benchmarks
MAX_TURNS = 6
TIMEOUT = 120  # 2 minutes


def check_ollama_available() -> bool:
    """Check if Ollama is running and model is available."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True, text=True, timeout=10
        )
        return FAST_MODEL.split(":")[0] in result.stdout
    except Exception as e:
        print(f"Ollama check failed: {e}")
        return False


def run_smoke_test() -> bool:
    """Run smoke test and return success status."""
    print("=" * 60)
    print("SMOKE TEST — Interpreter-Tutor Benchmarks")
    print("=" * 60)

    # Check Ollama
    print(f"\nChecking Ollama with {FAST_MODEL}...")
    if not check_ollama_available():
        print(f"ERROR: Ollama not running or {FAST_MODEL} not available")
        print(f"Run: ollama pull {FAST_MODEL}")
        return False
    print("Ollama OK")

    # Run benchmarks
    project_root = Path(__file__).parent.parent.parent.parent
    cmd = [
        sys.executable, "-m",
        "ragix_core.reasoning_tutor.benchmarks.scored_mode",
        "--models", FAST_MODEL,
        "--max-turns", str(MAX_TURNS),
        "--quiet",
    ]
    for b in SMOKE_BENCHMARKS:
        cmd.extend(["--benchmark", b])

    print(f"\nRunning B01+B02 with {FAST_MODEL}...")
    start = time.time()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=project_root,
            timeout=TIMEOUT
        )
    except subprocess.TimeoutExpired:
        print(f"ERROR: Timed out after {TIMEOUT}s")
        return False

    elapsed = time.time() - start
    print(f"Completed in {elapsed:.1f}s")

    # Check for obvious errors
    if result.returncode != 0:
        print(f"ERROR: Non-zero exit code: {result.returncode}")
        if result.stderr:
            print(f"STDERR: {result.stderr[:500]}")
        return False

    # Look for game_end events
    game_ends = result.stdout.count('"type": "game_end"')
    if game_ends < len(SMOKE_BENCHMARKS):
        print(f"WARNING: Expected {len(SMOKE_BENCHMARKS)} game_end events, found {game_ends}")

    print("\n" + "-" * 60)
    print("[PASS] Smoke test completed successfully")
    print("-" * 60)
    return True


def main():
    success = run_smoke_test()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
