#!/usr/bin/env python3
# =============================================================================
# Interpreter-Tutor Demo
# =============================================================================
#
# Demonstrates the proof game with a slim LLM (granite3.1-moe:3b).
# The LLM proposes moves; the Tutor validates and executes deterministically.
#
# Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
# Version: 0.1.0 (2025-12-21)
#
# Usage:
#   python -m ragix_core.reasoning_tutor.demo
#   python -m ragix_core.reasoning_tutor.demo --model qwen2.5-coder:7b
#
# =============================================================================

"""
Demo of the Interpreter-Tutor architecture.

This demo shows:
1. Game setup with goal and constraints
2. LLM proposing moves (shell commands)
3. Tutor validating and executing moves
4. Rules deriving truths from observations
5. CHECK protocol for claims
"""

import argparse
import json
import requests
from pathlib import Path

from .tutor import Tutor, MoveVerdict, CheckVerdict
from .moves import parse_move, generate_move_prompt
from .pcg import Status


# =============================================================================
# Ollama Interface
# =============================================================================

def call_ollama(prompt: str, model: str = "granite3.1-moe:3b",
                system: str = None, temperature: float = 0.3) -> str:
    """Call Ollama API."""
    url = "http://localhost:11434/api/generate"

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": 500,
        }
    }

    if system:
        payload["system"] = system

    try:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        return response.json().get("response", "")
    except Exception as e:
        return f"ERROR: {e}"


# =============================================================================
# Demo Game
# =============================================================================

def run_demo(model: str = "granite3.1-moe:3b", sandbox: str = "."):
    """Run the demo game."""
    print("=" * 70)
    print("INTERPRETER-TUTOR DEMO")
    print(f"Model: {model}")
    print(f"Sandbox: {sandbox}")
    print("=" * 70)
    print()

    # Initialize Tutor
    rules_path = Path(__file__).parent / "rules"
    tutor = Tutor(
        game_id="demo_001",
        sandbox_root=sandbox,
        rules_path=str(rules_path),
    )

    # Setup game
    goal = "Explore the current directory and report what Python files exist."
    tutor.setup_game(goal)

    print(f"GOAL: {goal}")
    print()
    print("CONSTRAINTS:")
    for c in tutor.pcg.get_active_constraints():
        print(f"  - {c.text}")
    print()
    print("-" * 70)

    # System prompt for LLM
    system_prompt = generate_move_prompt({})

    # Game loop
    max_turns = 5
    for turn in range(1, max_turns + 1):
        tutor.next_turn()
        print(f"\n{'='*70}")
        print(f"TURN {turn}")
        print("=" * 70)

        # Get context for LLM
        context = tutor.get_context_for_llm()

        # Build prompt
        if turn == 1:
            user_prompt = f"""GOAL: {goal}

Start by exploring what files are in the current directory.
Propose a shell command to list files.

Output a JSON move."""
        else:
            user_prompt = f"""{context}

Based on the evidence above, what should we do next to achieve the goal?
If we have enough information, summarize what we found.
Otherwise, propose another command to gather more evidence.

Output a JSON move."""

        # Call LLM
        print("\n[PLAYER → LLM]")
        print(f"Prompt: {user_prompt[:200]}...")
        print()

        llm_response = call_ollama(user_prompt, model=model, system=system_prompt)
        print(f"[LLM RESPONSE]")
        print(llm_response[:500])
        print()

        # Parse moves
        moves = parse_move(llm_response)
        if not moves:
            print("[TUTOR] No valid moves parsed. Skipping turn.")
            continue

        # Execute each move
        for move in moves:
            print(f"\n[TUTOR] Processing move: {move.move_type.value}")

            # Validate and execute
            result = tutor.execute_move(move)

            print(f"  Verdict: {result.verdict.value}")
            if result.reason:
                print(f"  Reason: {result.reason}")

            if result.observation:
                obs = result.observation
                print(f"  Observation [{obs.id}]:")
                print(f"    Command: {obs.command}")
                print(f"    RC: {obs.rc}")
                if obs.stdout:
                    print(f"    Stdout: {obs.stdout[:300]}...")

            if result.truths:
                print(f"  Derived truths:")
                for t in result.truths:
                    print(f"    [{t.id}] {t.text} (status={t.status.value})")

            if result.entities:
                print(f"  Extracted entities:")
                for e in result.entities:
                    print(f"    [{e.id}] {e.kind}: {e.value}")

        # Check goal
        if tutor.check_goal_satisfaction():
            print("\n" + "=" * 70)
            print("GOAL SATISFIED!")
            break

    # Final summary
    print("\n" + "=" * 70)
    print("GAME SUMMARY")
    print("=" * 70)
    state = tutor.get_state_summary()
    print(f"Turns played: {state['turn']}")
    print(f"Score: {state['score']}")
    print(f"Validated truths: {state['validated_truths']}")
    print(f"Open questions: {state['open_questions']}")
    print(f"Goal satisfied: {state['goal_satisfied']}")
    print()

    # Print validated truths
    validated = tutor.pcg.get_truths(Status.VALIDATED)
    if validated:
        print("VALIDATED TRUTHS:")
        for t in validated:
            print(f"  [{t.id}] {t.text}")

    # Export PCG
    print("\n" + "-" * 70)
    print("PCG TRACE (first 20 lines):")
    print("-" * 70)
    trace = tutor.pcg.to_jsonl()
    for i, line in enumerate(trace.split("\n")[:20]):
        print(line)

    return tutor


# =============================================================================
# Interactive Mode
# =============================================================================

def run_interactive(model: str = "granite3.1-moe:3b", sandbox: str = "."):
    """Run interactive game."""
    print("=" * 70)
    print("INTERPRETER-TUTOR - INTERACTIVE MODE")
    print(f"Model: {model}")
    print("=" * 70)
    print()
    print("Commands:")
    print("  <goal>     - Set goal and start game")
    print("  /status    - Show game state")
    print("  /truths    - Show validated truths")
    print("  /trace     - Show PCG trace")
    print("  /quit      - Exit")
    print()

    # Initialize
    rules_path = Path(__file__).parent / "rules"
    tutor = None
    system_prompt = generate_move_prompt({})

    while True:
        try:
            user_input = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        # Commands
        if user_input == "/quit":
            break
        elif user_input == "/status":
            if tutor:
                print(json.dumps(tutor.get_state_summary(), indent=2))
            else:
                print("No game in progress.")
            continue
        elif user_input == "/truths":
            if tutor:
                for t in tutor.pcg.get_truths(Status.VALIDATED):
                    print(f"  [{t.id}] {t.text}")
            else:
                print("No game in progress.")
            continue
        elif user_input == "/trace":
            if tutor:
                print(tutor.pcg.to_jsonl()[:2000])
            else:
                print("No game in progress.")
            continue

        # Start game or continue
        if tutor is None:
            # Treat as goal
            tutor = Tutor(
                game_id="interactive",
                sandbox_root=sandbox,
                rules_path=str(rules_path),
            )
            tutor.setup_game(user_input)
            print(f"\nGAME STARTED")
            print(f"Goal: {user_input}")
            continue

        # Play turn
        tutor.next_turn()
        context = tutor.get_context_for_llm()

        prompt = f"""{context}

User request: {user_input}

Propose a shell command or respond.
Output a JSON move."""

        print("\n[Thinking...]")
        llm_response = call_ollama(prompt, model=model, system=system_prompt)

        # Parse and execute
        moves = parse_move(llm_response)
        for move in moves:
            result = tutor.execute_move(move)
            if result.observation:
                print(f"\n[Command] {result.observation.command}")
                print(f"[RC] {result.observation.rc}")
                if result.observation.stdout:
                    print(f"[Output]\n{result.observation.stdout[:500]}")
            if result.truths:
                print(f"\n[Derived]")
                for t in result.truths:
                    print(f"  - {t.text}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Interpreter-Tutor Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model", "-m",
        default="granite3.1-moe:3b",
        help="Ollama model to use (default: granite3.1-moe:3b)",
    )
    parser.add_argument(
        "--sandbox", "-s",
        default=".",
        help="Sandbox directory (default: current directory)",
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode",
    )

    args = parser.parse_args()

    if args.interactive:
        run_interactive(model=args.model, sandbox=args.sandbox)
    else:
        run_demo(model=args.model, sandbox=args.sandbox)


if __name__ == "__main__":
    main()
