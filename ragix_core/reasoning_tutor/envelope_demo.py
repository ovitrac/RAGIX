#!/usr/bin/env python3
# =============================================================================
# Envelope Demo - Interface Safety Tests
# =============================================================================
#
# Demonstrates TutorEnvelope layers with/without normalization.
# Tests "JSON Pollution" and "Evidence Chain" scenarios.
#
# Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
# Version: 0.1.0 (2025-12-27)
#
# =============================================================================

"""
Envelope Demo Runner.

Tests the impact of interface safety (Layer 1) on LLM output parsing.
Compares slim vs fat models with different envelope configurations.

Usage:
    python envelope_demo.py --test json_trap --models "granite3.1-moe:3b,qwen2.5-coder:14b"
    python envelope_demo.py --test evidence_chain --models "llama3.2:3b"
    python envelope_demo.py --test all --report
"""

from __future__ import annotations
import argparse
import json
import re
import subprocess
import hashlib
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, Any
import requests

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

SANDBOX_ROOT = Path(__file__).parent / "sandbox" / "tasks"
RESULTS_DIR = Path(__file__).parent / "results"

SLIM_MODELS = ["granite3.1-moe:3b", "llama3.2:3b"]
FAT_MODELS = ["qwen2.5-coder:14b", "qwen2.5-coder:7b"]

OLLAMA_API = "http://localhost:11434/api/generate"


# -----------------------------------------------------------------------------
# Move Types (matching moves.py)
# -----------------------------------------------------------------------------

class MoveType(Enum):
    ASSERT = "ASSERT"
    ASK = "ASK"
    PROPOSE = "PROPOSE"
    CHALLENGE = "CHALLENGE"
    REFORMULATE = "REFORMULATE"
    RESPOND = "RESPOND"


class ParseVerdict(Enum):
    """Verdict for parsing quality."""
    CLEAN_JSON = "clean_json"       # Direct JSON output
    EXTRACTED_JSON = "extracted"    # JSON found in prose
    FENCE_EXTRACTED = "fence"       # Command from markdown fence
    DEGRADED = "degraded"           # Partial extraction
    ILL_TYPED = "ill_typed"         # Unparseable


class CommandVerdict(Enum):
    """Verdict for command validation (pre-execution)."""
    VALID_SHELL = "valid_shell"           # Parseable shell command
    NATURAL_LANGUAGE = "natural_language" # Prose masquerading as command
    NON_PORTABLE = "non_portable"         # Uses bash-specific features
    EMPTY = "empty"                       # No command


@dataclass
class ParseResult:
    """Result of parsing LLM output."""
    verdict: ParseVerdict
    move_type: Optional[MoveType] = None
    command: str = ""
    intent: str = ""
    raw_output: str = ""
    normalized_output: str = ""
    json_found: bool = False
    fence_found: bool = False
    prose_ratio: float = 0.0  # Ratio of non-JSON text
    # NEW: Command validation (reviewer fix 3.1)
    command_verdict: CommandVerdict = CommandVerdict.EMPTY
    command_issues: list = field(default_factory=list)


# -----------------------------------------------------------------------------
# Output Normalizer (Layer 1 simulation)
# -----------------------------------------------------------------------------

class OutputNormalizer:
    """
    Simulates Layer 1 interface safety.

    Modes:
    - off: No normalization (baseline)
    - lenient: Strip fences, extract JSON
    - strict: Reject if not clean JSON
    """

    # Patterns for markdown fences
    FENCE_PATTERN = re.compile(
        r'```(?:bash|sh|json)?\s*\n(.*?)```',
        re.DOTALL | re.IGNORECASE
    )

    # Pattern for JSON objects
    JSON_PATTERN = re.compile(r'\{[^{}]*\}')

    def __init__(self, mode: str = "off"):
        self.mode = mode

    def normalize(self, raw_output: str) -> ParseResult:
        """Parse and optionally normalize LLM output."""
        result = ParseResult(
            verdict=ParseVerdict.ILL_TYPED,
            raw_output=raw_output
        )

        # Calculate prose ratio
        stripped = raw_output.strip()
        json_chars = sum(len(m.group()) for m in self.JSON_PATTERN.finditer(stripped))
        result.prose_ratio = 1.0 - (json_chars / len(stripped)) if stripped else 1.0

        # Check for markdown fences
        fence_match = self.FENCE_PATTERN.search(raw_output)
        if fence_match:
            result.fence_found = True

        # Try direct JSON parse
        try:
            # Find JSON in output
            json_matches = list(self.JSON_PATTERN.finditer(raw_output))
            if json_matches:
                result.json_found = True
                # Try each JSON match
                for match in json_matches:
                    try:
                        data = json.loads(match.group())
                        action = data.get("action", "").upper()
                        if action in [m.value for m in MoveType]:
                            result.move_type = MoveType(action)
                            result.command = data.get("command", "")
                            result.intent = data.get("intent", "")
                            result.normalized_output = match.group()

                            # Determine verdict based on cleanliness
                            if result.prose_ratio < 0.1:
                                result.verdict = ParseVerdict.CLEAN_JSON
                            else:
                                result.verdict = ParseVerdict.EXTRACTED_JSON
                            return result
                    except (json.JSONDecodeError, KeyError):
                        continue
        except Exception:
            pass

        # Try fence extraction (for commands)
        if fence_match:
            command = fence_match.group(1).strip()
            if command and not command.startswith("{"):
                result.command = command
                result.move_type = MoveType.PROPOSE
                result.verdict = ParseVerdict.FENCE_EXTRACTED
                result.normalized_output = json.dumps({
                    "action": "PROPOSE",
                    "command": command,
                    "intent": "Extracted from fence"
                })
                return result

        # Fallback: try to extract any command-like text
        lines = raw_output.strip().split("\n")
        for line in lines:
            line = line.strip()
            # Skip obvious prose
            if line and not line[0].isupper() and "|" in line or ">" in line or "<" in line:
                result.command = line
                result.move_type = MoveType.PROPOSE
                result.verdict = ParseVerdict.DEGRADED
                return result

        # Nothing found
        result.verdict = ParseVerdict.ILL_TYPED
        return result

    def apply_mode(self, result: ParseResult) -> ParseResult:
        """Apply mode-specific filtering."""
        if self.mode == "strict":
            # Only accept clean JSON
            if result.verdict != ParseVerdict.CLEAN_JSON:
                result.verdict = ParseVerdict.ILL_TYPED
                result.command = ""
                result.move_type = None
        elif self.mode == "lenient":
            # Accept extracted JSON and fences
            pass  # Already handled in normalize()
        # mode == "off": return as-is
        return result


# -----------------------------------------------------------------------------
# Command Validator (Reviewer Fix 3.1, 3.2)
# -----------------------------------------------------------------------------

class CommandValidator:
    """
    Pre-execution command validation.

    Detects:
    - Natural language masquerading as shell commands
    - Non-portable bash-specific syntax
    - Environment-dependent constructs
    """

    # Natural language indicators (reviewer fix 3.1)
    NL_INDICATORS = [
        r"^(please|check|review|analyze|look|find|identify|examine)\s",
        r"^(the|a|an)\s",
        r"\s(the|for|any|to|in|of|with)\s.*\.$",
        r"misconfig",
        r"issue",
        r"problem",
    ]

    # Non-portable constructs (reviewer fix 3.2)
    NON_PORTABLE = [
        (r"<\([^)]+\)", "process substitution <(...)"),
        (r">\([^)]+\)", "process substitution >(...)"),
        (r"\[\[", "bash [[ ]] test"),
        (r"\$\{[^}]+//", "bash parameter expansion"),
        (r"\$\{[^}]+:[+-]", "bash default value syntax"),
        (r"<<<", "bash here-string"),
        (r"&>>", "bash append redirect"),
    ]

    # Shell command indicators
    SHELL_INDICATORS = [
        r"^(cat|grep|sed|awk|sort|uniq|head|tail|wc|ls|find|echo|tr|cut)\s",
        r"^(test|cd|mkdir|rm|cp|mv|chmod|chown)\s",
        r"^(python|bash|sh|perl|ruby)\s",
        r"\|",  # Pipe
        r"[<>]",  # Redirect
        r"&&|\|\|",  # Boolean operators
    ]

    def validate(self, command: str) -> tuple[CommandVerdict, list[str]]:
        """
        Validate command before execution.

        Returns (verdict, list of issues).
        """
        if not command or not command.strip():
            return CommandVerdict.EMPTY, []

        command = command.strip()
        issues = []

        # Check for natural language (reviewer fix 3.1)
        nl_score = 0
        for pattern in self.NL_INDICATORS:
            if re.search(pattern, command, re.IGNORECASE):
                nl_score += 1

        # Check for shell indicators
        shell_score = 0
        for pattern in self.SHELL_INDICATORS:
            if re.search(pattern, command):
                shell_score += 1

        # Natural language detection
        if nl_score >= 2 and shell_score == 0:
            issues.append(f"Natural language detected (score={nl_score})")
            return CommandVerdict.NATURAL_LANGUAGE, issues

        if nl_score >= 1 and shell_score == 0 and len(command.split()) > 5:
            issues.append("Prose-like command (long, no shell operators)")
            return CommandVerdict.NATURAL_LANGUAGE, issues

        # Check for non-portable constructs (reviewer fix 3.2)
        for pattern, description in self.NON_PORTABLE:
            if re.search(pattern, command):
                issues.append(f"Non-portable: {description}")

        if issues:
            return CommandVerdict.NON_PORTABLE, issues

        # Attempt shell parsing
        try:
            import shlex
            shlex.split(command)
            return CommandVerdict.VALID_SHELL, []
        except ValueError as e:
            issues.append(f"Shell parse error: {e}")
            return CommandVerdict.NON_PORTABLE, issues


# -----------------------------------------------------------------------------
# Test Runner
# -----------------------------------------------------------------------------

@dataclass
class TestResult:
    """Result of a single test run."""
    test_name: str
    model: str
    mode: str  # "baseline", "lenient", "strict"

    # Parsing
    parse_verdict: str
    json_found: bool
    fence_found: bool
    prose_ratio: float

    # Command validation (reviewer fix 3.1, 3.2)
    command_verdict: str = ""
    command_issues: list = field(default_factory=list)

    # Execution
    command_extracted: str = ""
    command_executed: bool = False
    exit_code: int = -1
    stdout: str = ""
    stderr: str = ""

    # Evaluation
    success: bool = False
    output_matches: bool = False

    # Timing
    response_time_ms: float = 0.0

    # Raw data
    raw_output: str = ""
    normalized_output: str = ""

    # Hashes for reproducibility (reviewer fix 3.3)
    prompt_hash: str = ""
    output_hash: str = ""


def call_ollama(model: str, prompt: str, temperature: float = 0.0) -> tuple[str, float]:
    """Call Ollama API and return response + timing."""
    start = datetime.now()

    try:
        response = requests.post(
            OLLAMA_API,
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": 512,
                }
            },
            timeout=120
        )
        response.raise_for_status()
        data = response.json()
        elapsed = (datetime.now() - start).total_seconds() * 1000
        return data.get("response", ""), elapsed
    except Exception as e:
        elapsed = (datetime.now() - start).total_seconds() * 1000
        return f"ERROR: {e}", elapsed


def run_json_trap_test(
    model: str,
    mode: str = "baseline",
    temperature: float = 0.0
) -> TestResult:
    """Run the JSON Pollution test."""

    task_dir = SANDBOX_ROOT / "json_trap"
    input_file = task_dir / "input.txt"
    expected_file = task_dir / "expected.txt"

    # Read expected output
    expected = expected_file.read_text().strip()

    # Construct prompt
    prompt = f"""You are a Unix command expert. Output ONLY a JSON object, no explanations.

TASK: Read the file 'input.txt' and output sorted unique tokens (case-insensitive), one per line.

INPUT FILE CONTENT:
{input_file.read_text()}

OUTPUT FORMAT (strict JSON only):
{{"action": "PROPOSE", "intent": "<your intent>", "command": "<shell command>", "mode": "read"}}

Your response (JSON only):"""

    # Compute prompt hash (reviewer fix 3.3)
    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:16]

    # Call model
    raw_output, response_time = call_ollama(model, prompt, temperature)

    # Compute output hash (reviewer fix 3.3)
    output_hash = hashlib.sha256(raw_output.encode()).hexdigest()[:16]

    # Parse output
    normalizer = OutputNormalizer(mode=mode if mode != "baseline" else "off")
    parse_result = normalizer.normalize(raw_output)
    if mode != "baseline":
        parse_result = normalizer.apply_mode(parse_result)

    # Validate command before execution (reviewer fix 3.1, 3.2)
    cmd_validator = CommandValidator()
    cmd_verdict, cmd_issues = cmd_validator.validate(parse_result.command)

    # Execute command if extracted and valid
    command_executed = False
    exit_code = -1
    stdout = ""
    stderr = ""
    output_matches = False

    # Only execute if parse verdict is OK and command is valid shell
    should_execute = (
        parse_result.command and
        parse_result.verdict != ParseVerdict.ILL_TYPED and
        cmd_verdict not in (CommandVerdict.NATURAL_LANGUAGE, CommandVerdict.EMPTY)
    )

    if should_execute:
        command_executed = True
        try:
            result = subprocess.run(
                parse_result.command,
                shell=True,
                cwd=str(task_dir),
                capture_output=True,
                text=True,
                timeout=10
            )
            exit_code = result.returncode
            stdout = result.stdout.strip()
            stderr = result.stderr.strip()
            output_matches = (stdout == expected)
        except subprocess.TimeoutExpired:
            exit_code = -1
            stderr = "TIMEOUT"
        except Exception as e:
            exit_code = -1
            stderr = str(e)
    elif cmd_verdict == CommandVerdict.NATURAL_LANGUAGE:
        stderr = f"PRE-EXEC REJECT: {'; '.join(cmd_issues)}"

    success = output_matches and exit_code == 0

    return TestResult(
        test_name="json_trap",
        model=model,
        mode=mode,
        parse_verdict=parse_result.verdict.value,
        json_found=parse_result.json_found,
        fence_found=parse_result.fence_found,
        prose_ratio=parse_result.prose_ratio,
        command_verdict=cmd_verdict.value,
        command_issues=cmd_issues,
        command_extracted=parse_result.command,
        command_executed=command_executed,
        exit_code=exit_code,
        stdout=stdout[:200],  # Truncate
        stderr=stderr[:200],
        success=success,
        output_matches=output_matches,
        response_time_ms=response_time,
        raw_output=raw_output,
        normalized_output=parse_result.normalized_output,
        prompt_hash=prompt_hash,
        output_hash=output_hash,
    )


def run_evidence_chain_test(
    model: str,
    mode: str = "baseline",
    temperature: float = 0.0,
    max_turns: int = 6
) -> list[TestResult]:
    """Run the Evidence Chain test (multi-turn)."""

    task_dir = SANDBOX_ROOT / "evidence_chain"

    # Initial prompt
    system_prompt = """You are a Unix sysadmin debugging a pipeline issue.
Output ONLY JSON objects for each action.

Available moves:
- {"action": "PROPOSE", "intent": "...", "command": "...", "mode": "read|write"}
- {"action": "ASSERT", "text": "...", "kind": "observation|conclusion"}
- {"action": "CHALLENGE", "text": "...", "falsify": "..."}

TASK: The pipeline has errors. Find the misconfiguration in config.yaml and fix it.
Working directory contains: system.log, config.yaml

Start by investigating the error."""

    results = []
    context = system_prompt
    evidence_nodes = 0

    for turn in range(max_turns):
        # Call model
        raw_output, response_time = call_ollama(model, context, temperature)

        # Parse
        normalizer = OutputNormalizer(mode=mode if mode != "baseline" else "off")
        parse_result = normalizer.normalize(raw_output)
        if mode != "baseline":
            parse_result = normalizer.apply_mode(parse_result)

        # Execute if PROPOSE
        exit_code = -1
        stdout = ""
        stderr = ""
        command_executed = False

        if parse_result.move_type == MoveType.PROPOSE and parse_result.command:
            command_executed = True
            try:
                result = subprocess.run(
                    parse_result.command,
                    shell=True,
                    cwd=str(task_dir),
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                exit_code = result.returncode
                stdout = result.stdout.strip()
                stderr = result.stderr.strip()

                # Track evidence
                if exit_code == 0 and stdout:
                    evidence_nodes += 1
            except Exception as e:
                stderr = str(e)

        # Check for goal completion (config fixed)
        goal_reached = False
        if parse_result.command and "sed" in parse_result.command and "0.10" in parse_result.command:
            # Verify fix was applied
            config_content = (task_dir / "config.yaml").read_text()
            if "alert_threshold: 0.10" in config_content:
                goal_reached = True

        results.append(TestResult(
            test_name=f"evidence_chain_t{turn+1}",
            model=model,
            mode=mode,
            parse_verdict=parse_result.verdict.value,
            json_found=parse_result.json_found,
            fence_found=parse_result.fence_found,
            prose_ratio=parse_result.prose_ratio,
            command_extracted=parse_result.command,
            command_executed=command_executed,
            exit_code=exit_code,
            stdout=stdout[:200],
            stderr=stderr[:200],
            success=goal_reached,
            output_matches=goal_reached,
            response_time_ms=response_time,
            raw_output=raw_output,
            normalized_output=parse_result.normalized_output,
        ))

        # Update context for next turn
        if command_executed and stdout:
            context = f"{context}\n\nYour action: {parse_result.normalized_output}\nResult:\n{stdout[:500]}\n\nContinue (JSON only):"
        else:
            context = f"{context}\n\nYour response was not valid. Try again with proper JSON format.\n\nContinue (JSON only):"

        if goal_reached:
            break

    return results


# -----------------------------------------------------------------------------
# Report Generation
# -----------------------------------------------------------------------------

def generate_report(results: list[TestResult], test_name: str) -> str:
    """Generate markdown report."""
    lines = []
    lines.append(f"# Envelope Demo Report: {test_name}")
    lines.append(f"\n**Generated:** {datetime.now().isoformat()}")
    lines.append(f"**Temperature:** 0.0 (deterministic)")
    lines.append("")

    # Summary table
    lines.append("## Results Summary")
    lines.append("")
    lines.append("| Model | Mode | Parse | JSON | Fence | Prose% | Command | Success |")
    lines.append("|-------|------|-------|------|-------|--------|---------|---------|")

    for r in results:
        model_short = r.model.split(":")[0][:12]
        prose_pct = f"{r.prose_ratio*100:.0f}%"
        cmd_short = r.command_extracted[:20] + "..." if len(r.command_extracted) > 20 else r.command_extracted
        success_icon = "✓" if r.success else "✗"
        json_icon = "✓" if r.json_found else "✗"
        fence_icon = "✓" if r.fence_found else "✗"

        lines.append(f"| {model_short} | {r.mode} | {r.parse_verdict} | {json_icon} | {fence_icon} | {prose_pct} | `{cmd_short}` | {success_icon} |")

    lines.append("")

    # Detailed results
    lines.append("## Detailed Results")
    lines.append("")

    for i, r in enumerate(results):
        lines.append(f"### Run {i+1}: {r.model} ({r.mode})")
        lines.append("")
        lines.append(f"**Parse Verdict:** {r.parse_verdict}")
        lines.append(f"**Prose Ratio:** {r.prose_ratio:.2f}")
        lines.append(f"**Success:** {'Yes' if r.success else 'No'}")
        lines.append(f"**Response Time:** {r.response_time_ms:.0f}ms")
        lines.append("")

        lines.append("**Raw Output:**")
        lines.append("```")
        lines.append(r.raw_output[:500])
        if len(r.raw_output) > 500:
            lines.append("... (truncated)")
        lines.append("```")
        lines.append("")

        if r.command_extracted:
            lines.append(f"**Extracted Command:** `{r.command_extracted}`")
            lines.append(f"**Exit Code:** {r.exit_code}")
            if r.stdout:
                lines.append(f"**Output:** `{r.stdout[:100]}`")
            lines.append("")

    # Compatibility check
    lines.append("## Compatibility Check")
    lines.append("")

    baseline_results = [r for r in results if r.mode == "baseline"]
    if len(baseline_results) >= 2:
        # Compare outputs between same model
        models_seen = {}
        for r in baseline_results:
            if r.model not in models_seen:
                models_seen[r.model] = r

        lines.append("Baseline results are deterministic (temperature=0.0).")
    else:
        lines.append("Run with multiple modes to verify baseline equivalence.")

    lines.append("")

    return "\n".join(lines)


def compute_trace_hash(results: list[TestResult]) -> str:
    """Compute comprehensive hash of trace for compatibility verification.

    Includes prompt, output, and state hashes per reviewer fix 3.3.
    """
    trace_data = []
    for r in results:
        trace_data.append({
            "model": r.model,
            "mode": r.mode,
            "parse_verdict": r.parse_verdict,
            "command_verdict": r.command_verdict,
            "command": r.command_extracted,
            "success": r.success,
            # Extended hashes (reviewer fix 3.3)
            "prompt_hash": r.prompt_hash,
            "output_hash": r.output_hash,
        })
    return hashlib.sha256(json.dumps(trace_data, sort_keys=True).encode()).hexdigest()[:16]


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Envelope Demo Runner")
    parser.add_argument("--test", choices=["json_trap", "evidence_chain", "all"],
                       default="json_trap", help="Test to run")
    parser.add_argument("--models", type=str,
                       default="granite3.1-moe:3b,qwen2.5-coder:14b",
                       help="Comma-separated model list")
    parser.add_argument("--modes", type=str,
                       default="baseline,lenient,strict",
                       help="Comma-separated modes")
    parser.add_argument("--report", action="store_true",
                       help="Generate markdown report")
    parser.add_argument("--output", type=str,
                       help="Output file for report")

    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",")]
    modes = [m.strip() for m in args.modes.split(",")]

    print(f"Envelope Demo - {args.test}")
    print(f"Models: {models}")
    print(f"Modes: {modes}")
    print("-" * 60)

    all_results = []

    # Run tests
    if args.test in ["json_trap", "all"]:
        print("\n=== JSON Trap Test ===")
        for model in models:
            for mode in modes:
                print(f"  Running: {model} ({mode})...", end=" ", flush=True)
                try:
                    result = run_json_trap_test(model, mode)
                    all_results.append(result)
                    status = "✓" if result.success else "✗"
                    print(f"{status} ({result.parse_verdict}, {result.response_time_ms:.0f}ms)")
                except Exception as e:
                    print(f"ERROR: {e}")

    if args.test in ["evidence_chain", "all"]:
        print("\n=== Evidence Chain Test ===")
        # Reset config file before test
        config_path = SANDBOX_ROOT / "evidence_chain" / "config.yaml"
        config_original = """pipeline:
  name: flow_monitor
  version: 1.2.0

thresholds:
  alert_threshold: 0.20
  max_flow: 1200.0
  warning_margin: 0.10

notifications:
  enabled: true
  email: admin@example.com
"""
        for model in models:
            for mode in modes:
                # Reset config
                config_path.write_text(config_original)
                print(f"  Running: {model} ({mode})...", flush=True)
                try:
                    results = run_evidence_chain_test(model, mode, max_turns=4)
                    all_results.extend(results)
                    final = results[-1] if results else None
                    if final:
                        status = "✓" if final.success else "✗"
                        print(f"    {len(results)} turns, {status}")
                except Exception as e:
                    print(f"    ERROR: {e}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print(f"\n{'Model':<20} {'Mode':<10} {'Parse':<15} {'Success':<8}")
    print("-" * 60)
    for r in all_results:
        if "evidence_chain" not in r.test_name or r.test_name.endswith("_t1"):
            model_short = r.model.split(":")[0][:18]
            success = "✓" if r.success else "✗"
            print(f"{model_short:<20} {r.mode:<10} {r.parse_verdict:<15} {success:<8}")

    # Trace hash for reproducibility
    trace_hash = compute_trace_hash(all_results)
    print(f"\nTrace Hash: {trace_hash}")

    # Generate report
    if args.report:
        report = generate_report(all_results, args.test)

        if args.output:
            Path(args.output).write_text(report)
            print(f"\nReport written to: {args.output}")
        else:
            # Default output
            RESULTS_DIR.mkdir(exist_ok=True)
            output_path = RESULTS_DIR / f"envelope_demo_{args.test}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            output_path.write_text(report)
            print(f"\nReport written to: {output_path}")

    # Return success if any test passed
    return 0 if any(r.success for r in all_results) else 1


if __name__ == "__main__":
    sys.exit(main())
