#!/usr/bin/env python3
# =============================================================================
# Policy Bundle Ablation Study v2 — Reviewer-Hardened Version
# =============================================================================
#
# Compares baseline (minimal guidance) vs. hand-optimized policy bundles.
# Addresses reviewer concerns about instrumentation and confounders.
#
# Key fixes (v2):
# - Renamed: "hand-optimized bundle" not "DSPy-optimized"
# - Added instrumentation: files_touched, repeat_count, edit_attempted
# - Fixed context update asymmetry (no hidden coaching)
# - Made success criterion less arbitrary (threshold < 0.15)
# - Added Layer config header to all reports
# - Support multiple max_turns variants
#
# Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
# Version: 2.0.0 (2025-12-27)
#
# =============================================================================

"""
Policy Bundle Ablation Study v2.

This is an ILLUSTRATIVE ablation, not a benchmark. It isolates the effect
of hand-crafted policy bundles on single-step command synthesis vs. multi-turn
strategic behavior.

What this tests:
- Policy bundle effect on command format/syntax
- Policy bundle effect on command semantics

What this does NOT test (explicitly stated):
- Layer 1 normalization (disabled)
- Layer 2 repair loops (not implemented)
- Evidence gain metrics (not computed, only proxied)

Usage:
    python ablation_study_v2.py --test json_trap --models "granite3.1-moe:3b"
    python ablation_study_v2.py --test evidence_chain --max-turns 4,8,12
"""

from __future__ import annotations
import argparse
import json
import hashlib
import subprocess
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional
import yaml
import requests

# Import from envelope_demo
from envelope_demo import (
    CommandValidator, CommandVerdict, ParseVerdict,
    OutputNormalizer, MoveType, SANDBOX_ROOT, RESULTS_DIR
)

POLICIES_DIR = Path(__file__).parent / "policies"
OLLAMA_API = "http://localhost:11434/api/generate"


# -----------------------------------------------------------------------------
# Configuration (explicit, logged)
# -----------------------------------------------------------------------------

@dataclass
class AblationConfig:
    """Explicit configuration for ablation run."""
    # Layer settings (reviewer fix 3.1)
    normalizer_mode: str = "off"  # Explicitly disabled — stated in report
    move_type_enforcement: str = "PROPOSE"  # Required for execution

    # Decoding
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 512

    # Evidence chain
    max_turns: int = 4

    # Success criterion (reviewer fix 3.4)
    # Accept any threshold < 0.15 (since log shows 0.15 is the failing margin)
    threshold_success_criterion: float = 0.15

    # Git info
    git_commit: str = ""

    def to_dict(self) -> dict:
        return {
            "normalizer_mode": self.normalizer_mode,
            "move_type_enforcement": self.move_type_enforcement,
            "temperature": self.temperature,
            "max_turns": self.max_turns,
            "threshold_success_criterion": self.threshold_success_criterion,
            "git_commit": self.git_commit or get_git_commit(),
        }


def get_git_commit() -> str:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


# -----------------------------------------------------------------------------
# Policy Bundle Loader
# -----------------------------------------------------------------------------

@dataclass
class PolicyBundle:
    """Loaded policy bundle with hash for reproducibility."""
    bundle_id: str
    bundle_type: str
    optimized: bool
    system_prompt: str
    task_template: str
    exemplars: list
    model_params: dict
    bundle_hash: str = ""

    @classmethod
    def load(cls, path: Path) -> "PolicyBundle":
        """Load policy bundle from YAML file."""
        with open(path) as f:
            content = f.read()
            data = yaml.safe_load(content)

        bundle_hash = hashlib.sha256(content.encode()).hexdigest()[:16]

        return cls(
            bundle_id=data.get("bundle_id", "unknown"),
            bundle_type=data.get("bundle_type", "unknown"),
            optimized=data.get("metadata", {}).get("optimized", False),
            system_prompt=data.get("prompts", {}).get("system", ""),
            task_template=data.get("prompts", {}).get("task_template", ""),
            exemplars=data.get("exemplars", []),
            model_params=data.get("model_params", {}),
            bundle_hash=bundle_hash,
        )

    def build_prompt(self, task_description: str, context: str = "") -> str:
        """Build full prompt from template."""
        prompt = self.system_prompt + "\n\n"
        prompt += self.task_template.format(
            task_description=task_description,
            context=context
        )
        return prompt


# -----------------------------------------------------------------------------
# Strategic Behavior Instrumentation (reviewer fix 4.1)
# -----------------------------------------------------------------------------

@dataclass
class StrategicMetrics:
    """Counters for strategic behavior analysis."""
    # File coverage
    read_config: bool = False
    read_log: bool = False
    edit_attempted: bool = False

    # Repetition (reviewer fix 4.1)
    command_signatures: list = field(default_factory=list)
    repetition_count: int = 0

    # Evidence proxy
    files_touched: set = field(default_factory=set)
    unique_commands: int = 0

    def add_command(self, command: str) -> None:
        """Track command and compute metrics."""
        if not command:
            return

        # Command signature (normalized)
        signature = self._compute_signature(command)
        self.command_signatures.append(signature)

        # Repetition detection (window k=3)
        if len(self.command_signatures) >= 2:
            window = self.command_signatures[-4:-1] if len(self.command_signatures) > 3 else self.command_signatures[:-1]
            if signature in window:
                self.repetition_count += 1

        # File coverage
        if re.search(r"(cat|less|head|tail|sed\s+-n|grep|awk).*config\.yaml", command):
            self.read_config = True
            self.files_touched.add("config.yaml")
        if re.search(r"(cat|less|head|tail|sed\s+-n|grep|awk).*system\.log", command):
            self.read_log = True
            self.files_touched.add("system.log")

        # Edit detection
        if re.search(r"(sed\s+-i|perl\s+-pi|python.*-c|>>|>\s*config)", command):
            self.edit_attempted = True

        self.unique_commands = len(set(self.command_signatures))

    def _compute_signature(self, command: str) -> str:
        """Compute normalized command signature."""
        # Remove quotes, normalize whitespace
        sig = re.sub(r"['\"]", "", command)
        sig = re.sub(r"\s+", " ", sig).strip()
        # Take first 50 chars as signature
        return sig[:50]

    def to_dict(self) -> dict:
        return {
            "read_config": self.read_config,
            "read_log": self.read_log,
            "edit_attempted": self.edit_attempted,
            "files_touched": list(self.files_touched),
            "unique_commands": self.unique_commands,
            "repetition_count": self.repetition_count,
        }


# -----------------------------------------------------------------------------
# Ablation Result (extended)
# -----------------------------------------------------------------------------

@dataclass
class AblationResult:
    """Result of a single ablation test run."""
    test_name: str
    model: str
    policy_id: str
    policy_hash: str
    policy_optimized: bool

    # Parsing
    parse_verdict: str
    command_verdict: str
    json_found: bool
    move_type: str = ""

    # Execution
    command: str = ""
    exit_code: int = -1
    output_correct: bool = False

    # Success
    success: bool = False

    # Timing
    response_time_ms: float = 0.0

    # Hashes
    prompt_hash: str = ""
    output_hash: str = ""

    # Strategic metrics (reviewer fix 4.1)
    strategic_metrics: dict = field(default_factory=dict)

    # Raw data
    raw_output: str = ""


# -----------------------------------------------------------------------------
# Test Runners
# -----------------------------------------------------------------------------

def call_ollama(model: str, prompt: str, config: AblationConfig) -> tuple[str, float]:
    """Call Ollama API with explicit config."""
    start = datetime.now()

    try:
        response = requests.post(
            OLLAMA_API,
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": config.temperature,
                    "top_p": config.top_p,
                    "num_predict": config.max_tokens,
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


def run_json_trap_ablation(
    model: str,
    policy: PolicyBundle,
    config: AblationConfig,
) -> AblationResult:
    """Run JSON trap test with specific policy."""

    task_dir = SANDBOX_ROOT / "json_trap"
    input_file = task_dir / "input.txt"
    expected_file = task_dir / "expected.txt"

    expected = expected_file.read_text().strip()
    input_content = input_file.read_text()

    # Build prompt from policy
    task_description = "Read the file 'input.txt' and output sorted unique tokens (case-insensitive), one per line."
    context = f"INPUT FILE CONTENT:\n{input_content}"

    prompt = policy.build_prompt(task_description, context)
    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:16]

    # Call model
    raw_output, response_time = call_ollama(model, prompt, config)
    output_hash = hashlib.sha256(raw_output.encode()).hexdigest()[:16]

    # Parse output (normalizer mode from config)
    normalizer = OutputNormalizer(mode=config.normalizer_mode)
    parse_result = normalizer.normalize(raw_output)

    # Validate command
    validator = CommandValidator()
    cmd_verdict, _ = validator.validate(parse_result.command)

    # Check move type enforcement (reviewer fix 3.2)
    move_type_ok = (
        parse_result.move_type == MoveType.PROPOSE or
        config.move_type_enforcement == "any"
    )

    # Execute if valid
    exit_code = -1
    output_correct = False

    if parse_result.command and cmd_verdict == CommandVerdict.VALID_SHELL and move_type_ok:
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
            output_correct = (result.stdout.strip() == expected)
        except Exception:
            pass

    success = output_correct and exit_code == 0

    return AblationResult(
        test_name="json_trap",
        model=model,
        policy_id=policy.bundle_id,
        policy_hash=policy.bundle_hash,
        policy_optimized=policy.optimized,
        parse_verdict=parse_result.verdict.value,
        command_verdict=cmd_verdict.value,
        json_found=parse_result.json_found,
        move_type=parse_result.move_type.value if parse_result.move_type else "none",
        command=parse_result.command,
        exit_code=exit_code,
        output_correct=output_correct,
        success=success,
        response_time_ms=response_time,
        prompt_hash=prompt_hash,
        output_hash=output_hash,
        raw_output=raw_output,
    )


def run_evidence_chain_ablation(
    model: str,
    policy: PolicyBundle,
    config: AblationConfig,
) -> list[AblationResult]:
    """Run evidence chain test with specific policy."""

    task_dir = SANDBOX_ROOT / "evidence_chain"
    results = []
    metrics = StrategicMetrics()

    # Reset config to known state
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
    (task_dir / "config.yaml").write_text(config_original)

    # Initial context (no hidden coaching - reviewer fix 3.3)
    task_description = "The pipeline has errors. Find the misconfiguration in config.yaml and fix it."
    base_context = "Working directory contains: system.log, config.yaml"

    conversation_history = []

    for turn in range(config.max_turns):
        # Build context from history (symmetric, no coaching - reviewer fix 3.3)
        if conversation_history:
            history_str = "\n\n".join(conversation_history[-3:])  # Last 3 turns
            full_context = f"{base_context}\n\nPrevious actions:\n{history_str}\n\nContinue:"
        else:
            full_context = f"{base_context}\n\nStart by investigating."

        prompt = policy.build_prompt(task_description, full_context)
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:16]

        # Call model
        raw_output, response_time = call_ollama(model, prompt, config)
        output_hash = hashlib.sha256(raw_output.encode()).hexdigest()[:16]

        # Parse
        normalizer = OutputNormalizer(mode=config.normalizer_mode)
        parse_result = normalizer.normalize(raw_output)

        # Validate
        validator = CommandValidator()
        cmd_verdict, _ = validator.validate(parse_result.command)

        # Track command for strategic metrics
        metrics.add_command(parse_result.command)

        # Check move type (reviewer fix 3.2)
        move_type_ok = (
            parse_result.move_type == MoveType.PROPOSE or
            config.move_type_enforcement == "any"
        )

        # Execute if valid PROPOSE with valid shell command
        exit_code = -1
        stdout = ""

        if parse_result.command and cmd_verdict == CommandVerdict.VALID_SHELL and move_type_ok:
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
            except Exception as e:
                stdout = f"ERROR: {e}"

        # Check for goal (reviewer fix 3.4 - accept any threshold < 0.15)
        goal_reached = False
        try:
            config_content = (task_dir / "config.yaml").read_text()
            threshold_match = re.search(r"alert_threshold:\s*([\d.]+)", config_content)
            if threshold_match:
                threshold_value = float(threshold_match.group(1))
                if threshold_value < config.threshold_success_criterion:
                    goal_reached = True
        except Exception:
            pass

        results.append(AblationResult(
            test_name=f"evidence_chain_t{turn+1}",
            model=model,
            policy_id=policy.bundle_id,
            policy_hash=policy.bundle_hash,
            policy_optimized=policy.optimized,
            parse_verdict=parse_result.verdict.value,
            command_verdict=cmd_verdict.value,
            json_found=parse_result.json_found,
            move_type=parse_result.move_type.value if parse_result.move_type else "none",
            command=parse_result.command,
            exit_code=exit_code,
            output_correct=goal_reached,
            success=goal_reached,
            response_time_ms=response_time,
            prompt_hash=prompt_hash,
            output_hash=output_hash,
            strategic_metrics=metrics.to_dict(),
            raw_output=raw_output,
        ))

        # Update conversation history (symmetric - reviewer fix 3.3)
        action_summary = f"Action: {parse_result.command[:60] if parse_result.command else 'none'}"
        if stdout:
            action_summary += f"\nOutput: {stdout[:200]}"
        conversation_history.append(action_summary)

        if goal_reached:
            break

    return results


# -----------------------------------------------------------------------------
# Report Generation (with Layer config header - reviewer fix 5.1)
# -----------------------------------------------------------------------------

def generate_ablation_report(
    baseline_results: list[AblationResult],
    optimized_results: list[AblationResult],
    test_name: str,
    config: AblationConfig,
    policies: tuple[PolicyBundle, PolicyBundle],
) -> str:
    """Generate markdown ablation report with full configuration."""
    lines = []
    lines.append(f"# Policy Bundle Ablation Study: {test_name}")
    lines.append(f"\n**Generated:** {datetime.now().isoformat()}")
    lines.append("**Type:** Illustrative ablation (not benchmark)")
    lines.append("")

    # Layer config header (reviewer fix 5.1)
    baseline_policy, optimized_policy = policies
    lines.append("## Configuration")
    lines.append("")
    lines.append("| Parameter | Value |")
    lines.append("|-----------|-------|")
    lines.append(f"| Normalizer mode | `{config.normalizer_mode}` (Layer 1 disabled) |")
    lines.append(f"| Move type enforcement | `{config.move_type_enforcement}` |")
    lines.append(f"| Temperature | {config.temperature} |")
    lines.append(f"| Max turns | {config.max_turns} |")
    lines.append(f"| Success criterion | threshold < {config.threshold_success_criterion} |")
    lines.append(f"| Git commit | `{get_git_commit()}` |")
    lines.append("")
    lines.append("| Policy | ID | Hash | Type |")
    lines.append("|--------|-----|------|------|")
    lines.append(f"| Baseline | {baseline_policy.bundle_id} | `{baseline_policy.bundle_hash}` | control |")
    lines.append(f"| Optimized | {optimized_policy.bundle_id} | `{optimized_policy.bundle_hash}` | hand-optimized |")
    lines.append("")

    # Explicit scope statement
    lines.append("### Scope Statement")
    lines.append("")
    lines.append("This ablation tests **policy bundle effect only**:")
    lines.append("- Layer 1 (interface normalization): **disabled**")
    lines.append("- Layer 2 (repair loops): **not implemented**")
    lines.append("- Evidence gain metrics: **proxied via counters**")
    lines.append("")
    lines.append("The optimized bundle is **hand-crafted**, not DSPy-compiled.")
    lines.append("")

    # Results table
    lines.append("## Results Summary")
    lines.append("")
    lines.append("| Model | Policy | Parse | Move Type | Cmd Valid | Success | Time |")
    lines.append("|-------|--------|-------|-----------|-----------|---------|------|")

    all_results = baseline_results + optimized_results
    for r in all_results:
        if "evidence_chain" not in r.test_name or r.test_name.endswith("_t1"):
            model_short = r.model.split(":")[0][:12]
            policy_short = "baseline" if not r.policy_optimized else "optimized"
            success = "Yes" if r.success else "No"
            cmd_valid = "Yes" if r.command_verdict == "valid_shell" else r.command_verdict[:8]
            lines.append(f"| {model_short} | {policy_short} | {r.parse_verdict[:10]} | {r.move_type[:8]} | {cmd_valid} | {success} | {r.response_time_ms:.0f}ms |")

    lines.append("")

    # Strategic metrics (reviewer fix 4.1)
    if any("evidence_chain" in r.test_name for r in all_results):
        lines.append("## Strategic Behavior Metrics")
        lines.append("")
        lines.append("| Model | Policy | Read Log | Read Config | Edit Attempted | Repetitions | Unique Cmds |")
        lines.append("|-------|--------|----------|-------------|----------------|-------------|-------------|")

        # Get final turn metrics for each model/policy
        for r in all_results:
            if r.test_name.endswith("_t4") or r.success:  # Final turn or success
                model_short = r.model.split(":")[0][:12]
                policy_short = "baseline" if not r.policy_optimized else "optimized"
                m = r.strategic_metrics
                lines.append(f"| {model_short} | {policy_short} | {m.get('read_log', False)} | {m.get('read_config', False)} | {m.get('edit_attempted', False)} | {m.get('repetition_count', 0)} | {m.get('unique_commands', 0)} |")

        lines.append("")

    # Per-model analysis
    lines.append("## Per-Model Analysis")
    lines.append("")

    models = sorted(set(r.model for r in all_results))
    for model in models:
        model_baseline = [r for r in baseline_results if r.model == model]
        model_optimized = [r for r in optimized_results if r.model == model]

        base_success = sum(1 for r in model_baseline if r.success)
        opt_success = sum(1 for r in model_optimized if r.success)

        lines.append(f"### {model}")
        lines.append("")
        lines.append(f"| Metric | Baseline | Optimized | Delta |")
        lines.append(f"|--------|----------|-----------|-------|")
        lines.append(f"| Success | {base_success}/{len(model_baseline)} | {opt_success}/{len(model_optimized)} | {opt_success - base_success:+d} |")

        # Add move type stats
        base_propose = sum(1 for r in model_baseline if r.move_type == "PROPOSE")
        opt_propose = sum(1 for r in model_optimized if r.move_type == "PROPOSE")
        lines.append(f"| PROPOSE moves | {base_propose}/{len(model_baseline)} | {opt_propose}/{len(model_optimized)} | {opt_propose - base_propose:+d} |")

        lines.append("")

    # Output samples
    lines.append("## Output Samples")
    lines.append("")

    for model in models:
        lines.append(f"### {model}")
        lines.append("")

        for r in baseline_results:
            if r.model == model and (r.test_name == "json_trap" or r.test_name.endswith("_t1")):
                lines.append("**Baseline:**")
                lines.append("```")
                lines.append(r.raw_output[:300])
                lines.append("```")
                lines.append(f"Command: `{r.command[:60] if r.command else 'none'}`")
                lines.append(f"Move type: {r.move_type}, Verdict: {r.command_verdict}")
                lines.append("")

        for r in optimized_results:
            if r.model == model and (r.test_name == "json_trap" or r.test_name.endswith("_t1")):
                lines.append("**Optimized:**")
                lines.append("```")
                lines.append(r.raw_output[:300])
                lines.append("```")
                lines.append(f"Command: `{r.command[:60] if r.command else 'none'}`")
                lines.append(f"Move type: {r.move_type}, Verdict: {r.command_verdict}")
                lines.append("")

    # Conclusion with caveats
    lines.append("## Conclusion")
    lines.append("")

    total_base = sum(1 for r in baseline_results if r.success)
    total_opt = sum(1 for r in optimized_results if r.success)

    lines.append(f"**Baseline success:** {total_base}/{len(baseline_results)}")
    lines.append(f"**Optimized success:** {total_opt}/{len(optimized_results)}")
    lines.append(f"**Delta:** {total_opt - total_base:+d}")
    lines.append("")

    lines.append("### Interpretation Caveats")
    lines.append("")
    lines.append("- N is small (illustrative, not benchmark)")
    lines.append("- Layer 1 normalization was disabled")
    lines.append("- Layer 2 repair was not tested")
    lines.append("- Policy bundle is hand-crafted, not optimizer-compiled")
    lines.append("- Evidence chain success = threshold edit, not general strategy")
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("*Policy Bundle Ablation Study v2.0.0*")

    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Policy Bundle Ablation Study v2")
    parser.add_argument("--models", type=str,
                       default="granite3.1-moe:3b,qwen2.5-coder:14b",
                       help="Comma-separated model list")
    parser.add_argument("--test", choices=["json_trap", "evidence_chain", "all"],
                       default="json_trap", help="Test to run")
    parser.add_argument("--max-turns", type=str, default="4",
                       help="Comma-separated max turns for evidence chain (e.g., '4,8,12')")
    parser.add_argument("--verbose", action="store_true", help="Show detailed output")

    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",")]
    max_turns_list = [int(t.strip()) for t in args.max_turns.split(",")]

    # Load policies
    baseline_policy = PolicyBundle.load(POLICIES_DIR / "baseline.yaml")
    optimized_policy = PolicyBundle.load(POLICIES_DIR / "optimized_v1.yaml")

    print("=" * 70)
    print("POLICY BUNDLE ABLATION STUDY v2")
    print("=" * 70)
    print(f"Models: {models}")
    print(f"Test: {args.test}")
    print(f"Max turns: {max_turns_list}")
    print(f"Baseline: {baseline_policy.bundle_id} ({baseline_policy.bundle_hash})")
    print(f"Optimized: {optimized_policy.bundle_id} ({optimized_policy.bundle_hash})")
    print("")
    print("NOTE: This is an ILLUSTRATIVE ablation, not a benchmark.")
    print("      Layer 1 normalization is DISABLED.")
    print("      Optimized bundle is HAND-CRAFTED, not DSPy-compiled.")
    print("=" * 70)

    # Run tests
    if args.test in ["json_trap", "all"]:
        print("\n=== JSON TRAP TEST ===")

        config = AblationConfig()
        baseline_results = []
        optimized_results = []

        for model in models:
            print(f"\n--- {model} ---")

            # Baseline
            print(f"  Baseline...", end=" ", flush=True)
            result = run_json_trap_ablation(model, baseline_policy, config)
            baseline_results.append(result)
            status = "SUCCESS" if result.success else "FAIL"
            print(f"{status} (move={result.move_type}, cmd={result.command_verdict})")

            # Optimized
            print(f"  Optimized...", end=" ", flush=True)
            result = run_json_trap_ablation(model, optimized_policy, config)
            optimized_results.append(result)
            status = "SUCCESS" if result.success else "FAIL"
            print(f"{status} (move={result.move_type}, cmd={result.command_verdict})")

        # Generate report
        RESULTS_DIR.mkdir(exist_ok=True)
        report = generate_ablation_report(
            baseline_results, optimized_results, "json_trap",
            config, (baseline_policy, optimized_policy)
        )
        output_path = RESULTS_DIR / f"ablation_v2_json_trap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        output_path.write_text(report)
        print(f"\nReport: {output_path}")

    if args.test in ["evidence_chain", "all"]:
        print("\n=== EVIDENCE CHAIN TEST ===")

        for max_turns in max_turns_list:
            print(f"\n--- Max turns: {max_turns} ---")

            config = AblationConfig(max_turns=max_turns)
            baseline_results = []
            optimized_results = []

            for model in models:
                print(f"\n  {model}:")

                # Baseline
                print(f"    Baseline...", end=" ", flush=True)
                results = run_evidence_chain_ablation(model, baseline_policy, config)
                baseline_results.extend(results)
                final = results[-1] if results else None
                if final:
                    status = "SUCCESS" if final.success else "FAIL"
                    m = final.strategic_metrics
                    print(f"{len(results)} turns, {status} "
                          f"(log={m.get('read_log')}, cfg={m.get('read_config')}, "
                          f"edit={m.get('edit_attempted')}, rep={m.get('repetition_count')})")

                # Optimized
                print(f"    Optimized...", end=" ", flush=True)
                results = run_evidence_chain_ablation(model, optimized_policy, config)
                optimized_results.extend(results)
                final = results[-1] if results else None
                if final:
                    status = "SUCCESS" if final.success else "FAIL"
                    m = final.strategic_metrics
                    print(f"{len(results)} turns, {status} "
                          f"(log={m.get('read_log')}, cfg={m.get('read_config')}, "
                          f"edit={m.get('edit_attempted')}, rep={m.get('repetition_count')})")

            # Generate report
            report = generate_ablation_report(
                baseline_results, optimized_results, f"evidence_chain_t{max_turns}",
                config, (baseline_policy, optimized_policy)
            )
            output_path = RESULTS_DIR / f"ablation_v2_evidence_t{max_turns}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            output_path.write_text(report)
            print(f"\n  Report: {output_path}")

    # Final summary
    print("\n" + "=" * 70)
    print("ABLATION COMPLETE")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
