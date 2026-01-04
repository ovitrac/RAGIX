#!/usr/bin/env python3
# =============================================================================
# DSPy Ablation Study - Policy Bundle Comparison
# =============================================================================
#
# Compares baseline (no optimization) vs. optimized policy bundles.
# Demonstrates the value of DSPy-style prompt optimization.
#
# Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
# Version: 1.0.0 (2025-12-27)
#
# =============================================================================

"""
DSPy Ablation Study.

Compares the same model running with:
1. Baseline policy (minimal guidance)
2. Optimized policy (DSPy-style prompt + exemplars)

This isolates the contribution of prompt optimization from model capability.

Usage:
    python ablation_study.py --models "granite3.1-moe:3b,qwen2.5-coder:14b"
    python ablation_study.py --test json_trap --verbose
"""

from __future__ import annotations
import argparse
import json
import hashlib
import subprocess
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
# Policy Bundle Loader
# -----------------------------------------------------------------------------

@dataclass
class PolicyBundle:
    """Loaded policy bundle."""
    bundle_id: str
    bundle_type: str
    optimized: bool
    system_prompt: str
    task_template: str
    exemplars: list
    anti_patterns: list
    model_params: dict

    @classmethod
    def load(cls, path: Path) -> "PolicyBundle":
        """Load policy bundle from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)

        return cls(
            bundle_id=data.get("bundle_id", "unknown"),
            bundle_type=data.get("bundle_type", "unknown"),
            optimized=data.get("metadata", {}).get("optimized", False),
            system_prompt=data.get("prompts", {}).get("system", ""),
            task_template=data.get("prompts", {}).get("task_template", ""),
            exemplars=data.get("exemplars", []),
            anti_patterns=data.get("anti_patterns", []),
            model_params=data.get("model_params", {}),
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
# Ablation Test Result
# -----------------------------------------------------------------------------

@dataclass
class AblationResult:
    """Result of a single ablation test run."""
    test_name: str
    model: str
    policy_id: str
    policy_optimized: bool

    # Parsing
    parse_verdict: str
    command_verdict: str
    json_found: bool

    # Execution
    command: str
    exit_code: int
    output_correct: bool

    # Success
    success: bool

    # Timing
    response_time_ms: float

    # Hashes
    prompt_hash: str
    output_hash: str

    # Raw data
    raw_output: str


# -----------------------------------------------------------------------------
# Test Runner
# -----------------------------------------------------------------------------

def call_ollama(model: str, prompt: str, params: dict) -> tuple[str, float]:
    """Call Ollama API with policy parameters."""
    start = datetime.now()

    try:
        response = requests.post(
            OLLAMA_API,
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": params.get("temperature", 0.0),
                    "top_p": params.get("top_p", 1.0),
                    "num_predict": params.get("max_tokens", 512),
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
    raw_output, response_time = call_ollama(model, prompt, policy.model_params)
    output_hash = hashlib.sha256(raw_output.encode()).hexdigest()[:16]

    # Parse output
    normalizer = OutputNormalizer(mode="off")
    parse_result = normalizer.normalize(raw_output)

    # Validate command
    validator = CommandValidator()
    cmd_verdict, _ = validator.validate(parse_result.command)

    # Execute if valid
    exit_code = -1
    output_correct = False

    if parse_result.command and cmd_verdict == CommandVerdict.VALID_SHELL:
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
        policy_optimized=policy.optimized,
        parse_verdict=parse_result.verdict.value,
        command_verdict=cmd_verdict.value,
        json_found=parse_result.json_found,
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
    max_turns: int = 4,
) -> list[AblationResult]:
    """Run evidence chain test with specific policy."""

    task_dir = SANDBOX_ROOT / "evidence_chain"
    results = []

    # Reset config
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

    # Initial context
    task_description = "The pipeline has errors. Find the misconfiguration in config.yaml and fix it."
    context = "Working directory contains: system.log, config.yaml\n\nStart by investigating the error."

    conversation_context = ""

    for turn in range(max_turns):
        # Build prompt
        full_context = context + conversation_context
        prompt = policy.build_prompt(task_description, full_context)
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:16]

        # Call model
        raw_output, response_time = call_ollama(model, prompt, policy.model_params)
        output_hash = hashlib.sha256(raw_output.encode()).hexdigest()[:16]

        # Parse
        normalizer = OutputNormalizer(mode="off")
        parse_result = normalizer.normalize(raw_output)

        # Validate
        validator = CommandValidator()
        cmd_verdict, _ = validator.validate(parse_result.command)

        # Execute if valid PROPOSE
        exit_code = -1
        stdout = ""

        if parse_result.move_type == MoveType.PROPOSE and cmd_verdict == CommandVerdict.VALID_SHELL:
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
            except Exception:
                pass

        # Check for goal
        goal_reached = False
        config_content = (task_dir / "config.yaml").read_text()
        if "alert_threshold: 0.10" in config_content or "alert_threshold: 0.1" in config_content:
            goal_reached = True

        results.append(AblationResult(
            test_name=f"evidence_chain_t{turn+1}",
            model=model,
            policy_id=policy.bundle_id,
            policy_optimized=policy.optimized,
            parse_verdict=parse_result.verdict.value,
            command_verdict=cmd_verdict.value,
            json_found=parse_result.json_found,
            command=parse_result.command,
            exit_code=exit_code,
            output_correct=goal_reached,
            success=goal_reached,
            response_time_ms=response_time,
            prompt_hash=prompt_hash,
            output_hash=output_hash,
            raw_output=raw_output,
        ))

        # Update context for next turn
        if stdout:
            conversation_context += f"\n\nPrevious action result:\n{stdout[:300]}\n\nContinue:"
        else:
            conversation_context += "\n\nPrevious action had no output. Try a different approach:"

        if goal_reached:
            break

    return results


# -----------------------------------------------------------------------------
# Report Generation
# -----------------------------------------------------------------------------

def generate_ablation_report(
    baseline_results: list[AblationResult],
    optimized_results: list[AblationResult],
    test_name: str
) -> str:
    """Generate markdown ablation report."""
    lines = []
    lines.append(f"# DSPy Ablation Study: {test_name}")
    lines.append(f"\n**Generated:** {datetime.now().isoformat()}")
    lines.append("**Purpose:** Compare baseline vs. optimized policy bundles")
    lines.append("")

    # Summary table
    lines.append("## Results Summary")
    lines.append("")
    lines.append("| Model | Policy | Parse | Cmd Valid | Success | Time (ms) |")
    lines.append("|-------|--------|-------|-----------|---------|-----------|")

    all_results = baseline_results + optimized_results
    for r in all_results:
        if "evidence_chain" not in r.test_name or r.test_name.endswith("_t1"):
            model_short = r.model.split(":")[0][:12]
            policy_short = "baseline" if not r.policy_optimized else "optimized"
            success = "Yes" if r.success else "No"
            cmd_valid = "Yes" if r.command_verdict == "valid_shell" else r.command_verdict[:8]
            lines.append(f"| {model_short} | {policy_short} | {r.parse_verdict[:10]} | {cmd_valid} | {success} | {r.response_time_ms:.0f} |")

    lines.append("")

    # Comparison analysis
    lines.append("## Ablation Analysis")
    lines.append("")

    # Group by model
    models = set(r.model for r in all_results)
    for model in sorted(models):
        model_baseline = [r for r in baseline_results if r.model == model]
        model_optimized = [r for r in optimized_results if r.model == model]

        base_success = sum(1 for r in model_baseline if r.success)
        opt_success = sum(1 for r in model_optimized if r.success)

        base_valid_cmd = sum(1 for r in model_baseline if r.command_verdict == "valid_shell")
        opt_valid_cmd = sum(1 for r in model_optimized if r.command_verdict == "valid_shell")

        lines.append(f"### {model}")
        lines.append("")
        lines.append(f"| Metric | Baseline | Optimized | Delta |")
        lines.append(f"|--------|----------|-----------|-------|")
        lines.append(f"| Success rate | {base_success}/{len(model_baseline)} | {opt_success}/{len(model_optimized)} | {opt_success - base_success:+d} |")
        lines.append(f"| Valid commands | {base_valid_cmd}/{len(model_baseline)} | {opt_valid_cmd}/{len(model_optimized)} | {opt_valid_cmd - base_valid_cmd:+d} |")
        lines.append("")

    # Detailed output comparison
    lines.append("## Output Comparison")
    lines.append("")

    for model in sorted(models):
        lines.append(f"### {model}")
        lines.append("")

        model_baseline = [r for r in baseline_results if r.model == model]
        model_optimized = [r for r in optimized_results if r.model == model]

        if model_baseline:
            r = model_baseline[0]
            lines.append("**Baseline output:**")
            lines.append("```")
            lines.append(r.raw_output[:400])
            lines.append("```")
            lines.append(f"Command: `{r.command[:80] if r.command else 'None'}`")
            lines.append(f"Verdict: {r.command_verdict}")
            lines.append("")

        if model_optimized:
            r = model_optimized[0]
            lines.append("**Optimized output:**")
            lines.append("```")
            lines.append(r.raw_output[:400])
            lines.append("```")
            lines.append(f"Command: `{r.command[:80] if r.command else 'None'}`")
            lines.append(f"Verdict: {r.command_verdict}")
            lines.append("")

    # Conclusion
    lines.append("## Conclusion")
    lines.append("")

    total_base_success = sum(1 for r in baseline_results if r.success)
    total_opt_success = sum(1 for r in optimized_results if r.success)

    if total_opt_success > total_base_success:
        lines.append(f"**Optimized policy improves success rate:** {total_base_success} → {total_opt_success}")
        lines.append("")
        lines.append("Key improvements from optimization:")
        lines.append("- Explicit format requirements reduce JSON pollution")
        lines.append("- Few-shot exemplars demonstrate correct command syntax")
        lines.append("- Anti-pattern warnings prevent common failures")
    elif total_opt_success == total_base_success:
        lines.append(f"**No change in success rate:** {total_base_success} = {total_opt_success}")
        lines.append("")
        lines.append("Possible explanations:")
        lines.append("- Model already capable (fat model)")
        lines.append("- Task too simple to differentiate")
        lines.append("- Optimization targets different failure mode")
    else:
        lines.append(f"**Unexpected: baseline outperforms optimized:** {total_base_success} > {total_opt_success}")
        lines.append("")
        lines.append("Requires investigation.")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*DSPy Ablation Study v1.0.0*")

    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="DSPy Ablation Study")
    parser.add_argument("--models", type=str,
                       default="granite3.1-moe:3b,qwen2.5-coder:14b",
                       help="Comma-separated model list")
    parser.add_argument("--test", choices=["json_trap", "evidence_chain", "all"],
                       default="json_trap", help="Test to run")
    parser.add_argument("--verbose", action="store_true", help="Show detailed output")

    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",")]

    # Load policies
    baseline_policy = PolicyBundle.load(POLICIES_DIR / "baseline.yaml")
    optimized_policy = PolicyBundle.load(POLICIES_DIR / "optimized_v1.yaml")

    print("=" * 70)
    print("DSPy ABLATION STUDY")
    print("=" * 70)
    print(f"Models: {models}")
    print(f"Test: {args.test}")
    print(f"Baseline policy: {baseline_policy.bundle_id}")
    print(f"Optimized policy: {optimized_policy.bundle_id}")
    print("=" * 70)

    baseline_results = []
    optimized_results = []

    # Run tests
    if args.test in ["json_trap", "all"]:
        print("\n=== JSON TRAP TEST ===")

        for model in models:
            print(f"\n--- {model} ---")

            # Baseline
            print(f"  Baseline...", end=" ", flush=True)
            result = run_json_trap_ablation(model, baseline_policy)
            baseline_results.append(result)
            status = "SUCCESS" if result.success else "FAIL"
            print(f"{status} (parse={result.parse_verdict}, cmd={result.command_verdict})")
            if args.verbose:
                print(f"    Command: {result.command[:60]}...")
                print(f"    Output: {result.raw_output[:100]}...")

            # Optimized
            print(f"  Optimized...", end=" ", flush=True)
            result = run_json_trap_ablation(model, optimized_policy)
            optimized_results.append(result)
            status = "SUCCESS" if result.success else "FAIL"
            print(f"{status} (parse={result.parse_verdict}, cmd={result.command_verdict})")
            if args.verbose:
                print(f"    Command: {result.command[:60]}...")
                print(f"    Output: {result.raw_output[:100]}...")

    if args.test in ["evidence_chain", "all"]:
        print("\n=== EVIDENCE CHAIN TEST ===")

        for model in models:
            print(f"\n--- {model} ---")

            # Baseline
            print(f"  Baseline...", end=" ", flush=True)
            results = run_evidence_chain_ablation(model, baseline_policy)
            baseline_results.extend(results)
            final = results[-1] if results else None
            if final:
                status = "SUCCESS" if final.success else "FAIL"
                print(f"{len(results)} turns, {status}")

            # Optimized
            print(f"  Optimized...", end=" ", flush=True)
            results = run_evidence_chain_ablation(model, optimized_policy)
            optimized_results.extend(results)
            final = results[-1] if results else None
            if final:
                status = "SUCCESS" if final.success else "FAIL"
                print(f"{len(results)} turns, {status}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    base_success = sum(1 for r in baseline_results if r.success)
    opt_success = sum(1 for r in optimized_results if r.success)

    print(f"\nBaseline success: {base_success}/{len(baseline_results)}")
    print(f"Optimized success: {opt_success}/{len(optimized_results)}")
    print(f"Improvement: {opt_success - base_success:+d}")

    # Generate report
    RESULTS_DIR.mkdir(exist_ok=True)
    report = generate_ablation_report(baseline_results, optimized_results, args.test)
    output_path = RESULTS_DIR / f"ablation_{args.test}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    output_path.write_text(report)
    print(f"\nReport written to: {output_path}")

    return 0 if opt_success >= base_success else 1


if __name__ == "__main__":
    sys.exit(main())
