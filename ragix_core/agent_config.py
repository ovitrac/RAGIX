"""
RAGIX Multi-Agent LLM Configuration
====================================

Configures different LLM models for Planner, Worker, and Verifier agents.
Supports Minimal (3B all), Strict (7B+ Planner), and Custom modes.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-11-28
"""

import os
import re
import subprocess
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# Agent Mode Enum
# =============================================================================

class AgentMode(Enum):
    """Agent LLM assignment modes."""
    MINIMAL = "minimal"   # All agents use 3B model (8GB VRAM / CPU)
    STRICT = "strict"     # Planner uses 7B+, Worker/Verifier use 3B
    CUSTOM = "custom"     # User-defined per-agent models


class AgentRole(Enum):
    """Agent roles in the Planner-Worker-Verifier architecture."""
    PLANNER = "planner"
    WORKER = "worker"
    VERIFIER = "verifier"


# =============================================================================
# Model Registry - Known Models with Metadata
# =============================================================================

# Model name -> (size_billions, category, description)
MODEL_REGISTRY: Dict[str, Tuple[float, str, str]] = {
    # Granite models (IBM)
    "granite3.1-moe:3b": (3.0, "3B", "Fast, deterministic, excellent tool-use"),
    "granite3.1-moe:1b": (1.0, "1B", "Ultra-light, basic tasks only"),

    # Mistral models
    "mistral:latest": (7.0, "7B", "Strong general reasoning"),
    "mistral:7b": (7.0, "7B", "Strong general reasoning"),
    "mistral:instruct": (7.0, "7B", "Instruction-tuned Mistral"),
    "dolphin-mistral:7b": (7.0, "7B", "Uncensored Mistral variant"),
    "dolphin-mistral:7b-v2.6-dpo-laser": (7.0, "7B", "DPO-tuned Mistral"),

    # Llama models
    "llama3:latest": (8.0, "8B", "Meta's Llama 3"),
    "llama3:8b": (8.0, "8B", "Meta's Llama 3 8B"),
    "llama3.2:3b": (3.0, "3B", "Meta's Llama 3.2 3B"),
    "llama3.2:1b": (1.0, "1B", "Meta's Llama 3.2 1B"),

    # DeepSeek models
    "deepseek-r1:14b": (14.0, "14B", "Strong reasoning, chain-of-thought"),
    "deepseek-r1:7b": (7.0, "7B", "Reasoning model"),
    "deepseek-coder:6.7b": (6.7, "7B", "Code-specialized"),

    # Qwen models
    "qwen2.5:3b": (3.0, "3B", "Alibaba's Qwen 2.5 3B"),
    "qwen2.5:7b": (7.0, "7B", "Alibaba's Qwen 2.5 7B"),
    "qwen2.5:14b": (14.0, "14B", "Alibaba's Qwen 2.5 14B"),

    # Phi models (Microsoft)
    "phi3:mini": (3.8, "4B", "Microsoft Phi-3 mini"),
    "phi3:medium": (14.0, "14B", "Microsoft Phi-3 medium"),
}

# Default fallback model
DEFAULT_MODEL = "granite3.1-moe:3b"


# =============================================================================
# Model Context Limits - Maximum context window per model (in tokens)
# =============================================================================

# Model name/pattern -> context window size in tokens
# Use patterns for model families (e.g., "qwen2.5" matches "qwen2.5:7b", "qwen2.5:14b")
MODEL_CONTEXT_LIMITS: Dict[str, int] = {
    # Granite models - 128k context
    "granite3.1-moe:3b": 128_000,
    "granite3.1-moe:1b": 128_000,
    "granite": 128_000,  # Default for granite family

    # Mistral models - 8k-32k context
    "mistral:latest": 32_000,
    "mistral:7b": 32_000,
    "mistral:instruct": 32_000,
    "dolphin-mistral": 32_000,
    "mistral": 32_000,  # Default for mistral family

    # Llama models - 8k-128k context
    "llama3:latest": 8_000,
    "llama3:8b": 8_000,
    "llama3.2:3b": 128_000,
    "llama3.2:1b": 128_000,
    "llama3.2": 128_000,
    "llama3": 8_000,  # Default for llama3 family

    # DeepSeek models - 32k-64k context
    "deepseek-r1:14b": 64_000,
    "deepseek-r1:7b": 64_000,
    "deepseek-coder": 32_000,
    "deepseek": 64_000,  # Default for deepseek family

    # Qwen models - 32k-128k context
    "qwen2.5:3b": 32_000,
    "qwen2.5:7b": 32_000,
    "qwen2.5:14b": 128_000,
    "qwen2.5": 32_000,  # Default for qwen2.5 family
    "qwen": 32_000,

    # Phi models - 4k-128k context
    "phi3:mini": 128_000,
    "phi3:medium": 128_000,
    "phi3": 128_000,
    "phi": 4_000,
}

# Default context limit for unknown models (conservative)
DEFAULT_CONTEXT_LIMIT = 8_000


def get_model_context_limit(model_name: str) -> int:
    """
    Get the context window limit for a model.

    Args:
        model_name: Model name (e.g., "qwen2.5:7b", "mistral:latest")

    Returns:
        Context limit in tokens
    """
    # Normalize model name
    model_lower = model_name.lower().strip()

    # Try exact match first
    if model_lower in MODEL_CONTEXT_LIMITS:
        return MODEL_CONTEXT_LIMITS[model_lower]

    # Try prefix match (for model families)
    for pattern, limit in MODEL_CONTEXT_LIMITS.items():
        if model_lower.startswith(pattern):
            return limit

    # Try partial match (model name contains pattern)
    for pattern, limit in MODEL_CONTEXT_LIMITS.items():
        if pattern in model_lower:
            return limit

    logger.warning(f"Unknown model '{model_name}', using default context limit {DEFAULT_CONTEXT_LIMIT}")
    return DEFAULT_CONTEXT_LIMIT


def get_model_info(model_name: str) -> Dict:
    """
    Get comprehensive model information including context limits.

    Args:
        model_name: Model name

    Returns:
        Dict with size, category, description, context_limit
    """
    model_lower = model_name.lower().strip()

    # Get from registry
    if model_lower in MODEL_REGISTRY:
        size, category, desc = MODEL_REGISTRY[model_lower]
    else:
        # Try to infer from name
        size = 0.0
        category = "unknown"
        desc = f"Model: {model_name}"

        # Extract size from name if possible (e.g., "7b", "14b")
        import re
        size_match = re.search(r'(\d+(?:\.\d+)?)\s*b', model_lower)
        if size_match:
            size = float(size_match.group(1))
            category = f"{int(size)}B"

    return {
        "name": model_name,
        "size_billions": size,
        "category": category,
        "description": desc,
        "context_limit": get_model_context_limit(model_name),
    }

# Size requirements by mode
SIZE_REQUIREMENTS = {
    AgentMode.MINIMAL: {
        AgentRole.PLANNER: 1.0,   # No minimum
        AgentRole.WORKER: 1.0,
        AgentRole.VERIFIER: 1.0,
    },
    AgentMode.STRICT: {
        AgentRole.PLANNER: 7.0,   # 7B minimum for planner
        AgentRole.WORKER: 3.0,    # 3B minimum for worker
        AgentRole.VERIFIER: 3.0,  # 3B minimum for verifier
    },
}


# =============================================================================
# Ollama Model Detection
# =============================================================================

@dataclass
class OllamaModel:
    """Represents an installed Ollama model."""
    name: str
    size_gb: float
    modified: str
    params_b: float = 0.0  # Estimated parameters in billions

    @property
    def category(self) -> str:
        """Get model category based on parameter count."""
        if self.params_b >= 14:
            return "14B+"
        elif self.params_b >= 7:
            return "7B"
        elif self.params_b >= 3:
            return "3B"
        elif self.params_b >= 1:
            return "1B"
        return "Unknown"


def detect_ollama_models(base_url: str = "http://localhost:11434") -> List[OllamaModel]:
    """
    Detect installed Ollama models via 'ollama list' command.

    Returns:
        List of OllamaModel instances
    """
    models = []

    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode != 0:
            logger.warning(f"ollama list failed: {result.stderr}")
            return models

        # Parse output: NAME  ID  SIZE  MODIFIED
        lines = result.stdout.strip().split('\n')
        if len(lines) < 2:
            return models

        for line in lines[1:]:  # Skip header
            parts = line.split()
            if len(parts) >= 4:
                name = parts[0]
                size_str = parts[2]
                modified = ' '.join(parts[3:])

                # Parse size (e.g., "2.0 GB", "4.1 GB")
                size_gb = _parse_size_gb(size_str)

                # Estimate parameters from registry or size
                params_b = _estimate_params(name, size_gb)

                models.append(OllamaModel(
                    name=name,
                    size_gb=size_gb,
                    modified=modified,
                    params_b=params_b
                ))

    except FileNotFoundError:
        logger.warning("Ollama not installed or not in PATH")
    except subprocess.TimeoutExpired:
        logger.warning("Ollama list timed out")
    except Exception as e:
        logger.warning(f"Error detecting Ollama models: {e}")

    return models


def _parse_size_gb(size_str: str) -> float:
    """Parse size string like '2.0' or '4.1' to float GB."""
    try:
        # Handle formats: "2.0", "2.0GB", "2.0 GB"
        match = re.match(r'([\d.]+)', size_str)
        if match:
            return float(match.group(1))
    except (ValueError, AttributeError):
        pass
    return 0.0


def _estimate_params(name: str, size_gb: float) -> float:
    """Estimate model parameters from name or size."""
    # Check registry first
    if name in MODEL_REGISTRY:
        return MODEL_REGISTRY[name][0]

    # Try to extract from name (e.g., "mistral:7b", "llama3:8b")
    match = re.search(r':(\d+)b', name.lower())
    if match:
        return float(match.group(1))

    # Estimate from size (rough heuristic: ~0.5GB per billion params for quantized)
    # This is very approximate
    return size_gb * 1.5


# =============================================================================
# Agent Configuration
# =============================================================================

@dataclass
class AgentConfig:
    """
    Configuration for the Planner-Worker-Verifier agent architecture.

    Attributes:
        mode: Assignment mode (minimal, strict, custom)
        planner_model: LLM model for Planner agent
        worker_model: LLM model for Worker agent
        verifier_model: LLM model for Verifier agent
        strict_enforcement: If True, validate models meet size requirements
        fallback_model: Model to use if assigned model unavailable
    """
    mode: AgentMode = AgentMode.MINIMAL
    planner_model: str = DEFAULT_MODEL
    worker_model: str = DEFAULT_MODEL
    verifier_model: str = DEFAULT_MODEL
    strict_enforcement: bool = False
    fallback_model: str = DEFAULT_MODEL

    def get_model(self, role: AgentRole) -> str:
        """
        Get the LLM model for a specific agent role.

        Applies mode rules:
        - MINIMAL: Always returns 3B model
        - STRICT: Planner gets 7B+, others get 3B
        - CUSTOM: Returns configured model

        Args:
            role: Agent role (planner, worker, verifier)

        Returns:
            Model name string
        """
        if self.mode == AgentMode.MINIMAL:
            return self.fallback_model  # Always 3B

        elif self.mode == AgentMode.STRICT:
            if role == AgentRole.PLANNER:
                return self.planner_model or "mistral:latest"
            return self.fallback_model  # 3B for worker/verifier

        else:  # CUSTOM
            model_map = {
                AgentRole.PLANNER: self.planner_model,
                AgentRole.WORKER: self.worker_model,
                AgentRole.VERIFIER: self.verifier_model,
            }
            return model_map.get(role, self.fallback_model)

    def validate(self, available_models: Optional[List[str]] = None) -> List[str]:
        """
        Validate configuration.

        Args:
            available_models: List of installed model names (auto-detected if None)

        Returns:
            List of warning messages (empty if valid)
        """
        warnings = []

        # Auto-detect models if not provided
        if available_models is None:
            detected = detect_ollama_models()
            available_models = [m.name for m in detected]

        # Check if models are available
        for role in AgentRole:
            model = self.get_model(role)
            if available_models and model not in available_models:
                warnings.append(
                    f"{role.value}: Model '{model}' not found. "
                    f"Available: {', '.join(available_models[:5])}"
                )

        # Check size requirements in strict mode
        if self.strict_enforcement and self.mode == AgentMode.STRICT:
            for role in AgentRole:
                model = self.get_model(role)
                required = SIZE_REQUIREMENTS[AgentMode.STRICT][role]
                actual = MODEL_REGISTRY.get(model, (0, "", ""))[0]

                if actual < required:
                    warnings.append(
                        f"{role.value}: Model '{model}' ({actual}B) below "
                        f"requirement ({required}B) for strict mode"
                    )

        return warnings

    @classmethod
    def detect_optimal(cls, vram_gb: Optional[float] = None) -> "AgentConfig":
        """
        Auto-configure based on available hardware.

        Args:
            vram_gb: Available VRAM in GB (auto-detected if None)

        Returns:
            Optimally configured AgentConfig
        """
        # Detect available models
        models = detect_ollama_models()
        model_names = {m.name for m in models}

        # Auto-detect VRAM (simplified - assumes GPU info)
        if vram_gb is None:
            vram_gb = _detect_vram()

        # Select mode based on VRAM
        if vram_gb <= 8:
            # Minimal mode - use smallest available 3B model
            fallback = _find_best_model(models, max_params=4.0) or DEFAULT_MODEL
            return cls(
                mode=AgentMode.MINIMAL,
                fallback_model=fallback
            )

        elif vram_gb <= 12:
            # Strict mode - use 7B for planner, 3B for others
            planner = _find_best_model(models, min_params=7.0, max_params=10.0)
            worker = _find_best_model(models, max_params=4.0) or DEFAULT_MODEL

            return cls(
                mode=AgentMode.STRICT,
                planner_model=planner or "mistral:latest",
                worker_model=worker,
                verifier_model=worker,
                fallback_model=worker
            )

        else:
            # Advanced mode - use larger models
            planner = _find_best_model(models, min_params=14.0) or \
                      _find_best_model(models, min_params=7.0) or "deepseek-r1:14b"
            worker = _find_best_model(models, min_params=7.0, max_params=10.0) or \
                     _find_best_model(models, max_params=4.0) or "mistral:latest"
            verifier = _find_best_model(models, max_params=4.0) or DEFAULT_MODEL

            return cls(
                mode=AgentMode.CUSTOM,
                planner_model=planner,
                worker_model=worker,
                verifier_model=verifier,
                fallback_model=verifier
            )

    def to_dict(self) -> Dict:
        """Convert to dictionary for YAML serialization."""
        return {
            "mode": self.mode.value,
            "strict_enforcement": self.strict_enforcement,
            "models": {
                "planner": self.planner_model,
                "worker": self.worker_model,
                "verifier": self.verifier_model,
            },
            "fallback_model": self.fallback_model,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "AgentConfig":
        """Create from dictionary (YAML deserialization)."""
        mode_str = data.get("mode", "minimal")
        try:
            mode = AgentMode(mode_str)
        except ValueError:
            logger.warning(f"Unknown agent mode '{mode_str}', using minimal")
            mode = AgentMode.MINIMAL

        models = data.get("models", {})

        return cls(
            mode=mode,
            planner_model=models.get("planner", DEFAULT_MODEL),
            worker_model=models.get("worker", DEFAULT_MODEL),
            verifier_model=models.get("verifier", DEFAULT_MODEL),
            strict_enforcement=data.get("strict_enforcement", False),
            fallback_model=data.get("fallback_model", DEFAULT_MODEL),
        )


def _detect_vram() -> float:
    """
    Detect available VRAM (simplified).

    Returns:
        Estimated VRAM in GB (defaults to 8 if unknown)
    """
    # Try nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            # Returns memory in MB
            mb = float(result.stdout.strip().split('\n')[0])
            return mb / 1024
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass

    # Default to conservative estimate
    return 8.0


def _find_best_model(
    models: List[OllamaModel],
    min_params: float = 0.0,
    max_params: float = float('inf')
) -> Optional[str]:
    """
    Find best model within parameter range.

    Prefers models in order: granite > mistral > llama > others
    """
    candidates = [
        m for m in models
        if min_params <= m.params_b <= max_params
    ]

    if not candidates:
        return None

    # Sort by preference (granite first, then by size descending)
    def sort_key(m: OllamaModel) -> Tuple[int, float]:
        # Lower priority number = better
        if "granite" in m.name.lower():
            priority = 0
        elif "mistral" in m.name.lower():
            priority = 1
        elif "llama" in m.name.lower():
            priority = 2
        elif "deepseek" in m.name.lower():
            priority = 3
        else:
            priority = 4

        return (priority, -m.params_b)  # Negative for descending size

    candidates.sort(key=sort_key)
    return candidates[0].name


# =============================================================================
# Agent Persona Prompts
# =============================================================================

AGENT_PERSONAS = {
    AgentRole.PLANNER: """You are a RAGIX Planner Agent.

ROLE: Structure complex tasks into actionable [PLAN] documents.

OUTPUT FORMAT:
[PLAN]
1. Define the objective.
2. Identify required data.
3. List steps.
4. Solve steps one by one.
5. Validate.
6. Summarize.

CONSTRAINTS:
- Break complex tasks into atomic steps
- Identify all required files and tools
- Consider dependencies between steps
- Be specific, not generic

You will hand off execution to Worker agents.""",

    AgentRole.WORKER: """You are a RAGIX Worker Agent operating in execution mode.

CRITICAL CONSTRAINTS:
- You MUST follow the provided [PLAN] exactly
- You MUST output valid JSON for tool calls
- You MUST NOT invent information outside provided context
- You MUST complete one atomic step at a time
- You MUST report step completion status

OUTPUT FORMAT FOR TOOL CALLS:
{{"action": "bash", "command": "..."}}
{{"action": "edit_file", "path": "...", "old": "...", "new": "..."}}
{{"action": "respond", "message": "..."}}

Execute the assigned step with precision. No creativity. No improvisation.""",

    AgentRole.VERIFIER: """You are a RAGIX Verifier Agent.

ROLE: Validate that work meets requirements and rules.

CHECKS TO PERFORM:
- Did we address ALL points from [PLAN]?
- Did we respect relevant rules?
- Are there obvious syntax or structural issues?
- Are there missing updates (README, CHANGELOG, tests)?

OUTPUT FORMAT:
[VERIFY]
- Scope checked: ...
- Conformity to rules: ...
- Potential issues / uncertainties: ...
- Recommended follow-up: ...

Be honest about uncertainties. Flag risks clearly."""
}


def get_agent_persona(role: AgentRole, tools: Optional[List[str]] = None) -> str:
    """
    Get the system prompt for an agent role.

    Args:
        role: Agent role
        tools: Optional list of available tools to include

    Returns:
        System prompt string
    """
    persona = AGENT_PERSONAS[role]

    if tools and role == AgentRole.WORKER:
        tools_list = "\n".join(f"- {t}" for t in tools)
        persona += f"\n\nAVAILABLE TOOLS:\n{tools_list}"

    return persona


# =============================================================================
# Convenience Functions
# =============================================================================

def print_model_summary():
    """Print summary of detected models and recommendations."""
    models = detect_ollama_models()

    print("\n" + "="*60)
    print("RAGIX Agent LLM Configuration")
    print("="*60)

    if not models:
        print("\nNo Ollama models detected. Run 'ollama pull granite3.1-moe:3b'")
        return

    print(f"\nDetected {len(models)} models:\n")
    print(f"{'Model':<35} {'Size':<8} {'Params':<8} {'Role'}")
    print("-"*60)

    for m in sorted(models, key=lambda x: -x.params_b):
        if m.params_b >= 14:
            role = "Advanced Planner"
        elif m.params_b >= 7:
            role = "Strict Planner"
        elif m.params_b >= 3:
            role = "Worker/Verifier"
        else:
            role = "Basic tasks"

        print(f"{m.name:<35} {m.size_gb:.1f} GB   {m.params_b:.1f}B     {role}")

    # Detect optimal config
    config = AgentConfig.detect_optimal()

    print("\n" + "-"*60)
    print(f"Recommended Mode: {config.mode.value.upper()}")
    print("-"*60)

    for role in AgentRole:
        model = config.get_model(role)
        print(f"  {role.value:<10}: {model}")

    print("="*60 + "\n")


# =============================================================================
# Module Entry Point
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print_model_summary()
