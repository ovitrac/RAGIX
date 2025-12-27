# =============================================================================
# Rules Engine - YAML-based declarative rules with typed operators
# =============================================================================
#
# Rules are defined in YAML with typed operators (eq, neq, contains, matches...)
# for deterministic execution. No arbitrary Python eval.
#
# Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
# Version: 0.1.0 (2025-12-21)
#
# =============================================================================

"""
Rule Engine for the Interpreter-Tutor system.

Rules are YAML-based with typed operators for safe, deterministic execution.
Supports both permanent rules (human-curated) and session rules (LLM-generated).

Typed Operators:
- eq: Exact equality
- neq: Not equal
- in: Value in list
- contains: Substring match
- matches: Regex match
- gt, lt, gte, lte: Numeric comparison
- exists: Field is present
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional, Union
from pathlib import Path
from datetime import datetime
import re
import yaml
import json


# =============================================================================
# Typed Operators
# =============================================================================

class Operator:
    """Base class for typed operators."""

    @staticmethod
    def evaluate(op: str, value: Any, expected: Any) -> bool:
        """Evaluate a typed operator."""
        if op == "eq":
            return value == expected
        elif op == "neq":
            return value != expected
        elif op == "in":
            return value in expected
        elif op == "contains":
            return expected in str(value) if value else False
        elif op == "matches":
            return bool(re.search(expected, str(value))) if value else False
        elif op == "gt":
            return float(value) > float(expected) if value is not None else False
        elif op == "lt":
            return float(value) < float(expected) if value is not None else False
        elif op == "gte":
            return float(value) >= float(expected) if value is not None else False
        elif op == "lte":
            return float(value) <= float(expected) if value is not None else False
        elif op == "exists":
            return (value is not None) == expected
        elif op == "startswith":
            return str(value).startswith(expected) if value else False
        elif op == "endswith":
            return str(value).endswith(expected) if value else False
        else:
            raise ValueError(f"Unknown operator: {op}")


# =============================================================================
# Rule Data Structures
# =============================================================================

@dataclass
class Match:
    """A match condition in a rule."""
    field: str                    # e.g., "obs.tool", "obs.rc", "obs.stdout"
    operator: str                 # eq, neq, contains, matches, etc.
    value: Any                    # Expected value

    @classmethod
    def from_dict(cls, field: str, condition: Union[dict, Any]) -> Match:
        """Parse a match from YAML format."""
        if isinstance(condition, dict):
            # e.g., {eq: "pytest"} or {contains: ".strip()"}
            if len(condition) != 1:
                raise ValueError(f"Match condition must have exactly one operator: {condition}")
            op, val = next(iter(condition.items()))
            return cls(field=field, operator=op, value=val)
        else:
            # Simple value = implicit eq
            return cls(field=field, operator="eq", value=condition)


@dataclass
class Extract:
    """Variable extraction from observation."""
    var: str                      # Variable name to bind
    regex: str                    # Regex with capture group
    source: str = "obs.stdout"   # Field to extract from

    @classmethod
    def from_dict(cls, var: str, spec: dict) -> Extract:
        """Parse an extraction from YAML format."""
        return cls(
            var=var,
            regex=spec.get("regex", ""),
            source=spec.get("from", "obs.stdout"),
        )


@dataclass
class Conclusion:
    """Conclusion of a rule."""
    type: str                     # "truth", "entity", "question"
    text: str                     # Text with {var} placeholders
    kind: str = ""                # Claim kind or entity kind
    domain: str = ""              # Domain for truths
    scope: str = ""               # Scope for truths

    @classmethod
    def from_dict(cls, type_: str, spec: dict) -> Conclusion:
        """Parse a conclusion from YAML format."""
        return cls(
            type=type_,
            text=spec.get("text", spec.get("value", "")),
            kind=spec.get("kind", ""),
            domain=spec.get("domain", ""),
            scope=spec.get("scope", ""),
        )


@dataclass
class Rule:
    """
    A deterministic inference rule.

    Rules are applied when all match conditions are satisfied.
    Variables can be extracted via regex and used in conclusions.
    """
    id: str
    soundness: str                # "sound" or "heuristic"
    matches: list[Match]
    extracts: list[Extract]
    conclusions: list[Conclusion]
    description: str = ""
    domain: str = ""              # Domain this rule applies to
    lifetime: str = "permanent"   # "permanent" or "session"
    game_id: str = ""             # If session, which game
    turn: int = 0                 # If session, when generated
    generated_by: str = ""        # LLM model ID (for audit)
    rationale: str = ""           # Why this rule was needed

    def _get_field(self, obs: dict, field_path: str) -> Any:
        """Get a field from observation using dot notation."""
        # Handle obs.X notation
        if field_path.startswith("obs."):
            field_path = field_path[4:]

        parts = field_path.split(".")
        value = obs
        for part in parts:
            if isinstance(value, dict):
                value = value.get(part)
            else:
                return None
        return value

    def _substitute(self, template: str, bindings: dict, obs: dict) -> str:
        """Substitute variables in template."""
        result = template

        # Substitute extracted variables
        for var, val in bindings.items():
            result = result.replace(f"{{{var}}}", str(val))

        # Substitute obs.X references
        obs_pattern = r"\{obs\.(\w+)\}"
        for match in re.finditer(obs_pattern, result):
            field = match.group(1)
            value = obs.get(field, "")
            result = result.replace(match.group(0), str(value))

        return result

    def apply(self, obs: dict) -> Optional[list[dict]]:
        """
        Apply rule to an observation.

        Returns list of conclusions if rule matches, None otherwise.
        All execution is deterministic (no LLM involvement).
        """
        bindings = {}

        # Check all match conditions
        for m in self.matches:
            value = self._get_field(obs, m.field)
            if not Operator.evaluate(m.operator, value, m.value):
                return None

        # Extract variables
        for e in self.extracts:
            source_value = self._get_field(obs, e.source)
            if source_value and e.regex:
                match = re.search(e.regex, str(source_value))
                if match:
                    bindings[e.var] = match.group(1) if match.groups() else match.group(0)

        # Generate conclusions with substitution
        results = []
        for c in self.conclusions:
            results.append({
                "type": c.type,
                "text": self._substitute(c.text, bindings, obs),
                "kind": self._substitute(c.kind, bindings, obs) if c.kind else "",
                "domain": self._substitute(c.domain, bindings, obs) if c.domain else self.domain,
                "scope": self._substitute(c.scope, bindings, obs) if c.scope else "",
            })

        return results

    @classmethod
    def from_dict(cls, data: dict) -> Rule:
        """Parse a rule from YAML dictionary."""
        # Parse matches
        matches = []
        for m in data.get("match", []):
            if isinstance(m, dict):
                for field, condition in m.items():
                    matches.append(Match.from_dict(field, condition))

        # Parse extracts
        extracts = []
        for var, spec in data.get("extract", {}).items():
            extracts.append(Extract.from_dict(var, spec))

        # Parse conclusions
        conclusions = []
        conclude = data.get("conclude", {})
        for type_, spec in conclude.items():
            if isinstance(spec, dict):
                conclusions.append(Conclusion.from_dict(type_, spec))

        return cls(
            id=data["id"],
            soundness=data.get("soundness", "sound"),
            matches=matches,
            extracts=extracts,
            conclusions=conclusions,
            description=data.get("description", ""),
            domain=data.get("domain", ""),
            lifetime=data.get("lifetime", "permanent"),
            game_id=data.get("game_id", ""),
            turn=data.get("turn", 0),
            generated_by=data.get("generated_by", ""),
            rationale=data.get("rationale", ""),
        )

    def to_dict(self) -> dict:
        """Convert rule to YAML-serializable dict."""
        d = {
            "id": self.id,
            "soundness": self.soundness,
            "description": self.description,
            "match": [
                {m.field: {m.operator: m.value}}
                for m in self.matches
            ],
        }
        if self.extracts:
            d["extract"] = {
                e.var: {"regex": e.regex, "from": e.source}
                for e in self.extracts
            }
        d["conclude"] = {}
        for c in self.conclusions:
            d["conclude"][c.type] = {
                "text": c.text,
                "kind": c.kind,
            }
            if c.domain:
                d["conclude"][c.type]["domain"] = c.domain
            if c.scope:
                d["conclude"][c.type]["scope"] = c.scope

        if self.lifetime != "permanent":
            d["lifetime"] = self.lifetime
            d["game_id"] = self.game_id
            d["turn"] = self.turn
            d["generated_by"] = self.generated_by
            d["rationale"] = self.rationale

        return d


# =============================================================================
# Rule Library
# =============================================================================

class RuleLibrary:
    """
    Library of rules, supporting both permanent and session rules.

    Permanent rules are loaded from YAML files.
    Session rules are added dynamically during a game.
    """

    def __init__(self):
        self.permanent_rules: dict[str, Rule] = {}
        self.session_rules: dict[str, dict[str, Rule]] = {}  # game_id -> {rule_id -> Rule}

    def load_file(self, path: Path) -> int:
        """Load rules from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)

        count = 0
        for rule_data in data.get("rules", []):
            rule = Rule.from_dict(rule_data)
            # Apply domain from meta if not specified
            if not rule.domain and "meta" in data:
                rule.domain = data["meta"].get("domain", "")
            self.permanent_rules[rule.id] = rule
            count += 1

        return count

    def load_directory(self, path: Path) -> int:
        """Load all .rules.yaml files from a directory."""
        count = 0
        for file in Path(path).glob("*.rules.yaml"):
            count += self.load_file(file)
        return count

    def add_session_rule(self, game_id: str, rule: Rule) -> None:
        """Add a session rule for a specific game."""
        rule.lifetime = "session"
        rule.game_id = game_id
        if game_id not in self.session_rules:
            self.session_rules[game_id] = {}
        self.session_rules[game_id][rule.id] = rule

    def clear_session(self, game_id: str) -> int:
        """Clear session rules for a game."""
        if game_id in self.session_rules:
            count = len(self.session_rules[game_id])
            del self.session_rules[game_id]
            return count
        return 0

    def get_rules(self, game_id: str = None, domain: str = None) -> list[Rule]:
        """Get applicable rules, optionally filtered by domain."""
        rules = list(self.permanent_rules.values())

        # Add session rules if game_id provided
        if game_id and game_id in self.session_rules:
            rules.extend(self.session_rules[game_id].values())

        # Filter by domain if specified
        if domain:
            rules = [r for r in rules if not r.domain or r.domain == domain]

        return rules

    def apply_all(self, obs: dict, game_id: str = None, domain: str = None) -> list[tuple[Rule, list[dict]]]:
        """
        Apply all matching rules to an observation.

        Returns list of (rule, conclusions) tuples for rules that matched.
        """
        results = []
        for rule in self.get_rules(game_id=game_id, domain=domain):
            conclusions = rule.apply(obs)
            if conclusions:
                results.append((rule, conclusions))
        return results

    def get_rule(self, rule_id: str, game_id: str = None) -> Optional[Rule]:
        """Get a specific rule by ID."""
        if rule_id in self.permanent_rules:
            return self.permanent_rules[rule_id]
        if game_id and game_id in self.session_rules:
            return self.session_rules[game_id].get(rule_id)
        return None

    def summary(self) -> dict:
        """Get library summary."""
        return {
            "permanent_rules": len(self.permanent_rules),
            "session_games": len(self.session_rules),
            "session_rules": sum(len(rules) for rules in self.session_rules.values()),
            "domains": list(set(r.domain for r in self.permanent_rules.values() if r.domain)),
        }


def load_rules(path: Union[str, Path]) -> RuleLibrary:
    """
    Load rules from a file or directory.

    If path is a directory, loads all .rules.yaml files.
    If path is a file, loads just that file.
    """
    library = RuleLibrary()
    path = Path(path)

    if path.is_dir():
        library.load_directory(path)
    elif path.is_file():
        library.load_file(path)
    else:
        raise FileNotFoundError(f"Rules path not found: {path}")

    return library


# =============================================================================
# Rule Generation (for fat LLM)
# =============================================================================

def generate_rule_prompt(claim_text: str, obs: dict, context: str = "") -> str:
    """
    Generate a prompt for a fat LLM to create a rule.

    The LLM should return YAML that matches the rule schema.
    """
    prompt = f"""You are a rule engineer. Generate a YAML rule to derive this claim from the evidence.

CLAIM to derive:
  "{claim_text}"

OBSERVATION (evidence):
  tool: {obs.get('tool', '')}
  rc: {obs.get('rc', '')}
  stdout (excerpt): {obs.get('stdout', '')[:500]}
  scope: {obs.get('scope', '')}

{f"CONTEXT: {context}" if context else ""}

Generate a YAML rule in this exact format:

```yaml
- id: R_session_<unique_suffix>
  soundness: sound
  description: "<brief description>"
  match:
    - obs.tool: {{eq: "<tool>"}}
    - obs.stdout: {{contains: "<pattern>"}}
  extract:
    var_name:
      regex: "<capture_group_regex>"
      from: obs.stdout
  conclude:
    truth:
      text: "<claim text with {{var_name}} placeholders>"
      kind: property
      domain: bash
      scope: "{{obs.scope}}"
```

Rules:
1. Use ONLY these operators: eq, neq, in, contains, matches, gt, lt, gte, lte, exists, startswith, endswith
2. Extract variables with regex capture groups: (\\w+) or similar
3. Reference extracted vars as {{var_name}} in conclusions
4. Reference observation fields as {{obs.field}}
5. soundness must be "sound" (provable) or "heuristic" (best-effort)
6. Output ONLY the YAML block, nothing else
"""
    return prompt


def parse_rule_from_llm(yaml_text: str, game_id: str, turn: int, model: str) -> Rule:
    """
    Parse a rule from LLM-generated YAML.

    Adds session metadata for audit trail.
    """
    # Extract YAML from markdown code block if present
    if "```yaml" in yaml_text:
        start = yaml_text.find("```yaml") + 7
        end = yaml_text.find("```", start)
        yaml_text = yaml_text[start:end].strip()
    elif "```" in yaml_text:
        start = yaml_text.find("```") + 3
        end = yaml_text.find("```", start)
        yaml_text = yaml_text[start:end].strip()

    # Parse YAML
    data = yaml.safe_load(yaml_text)

    # Handle list format
    if isinstance(data, list) and len(data) > 0:
        data = data[0]

    # Add session metadata
    data["lifetime"] = "session"
    data["game_id"] = game_id
    data["turn"] = turn
    data["generated_by"] = model

    return Rule.from_dict(data)
