"""
JSON Validator Plugin - JSON/YAML validation and formatting tools

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-11-26
"""

import json
from typing import Any, Dict, List, Optional, Union


def validate_json(content: str, format: bool = False) -> Dict[str, Any]:
    """
    Validate JSON content.

    Args:
        content: JSON string to validate
        format: If True, return formatted JSON

    Returns:
        Validation result with parsed data
    """
    try:
        parsed = json.loads(content)

        result = {
            "valid": True,
            "type": type(parsed).__name__,
        }

        if format:
            result["formatted"] = json.dumps(parsed, indent=2, ensure_ascii=False)

        if isinstance(parsed, dict):
            result["keys"] = list(parsed.keys())
            result["key_count"] = len(parsed)
        elif isinstance(parsed, list):
            result["length"] = len(parsed)

        return result

    except json.JSONDecodeError as e:
        return {
            "valid": False,
            "error": str(e),
            "line": e.lineno,
            "column": e.colno,
            "position": e.pos,
        }


def validate_yaml(content: str) -> Dict[str, Any]:
    """
    Validate YAML content and convert to JSON.

    Args:
        content: YAML string to validate

    Returns:
        Validation result with JSON output
    """
    try:
        import yaml
    except ImportError:
        return {
            "valid": False,
            "error": "PyYAML not installed",
        }

    try:
        parsed = yaml.safe_load(content)

        result = {
            "valid": True,
            "type": type(parsed).__name__,
            "json": json.dumps(parsed, indent=2, ensure_ascii=False),
        }

        if isinstance(parsed, dict):
            result["keys"] = list(parsed.keys())
        elif isinstance(parsed, list):
            result["length"] = len(parsed)

        return result

    except yaml.YAMLError as e:
        return {
            "valid": False,
            "error": str(e),
        }


def json_diff(json1: str, json2: str) -> Dict[str, Any]:
    """
    Compare two JSON objects and show differences.

    Args:
        json1: First JSON string
        json2: Second JSON string

    Returns:
        Difference report
    """
    try:
        obj1 = json.loads(json1)
        obj2 = json.loads(json2)
    except json.JSONDecodeError as e:
        return {
            "success": False,
            "error": f"JSON parse error: {e}",
        }

    differences = _compare_objects(obj1, obj2, "")

    return {
        "success": True,
        "identical": len(differences) == 0,
        "differences": differences,
        "diff_count": len(differences),
    }


def _compare_objects(
    obj1: Any,
    obj2: Any,
    path: str
) -> List[Dict[str, Any]]:
    """
    Recursively compare two objects.

    Args:
        obj1: First object
        obj2: Second object
        path: Current path

    Returns:
        List of differences
    """
    differences = []

    # Type mismatch
    if type(obj1) != type(obj2):
        differences.append({
            "path": path or "/",
            "type": "type_change",
            "from_type": type(obj1).__name__,
            "to_type": type(obj2).__name__,
            "from_value": _safe_repr(obj1),
            "to_value": _safe_repr(obj2),
        })
        return differences

    # Dictionary comparison
    if isinstance(obj1, dict):
        all_keys = set(obj1.keys()) | set(obj2.keys())

        for key in all_keys:
            key_path = f"{path}/{key}" if path else f"/{key}"

            if key not in obj1:
                differences.append({
                    "path": key_path,
                    "type": "added",
                    "value": _safe_repr(obj2[key]),
                })
            elif key not in obj2:
                differences.append({
                    "path": key_path,
                    "type": "removed",
                    "value": _safe_repr(obj1[key]),
                })
            else:
                differences.extend(_compare_objects(obj1[key], obj2[key], key_path))

    # List comparison
    elif isinstance(obj1, list):
        max_len = max(len(obj1), len(obj2))

        for i in range(max_len):
            idx_path = f"{path}[{i}]"

            if i >= len(obj1):
                differences.append({
                    "path": idx_path,
                    "type": "added",
                    "value": _safe_repr(obj2[i]),
                })
            elif i >= len(obj2):
                differences.append({
                    "path": idx_path,
                    "type": "removed",
                    "value": _safe_repr(obj1[i]),
                })
            else:
                differences.extend(_compare_objects(obj1[i], obj2[i], idx_path))

    # Scalar comparison
    else:
        if obj1 != obj2:
            differences.append({
                "path": path or "/",
                "type": "changed",
                "from_value": _safe_repr(obj1),
                "to_value": _safe_repr(obj2),
            })

    return differences


def _safe_repr(obj: Any, max_len: int = 50) -> str:
    """Safe string representation of an object."""
    s = repr(obj)
    if len(s) > max_len:
        return s[:max_len - 3] + "..."
    return s
