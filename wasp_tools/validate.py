"""
WASP Validation Tools - JSON/YAML validation and formatting

Deterministic validation tools for structured data.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-11-26
"""

import json
import re
from typing import Any, Dict, List, Optional, Union


def validate_json(
    content: str,
    schema: Optional[str] = None,
    strict: bool = False,
) -> Dict[str, Any]:
    """
    Validate JSON content and optionally check against a schema.

    Args:
        content: JSON string to validate
        schema: Optional JSON Schema string to validate against
        strict: If True, fail on trailing commas and comments

    Returns:
        dict with keys:
            - valid: bool
            - data: parsed data (if valid)
            - error: error message (if invalid)
            - error_line: line number of error (if available)
            - error_col: column of error (if available)
    """
    result = {"valid": False}

    # Pre-process for lenient mode
    if not strict:
        # Remove single-line comments (not standard JSON but common)
        content = re.sub(r'//.*$', '', content, flags=re.MULTILINE)
        # Remove trailing commas before ] or }
        content = re.sub(r',(\s*[}\]])', r'\1', content)

    try:
        data = json.loads(content)
        result["valid"] = True
        result["data"] = data
        result["type"] = type(data).__name__

        # Basic stats
        if isinstance(data, dict):
            result["key_count"] = len(data)
            result["keys"] = list(data.keys())[:20]  # First 20 keys
        elif isinstance(data, list):
            result["item_count"] = len(data)

    except json.JSONDecodeError as e:
        result["error"] = str(e)
        result["error_line"] = e.lineno
        result["error_col"] = e.colno
        result["error_pos"] = e.pos

        # Try to provide context
        lines = content.splitlines()
        if 0 < e.lineno <= len(lines):
            result["error_context"] = lines[e.lineno - 1]

    # Schema validation if provided
    if result["valid"] and schema:
        schema_result = _validate_schema(result["data"], schema)
        if not schema_result["valid"]:
            result["valid"] = False
            result["schema_errors"] = schema_result["errors"]

    return result


def validate_yaml(
    content: str,
    schema: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Validate YAML content and optionally check against a schema.

    Args:
        content: YAML string to validate
        schema: Optional JSON Schema string to validate against

    Returns:
        dict with keys:
            - valid: bool
            - data: parsed data (if valid)
            - error: error message (if invalid)
    """
    result = {"valid": False}

    try:
        # Use safe YAML loading
        import yaml
        data = yaml.safe_load(content)
        result["valid"] = True
        result["data"] = data
        result["type"] = type(data).__name__ if data is not None else "null"

        # Basic stats
        if isinstance(data, dict):
            result["key_count"] = len(data)
            result["keys"] = list(data.keys())[:20]
        elif isinstance(data, list):
            result["item_count"] = len(data)

    except yaml.YAMLError as e:
        result["error"] = str(e)
        if hasattr(e, 'problem_mark'):
            mark = e.problem_mark
            result["error_line"] = mark.line + 1
            result["error_col"] = mark.column + 1

    except ImportError:
        result["error"] = "PyYAML not installed. Install with: pip install pyyaml"

    # Schema validation if provided
    if result["valid"] and schema:
        schema_result = _validate_schema(result["data"], schema)
        if not schema_result["valid"]:
            result["valid"] = False
            result["schema_errors"] = schema_result["errors"]

    return result


def format_json(
    content: str,
    indent: int = 2,
    sort_keys: bool = False,
    compact: bool = False,
) -> Dict[str, Any]:
    """
    Format/prettify JSON content.

    Args:
        content: JSON string to format
        indent: Indentation level (default: 2)
        sort_keys: Sort object keys alphabetically
        compact: Output compact single-line JSON

    Returns:
        dict with keys:
            - success: bool
            - formatted: formatted JSON string
            - error: error message (if failed)
    """
    result = {"success": False}

    try:
        # Parse first
        data = json.loads(content)

        # Format
        if compact:
            formatted = json.dumps(data, sort_keys=sort_keys, separators=(',', ':'))
        else:
            formatted = json.dumps(
                data,
                indent=indent,
                sort_keys=sort_keys,
                ensure_ascii=False,
            )

        result["success"] = True
        result["formatted"] = formatted
        result["original_size"] = len(content)
        result["formatted_size"] = len(formatted)

    except json.JSONDecodeError as e:
        result["error"] = f"Invalid JSON: {e}"

    return result


def format_yaml(
    content: str,
    default_flow_style: bool = False,
    indent: int = 2,
) -> Dict[str, Any]:
    """
    Format/prettify YAML content.

    Args:
        content: YAML string to format
        default_flow_style: Use flow style for collections
        indent: Indentation level

    Returns:
        dict with keys:
            - success: bool
            - formatted: formatted YAML string
            - error: error message (if failed)
    """
    result = {"success": False}

    try:
        import yaml

        # Parse
        data = yaml.safe_load(content)

        # Format
        formatted = yaml.dump(
            data,
            default_flow_style=default_flow_style,
            indent=indent,
            allow_unicode=True,
            sort_keys=False,
        )

        result["success"] = True
        result["formatted"] = formatted
        result["original_size"] = len(content)
        result["formatted_size"] = len(formatted)

    except yaml.YAMLError as e:
        result["error"] = f"Invalid YAML: {e}"

    except ImportError:
        result["error"] = "PyYAML not installed"

    return result


def json_to_yaml(
    content: str,
    indent: int = 2,
) -> Dict[str, Any]:
    """
    Convert JSON to YAML.

    Args:
        content: JSON string to convert
        indent: YAML indentation level

    Returns:
        dict with keys:
            - success: bool
            - yaml: converted YAML string
            - error: error message (if failed)
    """
    result = {"success": False}

    try:
        import yaml

        # Parse JSON
        data = json.loads(content)

        # Convert to YAML
        yaml_str = yaml.dump(
            data,
            default_flow_style=False,
            indent=indent,
            allow_unicode=True,
            sort_keys=False,
        )

        result["success"] = True
        result["yaml"] = yaml_str

    except json.JSONDecodeError as e:
        result["error"] = f"Invalid JSON: {e}"

    except ImportError:
        result["error"] = "PyYAML not installed"

    return result


def yaml_to_json(
    content: str,
    indent: int = 2,
    compact: bool = False,
) -> Dict[str, Any]:
    """
    Convert YAML to JSON.

    Args:
        content: YAML string to convert
        indent: JSON indentation level
        compact: Output compact JSON

    Returns:
        dict with keys:
            - success: bool
            - json: converted JSON string
            - error: error message (if failed)
    """
    result = {"success": False}

    try:
        import yaml

        # Parse YAML
        data = yaml.safe_load(content)

        # Convert to JSON
        if compact:
            json_str = json.dumps(data, separators=(',', ':'), ensure_ascii=False)
        else:
            json_str = json.dumps(data, indent=indent, ensure_ascii=False)

        result["success"] = True
        result["json"] = json_str

    except yaml.YAMLError as e:
        result["error"] = f"Invalid YAML: {e}"

    except ImportError:
        result["error"] = "PyYAML not installed"

    return result


def _validate_schema(data: Any, schema_str: str) -> Dict[str, Any]:
    """
    Validate data against JSON Schema.

    Simple schema validation without external dependencies.
    Supports: type, required, properties, items, enum, minimum, maximum,
              minLength, maxLength, pattern.
    """
    result = {"valid": True, "errors": []}

    try:
        schema = json.loads(schema_str)
    except json.JSONDecodeError as e:
        result["valid"] = False
        result["errors"].append(f"Invalid schema JSON: {e}")
        return result

    errors = _check_schema(data, schema, "")
    if errors:
        result["valid"] = False
        result["errors"] = errors

    return result


def _check_schema(
    data: Any,
    schema: Dict,
    path: str,
) -> List[str]:
    """Recursively check data against schema."""
    errors = []

    # Type check
    if "type" in schema:
        expected_type = schema["type"]
        actual_type = _get_json_type(data)

        if isinstance(expected_type, list):
            if actual_type not in expected_type:
                errors.append(f"{path or 'root'}: expected {expected_type}, got {actual_type}")
        elif actual_type != expected_type:
            errors.append(f"{path or 'root'}: expected {expected_type}, got {actual_type}")

    # Enum check
    if "enum" in schema:
        if data not in schema["enum"]:
            errors.append(f"{path or 'root'}: value not in enum {schema['enum']}")

    # Object checks
    if isinstance(data, dict):
        # Required fields
        if "required" in schema:
            for field in schema["required"]:
                if field not in data:
                    errors.append(f"{path or 'root'}: missing required field '{field}'")

        # Property validation
        if "properties" in schema:
            for key, value in data.items():
                if key in schema["properties"]:
                    prop_path = f"{path}.{key}" if path else key
                    errors.extend(_check_schema(value, schema["properties"][key], prop_path))

    # Array checks
    if isinstance(data, list):
        # Item validation
        if "items" in schema:
            for i, item in enumerate(data):
                item_path = f"{path}[{i}]"
                errors.extend(_check_schema(item, schema["items"], item_path))

        # Length checks
        if "minItems" in schema and len(data) < schema["minItems"]:
            errors.append(f"{path or 'root'}: array length {len(data)} < minItems {schema['minItems']}")
        if "maxItems" in schema and len(data) > schema["maxItems"]:
            errors.append(f"{path or 'root'}: array length {len(data)} > maxItems {schema['maxItems']}")

    # String checks
    if isinstance(data, str):
        if "minLength" in schema and len(data) < schema["minLength"]:
            errors.append(f"{path or 'root'}: string length {len(data)} < minLength {schema['minLength']}")
        if "maxLength" in schema and len(data) > schema["maxLength"]:
            errors.append(f"{path or 'root'}: string length {len(data)} > maxLength {schema['maxLength']}")
        if "pattern" in schema:
            if not re.search(schema["pattern"], data):
                errors.append(f"{path or 'root'}: string does not match pattern '{schema['pattern']}'")

    # Number checks
    if isinstance(data, (int, float)) and not isinstance(data, bool):
        if "minimum" in schema and data < schema["minimum"]:
            errors.append(f"{path or 'root'}: value {data} < minimum {schema['minimum']}")
        if "maximum" in schema and data > schema["maximum"]:
            errors.append(f"{path or 'root'}: value {data} > maximum {schema['maximum']}")

    return errors


def _get_json_type(value: Any) -> str:
    """Get JSON Schema type name for a Python value."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "unknown"
