"""
Orchestrator and Action Protocol for RAGIX

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-11-24
"""

import json
import re
from typing import Optional, Dict, Any, Tuple


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract the first JSON object from a text response.

    The model is instructed to respond with pure JSON, but this function
    is defensive in case extra text appears.

    Args:
        text: Raw text response from LLM

    Returns:
        Parsed JSON dict or None if extraction fails
    """
    start = text.find("{")
    if start == -1:
        return None

    # Try to find matching closing brace by counting braces
    # This handles nested JSON in message content
    brace_count = 0
    in_string = False
    escape_next = False
    end = -1

    for i, char in enumerate(text[start:], start):
        if escape_next:
            escape_next = False
            continue

        if char == '\\' and in_string:
            escape_next = True
            continue

        if char == '"' and not escape_next:
            in_string = not in_string
            continue

        if not in_string:
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    end = i
                    break

    if end == -1:
        # Fallback to rfind if parsing fails
        end = text.rfind("}")
        if end == -1:
            return None

    candidate = text[start:end + 1]

    # Try standard JSON parsing first
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # Try to fix common LLM JSON errors
    try:
        import re
        fixed = candidate

        # Fix 1: Unquoted keys like `message:` -> `"message":`
        fixed = re.sub(r'([{,]\s*)(\w+)(\s*:)', r'\1"\2"\3', fixed)

        # Fix 2: Single quotes to double quotes (but not inside strings)
        # This is tricky - do a simple replacement if no double quotes present
        if '"' not in fixed.replace('\\"', ''):
            fixed = fixed.replace("'", '"')
        # If we have mixed quotes, try to fix single-quoted string values
        elif "'" in fixed:
            # Replace 'value' patterns with "value" but be careful
            fixed = re.sub(r":\s*'([^']*)'", r': "\1"', fixed)

        # Fix 3: Trailing commas (common LLM error)
        fixed = re.sub(r',\s*([}\]])', r'\1', fixed)

        return json.loads(fixed)
    except (json.JSONDecodeError, Exception):
        return None


def extract_json_with_diagnostics(text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Extract JSON object with detailed error diagnostics.

    Args:
        text: Raw text response from LLM

    Returns:
        Tuple of (parsed_json, error_message)
        - parsed_json: Dict if successful, None if failed
        - error_message: None if successful, detailed error if failed
    """
    # Try basic extraction first
    result = extract_json_object(text)
    if result is not None:
        return result, None

    # Diagnostics for failure cases
    if "{" not in text:
        return None, "No JSON object found (no opening brace)"

    if "}" not in text:
        return None, "No JSON object found (no closing brace)"

    # Try to extract and show JSON error
    start = text.find("{")
    end = text.rfind("}")
    candidate = text[start:end + 1]

    try:
        json.loads(candidate)
        return None, "Unexpected: JSON parsed but extraction failed"
    except json.JSONDecodeError as e:
        # Provide helpful error with context
        line_start = max(0, e.pos - 40)
        line_end = min(len(candidate), e.pos + 40)
        context = candidate[line_start:line_end]
        return None, f"JSON parse error at position {e.pos}: {e.msg}\nContext: ...{context}..."


def validate_action_schema(action: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    Validate that an action dict conforms to the RAGIX action protocol.

    Valid actions:
    - {"action": "bash", "command": "..."}
    - {"action": "respond", "message": "..."}
    - {"action": "bash_and_respond", "command": "...", "message": "..."}
    - {"action": "edit_file", "path": "...", "old": "...", "new": "..."}

    Args:
        action: Parsed JSON dict

    Returns:
        Tuple of (is_valid, error_message)
        - is_valid: True if valid, False otherwise
        - error_message: None if valid, description if invalid
    """
    if not isinstance(action, dict):
        return False, f"Action must be a dict, got {type(action).__name__}"

    action_type = action.get("action")
    if action_type is None:
        return False, "Missing required field: 'action'"

    if not isinstance(action_type, str):
        return False, f"Field 'action' must be string, got {type(action_type).__name__}"

    # Validate each action type
    if action_type == "bash":
        if "command" not in action:
            return False, "Action 'bash' requires 'command' field"
        if not isinstance(action["command"], str):
            return False, "Field 'command' must be string"
        return True, None

    elif action_type == "respond":
        if "message" not in action:
            return False, "Action 'respond' requires 'message' field"
        if not isinstance(action["message"], str):
            return False, "Field 'message' must be string"
        return True, None

    elif action_type == "bash_and_respond":
        if "command" not in action:
            return False, "Action 'bash_and_respond' requires 'command' field"
        if "message" not in action:
            return False, "Action 'bash_and_respond' requires 'message' field"
        if not isinstance(action["command"], str):
            return False, "Field 'command' must be string"
        if not isinstance(action["message"], str):
            return False, "Field 'message' must be string"
        return True, None

    elif action_type == "edit_file":
        required = ["path", "old", "new"]
        for field in required:
            if field not in action:
                return False, f"Action 'edit_file' requires '{field}' field"
            if not isinstance(action[field], str):
                return False, f"Field '{field}' must be string"
        return True, None

    elif action_type == "wasp_task":
        # WASP tool execution
        if "tool" not in action:
            return False, "Action 'wasp_task' requires 'tool' field"
        if not isinstance(action["tool"], str):
            return False, "Field 'tool' must be string"
        # Optional inputs dict
        if "inputs" in action and not isinstance(action["inputs"], dict):
            return False, "Field 'inputs' must be a dict"
        return True, None

    else:
        return False, f"Unknown action type: '{action_type}'"


def create_retry_prompt(raw_response: str, error_message: str) -> str:
    """
    Create a retry prompt to send back to the LLM when JSON parsing fails.

    Args:
        raw_response: The invalid response from LLM
        error_message: Diagnostic error message

    Returns:
        Formatted retry prompt explaining the error
    """
    return f"""Your previous response could not be parsed as valid JSON.

Error: {error_message}

Please respond with a valid JSON object following the action protocol:

Valid actions:
1. {{"action": "bash", "command": "..."}}
2. {{"action": "respond", "message": "..."}}
3. {{"action": "bash_and_respond", "command": "...", "message": "..."}}
4. {{"action": "edit_file", "path": "...", "old": "...", "new": "..."}}
5. {{"action": "wasp_task", "tool": "...", "inputs": {{}}}}

Requirements:
- Respond with ONLY the JSON object, no markdown fences, no extra text
- Use double quotes for strings
- Escape special characters properly
- Ensure all braces are balanced

Previous response (for reference):
{raw_response[:200]}{"..." if len(raw_response) > 200 else ""}
"""
