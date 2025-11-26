# WASP Guide - WebAssembly-ready Agentic System Protocol

**Author:** Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-11-26

---

## Overview

WASP (WebAssembly-ready Agentic System Protocol) provides deterministic, sandboxed tools for RAGIX agents. These tools can run:

1. **Server-side** (Python) - Full-featured, production-ready
2. **Browser-side** (JavaScript) - Client-side execution for offline capability

## Quick Start

### Using WASP Tools from CLI

```bash
# List available tools
ragix-wasp list

# Get tool information
ragix-wasp info validate_json

# Run a tool
ragix-wasp run validate_json '{"name": "test"}'

# Run with file input
ragix-wasp run validate_json -f data.json

# Pipe content
cat document.md | ragix-wasp run extract_headers
```

### Using WASP Tools from Python

```python
from wasp_tools import validate_json, extract_headers, search_pattern

# Validate JSON
result = validate_json('{"name": "test", "value": 42}')
print(f"Valid: {result['valid']}")

# Extract headers from Markdown
md = "# Title\n## Section 1\n## Section 2"
result = extract_headers(md)
print(f"Found {result['count']} headers")

# Search for patterns
result = search_pattern(r"\d+", "Line 1, Line 2, Line 3")
print(f"Matches: {result['count']}")
```

### Using WASP in Agent Actions

```python
from ragix_core import execute_wasp_action

# Execute a WASP tool via the agent protocol
response, result = execute_wasp_action({
    "action": "wasp_task",
    "tool": "validate_json",
    "inputs": {"content": '{"test": 123}'}
})

print(f"Success: {result.success}")
print(f"Duration: {result.duration_ms}ms")
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     RAGIX Agent                              │
│  {"action": "wasp_task", "tool": "...", "inputs": {...}}     │
└─────────────────────────┬───────────────────────────────────┘
                          │
            ┌─────────────▼─────────────┐
            │      WaspExecutor         │
            │  - Tool registry          │
            │  - Input validation       │
            │  - Execution timing       │
            └─────────────┬─────────────┘
                          │
       ┌──────────────────┼──────────────────┐
       │                  │                  │
       ▼                  ▼                  ▼
  ┌─────────┐       ┌─────────┐       ┌─────────┐
  │Validation│       │Markdown │       │ Search  │
  │  Tools   │       │  Tools  │       │  Tools  │
  └─────────┘       └─────────┘       └─────────┘
```

## Tool Categories

### Validation Tools

| Tool | Description |
|------|-------------|
| `validate_json` | Validate JSON content with optional schema |
| `validate_yaml` | Validate YAML content with optional schema |
| `format_json` | Format/prettify JSON |
| `format_yaml` | Format/prettify YAML |
| `json_to_yaml` | Convert JSON to YAML |
| `yaml_to_json` | Convert YAML to JSON |

### Markdown Tools

| Tool | Description |
|------|-------------|
| `parse_markdown` | Parse Markdown to structured AST |
| `extract_headers` | Extract headers from Markdown |
| `extract_code_blocks` | Extract code blocks from Markdown |
| `extract_links` | Extract links from Markdown |
| `extract_frontmatter` | Extract YAML frontmatter |
| `renumber_sections` | Renumber section headers |
| `generate_toc` | Generate table of contents |

### Search Tools

| Tool | Description |
|------|-------------|
| `search_pattern` | Search for regex pattern |
| `search_lines` | Search with line context |
| `count_matches` | Count pattern matches |
| `extract_matches` | Extract all matches with groups |
| `replace_pattern` | Replace pattern matches |

## Tool Reference

### validate_json

Validate JSON content and optionally check against a schema.

**Parameters:**
- `content` (string, required): JSON string to validate
- `schema` (string, optional): JSON Schema to validate against
- `strict` (boolean, optional): Fail on trailing commas/comments

**Returns:**
```json
{
  "valid": true,
  "type": "dict",
  "key_count": 2,
  "keys": ["name", "value"]
}
```

### extract_headers

Extract all headers from Markdown content.

**Parameters:**
- `content` (string, required): Markdown string

**Returns:**
```json
{
  "success": true,
  "headers": [
    {"level": 1, "text": "Title", "line": 1},
    {"level": 2, "text": "Section", "line": 3}
  ],
  "count": 2
}
```

### search_pattern

Search for regex pattern in content.

**Parameters:**
- `pattern` (string, required): Regular expression
- `content` (string, required): Text to search
- `flags` (string, optional): Regex flags (i=ignorecase, m=multiline)
- `max_matches` (integer, optional): Maximum matches (default: 100)

**Returns:**
```json
{
  "success": true,
  "matches": [
    {
      "text": "matched",
      "start": 10,
      "end": 17,
      "line": 1,
      "column": 11
    }
  ],
  "count": 1
}
```

## JSON Protocol Integration

The `wasp_task` action integrates WASP tools into the RAGIX agent protocol:

```json
{
  "action": "wasp_task",
  "tool": "validate_json",
  "inputs": {
    "content": "{\"name\": \"test\"}"
  }
}
```

Valid alongside other RAGIX actions:
- `bash` - Shell command execution
- `respond` - Send message to user
- `bash_and_respond` - Execute and respond
- `edit_file` - File editing
- **`wasp_task`** - WASP tool execution

## Browser Runtime

For client-side execution, include the JavaScript runtime:

```html
<script src="/static/js/virtual_fs.js"></script>
<script src="/static/js/wasp_runtime.js"></script>
<script src="/static/js/browser_tools.js"></script>

<script>
  const runtime = new WaspRuntime();

  // Execute a tool
  const result = await runtime.execute('validate_json', {
    content: '{"test": 123}'
  });

  console.log(result.success);  // true
</script>
```

### Browser UI Integration

```html
<div id="browser-tools"></div>

<script>
  const ui = new BrowserToolsUI({
    containerId: 'browser-tools',
    runtime: new WaspRuntime(),
    executionMode: 'browser'  // or 'server'
  });
  ui.init();
</script>
```

## Creating Custom Tools

### Python Tool

```python
from wasp_tools import WASP_TOOLS

def my_custom_tool(text: str, uppercase: bool = False) -> dict:
    """My custom tool."""
    result = text.upper() if uppercase else text.lower()
    return {
        "success": True,
        "result": result,
        "original_length": len(text)
    }

# Register the tool
WASP_TOOLS["my_tool"] = {
    "func": my_custom_tool,
    "description": "Transform text",
    "category": "custom"
}
```

### Via Executor

```python
from ragix_core import get_wasp_executor

executor = get_wasp_executor()
executor.register_tool(
    "custom_tool",
    lambda text: {"result": text.upper()},
    description="Convert to uppercase",
    category="custom"
)
```

## Best Practices

### 1. Use Appropriate Tools

- **Validation tools** for data integrity checks
- **Markdown tools** for documentation processing
- **Search tools** for pattern matching and text analysis

### 2. Handle Errors

```python
result = validate_json(content)
if not result["valid"]:
    print(f"Error: {result.get('error', 'Unknown error')}")
    if "error_line" in result:
        print(f"At line {result['error_line']}")
```

### 3. Use Schema Validation

```python
schema = '''{
  "type": "object",
  "required": ["name", "version"],
  "properties": {
    "name": {"type": "string"},
    "version": {"type": "string", "pattern": "^\\\\d+\\\\.\\\\d+\\\\.\\\\d+$"}
  }
}'''

result = validate_json(config_json, schema=schema)
```

### 4. Combine Tools

```python
# Extract code blocks and validate JSON in each
md_result = extract_code_blocks(markdown)
for block in md_result["blocks"]:
    if block["language"] == "json":
        json_result = validate_json(block["content"])
        print(f"Block at line {block['line']}: valid={json_result['valid']}")
```

## Performance

WASP tools are designed for:

- **Low latency**: Typical execution < 10ms
- **Deterministic**: Same input always produces same output
- **Memory efficient**: Stream processing where possible
- **Parallelizable**: Multiple tools can run concurrently

## Security

- Tools operate on content passed to them (no file access)
- No shell command execution within tools
- Input size limits to prevent DoS
- Schema validation for structured data

## Troubleshooting

### Tool Not Found

```python
from wasp_tools import list_tools
print(list_tools())  # See available tools
```

### Invalid Regex

```python
result = search_pattern(r"[invalid", text)
if not result["success"]:
    print(f"Regex error: {result['error']}")
```

### Large Output Truncation

The executor truncates outputs > 50KB. Check for `_truncated` flag:

```python
if result.result.get("_truncated"):
    print(f"Output truncated from {result.result['_original_size']} bytes")
```
