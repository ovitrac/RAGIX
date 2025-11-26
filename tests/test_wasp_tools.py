"""
Tests for WASP Tools

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-11-26
"""

import pytest
from wasp_tools import (
    # Validation
    validate_json,
    validate_yaml,
    format_json,
    format_yaml,
    json_to_yaml,
    yaml_to_json,
    # Markdown
    parse_markdown,
    extract_headers,
    extract_code_blocks,
    extract_links,
    extract_frontmatter,
    renumber_sections,
    generate_toc,
    # Search
    search_pattern,
    search_lines,
    count_matches,
    extract_matches,
    replace_pattern,
    # Registry
    WASP_TOOLS,
    get_tool,
    list_tools,
    get_tool_info,
)


class TestValidationTools:
    """Tests for JSON/YAML validation tools."""

    def test_validate_json_valid(self):
        result = validate_json('{"name": "test", "value": 42}')
        assert result["valid"] is True
        assert result["type"] == "dict"
        assert result["key_count"] == 2
        assert "name" in result["keys"]

    def test_validate_json_invalid(self):
        result = validate_json('{"name": "test"')
        assert result["valid"] is False
        assert "error" in result

    def test_validate_json_array(self):
        result = validate_json('[1, 2, 3]')
        assert result["valid"] is True
        assert result["type"] == "list"
        assert result["item_count"] == 3

    def test_validate_json_with_comments(self):
        # Comments removed in lenient mode
        result = validate_json('{"name": "test"} // comment')
        assert result["valid"] is True

    def test_validate_json_strict_mode(self):
        # Strict mode fails on comments
        result = validate_json('{"name": "test",}', strict=True)
        assert result["valid"] is False

    def test_validate_json_with_schema(self):
        schema = '{"type": "object", "required": ["name"]}'
        result = validate_json('{"name": "test"}', schema=schema)
        assert result["valid"] is True

        result = validate_json('{"value": 42}', schema=schema)
        assert result["valid"] is False
        assert "schema_errors" in result

    def test_validate_yaml_valid(self):
        result = validate_yaml("name: test\nvalue: 42")
        assert result["valid"] is True
        assert result["type"] == "dict"

    def test_validate_yaml_invalid(self):
        # Use clearly invalid YAML syntax
        result = validate_yaml("name: test\n  - item\n bad: [unclosed")
        assert result["valid"] is False

    def test_format_json(self):
        result = format_json('{"b":1,"a":2}')
        assert result["success"] is True
        assert "formatted" in result

    def test_format_json_sorted(self):
        result = format_json('{"b":1,"a":2}', sort_keys=True)
        assert result["success"] is True
        # a should come before b
        assert result["formatted"].index('"a"') < result["formatted"].index('"b"')

    def test_format_json_compact(self):
        result = format_json('{"a": 1, "b": 2}', compact=True)
        assert result["success"] is True
        assert "\n" not in result["formatted"]

    def test_json_to_yaml(self):
        result = json_to_yaml('{"name": "test", "value": 42}')
        assert result["success"] is True
        assert "name: test" in result["yaml"]

    def test_yaml_to_json(self):
        result = yaml_to_json("name: test\nvalue: 42")
        assert result["success"] is True
        assert '"name"' in result["json"]


class TestMarkdownTools:
    """Tests for Markdown parsing tools."""

    SAMPLE_MD = """---
title: Test Document
author: Test
---

# Main Title

Some introductory text.

## Section 1

Content for section 1.

```python
def hello():
    print("Hello, World!")
```

## Section 2

More content with a [link](https://example.com "Example").

### Subsection 2.1

Nested content.

- Item 1
- Item 2
- Item 3
"""

    def test_parse_markdown(self):
        result = parse_markdown(self.SAMPLE_MD)
        assert result["success"] is True
        assert "ast" in result
        assert "stats" in result
        assert result["stats"]["heading_count"] == 4
        assert result["stats"]["has_frontmatter"] is True

    def test_extract_headers(self):
        result = extract_headers(self.SAMPLE_MD)
        assert result["success"] is True
        assert result["count"] == 4
        assert result["headers"][0]["text"] == "Main Title"
        assert result["headers"][0]["level"] == 1

    def test_extract_code_blocks(self):
        result = extract_code_blocks(self.SAMPLE_MD)
        assert result["success"] is True
        assert result["count"] == 1
        assert result["blocks"][0]["language"] == "python"
        assert "hello" in result["blocks"][0]["content"]

    def test_extract_links(self):
        result = extract_links(self.SAMPLE_MD)
        assert result["success"] is True
        assert result["count"] >= 1
        links = [l for l in result["links"] if l["type"] == "inline"]
        assert len(links) >= 1
        assert links[0]["url"] == "https://example.com"

    def test_extract_frontmatter(self):
        result = extract_frontmatter(self.SAMPLE_MD)
        assert result["success"] is True
        assert result["has_frontmatter"] is True
        assert result["frontmatter"]["title"] == "Test Document"

    def test_extract_frontmatter_none(self):
        result = extract_frontmatter("# No frontmatter here")
        assert result["success"] is True
        assert result["has_frontmatter"] is False

    def test_renumber_sections(self):
        md = """# First
## Sub A
## Sub B
# Second
## Sub C
"""
        result = renumber_sections(md)
        assert result["success"] is True
        assert "1. First" in result["content"]
        assert "1.1. Sub A" in result["content"]
        assert "2. Second" in result["content"]

    def test_generate_toc(self):
        result = generate_toc(self.SAMPLE_MD, max_level=2)
        assert result["success"] is True
        assert "Main Title" in result["toc"]
        assert "Section 1" in result["toc"]
        # Subsection should not be included (max_level=2)
        assert "Subsection" not in result["toc"]


class TestSearchTools:
    """Tests for text search tools."""

    SAMPLE_TEXT = """Line 1: Hello World
Line 2: foo bar
Line 3: Hello again
Line 4: testing 123
Line 5: foo baz"""

    def test_search_pattern(self):
        result = search_pattern(r"Hello", self.SAMPLE_TEXT)
        assert result["success"] is True
        assert result["count"] == 2

    def test_search_pattern_with_groups(self):
        result = search_pattern(r"Line (\d+)", self.SAMPLE_TEXT)
        assert result["success"] is True
        assert result["count"] == 5
        assert result["matches"][0]["groups"] == ["1"]

    def test_search_pattern_case_insensitive(self):
        result = search_pattern(r"hello", self.SAMPLE_TEXT, flags="i")
        assert result["success"] is True
        assert result["count"] == 2

    def test_search_pattern_invalid_regex(self):
        result = search_pattern(r"[invalid", self.SAMPLE_TEXT)
        assert result["success"] is False
        assert "error" in result

    def test_search_lines(self):
        result = search_lines(r"foo", self.SAMPLE_TEXT)
        assert result["success"] is True
        assert result["count"] == 2

    def test_search_lines_with_context(self):
        result = search_lines(r"foo", self.SAMPLE_TEXT, context_before=1, context_after=1)
        assert result["success"] is True
        assert "context_before" in result["lines"][0]
        assert "context_after" in result["lines"][0]

    def test_count_matches(self):
        result = count_matches(r"Line", self.SAMPLE_TEXT)
        assert result["success"] is True
        assert result["count"] == 5
        assert result["line_count"] == 5

    def test_extract_matches(self):
        result = extract_matches(r"\d+", self.SAMPLE_TEXT)
        assert result["success"] is True
        assert "1" in result["matches"]
        assert "123" in result["matches"]

    def test_extract_matches_unique(self):
        text = "a b a c a d"
        result = extract_matches(r"a", text, unique=True)
        assert result["success"] is True
        assert result["count"] == 1

    def test_replace_pattern(self):
        result = replace_pattern(r"foo", "FOO", self.SAMPLE_TEXT)
        assert result["success"] is True
        assert result["replacements"] == 2
        assert "FOO bar" in result["content"]
        assert "FOO baz" in result["content"]

    def test_replace_pattern_with_backreference(self):
        result = replace_pattern(r"Line (\d+)", r"Row \1", self.SAMPLE_TEXT)
        assert result["success"] is True
        assert "Row 1" in result["content"]

    def test_replace_pattern_limited(self):
        result = replace_pattern(r"foo", "FOO", self.SAMPLE_TEXT, count=1)
        assert result["success"] is True
        assert result["replacements"] == 1


class TestToolRegistry:
    """Tests for WASP tool registry."""

    def test_wasp_tools_registry(self):
        assert len(WASP_TOOLS) > 0
        assert "validate_json" in WASP_TOOLS
        assert "parse_markdown" in WASP_TOOLS
        assert "search_pattern" in WASP_TOOLS

    def test_get_tool(self):
        func = get_tool("validate_json")
        assert callable(func)
        result = func('{"test": 1}')
        assert result["valid"] is True

    def test_get_tool_nonexistent(self):
        func = get_tool("nonexistent_tool")
        assert func is None

    def test_list_tools(self):
        tools = list_tools()
        assert len(tools) > 0
        assert "validate_json" in tools

    def test_list_tools_by_category(self):
        validation_tools = list_tools(category="validation")
        assert "validate_json" in validation_tools
        assert "parse_markdown" not in validation_tools

    def test_get_tool_info(self):
        info = get_tool_info("validate_json")
        assert info is not None
        assert info["name"] == "validate_json"
        assert "description" in info
        assert "category" in info


class TestSchemaValidation:
    """Tests for JSON Schema validation."""

    def test_schema_type_validation(self):
        schema = '{"type": "string"}'
        result = validate_json('"hello"', schema=schema)
        assert result["valid"] is True

        result = validate_json('123', schema=schema)
        assert result["valid"] is False

    def test_schema_required_fields(self):
        schema = '{"type": "object", "required": ["name", "age"]}'
        result = validate_json('{"name": "Test", "age": 25}', schema=schema)
        assert result["valid"] is True

        result = validate_json('{"name": "Test"}', schema=schema)
        assert result["valid"] is False

    def test_schema_nested_properties(self):
        schema = '''{
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"}
                    }
                }
            }
        }'''
        result = validate_json('{"user": {"name": "Test"}}', schema=schema)
        assert result["valid"] is True

        result = validate_json('{"user": {"name": 123}}', schema=schema)
        assert result["valid"] is False

    def test_schema_array_items(self):
        # JSON Schema integer works for whole numbers
        schema = '{"type": "array", "items": {"type": "integer"}}'
        result = validate_json('[1, 2, 3]', schema=schema)
        assert result["valid"] is True

        result = validate_json('[1, "two", 3]', schema=schema)
        assert result["valid"] is False

    def test_schema_string_constraints(self):
        schema = '{"type": "string", "minLength": 3, "maxLength": 10}'
        result = validate_json('"hello"', schema=schema)
        assert result["valid"] is True

        result = validate_json('"hi"', schema=schema)
        assert result["valid"] is False

    def test_schema_number_constraints(self):
        # Use integer type which properly matches JSON integers
        schema = '{"type": "integer", "minimum": 0, "maximum": 100}'
        result = validate_json('50', schema=schema)
        assert result["valid"] is True

        result = validate_json('150', schema=schema)
        assert result["valid"] is False

    def test_schema_enum(self):
        schema = '{"type": "string", "enum": ["red", "green", "blue"]}'
        result = validate_json('"green"', schema=schema)
        assert result["valid"] is True

        result = validate_json('"yellow"', schema=schema)
        assert result["valid"] is False
