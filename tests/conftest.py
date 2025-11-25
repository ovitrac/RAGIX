"""
Pytest Configuration and Fixtures

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-11-25
"""

import asyncio
import json
import sys
import tempfile
from pathlib import Path
from typing import Generator, Dict, Any

import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_project(temp_dir: Path) -> Path:
    """Create a sample project structure for testing."""
    # Create directory structure
    (temp_dir / "src").mkdir()
    (temp_dir / "tests").mkdir()
    (temp_dir / "docs").mkdir()

    # Create sample Python files
    (temp_dir / "src" / "main.py").write_text("""
def main():
    \"\"\"Main entry point.\"\"\"
    print("Hello, RAGIX!")

def calculate_sum(a: int, b: int) -> int:
    \"\"\"Calculate sum of two numbers.\"\"\"
    return a + b

class DataProcessor:
    \"\"\"Process data items.\"\"\"

    def __init__(self, name: str):
        self.name = name
        self.items = []

    def add_item(self, item):
        self.items.append(item)

    def process(self):
        return [str(item).upper() for item in self.items]

if __name__ == "__main__":
    main()
""")

    (temp_dir / "src" / "utils.py").write_text("""
import os
import json

def load_config(path: str) -> dict:
    \"\"\"Load configuration from JSON file.\"\"\"
    with open(path, 'r') as f:
        return json.load(f)

def save_config(path: str, config: dict) -> None:
    \"\"\"Save configuration to JSON file.\"\"\"
    with open(path, 'w') as f:
        json.dump(config, f, indent=2)

def get_env(key: str, default: str = "") -> str:
    \"\"\"Get environment variable.\"\"\"
    return os.environ.get(key, default)
""")

    (temp_dir / "tests" / "test_main.py").write_text("""
import pytest
from src.main import calculate_sum, DataProcessor

def test_calculate_sum():
    assert calculate_sum(1, 2) == 3
    assert calculate_sum(-1, 1) == 0

def test_data_processor():
    processor = DataProcessor("test")
    processor.add_item("hello")
    processor.add_item("world")
    result = processor.process()
    assert result == ["HELLO", "WORLD"]
""")

    (temp_dir / "README.md").write_text("""
# Sample Project

This is a sample project for testing RAGIX.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from src.main import main
main()
```
""")

    (temp_dir / "pyproject.toml").write_text("""
[project]
name = "sample-project"
version = "0.1.0"
description = "Sample project for testing"

[tool.pytest.ini_options]
testpaths = ["tests"]
""")

    return temp_dir


@pytest.fixture
def sample_chunks() -> list:
    """Create sample code chunks for testing."""
    return [
        {
            "id": "chunk_001",
            "file_path": "src/main.py",
            "chunk_type": "function",
            "name": "main",
            "content": "def main():\n    print('Hello, RAGIX!')",
            "start_line": 1,
            "end_line": 3,
        },
        {
            "id": "chunk_002",
            "file_path": "src/main.py",
            "chunk_type": "function",
            "name": "calculate_sum",
            "content": "def calculate_sum(a, b):\n    return a + b",
            "start_line": 5,
            "end_line": 7,
        },
        {
            "id": "chunk_003",
            "file_path": "src/main.py",
            "chunk_type": "class",
            "name": "DataProcessor",
            "content": "class DataProcessor:\n    def __init__(self, name):\n        self.name = name",
            "start_line": 9,
            "end_line": 15,
        },
        {
            "id": "chunk_004",
            "file_path": "src/utils.py",
            "chunk_type": "function",
            "name": "load_config",
            "content": "def load_config(path):\n    with open(path, 'r') as f:\n        return json.load(f)",
            "start_line": 1,
            "end_line": 4,
        },
    ]


@pytest.fixture
def mock_llm_response() -> str:
    """Create mock LLM response with tool calls."""
    return """Let me analyze this bug.

**Thinking**: The error seems to be in the utils.py file where we load the config.

{"action": "read_file", "path": "src/utils.py"}

After checking the file, I need to fix the issue.

{"action": "edit_file", "path": "src/utils.py", "old_text": "return json.load(f)", "new_text": "return json.load(f) or {}"}

Task completed: I've fixed the bug by adding a fallback empty dict.
"""


@pytest.fixture
def mock_tool_calls() -> list:
    """Create mock tool calls for testing."""
    from ragix_core.agent_llm_bridge import ToolCall

    return [
        ToolCall(
            tool_name="read_file",
            parameters={"path": "src/main.py"},
            raw_text='{"action": "read_file", "path": "src/main.py"}',
        ),
        ToolCall(
            tool_name="grep_search",
            parameters={"pattern": "def main", "path": "."},
            raw_text='{"action": "grep_search", "pattern": "def main", "path": "."}',
        ),
    ]
