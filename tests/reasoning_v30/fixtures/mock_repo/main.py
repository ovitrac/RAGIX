"""
Mock repository main module for testing.
"""

def hello_world():
    """Print hello world."""
    print("Hello, World!")


def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


class Calculator:
    """Simple calculator class."""

    def __init__(self):
        self.result = 0

    def add(self, value: int) -> "Calculator":
        """Add value to result."""
        self.result += value
        return self

    def subtract(self, value: int) -> "Calculator":
        """Subtract value from result."""
        self.result -= value
        return self

    def get_result(self) -> int:
        """Get current result."""
        return self.result


if __name__ == "__main__":
    hello_world()
