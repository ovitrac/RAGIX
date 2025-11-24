"""
Pre-Built Agent Roles for Multi-Agent Workflows

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-11-24
"""

from .base_agent import BaseAgent, AgentCapability
from .code_agent import CodeAgent
from .doc_agent import DocAgent
from .git_agent import GitAgent
from .test_agent import TestAgent

__all__ = [
    "BaseAgent",
    "AgentCapability",
    "CodeAgent",
    "DocAgent",
    "GitAgent",
    "TestAgent",
]
