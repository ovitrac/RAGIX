"""
RAGIX Web Routers - Modular API endpoints

This module organizes API endpoints into logical groups using FastAPI APIRouter.

Two-Level RAG Architecture:
    - rag_router: Chat RAG (Level 2) - light, session-scoped, BM25-based (.ragix/)
    - rag_project_router: Project RAG (Level 1) - massive, persistent, ChromaDB-based (.RAG/)

Audit Module (v0.4):
    - audit_router: Code audit with timeline, risk, and drift analysis

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-12-09
"""

from .sessions import router as sessions_router, get_agent_config_store
from .memory import router as memory_router
from .context import router as context_router
from .agents import router as agents_router
from .logs import router as logs_router
from .reasoning import router as reasoning_router
from .threads import router as threads_router
from .rag import router as rag_router
from .rag_project import router as rag_project_router
from .audit import router as audit_router

__all__ = [
    "sessions_router",
    "memory_router",
    "context_router",
    "agents_router",
    "logs_router",
    "reasoning_router",
    "threads_router",
    "rag_router",
    "rag_project_router",
    "audit_router",
    "get_agent_config_store",
]
