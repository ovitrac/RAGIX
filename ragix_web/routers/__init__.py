"""
RAGIX Web Routers - Modular API endpoints

This module organizes API endpoints into logical groups using FastAPI APIRouter.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-11-28
"""

from .sessions import router as sessions_router, get_agent_config_store
from .memory import router as memory_router
from .context import router as context_router
from .agents import router as agents_router
from .logs import router as logs_router
from .reasoning import router as reasoning_router
from .threads import router as threads_router
from .rag import router as rag_router

__all__ = [
    "sessions_router",
    "memory_router",
    "context_router",
    "agents_router",
    "logs_router",
    "reasoning_router",
    "threads_router",
    "rag_router",
    "get_agent_config_store",
]
