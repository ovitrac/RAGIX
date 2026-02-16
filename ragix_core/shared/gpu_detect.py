"""
NVIDIA GPU Detection for RAGIX

Lightweight GPU availability detection via:
  1. nvidia-smi subprocess check (driver + device info)
  2. faiss GPU import probe (Python bindings with get_num_gpus)
  3. Cached result (single check per process lifetime)

Usage:
    from ragix_core.shared.gpu_detect import gpu_info, has_gpu, has_faiss_gpu

    if has_faiss_gpu():
        # use GPU-accelerated FAISS
    info = gpu_info()  # GPUInfo with device_name, memory_mb, driver_version

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-02-15
"""

from __future__ import annotations

import dataclasses
import functools
import logging
import subprocess
from typing import Tuple

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class GPUInfo:
    """NVIDIA GPU detection result."""

    available: bool
    device_name: str = ""
    memory_mb: int = 0
    driver_version: str = ""
    faiss_gpu: bool = False
    unified_memory: bool = False  # True when GPU shares system RAM (Grace Blackwell, etc.)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_nvidia_smi(stdout: str) -> Tuple[str, int, str]:
    """
    Parse nvidia-smi CSV output (first GPU line only).

    Expected format (--format=csv,noheader,nounits):
        NVIDIA GB10, 1234, 580.126.09

    Returns (device_name, memory_mb, driver_version).
    Handles [N/A] fields gracefully (some architectures report [N/A] for memory).
    """
    lines = stdout.strip().splitlines()
    if not lines:
        return ("", 0, "")
    parts = lines[0].split(",")
    if len(parts) < 3:
        return ("", 0, "")
    device_name = parts[0].strip()
    try:
        memory_mb = int(float(parts[1].strip()))
    except (ValueError, TypeError):
        memory_mb = 0  # [N/A] on some architectures (Grace Hopper, etc.)
    driver_version = parts[2].strip()
    return (device_name, memory_mb, driver_version)


def _probe_nvidia_smi() -> Tuple[str, int, str]:
    """Run nvidia-smi and return parsed (device_name, memory_mb, driver_version)."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return _parse_nvidia_smi(result.stdout)
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return ("", 0, "")


def _probe_system_memory_mb() -> int:
    """
    Read total system memory from /proc/meminfo (Linux only).

    Used as fallback for unified memory architectures (Grace Blackwell, etc.)
    where nvidia-smi reports [N/A] for GPU memory because CPU and GPU share
    the same physical RAM.
    """
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return kb // 1024  # kB → MB
    except (OSError, ValueError, IndexError):
        pass
    return 0


def _probe_faiss_gpu() -> bool:
    """Check if faiss Python bindings support GPU (get_num_gpus > 0)."""
    try:
        import faiss  # type: ignore[import-untyped]

        if hasattr(faiss, "get_num_gpus") and faiss.get_num_gpus() > 0:
            return True
    except (ImportError, Exception):
        pass
    return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def gpu_info() -> GPUInfo:
    """
    Detect NVIDIA GPU and FAISS GPU support.

    Result is cached for the lifetime of the process.  To force re-detection
    (e.g. in tests), call ``gpu_info.cache_clear()``.
    """
    device_name, memory_mb, driver_version = _probe_nvidia_smi()
    if not device_name:
        logger.debug("No NVIDIA GPU detected (nvidia-smi not found or failed)")
        return GPUInfo(available=False)

    # Unified memory fallback (Grace Blackwell, Jetson, etc.)
    unified = False
    if memory_mb == 0:
        sys_mem = _probe_system_memory_mb()
        if sys_mem > 0:
            memory_mb = sys_mem
            unified = True
            logger.debug(
                f"Unified memory architecture — using system RAM: {memory_mb} MB"
            )

    faiss_gpu = _probe_faiss_gpu()
    logger.info(
        f"GPU detected: {device_name} ({memory_mb} MB"
        f"{' unified' if unified else ''}, driver {driver_version}, "
        f"faiss_gpu={'yes' if faiss_gpu else 'no'})"
    )
    return GPUInfo(
        available=True,
        device_name=device_name,
        memory_mb=memory_mb,
        driver_version=driver_version,
        faiss_gpu=faiss_gpu,
        unified_memory=unified,
    )


def has_gpu() -> bool:
    """True if NVIDIA GPU is detected via nvidia-smi."""
    return gpu_info().available


def has_faiss_gpu() -> bool:
    """True if NVIDIA GPU detected AND faiss GPU bindings are available."""
    return gpu_info().faiss_gpu
