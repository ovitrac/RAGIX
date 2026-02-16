"""
Tests for ragix_core.shared.gpu_detect

Covers:
  - GPUInfo dataclass
  - nvidia-smi output parsing (_parse_nvidia_smi)
  - Public API (has_gpu, has_faiss_gpu, gpu_info)
  - Mock scenarios (no GPU, GPU without faiss-gpu)

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-02-15
"""

from __future__ import annotations

from unittest import mock

import pytest

from ragix_core.shared.gpu_detect import (
    GPUInfo,
    _parse_nvidia_smi,
    _probe_system_memory_mb,
    gpu_info,
    has_faiss_gpu,
    has_gpu,
)


# ---------------------------------------------------------------------------
# GPUInfo dataclass
# ---------------------------------------------------------------------------

class TestGPUInfo:
    """GPUInfo dataclass construction and immutability."""

    def test_no_gpu(self):
        info = GPUInfo(available=False)
        assert not info.available
        assert info.device_name == ""
        assert info.memory_mb == 0
        assert info.driver_version == ""
        assert not info.faiss_gpu

    def test_full_gpu(self):
        info = GPUInfo(
            available=True,
            device_name="NVIDIA A100",
            memory_mb=40960,
            driver_version="535.129.03",
            faiss_gpu=True,
        )
        assert info.available
        assert info.device_name == "NVIDIA A100"
        assert info.memory_mb == 40960
        assert info.faiss_gpu
        assert not info.unified_memory

    def test_unified_memory(self):
        info = GPUInfo(
            available=True,
            device_name="NVIDIA GB10",
            memory_mb=122503,
            driver_version="580.126.09",
            unified_memory=True,
        )
        assert info.available
        assert info.unified_memory
        assert info.memory_mb == 122503

    def test_frozen(self):
        info = GPUInfo(available=False)
        with pytest.raises(AttributeError):
            info.available = True  # type: ignore[misc]


# ---------------------------------------------------------------------------
# nvidia-smi output parsing
# ---------------------------------------------------------------------------

class TestParseNvidiaSmi:
    """Unit tests for _parse_nvidia_smi (pure function, no subprocess)."""

    def test_standard_output(self):
        stdout = "NVIDIA A100-SXM4-80GB, 81920, 535.129.03\n"
        name, mem, driver = _parse_nvidia_smi(stdout)
        assert name == "NVIDIA A100-SXM4-80GB"
        assert mem == 81920
        assert driver == "535.129.03"

    def test_gb10_na_memory(self):
        """Grace Hopper / Blackwell architectures report [N/A] for memory."""
        stdout = "NVIDIA GB10, [N/A], 580.126.09\n"
        name, mem, driver = _parse_nvidia_smi(stdout)
        assert name == "NVIDIA GB10"
        assert mem == 0  # [N/A] → 0
        assert driver == "580.126.09"

    def test_float_memory(self):
        stdout = "NVIDIA T4, 15360.5, 525.85.12\n"
        name, mem, driver = _parse_nvidia_smi(stdout)
        assert mem == 15360  # int(float(...))

    def test_multi_gpu_first_line(self):
        stdout = "NVIDIA A100, 40960, 535.129.03\nNVIDIA A100, 40960, 535.129.03\n"
        name, mem, driver = _parse_nvidia_smi(stdout)
        assert name == "NVIDIA A100"

    def test_empty_output(self):
        name, mem, driver = _parse_nvidia_smi("")
        assert name == ""
        assert mem == 0
        assert driver == ""

    def test_malformed_output(self):
        name, mem, driver = _parse_nvidia_smi("some garbage")
        assert name == ""

    def test_whitespace_trimming(self):
        stdout = "  NVIDIA RTX 4090 ,  24564 ,  545.23.08  \n"
        name, mem, driver = _parse_nvidia_smi(stdout)
        assert name == "NVIDIA RTX 4090"
        assert mem == 24564
        assert driver == "545.23.08"


# ---------------------------------------------------------------------------
# Public API with mocking
# ---------------------------------------------------------------------------

class TestSystemMemoryProbe:
    """Test _probe_system_memory_mb (Linux /proc/meminfo)."""

    def test_returns_positive_on_linux(self):
        """On Linux, system memory should be > 0."""
        import platform
        if platform.system() != "Linux":
            pytest.skip("Linux-only test")
        mem = _probe_system_memory_mb()
        assert mem > 0

    @mock.patch("builtins.open", side_effect=OSError("no /proc"))
    def test_returns_zero_on_failure(self, _):
        assert _probe_system_memory_mb() == 0


class TestPublicAPI:
    """Test has_gpu / has_faiss_gpu / gpu_info with mocked subprocess."""

    def setup_method(self):
        # Clear LRU cache before each test
        gpu_info.cache_clear()

    def teardown_method(self):
        gpu_info.cache_clear()

    @mock.patch("ragix_core.shared.gpu_detect._probe_nvidia_smi")
    @mock.patch("ragix_core.shared.gpu_detect._probe_faiss_gpu")
    def test_no_gpu_detected(self, mock_faiss, mock_smi):
        mock_smi.return_value = ("", 0, "")
        mock_faiss.return_value = False
        info = gpu_info()
        assert not info.available
        assert not has_gpu()
        assert not has_faiss_gpu()

    @mock.patch("ragix_core.shared.gpu_detect._probe_nvidia_smi")
    @mock.patch("ragix_core.shared.gpu_detect._probe_faiss_gpu")
    def test_gpu_without_faiss(self, mock_faiss, mock_smi):
        mock_smi.return_value = ("NVIDIA T4", 15360, "525.85.12")
        mock_faiss.return_value = False
        info = gpu_info()
        assert info.available
        assert info.device_name == "NVIDIA T4"
        assert has_gpu()
        assert not has_faiss_gpu()

    @mock.patch("ragix_core.shared.gpu_detect._probe_nvidia_smi")
    @mock.patch("ragix_core.shared.gpu_detect._probe_faiss_gpu")
    def test_gpu_with_faiss(self, mock_faiss, mock_smi):
        mock_smi.return_value = ("NVIDIA A100", 40960, "535.129.03")
        mock_faiss.return_value = True
        info = gpu_info()
        assert info.available
        assert info.faiss_gpu
        assert has_gpu()
        assert has_faiss_gpu()

    @mock.patch("ragix_core.shared.gpu_detect._probe_nvidia_smi")
    @mock.patch("ragix_core.shared.gpu_detect._probe_faiss_gpu")
    @mock.patch("ragix_core.shared.gpu_detect._probe_system_memory_mb")
    def test_unified_memory_fallback(self, mock_sysmem, mock_faiss, mock_smi):
        """When nvidia-smi reports 0 memory, fall back to system memory."""
        mock_smi.return_value = ("NVIDIA GB10", 0, "580.126.09")
        mock_sysmem.return_value = 122503
        mock_faiss.return_value = False
        info = gpu_info()
        assert info.available
        assert info.device_name == "NVIDIA GB10"
        assert info.memory_mb == 122503
        assert info.unified_memory

    @mock.patch("ragix_core.shared.gpu_detect._probe_nvidia_smi")
    @mock.patch("ragix_core.shared.gpu_detect._probe_faiss_gpu")
    def test_cache_consistency(self, mock_faiss, mock_smi):
        """gpu_info() is cached — second call returns same object."""
        mock_smi.return_value = ("NVIDIA T4", 15360, "525.85.12")
        mock_faiss.return_value = False
        info1 = gpu_info()
        info2 = gpu_info()
        assert info1 is info2
        # _probe_nvidia_smi should be called only once
        mock_smi.assert_called_once()


class TestLiveDetection:
    """Non-mocked test that runs the actual detection on this machine."""

    def setup_method(self):
        gpu_info.cache_clear()

    def teardown_method(self):
        gpu_info.cache_clear()

    def test_gpu_info_returns_gpuinfo(self):
        """gpu_info() always returns a GPUInfo, regardless of hardware."""
        info = gpu_info()
        assert isinstance(info, GPUInfo)
        assert isinstance(info.available, bool)
        assert isinstance(info.faiss_gpu, bool)

    def test_has_gpu_returns_bool(self):
        assert isinstance(has_gpu(), bool)

    def test_has_faiss_gpu_returns_bool(self):
        assert isinstance(has_faiss_gpu(), bool)
