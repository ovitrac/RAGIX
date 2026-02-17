"""
Kernel: Volumetry
Stage: 1 (Collection)

Ingests operational volumetry data from YAML/JSON configuration.
Normalizes volumes, identifies peak patterns, and calculates throughput requirements.

This kernel bridges the gap between code metrics and production workload,
enabling volumetry-weighted risk assessment.

Input sources (in order of precedence):
1. workspace/data/volumetry.yaml
2. workspace/data/volumetry.json
3. manifest config (inline)

Output:
- flows: Normalized flow data (volume/day, volume/sec, peaks)
- module_volumetry: Volume attribution per module
- peak_patterns: Peak timing and multipliers
- incidents: Historical incidents data

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-12-16
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
import json

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

from ragix_kernels.base import Kernel, KernelInput

logger = logging.getLogger(__name__)


# Default normalization parameters
DEFAULT_NORMALIZATION = {
    "volume_max": 4_000_000,     # 4M/day → score 10
    "peak_multiplier_max": 20,   # 20x peak → score 10
    "incident_weight": 2.0,      # Each incident adds to criticality
}


class VolumetryKernel(Kernel):
    """
    Ingest and normalize operational volumetry data.

    This kernel accepts volumetry data from multiple sources:
    - YAML/JSON file in workspace/data/
    - Inline configuration in manifest

    The output provides:
    - Normalized volume scores (0-10 scale)
    - Peak pattern identification
    - Module-level volume attribution
    - Historical incident tracking

    Configuration options:
        data_file: Path to volumetry file (relative to workspace)
        normalization: Custom normalization parameters
        flows: Inline flow definitions (if no file)
        modules: Inline module mappings (if no file)

    Example manifest config:
        volumetry:
          enabled: true
          options:
            data_file: "data/volumetry.yaml"
            normalization:
              volume_max: 4000000

    Example volumetry.yaml:
        flows:
          - name: MSG-HUB
            volume_day: 4000000
            peak_hour: 5
            peak_window: "00:00-10:00"
            peak_multiplier: 10
        modules:
          - name: acme-msg-hub
            flows: [MSG-HUB]
            role: entry_point
    """

    name = "volumetry"
    version = "1.0.0"
    category = "audit"
    stage = 1
    description = "Operational volumetry data collection"

    requires = []  # No dependencies - foundation kernel
    provides = ["volumetry", "peak_patterns", "throughput", "module_volumetry"]

    def compute(self, input: KernelInput) -> Dict[str, Any]:
        """Load and normalize volumetry data."""

        # Load volumetry data from file or config
        volumetry_data = self._load_volumetry_data(input)

        if not volumetry_data:
            logger.warning("[volumetry] No volumetry data found")
            return {
                "flows": {},
                "module_volumetry": {},
                "peak_patterns": {},
                "incidents": [],
                "statistics": {"total_flows": 0, "has_data": False},
            }

        # Get normalization parameters
        norm_config = input.config.get("normalization", {})
        normalization = {**DEFAULT_NORMALIZATION, **norm_config}

        # Process flows
        flows_data = self._process_flows(
            volumetry_data.get("flows", []),
            normalization
        )

        # Process module mappings
        module_volumetry = self._process_modules(
            volumetry_data.get("modules", []),
            flows_data
        )

        # Extract peak patterns
        peak_patterns = self._extract_peak_patterns(flows_data)

        # Process incidents
        incidents = self._process_incidents(
            volumetry_data.get("incidents", [])
        )

        # Update criticality based on incidents
        module_volumetry = self._apply_incident_criticality(
            module_volumetry, incidents, normalization
        )

        # Statistics
        statistics = {
            "total_flows": len(flows_data),
            "total_modules": len(module_volumetry),
            "total_incidents": len(incidents),
            "has_data": True,
            "max_volume_day": max((f.get("volume_day", 0) for f in flows_data.values()), default=0),
            "max_peak_multiplier": max((f.get("peak_multiplier", 1) for f in flows_data.values()), default=1),
        }

        return {
            "flows": flows_data,
            "module_volumetry": module_volumetry,
            "peak_patterns": peak_patterns,
            "incidents": incidents,
            "normalization": normalization,
            "statistics": statistics,
        }

    def _load_volumetry_data(self, input: KernelInput) -> Optional[Dict[str, Any]]:
        """Load volumetry data from file or config."""

        # Check for data file
        data_file = input.config.get("data_file")
        if data_file:
            file_path = input.workspace / data_file
            if file_path.exists():
                return self._load_file(file_path)
            else:
                logger.warning(f"[volumetry] Data file not found: {file_path}")

        # Check default locations
        for filename in ["volumetry.yaml", "volumetry.yml", "volumetry.json"]:
            file_path = input.workspace / "data" / filename
            if file_path.exists():
                logger.info(f"[volumetry] Loading from {file_path}")
                return self._load_file(file_path)

        # Use inline config
        if "flows" in input.config or "modules" in input.config:
            logger.info("[volumetry] Using inline configuration")
            return {
                "flows": input.config.get("flows", []),
                "modules": input.config.get("modules", []),
                "incidents": input.config.get("incidents", []),
            }

        return None

    def _load_file(self, path: Path) -> Dict[str, Any]:
        """Load YAML or JSON file."""
        content = path.read_text(encoding="utf-8")

        if path.suffix in (".yaml", ".yml"):
            if not YAML_AVAILABLE:
                raise RuntimeError("PyYAML not installed. Install with: pip install pyyaml")
            return yaml.safe_load(content) or {}
        else:
            return json.loads(content)

    def _process_flows(
        self,
        flows: List[Dict[str, Any]],
        normalization: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """Process and normalize flow data."""
        result = {}
        volume_max = normalization["volume_max"]

        for flow in flows:
            name = flow.get("name", "unknown")

            volume_day = flow.get("volume_day", 0)
            unit = flow.get("unit", "messages")
            peak_hour = flow.get("peak_hour")
            peak_window = flow.get("peak_window", "")
            peak_multiplier = flow.get("peak_multiplier", 1)

            # Calculate throughput
            volume_sec_avg = volume_day / 86400 if volume_day > 0 else 0

            # Calculate peak throughput
            if peak_window:
                # Parse window like "00:00-10:00"
                try:
                    start, end = peak_window.split("-")
                    start_hour = int(start.split(":")[0])
                    end_hour = int(end.split(":")[0])
                    window_hours = (end_hour - start_hour) % 24
                    if window_hours > 0:
                        window_seconds = window_hours * 3600
                        volume_sec_window = volume_day / window_seconds
                    else:
                        volume_sec_window = volume_sec_avg
                except (ValueError, IndexError):
                    volume_sec_window = volume_sec_avg
            else:
                volume_sec_window = volume_sec_avg

            # Peak throughput
            volume_sec_peak = volume_sec_window * peak_multiplier

            # Normalized score (0-10 scale)
            normalized_score = min(10.0, (volume_day / volume_max) * 10)

            result[name] = {
                "volume_day": volume_day,
                "unit": unit,
                "volume_sec_avg": round(volume_sec_avg, 1),
                "volume_sec_window": round(volume_sec_window, 1),
                "volume_sec_peak": round(volume_sec_peak, 1),
                "peak_hour": peak_hour,
                "peak_window": peak_window,
                "peak_multiplier": peak_multiplier,
                "normalized_score": round(normalized_score, 2),
            }

        return result

    def _process_modules(
        self,
        modules: List[Dict[str, Any]],
        flows_data: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """Map modules to their volumetry."""
        result = {}

        for module in modules:
            name = module.get("name", "unknown")
            module_flows = module.get("flows", [])
            role = module.get("role", "unknown")

            # Aggregate volume from all flows
            total_volume = 0
            max_peak = 0
            flow_details = []

            for flow_name in module_flows:
                if flow_name in flows_data:
                    flow = flows_data[flow_name]
                    total_volume += flow.get("volume_day", 0)
                    max_peak = max(max_peak, flow.get("volume_sec_peak", 0))
                    flow_details.append({
                        "flow": flow_name,
                        "volume_day": flow.get("volume_day", 0),
                    })

            # Determine criticality based on volume and role
            criticality = self._determine_criticality(total_volume, max_peak, role)

            result[name] = {
                "flows": module_flows,
                "role": role,
                "volume_day": total_volume,
                "volume_sec_peak": round(max_peak, 1),
                "flow_details": flow_details,
                "criticality": criticality,
                "incident_count": 0,  # Will be updated by incident processing
            }

        return result

    def _determine_criticality(
        self,
        volume_day: int,
        peak_rate: float,
        role: str
    ) -> str:
        """Determine module criticality based on volume and role."""
        # Entry points and high-volume modules are more critical
        if role in ("entry_point", "gateway") or peak_rate > 500:
            return "CRITICAL"
        elif volume_day > 1_000_000 or peak_rate > 100:
            return "HIGH"
        elif volume_day > 100_000 or peak_rate > 50:
            return "MEDIUM"
        else:
            return "LOW"

    def _extract_peak_patterns(
        self,
        flows_data: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Extract peak timing patterns from flows."""
        peak_hours = []
        peak_windows = []

        for flow in flows_data.values():
            if flow.get("peak_hour") is not None:
                peak_hours.append(flow["peak_hour"])
            if flow.get("peak_window"):
                peak_windows.append(flow["peak_window"])

        # Find common peak window
        common_window = None
        if peak_windows:
            # Use most common window
            from collections import Counter
            window_counts = Counter(peak_windows)
            common_window = window_counts.most_common(1)[0][0]

        # Calculate off-peak window
        off_peak_window = None
        off_peak_seconds = 0
        if common_window:
            try:
                start, end = common_window.split("-")
                end_hour = int(end.split(":")[0])
                # Off-peak is from end of peak to midnight + midnight to start of peak
                off_peak_window = f"{end}-{start}"
                off_peak_hours = (24 - end_hour) + int(start.split(":")[0])
                off_peak_seconds = off_peak_hours * 3600
            except (ValueError, IndexError):
                pass

        return {
            "peak_hours": sorted(set(peak_hours)),
            "peak_window": common_window,
            "off_peak_window": off_peak_window,
            "off_peak_seconds": off_peak_seconds,
            "max_peak_multiplier": max(
                (f.get("peak_multiplier", 1) for f in flows_data.values()),
                default=1
            ),
        }

    def _process_incidents(
        self,
        incidents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Process historical incidents."""
        result = []

        for incident in incidents:
            result.append({
                "date": incident.get("date", "unknown"),
                "type": incident.get("type", "unknown"),
                "module": incident.get("module", "unknown"),
                "cause": incident.get("cause", "unknown"),
                "description": incident.get("description", ""),
            })

        return result

    def _apply_incident_criticality(
        self,
        module_volumetry: Dict[str, Dict[str, Any]],
        incidents: List[Dict[str, Any]],
        normalization: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """Update module criticality based on incidents."""
        incident_weight = normalization.get("incident_weight", 2.0)

        # Count incidents per module
        incident_counts = {}
        for incident in incidents:
            module = incident.get("module", "")
            if module:
                incident_counts[module] = incident_counts.get(module, 0) + 1

        # Update module data
        for name, data in module_volumetry.items():
            count = incident_counts.get(name, 0)
            data["incident_count"] = count

            # Escalate criticality if incidents occurred
            if count > 0:
                current = data["criticality"]
                if current == "LOW":
                    data["criticality"] = "MEDIUM"
                elif current == "MEDIUM":
                    data["criticality"] = "HIGH"
                elif current == "HIGH" and count >= 2:
                    data["criticality"] = "CRITICAL"

        return module_volumetry

    def summarize(self, data: Dict[str, Any]) -> str:
        """Generate LLM-consumable summary."""
        stats = data.get("statistics", {})

        if not stats.get("has_data"):
            return "Volumetry: No operational data available."

        flows = data.get("flows", {})
        modules = data.get("module_volumetry", {})
        incidents = data.get("incidents", [])
        peaks = data.get("peak_patterns", {})

        # Count by criticality
        criticality_counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
        for m in modules.values():
            crit = m.get("criticality", "LOW")
            criticality_counts[crit] = criticality_counts.get(crit, 0) + 1

        # Top flow
        top_flow = max(flows.items(), key=lambda x: x[1].get("volume_day", 0), default=("none", {}))
        top_flow_name = top_flow[0]
        top_flow_vol = top_flow[1].get("volume_day", 0)

        return (
            f"Volumetry: {len(flows)} flows, {len(modules)} modules. "
            f"Top: {top_flow_name} ({top_flow_vol:,}/day). "
            f"Peak: {peaks.get('peak_window', 'N/A')} ({peaks.get('max_peak_multiplier', 1)}x). "
            f"Criticality: {criticality_counts['CRITICAL']} CRIT, {criticality_counts['HIGH']} HIGH. "
            f"Incidents: {len(incidents)}."
        )
