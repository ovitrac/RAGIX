"""
Kernel: Spring DI Wiring Resolution
Stage: 2 (Analysis)

Resolves Spring dependency injection wiring from ast_scan annotations
to identify implicit entry points invisible to standard reachability
analysis. Addresses the dead code false positive problem (98% → realistic).

Detects:
- Spring beans (@Service, @Component, @Repository, @Controller, @Configuration)
- Factory beans (@Bean methods in @Configuration classes)
- Implicit entry points (@JmsListener, @EventListener, @Scheduled, @RequestMapping)
- Injection wiring (@Autowired, @Inject fields)

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-02-09
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, List, Optional, Set
import logging

from ragix_kernels.base import Kernel, KernelInput

logger = logging.getLogger(__name__)

# Spring annotation classification
SPRING_BEAN_ANNOTATIONS = frozenset({
    "Service", "Component", "Repository", "Controller",
    "RestController", "Configuration",
})

SPRING_ENTRY_ANNOTATIONS = frozenset({
    "JmsListener", "EventListener", "Scheduled",
    "RequestMapping", "GetMapping", "PostMapping",
    "PutMapping", "DeleteMapping", "PatchMapping",
})

SPRING_INJECT_ANNOTATIONS = frozenset({
    "Autowired", "Inject", "Value", "Resource",
})

SPRING_FACTORY_ANNOTATIONS = frozenset({
    "Bean",
})


class SpringWiringKernel(Kernel):
    """
    Resolve Spring DI wiring and implicit entry points.

    Reads ast_scan output (enriched with annotations) to build a
    Spring bean registry and wiring graph. Identifies entry points
    that would otherwise be missed by standard reachability analysis.
    """

    name = "spring_wiring"
    version = "1.0.0"
    category = "audit"
    stage = 2
    description = "Resolve Spring DI wiring and implicit entry points"
    requires = ["ast_scan"]
    provides = ["spring_beans", "spring_entry_points"]

    def compute(self, input: KernelInput) -> Dict[str, Any]:
        # Load ast_scan output
        ast_path = input.dependencies.get("ast_scan")
        if not ast_path or not ast_path.exists():
            raise RuntimeError("ast_scan output not found")

        with open(ast_path) as f:
            ast_data = json.load(f).get("data", {})

        symbols = ast_data.get("symbols", [])

        # Classify symbols by type
        classes = [s for s in symbols if s.get("type") == "class"]
        methods = [s for s in symbols if s.get("type") in ("method", "constructor")]
        fields = [s for s in symbols if s.get("type") == "field"]

        # Build class → annotations map
        class_annotations: Dict[str, List[str]] = {}
        for cls in classes:
            ann = cls.get("annotations", [])
            if ann:
                class_annotations[cls["qualified_name"]] = ann

        # Build method → annotations map
        method_annotations: Dict[str, List[str]] = {}
        for meth in methods:
            ann = meth.get("annotations", [])
            if ann:
                method_annotations[meth["qualified_name"]] = ann

        # Build field → annotations map
        field_annotations: Dict[str, List[str]] = {}
        for fld in fields:
            ann = fld.get("annotations", [])
            if ann:
                field_annotations[fld["qualified_name"]] = ann

        # Identify Spring beans
        beans = []
        bean_classes: Set[str] = set()
        for cls in classes:
            qname = cls["qualified_name"]
            ann = cls.get("annotations", [])
            bean_type = self._classify_bean(ann)
            if bean_type:
                beans.append({
                    "class": qname,
                    "type": bean_type,
                    "annotations": ann,
                    "file": cls.get("file", ""),
                    "line": cls.get("line", 0),
                    "entry_point": False,  # updated below
                })
                bean_classes.add(qname)

        # Identify factory beans (@Bean methods in @Configuration classes)
        for meth in methods:
            ann = meth.get("annotations", [])
            if any(a in SPRING_FACTORY_ANNOTATIONS for a in ann):
                # Find parent class
                parent_class = self._get_parent_class(meth["qualified_name"])
                if parent_class in class_annotations:
                    parent_ann = class_annotations[parent_class]
                    if "Configuration" in parent_ann:
                        beans.append({
                            "class": meth["qualified_name"],
                            "type": "factory_bean",
                            "annotations": ann,
                            "file": meth.get("file", ""),
                            "line": meth.get("line", 0),
                            "entry_point": False,
                            "factory_class": parent_class,
                        })
                        bean_classes.add(meth["qualified_name"])

        # Identify entry points (implicit reachability)
        entry_points = {
            "jms_listeners": [],
            "event_listeners": [],
            "scheduled": [],
            "rest_endpoints": [],
            "main_methods": [],
        }

        for meth in methods:
            ann = meth.get("annotations", [])
            qname = meth["qualified_name"]

            if "JmsListener" in ann:
                entry_points["jms_listeners"].append({
                    "method": qname,
                    "file": meth.get("file", ""),
                    "line": meth.get("line", 0),
                })
            if "EventListener" in ann:
                entry_points["event_listeners"].append({
                    "method": qname,
                    "file": meth.get("file", ""),
                    "line": meth.get("line", 0),
                })
            if "Scheduled" in ann:
                entry_points["scheduled"].append({
                    "method": qname,
                    "file": meth.get("file", ""),
                    "line": meth.get("line", 0),
                })
            if any(a in ("RequestMapping", "GetMapping", "PostMapping",
                         "PutMapping", "DeleteMapping", "PatchMapping") for a in ann):
                entry_points["rest_endpoints"].append({
                    "method": qname,
                    "file": meth.get("file", ""),
                    "line": meth.get("line", 0),
                })

            # main() methods
            if meth.get("name") == "main":
                entry_points["main_methods"].append({
                    "method": qname,
                    "file": meth.get("file", ""),
                    "line": meth.get("line", 0),
                })

        # Mark beans that are entry points
        entry_classes = set()
        for category in entry_points.values():
            for ep in category:
                parent = self._get_parent_class(ep["method"])
                entry_classes.add(parent)

        for bean in beans:
            if bean["class"] in entry_classes:
                bean["entry_point"] = True

        # Build wiring graph (field injection)
        wiring = []
        for fld in fields:
            ann = fld.get("annotations", [])
            if any(a in SPRING_INJECT_ANNOTATIONS for a in ann):
                parent = self._get_parent_class(fld["qualified_name"])
                wiring.append({
                    "from": parent,
                    "field": fld["qualified_name"],
                    "type": "autowired" if "Autowired" in ann else "inject",
                    "annotations": [a for a in ann if a in SPRING_INJECT_ANNOTATIONS],
                })

        # Compute entry point totals
        total_entry_points = sum(len(v) for v in entry_points.values())

        # Estimate reachability improvement
        # Before: only main() and controllers (from dead_code kernel)
        before_ep = len(entry_points["main_methods"]) + len(entry_points["rest_endpoints"])
        after_ep = total_entry_points + len(bean_classes)

        return {
            "beans": beans,
            "wiring": wiring,
            "entry_points": entry_points,
            "entry_point_summary": {
                "jms_listeners": len(entry_points["jms_listeners"]),
                "event_listeners": len(entry_points["event_listeners"]),
                "scheduled": len(entry_points["scheduled"]),
                "rest_endpoints": len(entry_points["rest_endpoints"]),
                "main_methods": len(entry_points["main_methods"]),
                "total": total_entry_points,
            },
            "reachability_delta": {
                "before": max(before_ep, 1),
                "after": max(after_ep, 1),
                "improvement_ratio": round(after_ep / max(before_ep, 1), 1),
                "new_bean_classes": len(bean_classes),
            },
            "statistics": {
                "total_beans": len(beans),
                "bean_classes": len(bean_classes),
                "by_type": self._count_by_type(beans),
                "total_wiring": len(wiring),
                "total_entry_points": total_entry_points,
                "classes_with_annotations": len(class_annotations),
                "methods_with_annotations": len(method_annotations),
                "fields_with_annotations": len(field_annotations),
            },
        }

    def summarize(self, data: Dict[str, Any]) -> str:
        stats = data.get("statistics", {})
        ep = data.get("entry_point_summary", {})
        delta = data.get("reachability_delta", {})
        return (
            f"Spring wiring: {stats.get('total_beans', 0)} beans, "
            f"{stats.get('total_wiring', 0)} injection points. "
            f"Entry points: {ep.get('total', 0)} "
            f"(JMS:{ep.get('jms_listeners', 0)}, REST:{ep.get('rest_endpoints', 0)}, "
            f"Scheduled:{ep.get('scheduled', 0)}, Main:{ep.get('main_methods', 0)}). "
            f"Reachability improvement: {delta.get('improvement_ratio', 1)}x."
        )

    def _classify_bean(self, annotations: List[str]) -> Optional[str]:
        """Classify a class as a Spring bean type based on its annotations."""
        for ann in annotations:
            if ann in ("Service",):
                return "service"
            if ann in ("Component",):
                return "component"
            if ann in ("Repository",):
                return "repository"
            if ann in ("Controller", "RestController"):
                return "controller"
            if ann in ("Configuration",):
                return "configuration"
        return None

    def _get_parent_class(self, qualified_name: str) -> str:
        """Get parent class qualified name from method/field qualified name."""
        parts = qualified_name.rsplit(".", 1)
        return parts[0] if len(parts) > 1 else qualified_name

    def _count_by_type(self, beans: List[Dict[str, Any]]) -> Dict[str, int]:
        """Count beans by type."""
        counts: Dict[str, int] = {}
        for bean in beans:
            t = bean.get("type", "unknown")
            counts[t] = counts.get(t, 0) + 1
        return counts
