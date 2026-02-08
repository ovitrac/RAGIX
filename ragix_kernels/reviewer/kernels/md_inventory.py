"""
Kernel: md_inventory
Stage: 1 (Collection)

File stats, SHA-256 hash, front-matter detection, code fence count, table count.
Creates immutable snapshot of the input document.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-02-06
"""

import hashlib
import re
from pathlib import Path
from typing import Any, Dict, List

from ragix_kernels.base import Kernel, KernelInput

import logging

logger = logging.getLogger(__name__)


class MdInventoryKernel(Kernel):
    """Document intake and fingerprint for Markdown review."""

    name = "md_inventory"
    version = "1.0.0"
    category = "reviewer"
    stage = 1
    description = "File stats, SHA-256, front-matter detection"

    requires: List[str] = []
    provides: List[str] = ["file_stats", "doc_hash"]

    def validate_input(self, input: KernelInput) -> List[str]:
        errors = super().validate_input(input)
        doc_path = input.config.get("doc_path", "")
        if not doc_path:
            errors.append("Missing required config: doc_path")
        elif not Path(doc_path).exists():
            errors.append(f"Document not found: {doc_path}")
        return errors

    def compute(self, input: KernelInput) -> Dict[str, Any]:
        doc_path = Path(input.config["doc_path"])
        text = doc_path.read_text(encoding="utf-8")
        lines = text.splitlines()

        # SHA-256
        file_hash = f"sha256:{hashlib.sha256(text.encode('utf-8')).hexdigest()}"

        # Basic stats
        line_count = len(lines)
        size_bytes = len(text.encode("utf-8"))

        # Front matter detection (YAML: starts with ---)
        has_frontmatter = False
        frontmatter_end = 0
        if lines and lines[0].strip() == "---":
            for i in range(1, len(lines)):
                if lines[i].strip() == "---":
                    has_frontmatter = True
                    frontmatter_end = i + 1
                    break

        # Code fence count
        code_fence_count = 0
        in_fence = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("```") or stripped.startswith("~~~"):
                if not in_fence:
                    code_fence_count += 1
                    in_fence = True
                else:
                    in_fence = False

        # Table count (heuristic: separator lines with |---|)
        table_sep_re = re.compile(r"^\|?[\s:]*-[-:|\s]*\|")
        table_count = sum(1 for line in lines if table_sep_re.match(line.strip()))

        # Save immutable snapshot
        stage_dir = input.workspace / "stage1"
        stage_dir.mkdir(parents=True, exist_ok=True)
        snapshot_path = stage_dir / "doc.raw.md"
        snapshot_path.write_text(text, encoding="utf-8")

        logger.info(
            f"[md_inventory] {doc_path.name}: {line_count} lines, "
            f"{size_bytes} bytes, {code_fence_count} code fences, "
            f"{table_count} tables"
        )

        return {
            "doc_path": str(doc_path),
            "doc_name": doc_path.name,
            "file_hash": file_hash,
            "line_count": line_count,
            "size_bytes": size_bytes,
            "has_frontmatter": has_frontmatter,
            "frontmatter_end_line": frontmatter_end,
            "code_fence_count": code_fence_count,
            "table_count": table_count,
            "snapshot_path": str(snapshot_path),
        }

    def summarize(self, data: Dict[str, Any]) -> str:
        return (
            f"Document {data['doc_name']}: {data['line_count']} lines, "
            f"{data['size_bytes']} bytes, {data['code_fence_count']} code fences, "
            f"{data['table_count']} tables, "
            f"frontmatter={'yes' if data['has_frontmatter'] else 'no'}"
        )
