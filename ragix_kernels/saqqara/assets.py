"""
saqqara.assets — where a document's pictures live, which is not in its tree.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-08-28

Specified by K6.2, K6.3 and K6.6 in SPEC.md.

A tree says what a document is made of. Bytes of a picture are not that: they are large, they are
opaque, and putting them in the tree would make every tree that holds a figure expensive to move,
awkward to read and impossible to compare with one that does not. So the tree carries a hash and
the bytes live here, beside it.

**Identity comes from the source, never from a rendering.** An image is identified by the bytes the
file stores; a region of drawing, when P6 reaches it, by the operators that draw it together with
their extent. A raster produced from either is a *derived* artefact — cached under its own key of
source, renderer, version and resolution — and never part of what a tree is. That is what lets two
versions of a renderer disagree about pixels without disturbing the reproducibility K4 rests on.

**A missing asset is a failure, not an absence.** `read` raises rather than returning nothing, and
verifies that what it read still hashes to what it was asked for. A store that quietly returned the
wrong bytes would be worse than one that lost them.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

__all__ = [
    "AssetStore",
    "MissingAsset",
    "asset_id",
]


class MissingAsset(KeyError):
    """An asset a node refers to is absent, or is not what it claims to be."""


def asset_id(payload: bytes) -> str:
    """The identity of a source: the sha256 of its bytes, as hex."""
    return hashlib.sha256(payload).hexdigest()


class AssetStore:
    """Bytes addressed by their own hash, with a manifest of who refers to them.

    The manifest is not bookkeeping. Content addressing means one picture used in
    two places is stored once, and that saving would quietly become a claim —
    that the two places hold *the same* evidence — if nothing recorded that there
    were two of them. K2.12 already refuses to let a duplicate document count as
    independent corroboration; a shared picture is the same trap one level down,
    and the list of references is what keeps it visible.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._manifest_path = self.root / "manifest.json"

    # ------------------------------------------------------------------ write

    def put(self, payload: bytes, media_type: str, reference: dict[str, Any]) -> str:
        """Store bytes under their hash and record one reference to them.

        Returns the identity the tree will carry. Storing the same bytes twice
        writes the file once and appends a second reference.
        """
        digest = asset_id(payload)
        target = self.root / digest
        if not target.exists():
            target.write_bytes(payload)

        manifest = self.manifest()
        entry = manifest.setdefault(
            digest, {"media_type": media_type, "bytes": len(payload), "references": []}
        )
        entry["references"].append(reference)
        self._write(manifest)
        return digest

    # ------------------------------------------------------------------- read

    def read(self, digest: str) -> bytes:
        """The bytes for an identity. Raises if absent, or if they have changed."""
        target = self.root / digest
        if not target.is_file():
            raise MissingAsset(f"asset {digest[:8]} is not in the store at {self.root}")
        payload = target.read_bytes()
        found = asset_id(payload)
        if found != digest:
            raise MissingAsset(
                f"asset {digest[:8]} holds bytes that hash to {found[:8]}: "
                "the store no longer describes what it contains"
            )
        return payload

    def ids(self) -> list[str]:
        return sorted(self.manifest())

    def manifest(self) -> dict[str, dict[str, Any]]:
        if not self._manifest_path.is_file():
            return {}
        return json.loads(self._manifest_path.read_text(encoding="utf-8"))

    def verify(self, digests: Iterable[str]) -> list[str]:
        """Which of these identities the store cannot honour. Counted, never silent."""
        missing = []
        for digest in digests:
            try:
                self.read(digest)
            except MissingAsset:
                missing.append(digest)
        return missing

    # ---------------------------------------------------------------- private

    def _write(self, manifest: dict[str, dict[str, Any]]) -> None:
        self._manifest_path.write_text(
            json.dumps(manifest, indent=1, sort_keys=True, ensure_ascii=False),
            encoding="utf-8",
        )

    def __repr__(self) -> str:
        return f"<AssetStore {self.root} n={len(self.manifest())}>"
