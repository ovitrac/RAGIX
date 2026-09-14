"""Row 26: every pass-1 abstract embedded once, deterministically, on the executor's own embedder.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-09-14

A fake Ollama stands in for the executor's: /api/tags lists the embedder, /api/ps shows nothing resident,
/api/embed returns a vector seeded by the text, so the same text always gets the same bytes. The script
must embed only abstracts that exist, write jsonl and npy in input order, resume without re-embedding,
pass a byte-identical check, send truncate false, and refuse a remote host, an absent embedder and a
vector that is not 1 024-d.
"""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import struct
from pathlib import Path

import numpy as np
import pytest

LAB = Path(__file__).resolve().parents[1]
SENT: list[dict] = []


def _load():
    from ragix_kernels.harvest import embed_abstracts
    return embed_abstracts


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _fake(dim=1024, tags=("snowflake-arctic-embed2:latest",)):
    def urlopen(request, timeout=None):
        url = request.full_url if hasattr(request, "full_url") else request
        if url.endswith("/api/tags"):
            return _Resp(json.dumps({"models": [{"name": t, "digest": "5de93a84837d" + "0" * 52} for t in tags]}).encode())
        if url.endswith("/api/ps"):
            return _Resp(json.dumps({"models": []}).encode())
        body = json.loads(request.data.decode("utf-8"))
        SENT.append(body)
        if not body.get("input"):                            # the release: keep_alive 0, no input
            return _Resp(b"{}")
        vecs = []
        for text in body["input"]:
            seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16) / 2**32
            vecs.append([seed + i / dim for i in range(dim)])
        return _Resp(json.dumps({"embeddings": vecs}).encode())
    return urlopen


def _abstracts(tmp_path):
    rows = [{"node_id": "w1", "level": "window", "ok": True, "abstract": "Une première phrase.", "source_sha256": "a" * 64},
            {"node_id": "w2", "level": "window", "ok": True, "abstract": "Une deuxième phrase.", "source_sha256": "b" * 64},
            {"node_id": "w3", "level": "window", "ok": False, "abstract": "", "refusal": "timeout"},
            {"node_id": "d1", "level": "document", "ok": True, "abstract": "Un document entier.", "source_sha256": "c" * 64}]
    path = tmp_path / "core_abstracts.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return path


def _run(module, monkeypatch, tmp_path, *extra, fake=None):
    monkeypatch.setattr(module.urllib.request, "urlopen", fake or _fake())
    return module.main(["--abstracts", str(_abstracts(tmp_path)), "--out", str(tmp_path / "out"), *extra])


def test_the_abstracts_that_exist_are_embedded_in_order(monkeypatch, tmp_path):
    SENT.clear()
    m = _load()
    assert _run(m, monkeypatch, tmp_path) == 0
    rows = [json.loads(l) for l in (tmp_path / "out/node_embeddings.jsonl").read_text().splitlines()]
    assert [r["node_id"] for r in rows] == ["w1", "w2", "d1"]
    assert np.load(tmp_path / "out/node_embeddings.npy").shape == (3, 1024)
    assert all(r["dim"] == 1024 and r["embedder"] == "snowflake-arctic-embed2:latest" for r in rows)
    assert any(b.get("input") and b.get("truncate") is False for b in SENT)


def test_a_resume_embeds_nothing_twice_and_the_check_is_byte_identical(monkeypatch, tmp_path):
    m = _load()
    assert _run(m, monkeypatch, tmp_path) == 0
    SENT.clear()
    assert _run(m, monkeypatch, tmp_path, "--resume", "--check", "3") == 0
    assert sum(len(b.get("input") or []) for b in SENT) == 3             # only the check's three re-embeddings
    check = json.loads((tmp_path / "out/check.json").read_text())
    assert check["identical"] == check["n"] == 3


@pytest.mark.parametrize("extra, fake, why", [
    (("--host", "http://203.0.113.9:11434"), None, "remote"),
    ((), _fake(tags=("nomic-embed-text:latest",)), "absent"),
    ((), _fake(dim=512), "1024"),
])
def test_refusals(monkeypatch, tmp_path, capsys, extra, fake, why):
    m = _load()
    with pytest.raises(SystemExit) as caught:
        _run(m, monkeypatch, tmp_path, *extra, fake=fake)
    assert caught.value.code == 2
    assert why in capsys.readouterr().err
