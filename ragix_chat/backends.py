"""Context backends — pluggable document sources for the chat engine.

A backend exposes `tools` (Ollama tool schemas) + `dispatch(name, args)`. The engine is
mode-agnostic: CLEAR (plaintext, for development/quality assessment) and SEALED (confidential)
backends share this interface, so the SAME conversation engine drives both.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-06-28
"""
from __future__ import annotations
import math, os, re, secrets
from abc import ABC, abstractmethod

WORD = re.compile(r"[a-zA-Zàâçéèêëîïôûùüÿñæœ0-9]+")


def tokenize(s):
    return WORD.findall(s.lower())


def chunk_text(text):
    """Content blocks with exact char offsets (c0, c1, block)."""
    chunks, pos = [], 0
    for block in re.split(r"(\n\n+)", text):
        if block and not block.startswith("\n") and block.strip():
            chunks.append((pos, pos + len(block), block))
        pos += len(block)
    return chunks or [(0, len(text), text)]


class BM25:
    def __init__(self, docs_tokens, k1=1.5, b=0.75):
        self.docs = docs_tokens
        self.k1, self.b = k1, b
        self.avg = sum(len(d) for d in self.docs) / max(1, len(self.docs))
        self.df = {}
        for d in self.docs:
            for t in set(d):
                self.df[t] = self.df.get(t, 0) + 1
        self.N = len(self.docs)

    def search(self, query, top=5):
        q = tokenize(query)
        out = []
        for i, d in enumerate(self.docs):
            if not d:
                continue
            s, dl = 0.0, len(d)
            for t in q:
                df = self.df.get(t)
                if not df:
                    continue
                f = d.count(t)
                if not f:
                    continue
                idf = math.log(1 + (self.N - df + 0.5) / (df + 0.5))
                s += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avg))
            if s > 0:
                out.append((i, s))
        out.sort(key=lambda x: -x[1])
        return out[:top]


def tool_schema(name, desc, props, req):
    return {"type": "function", "function": {"name": name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": req}}}


class ContextBackend(ABC):
    @property
    @abstractmethod
    def tools(self):
        ...

    @abstractmethod
    def dispatch(self, name, args):
        ...

    def finish(self):
        """Hook for sealed backends to seal derivatives. No-op for clear."""
        return None


class ClearBackend(ContextBackend):
    """Plaintext backend (DEVELOPMENT/QUALITY mode). No sealing, no budget — readable output.

    docs: dict {name: text}  OR  a directory of .md/.txt files.
    """
    def __init__(self, docs):
        if isinstance(docs, str):
            docs = self._load_dir(docs)
        self.docs = docs
        self.chunks = []                       # (doc_name, idx, c0, c1, block)
        self._docchunks = {}
        for name, text in docs.items():
            cl = chunk_text(text)
            self._docchunks[name] = cl
            for i, (c0, c1, blk) in enumerate(cl):
                self.chunks.append((name, i, c0, c1, blk))
        self.bm25 = BM25([tokenize(c[4]) for c in self.chunks])
        self.tickets = {}

    @staticmethod
    def _load_dir(d):
        out = {}
        for root, _, files in os.walk(d):
            for f in files:
                if f.endswith((".md", ".txt")):
                    p = os.path.join(root, f)
                    out[os.path.relpath(p, d)] = open(p, encoding="utf-8", errors="replace").read()
        return out

    @property
    def tools(self):
        return [
            tool_schema("search_sources",
                        "Search the document corpus; returns ranked snippets with ticket ids.",
                        {"query": {"type": "string"}}, ["query"]),
            tool_schema("open_context",
                        "Read the full text behind a ticket id from search_sources. expansion: none|small|medium.",
                        {"ticket_id": {"type": "string"}, "expansion": {"type": "string"}}, ["ticket_id"]),
        ]

    def dispatch(self, name, args):
        try:
            if name == "search_sources":
                return self._search(**(args or {}))
            if name == "open_context":
                return self._open(**(args or {}))
            return {"error": f"unknown tool {name}"}
        except Exception as e:
            return {"error": str(e)[:160]}

    def _search(self, query="", **_):
        out = []
        for rank, (ci, score) in enumerate(self.bm25.search(query), 1):
            name, idx, c0, c1, blk = self.chunks[ci]
            tid = "tkt_" + secrets.token_hex(6)
            self.tickets[tid] = ci
            out.append({"ticket_id": tid, "rank": rank, "doc": name,
                        "snippet": blk.strip().replace("\n", " ")[:120]})
        return {"tickets": out}

    def _open(self, ticket_id=None, expansion="none", doc=None, **_):
        # primary path: a ticket from search_sources
        if ticket_id in self.tickets:
            name, idx, *_r = self.chunks[self.tickets[ticket_id]]
        elif doc and doc in self._docchunks:                  # forgiving fallback: by doc name
            name, idx = doc, 0
        elif doc:                                             # fuzzy doc match
            cand = [n for n in self._docchunks if doc.lower() in n.lower()]
            if not cand:
                return {"error": f"no ticket '{ticket_id}' and no doc matching '{doc}'"}
            name, idx = cand[0], 0
        else:
            return {"error": "invalid ticket_id (use a ticket from search_sources, or pass doc=<name>)"}
        dc = self._docchunks[name]
        depth = {"none": 0, "small": 1, "medium": 2}.get(expansion, 1)
        lo, hi = max(0, idx - depth), min(len(dc) - 1, idx + depth)
        text = "\n\n".join(dc[j][2] for j in range(lo, hi + 1))
        return {"doc": name, "text": text}
