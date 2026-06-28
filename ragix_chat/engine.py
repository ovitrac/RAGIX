"""ChatEngine — multi-turn, tool-using conversation over a pluggable ContextBackend.

Mode-agnostic: the same engine runs CLEAR or SEALED backends. The system prompt is fully
customizable. Sealing (if any) is the backend's concern (finish()), not the engine's — so
the conversation is fluid, not forced to "seal every turn".

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-06-28
"""
from __future__ import annotations
import json
from .transport import chat as _chat

DEFAULT_SYSTEM = (
    "You are a careful analysis assistant working over a private document corpus. "
    "You can ONLY see documents through the tools. For each question: first call "
    "search_sources(query), then call open_context with the EXACT ticket_id string returned "
    "by search_sources (never a filename) to read the passages that look relevant, THEN write "
    "your answer. Ground every claim in passages you actually opened; if the corpus does not "
    "contain the answer, say so plainly rather than guessing. "
    "Make tool calls through the tool interface only — never write tool-call JSON in your reply. "
    "When you have read enough, reply with a concise, conversational answer naming the source "
    "document(s). You may make several tool calls per turn."
)


def extract_text_tool_calls(content):
    """Fallback (T1 JSON-action): some models emit tool calls as JSON in the reply text.

    Scan for top-level objects carrying a tool name + params/arguments and return
    [(name, args), ...]. Lets the engine drive models that don't use native tool_calls.
    """
    out, i, dec = [], 0, json.JSONDecoder()
    while i < len(content):
        if content[i] == "{":
            try:
                obj, end = dec.raw_decode(content[i:])
            except Exception:
                i += 1
                continue
            if isinstance(obj, dict):
                name = obj.get("name") or obj.get("tool") or obj.get("function")
                args = obj.get("parameters") or obj.get("arguments") or obj.get("args") or {}
                if isinstance(name, str) and isinstance(args, dict):
                    out.append((name, args))
            i += max(end, 1)
        else:
            i += 1
    return out


class ChatEngine:
    def __init__(self, backend, *, model, host=None, system_prompt=None,
                 temperature=0.3, max_tool_steps=8, verbose=False):
        self.backend = backend
        self.model = model
        self.host = host
        self.temperature = temperature
        self.max_tool_steps = max_tool_steps
        self.verbose = verbose
        self.system_prompt = system_prompt or DEFAULT_SYSTEM
        self.messages = [{"role": "system", "content": self.system_prompt}]
        self.tool_log = []

    def ask(self, user_text):
        """One user turn. Runs the tool loop internally; returns the assistant's text."""
        self.messages.append({"role": "user", "content": user_text})
        for _ in range(self.max_tool_steps):
            msg = _chat(self.messages, tools=self.backend.tools, model=self.model,
                        host=self.host, temperature=self.temperature)
            self.messages.append(msg)
            content = msg.get("content") or ""
            actions = [(c["function"]["name"], c["function"].get("arguments") or {})
                       for c in (msg.get("tool_calls") or [])]
            if not actions and content:
                actions = extract_text_tool_calls(content)     # T1 fallback for text tool calls
            if not actions:
                return content
            for fn, args in actions:
                res = self.backend.dispatch(fn, args)
                self.tool_log.append((fn, args, res))
                if self.verbose:
                    shown = {k: (f"<{len(v)} chars>" if isinstance(v, str) and k in ("text", "context")
                                 else v) for k, v in res.items()}
                    print(f"   · {fn}({json.dumps(args, ensure_ascii=False)[:90]}) "
                          f"-> {json.dumps(shown, ensure_ascii=False)[:150]}")
                self.messages.append({"role": "tool", "tool_name": fn,
                                      "content": json.dumps(res, ensure_ascii=False)})
        return "(reached tool-step limit without a final answer)"

    def reset(self):
        self.messages = self.messages[:1]
        self.tool_log = []
