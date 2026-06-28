"""Ollama transport — local (http loopback) or remote (ssh -> loopback curl).

host=None/localhost  -> POST http://127.0.0.1:port/api/chat directly.
host="user@h"        -> ssh user@h 'curl 127.0.0.1:port/api/chat -d @-'  (loopback-bound,
                        nothing written to the remote disk; only the request body crosses).

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-06-28
"""
from __future__ import annotations
import json, subprocess, urllib.request

_LOCAL = {None, "", "local", "localhost", "127.0.0.1"}


def chat(messages, tools=None, *, model, host=None, port=11434,
         temperature=0.2, keep_alive="5m", timeout=300):
    payload = {"model": model, "messages": messages, "stream": False,
               "keep_alive": keep_alive, "options": {"temperature": temperature}}
    if tools:
        payload["tools"] = tools
    body = json.dumps(payload).encode()
    if host in _LOCAL:
        req = urllib.request.Request(f"http://127.0.0.1:{port}/api/chat", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            resp = json.loads(r.read().decode())
    else:
        p = subprocess.run(["ssh", "-o", "ConnectTimeout=10", host,
                            f"curl -s http://127.0.0.1:{port}/api/chat -d @-"],
                           input=body, capture_output=True, timeout=timeout)
        if p.returncode != 0:
            raise RuntimeError(f"ssh/curl failed: {p.stderr.decode()[:300]}")
        resp = json.loads(p.stdout.decode())
    if "error" in resp:
        raise RuntimeError(f"ollama error: {resp['error']}")
    return resp["message"]
