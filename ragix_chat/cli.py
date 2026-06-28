"""ragix-chat CLI — interactive (or scripted) chat over a CLEAR document corpus.

  python -m ragix_chat.cli --docs <dir> --model <m> [--host user@h] [--system file] [--ask Q]

CLEAR mode only (plaintext). The SEALED backend lives with the confidential data and is
launched from there, reusing ChatEngine. Multi-turn: stdin lines are successive turns.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-06-28
"""
from __future__ import annotations
import argparse, sys
from .engine import ChatEngine
from .backends import ClearBackend


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", required=True, help="directory of .md/.txt documents")
    ap.add_argument("--model", default="mistral:latest")
    ap.add_argument("--host", default=None, help="ssh host for remote ollama (default: localhost)")
    ap.add_argument("--system", help="file with a custom system prompt")
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--ask", help="one-shot question (non-interactive)")
    ap.add_argument("--verbose", action="store_true", help="show tool calls")
    args = ap.parse_args(argv)

    system = open(args.system, encoding="utf-8").read() if args.system else None
    backend = ClearBackend(args.docs)
    eng = ChatEngine(backend, model=args.model, host=args.host, system_prompt=system,
                     temperature=args.temperature, verbose=args.verbose)
    print(f"[ragix-chat CLEAR] {len(backend.docs)} docs, {len(backend.chunks)} chunks, "
          f"model={args.model} host={args.host or 'localhost'}")

    if args.ask:
        print(f"\nyou> {args.ask}\nbot> {eng.ask(args.ask)}")
        return 0
    for line in sys.stdin:
        q = line.strip()
        if q in ("", "quit", "exit"):
            break
        print(f"\nyou> {q}")
        print(f"bot> {eng.ask(q)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
