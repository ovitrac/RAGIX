# harvest/models — the Granite 4.2 Modelfiles and the server monitors

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-09-14

Not Python: this directory has no `__init__.py`, so the kernel registry never walks it.

## The Granite 4.2 Modelfiles

`granite4.2` is published on the Ollama registry **without a chat template**: its manifests carry
the model, its licence and its parameters, and Ollama falls back to `{{ .Prompt }}`. The model is
trained on ChatML with thinking on by default, so fed raw text it opens a think block of its own, and
under a JSON grammar the constrained content comes back empty. The fix is a template, not a prompt.

| file | what it builds |
|---|---|
| `granite42/Modelfile.8b-chat` | `granite4.2:8b-chat` — `FROM granite4.2:8b` with the ChatML template |
| `granite42/Modelfile.30b-chat` | `granite4.2:30b-chat` — `FROM granite4.2:30b` with the ChatML template |
| `granite42/granite42_chatml.tmpl` | the template alone |

    ollama create granite4.2:8b-chat -f granite42/Modelfile.8b-chat
    ollama create granite4.2:30b-chat -f granite42/Modelfile.30b-chat

The template follows the ChatML convention (`<|im_start|>role … <|im_end|>`) and keeps Ollama's
`.IsThinkSet` / `.Think` / `.Thinking` handling. It is derived from the ChatML template Ollama ships
for Qwen3, with the two Qwen-specific parts removed (the `/think` and `/no_think` suffixes, and the
tools block, since Granite 4.2's tool syntax is different). With `think: false` it prefills an empty
think block, so the model cannot open one of its own; the built tag then declares the `completion`
and `thinking` capabilities and no `tools`. `runner.call_node` sends `think` only to a model whose
`/api/show` declares `thinking`.

A tag built by grafting the Granite 4.0/4.1 template onto 4.2 is wrong on three counts — role
tokens, stop token and tool syntax — and is not provided.

## The monitors

| file | what it does | reads |
|---|---|---|
| `ps_watch.sh` | prints the set of resident models each time it changes | `GET /api/ps` on localhost |
| `conc_watch.py` | fires the first time N model calls overlap in time | `journalctl -u ollama` |
| `ollama_journal.sh` | per-model load facts, errors, and per-call outcomes over a time window | `journalctl -u ollama` |
| `manifest.py` | prints the layers of a registry manifest read on stdin | stdin |

All four are read-only. They are how the resident memory and the real concurrency of a job are
measured rather than assumed: a server configured for one call at a time queues the others, and a
pool of workers then changes the wall clock and nothing else.
