#!/usr/bin/env bash
# Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-09-14
# Harvest per-model load facts and per-call outcomes from the ollama journal.
# Usage: harvest.sh "HH:MM" ["HH:MM"]   (read-only; journalctl only)
SINCE="${1:?since HH:MM required}"; UNTIL="${2:-now}"
J() { journalctl -u ollama --since "$SINCE" --until "$UNTIL" --no-pager 2>/dev/null; }

echo "=== LOADS (model, ctx, buffers, runner start) ==="
J | grep -E 'general\.name +=|llama_context: n_ctx +=|load_tensors: +CUDA0 model buffer|load_tensors: offloaded|llama runner started in|system memory|requested context size too large' \
  | sed -E 's/^[A-Za-z]{3} [0-9]+ ([0-9:]+).*ollama\[[0-9]+\]: /\1 /' | uniq

echo
echo "=== ERRORS / WARNINGS / TRUNCATION ==="
J | grep -iE 'level=(ERROR|WARN)|[Tt]runcat|[Tt]imeout|panic|OOM|out of memory|context canceled|no slots|server busy|does not support|exceed' \
  | grep -v 'source=routes.go:1742' \
  | sed -E 's/^[A-Za-z]{3} [0-9]+ ([0-9:]+).*ollama\[[0-9]+\]: /\1 /' | uniq
echo "(empty above = none)"

echo
echo "=== MODEL CALLS (status, duration) ==="
J | grep -E '\[GIN\].*(api/chat|api/generate|api/embed)' \
  | sed -E 's/.*\| +([0-9]{3}) \| +([0-9.]+[a-zµ]+) *\|.*\| +([A-Z]+) +"(.*)"/\1  \2  \3 \4/' | uniq -c

echo
echo "=== NON-2xx CALLS (detail) ==="
J | grep -E '\[GIN\].*\| *[45][0-9][0-9] \|' | sed -E 's/^[A-Za-z]{3} [0-9]+ ([0-9:]+).*ollama\[[0-9]+\]: /\1 /'
echo "(empty above = none)"
