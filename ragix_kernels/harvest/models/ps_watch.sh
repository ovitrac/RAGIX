#!/usr/bin/env bash
# Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-09-14
# Emits only when the SET of resident models changes (expiry ticks ignored). Read-only: GET /api/ps.
prev=""
while true; do
  cur=$(curl -s --max-time 5 http://localhost:11434/api/ps 2>/dev/null | python3 -c '
import json,sys
try: d=json.load(sys.stdin)
except Exception: sys.exit(0)
ms=d.get("models",[])
if not ms:
    print("(none)"); sys.exit(0)
for m in sorted(ms, key=lambda x: x.get("name","")):
    print("%s | size=%.1f GB | vram=%.1f GB | ctx=%s" % (
        m.get("name"), m.get("size",0)/2**30, m.get("size_vram",0)/2**30, m.get("context_length")))
' 2>/dev/null)
  if [ -n "$cur" ] && [ "$cur" != "$prev" ]; then
    echo "[$(date +%H:%M:%S)] RESIDENT: $cur"
    prev="$cur"
  fi
  sleep 5
done
