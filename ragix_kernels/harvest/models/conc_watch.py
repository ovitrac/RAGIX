#!/usr/bin/env python3
"""One-shot: fire the first time N model calls overlap in time, inferred from Ollama's GIN completion lines.
A GIN line gives the completion time (1 s resolution) and the duration; start = end - duration. Read-only.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-09-14

Usage: conc_watch.py SINCE N LIMIT   (journalctl --since SINCE; fire at N overlapping calls; give up after LIMIT s)
"""
import re, subprocess, sys, time, datetime as dt
SINCE, N, LIMIT = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])   # since, overlap target, give-up seconds
PAT = re.compile(r'\[GIN\] (\d{4}/\d\d/\d\d - \d\d:\d\d:\d\d) \| (\d{3}) \| +([0-9.]+(?:µs|ms|s|m[0-9.]*s|h[0-9.m]*s)) +\|.*POST +"(/api/(?:generate|chat))"')
def secs(d):
    m = re.fullmatch(r'(?:(\d+)h)?(?:(\d+)m)?([\d.]+)(µs|ms|s)', d)
    if not m: return None
    h, mi, v, u = m.groups(); v = float(v)
    v = v/1e6 if u == 'µs' else v/1e3 if u == 'ms' else v
    return (int(h or 0))*3600 + (int(mi or 0))*60 + v
t0 = time.time()
while True:
    out = subprocess.run(['journalctl', '-u', 'ollama', '--since', SINCE, '--no-pager', '-o', 'cat'],
                         capture_output=True, text=True).stdout
    calls = []
    for m in PAT.finditer(out):
        end = dt.datetime.strptime(m.group(1), '%Y/%m/%d - %H:%M:%S').timestamp()
        d = secs(m.group(3))
        if d is None or d < 0.5: continue          # skip keep_alive releases
        calls.append((end - d, end, m.group(2), d))
    ev = sorted([(s, 1) for s, e, *_ in calls] + [(e, -1) for s, e, *_ in calls])
    cur = best = 0; at = None
    for t, k in ev:
        cur += k
        if cur > best: best, at = cur, t
    if best >= N or time.time() - t0 > LIMIT:
        now = time.time()
        recent = [c for c in calls if c[1] >= now - 300]
        stat = sorted({c[2] for c in calls})
        print(f"{dt.datetime.now().isoformat(timespec='seconds')} max_overlap={best} "
              f"first_at={dt.datetime.fromtimestamp(at).strftime('%H:%M:%S') if at else None} "
              f"calls={len(calls)} statuses={stat} last5min={len(recent)} ({len(recent)/5:.1f}/min) "
              f"dur_median={sorted(c[3] for c in calls)[len(calls)//2] if calls else None}"
              + ("" if best >= N else "  (TIMED OUT before reaching the target)"))
        break
    time.sleep(10)
