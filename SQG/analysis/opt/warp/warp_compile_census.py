"""warp_compile_census.py -- Warp's cold-compile cost per job, from the 'Module <name> <hash> load on device ...
took X ms (compiled|cached)' lines every SQG result log already holds (WARP_SKILLS_REVIEW_2026-11-19.md, part 1).
Standard library only; run from the repo root:  python3 SQG/analysis/opt/warp/warp_compile_census.py"""
import collections
import os
import re

ROOT = "SQG/analysis/results"
PAT = re.compile(r"Module (\S+) (\w+) load on device '([^']+)' took ([\d.]+) ms\s+\((\w+)\)")
per_job = collections.OrderedDict()
names = collections.defaultdict(set)
for d in sorted(os.listdir(ROOT)):
    p = os.path.join(ROOT, d)
    if not os.path.isdir(p):
        continue
    for dp, _, fn in os.walk(p):
        for f in fn:
            if not (f.endswith(".log") or f.endswith(".err") or f.endswith(".OU")):
                continue
            try:
                txt = open(os.path.join(dp, f), errors="replace").read()
            except OSError:
                continue
            for m in PAT.finditer(txt):
                mod, h, dev, ms, how = m.groups()
                base = re.sub(r"_[0-9a-f]{8}$", "", mod)
                j = per_job.setdefault(d, collections.defaultdict(list))
                j[(base, how)].append(float(ms))
                names[base].add(h)
print("| job dir | module | how | loads | median ms | max ms | total s |")
print("|---|---|---|---|---|---|---|")
tot_by_job = {}
for d, j in per_job.items():
    for (base, how), v in sorted(j.items()):
        v = sorted(v)
        print(f"| {d} | {base} | {how} | {len(v)} | {v[len(v)//2]:.0f} | {v[-1]:.0f} | {sum(v)/1000:.1f} |")
        tot_by_job[d] = tot_by_job.get(d, 0) + sum(v) / 1000
print()
print("total Warp module-load seconds per job dir (summed over workers):")
for d, s in tot_by_job.items():
    print(f"  {d}: {s:.1f} s")
print()
print("distinct hashes per module name (all jobs):")
for b, hs in sorted(names.items()):
    print(f"  {b}: {len(hs)} {sorted(hs)[:8]}")
