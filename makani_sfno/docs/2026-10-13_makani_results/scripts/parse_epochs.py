"""Extract makani per-epoch summary blocks from a training log (bytes-safe, rank-0 lines only).

usage: python3 parse_epochs.py <log> [keys...]   -> TSV: line, epoch, then each key
"""
import re
import sys

KV = re.compile(r"INFO:root:\s+([A-Za-z][^:]*?):\s+([-+0-9.eE]+|nan|inf)\s*$")
EP = re.compile(r"INFO:root:Epoch (\d+) summary:")

path = sys.argv[1]
want = sys.argv[2:] or ["training steps", "epoch time [s]", "training loss", "validation loss"]
rows, cur = [], None
with open(path, "rb") as fh:
    for n, raw in enumerate(fh, 1):
        line = raw.decode("utf-8", "replace").rstrip("\n")
        if " 0: " not in line[:40]:
            continue
        m = EP.search(line)
        if m:
            if cur:
                rows.append(cur)
            cur = {"_line": n, "_epoch": int(m.group(1))}
            continue
        if cur is None:
            continue
        if "-----" in line and len(cur) > 3:
            rows.append(cur)
            cur = None
            continue
        k = KV.search(line)
        if k:
            cur.setdefault(k.group(1).strip(), k.group(2))
if cur:
    rows.append(cur)
print("line\tepoch\t" + "\t".join(want))
for r in rows:
    print("%d\t%d\t%s" % (r["_line"], r["_epoch"], "\t".join(r.get(k, "") for k in want)))
