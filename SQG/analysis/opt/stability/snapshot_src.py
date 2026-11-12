"""snapshot_src.py -- freeze the code a queued job will import (SQG/analysis minus results/, SQG/fixed_ring, and
o1.2's own file) into <dest>, with MANIFEST.sha256 and the git commit, so edits to the worktree between qsub and job
start cannot change the job. Standard library only (run from the repo root on a login node).

  python3 SQG/analysis/opt/stability/snapshot_src.py <dest>
  qsub -v STAB_SRC=<dest> ... run_stability_polaris.pbs          (or O12_SRC=<dest> ... run_o12_tilerows_polaris.pbs)
"""
import hashlib
import os
import shutil
import subprocess
import sys

TREES = ("SQG/analysis", "SQG/fixed_ring")
FILES = ("SQG/o1.2-TrainUViT_m16_tendency.py",)                       # o12run/o12_polaris.py reads it from SQG/
SKIP_DIRS = {"results", "__pycache__"}


def sources():
    for root in TREES:
        for dp, dn, fn in os.walk(root):
            dn[:] = sorted(d for d in dn if d not in SKIP_DIRS)
            for f in sorted(fn):
                yield os.path.join(dp, f)
    for f in FILES:
        if os.path.exists(f):
            yield f


def main():
    dest = os.path.abspath(sys.argv[1])
    if os.path.exists(dest):
        raise SystemExit(f"ERROR SNAPSHOT_EXISTS {dest}")
    lines = []
    for src in sources():
        dst = os.path.join(dest, src)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        lines.append(f"{hashlib.sha256(open(src, 'rb').read()).hexdigest()}  {src}")
    head = subprocess.run(["git", "-c", "core.preloadIndex=false", "rev-parse", "HEAD"], stdout=subprocess.PIPE,
                          universal_newlines=True).stdout.strip()             # python 3.6 (the login node's python3)
    with open(os.path.join(dest, "MANIFEST.sha256"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(dest, "GIT_HEAD"), "w") as f:
        f.write(head + "\n")
    print(f"SNAPSHOT_OK {dest} files={len(lines)} head={head}")


if __name__ == "__main__":
    main()
