#!/usr/bin/env python3
"""Per-job training rate before vs after each job's FIRST epoch boundary.

    python per_job_rates.py /work/nvme/bdiu/krucker/ace_zero_co2/run4/logs/errors_*.err

Question it answers: is the slowdown tied to the epoch NUMBER (only the run's
epoch 1 is fast) or to the PROCESS (every job, including resumed ones, is fast
until its own first epoch boundary, then slow)? Read-only, pure stdlib.

Per log, one line:
    JOB <file> first_epoch=<E> pre=<median rate before boundary> (n)
        post=<median rate after> (n) ratio=<post/pre> recompile_msgs=<count>
followed by up to 3 dynamo recompile/limit messages verbatim.
"""
import re
import statistics as st
import sys

STEP = re.compile(r"Step (\d+): .*'training_samples_per_second_on_rank_0': ([0-9.eE+-]+)")
EPOCH_END = re.compile(r"Time taken for epoch (\d+) is")
DYNAMO = re.compile(r"recompile_limit|cache_size_limit|[Rr]ecompiling function|torch\._dynamo hit")


def scan(path):
    pre, post, dyn = [], [], []
    first_epoch = None
    for line in open(path, errors="ignore"):
        m = EPOCH_END.search(line)
        if m and first_epoch is None:
            first_epoch = int(m.group(1))
            continue
        m = STEP.search(line)
        if m:
            (pre if first_epoch is None else post).append(float(m.group(2)))
        if DYNAMO.search(line):
            dyn.append(line.strip()[:200])

    def med(v):
        return f"{st.median(v):.3f}" if v else "NA"

    ratio = f"{st.median(post) / st.median(pre):.3f}" if pre and post else "NA"
    print(f"JOB {path} first_epoch={first_epoch} pre={med(pre)} (n={len(pre)}) "
          f"post={med(post)} (n={len(post)}) ratio={ratio} recompile_msgs={len(dyn)}")
    for d in dyn[:3]:
        print(f"  DYNAMO {d}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for p in sys.argv[1:]:
        scan(p)
