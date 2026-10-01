#!/usr/bin/env python3
"""Score an fme (ACE) training run for a one-time slowdown after epoch 1.

    python parse_steptest.py <experiment_dir> [<experiment_dir> ...]

Reads <experiment_dir>/metrics/*.jsonl (written when the config sets
logging.metrics_log_dir) and, if present, <experiment_dir>/out.log. Pure stdlib.

R = median training_samples_per_second_on_rank_0 over epoch 2
    / median over the SECOND HALF of epoch 1 (skips warmup / compile).
    (Polaris job 7671383 pooled epochs 2-3; epoch 3 is now reported separately
    as a RATIO line, because arm A runs it in a relaunched process.)
Registered thresholds (same as Polaris job 7671383):
    R <= 0.77 -> REPRODUCED   R >= 0.91 -> NO_STEP   otherwise INCONCLUSIVE
"""
import glob
import json
import os
import statistics as st
import sys


def score(exp):
    name = os.path.basename(os.path.normpath(exp))
    rows = []
    for f in glob.glob(os.path.join(exp, "metrics", "*.jsonl")):
        rows += [json.loads(line) for line in open(f) if line.strip()]
    rows.sort(key=lambda r: r.get("step", 0))
    ep_rows = [r for r in rows if "epoch_train_seconds" in r]
    bounds = [r["step"] for r in ep_rows]

    def epoch_of(step):
        for i, b in enumerate(bounds):
            if step <= b:
                return i + 1
        return len(bounds) + 1

    rate = {}
    for r in rows:
        if "training_samples_per_second_on_rank_0" in r and "epoch_train_seconds" not in r:
            rate.setdefault(epoch_of(r["step"]), []).append(r["training_samples_per_second_on_rank_0"])
    for r in ep_rows:
        print(f"EPOCH {name} e{r['epoch']} train_s={r['epoch_train_seconds']:.0f} "
              f"val_s={r.get('epoch_validation_seconds', 0):.0f} "
              f"inf_s={r.get('epoch_inference_seconds', 0):.0f} total_s={r.get('epoch_total_seconds', 0):.0f}")
    for e, v in sorted(rate.items()):
        print(f"RATE {name} e{e} n={len(v)} median={st.median(v):.3f} samples/s/rank0")
    log = os.path.join(exp, "out.log")
    if os.path.exists(log):
        for line in open(log, errors="ignore"):
            if "reclaim]" in line:
                print(f"MEM {name} {line.strip()[-150:]}")
    val = [r.get("epoch_validation_seconds") for r in ep_rows if r.get("epoch_validation_seconds")]
    if len(val) >= 2:
        print(f"VAL_RATIO {name} later/e1 = {st.median(val[1:]) / val[0]:.3f}")
    e1 = rate.get(1, [])
    late1 = e1[len(e1) // 2:]
    after = rate.get(2, [])
    if late1:
        for e in sorted(k for k in rate if k >= 2):
            print(f"RATIO {name} e{e}/late_e1 = {st.median(rate[e]) / st.median(late1):.3f}")
    if late1 and after:
        R = st.median(after) / st.median(late1)
        verdict = "REPRODUCED" if R <= 0.77 else ("NO_STEP" if R >= 0.91 else "INCONCLUSIVE")
        print(f"STEPTEST_RESULT run={name} late_e1={st.median(late1):.3f} e2={st.median(after):.3f} "
              f"R={R:.3f} {verdict} epochs_done={len(ep_rows)}")
    else:
        print(f"ERROR STEPTEST_NO_DATA run={name} epochs_done={len(ep_rows)} rate_epochs={sorted(rate)}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for d in sys.argv[1:]:
        score(d)
