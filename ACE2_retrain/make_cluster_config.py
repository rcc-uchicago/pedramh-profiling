"""Rewrite ACE2's data paths for another cluster.

WHY A SCRIPT AND NOT --override: three of the training data paths live inside a
`concat:` LIST, and OmegaConf's dotlist cannot index into lists --
`train_loader.dataset.concat.0.data_path=...` fails with
"Cannot merge DictConfig with ListConfig". So the paths have to be rewritten in
the file. This is the same substitution used to make config_midway.yaml, kept as
a script so nobody hand-edits 11 paths and misses one.

Usage (on Delta, once you know where your copy of the data lives):

    python make_cluster_config.py \\
        --data-root /work/nvme/bdiu/jlandsberg \\
        --experiment-dir $SCRATCH/ace2_profile \\
        --out config_delta.yaml

`--data-root` must contain `ace_training/` and `normalization/`, mirroring the
layout of the original staging directory. The script prints every path it wrote,
checks whether each exists, and refuses to emit a half-retargeted config.
"""

import argparse
import os
import re
import sys

SRC_ROOT = "/project/pedramh/shared/ACE2_retrain"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="config_nsight.yaml")
    ap.add_argument("--data-root", required=True,
                    help="directory containing ace_training/ and normalization/")
    ap.add_argument("--experiment-dir", required=True, help="where the run writes output")
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep-wandb", action="store_true",
                    help="leave log_to_wandb as-is (default: turn it off)")
    a = ap.parse_args()

    text = open(a.source).read()
    n = text.count(SRC_ROOT)
    if n == 0:
        sys.exit(f"ERROR {a.source} contains no {SRC_ROOT} paths -- already retargeted?")

    text = text.replace(SRC_ROOT, a.data_root.rstrip("/"))
    text = re.sub(r"^experiment_dir: .*$", f"experiment_dir: {a.experiment_dir}",
                  text, count=1, flags=re.M)
    if not a.keep_wandb:
        text = text.replace("log_to_wandb: true", "log_to_wandb: false")

    if SRC_ROOT in text:
        sys.exit(f"ERROR output still contains {SRC_ROOT} -- refusing to write it half-retargeted")
    open(a.out, "w").write(text)
    print(f"wrote {a.out} ({n} data paths retargeted)")

    missing = []
    for m in sorted(set(re.findall(r"(/\S+\.nc|/\S+/ace_training)\b", text))):
        ok = os.path.exists(m)
        print(f"  {'ok     ' if ok else 'MISSING'} {m}")
        if not ok:
            missing.append(m)
    if missing:
        print(f"\nWARNING {len(missing)} path(s) not found on this host. Fine if you "
              "are generating the config off the compute node; otherwise fix --data-root.")


if __name__ == "__main__":
    main()
