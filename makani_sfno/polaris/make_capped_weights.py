"""Ports B+C as ONE config-only change: an explicit, capped per-channel loss-weight list.

Architect review §3. `temp_diff_normalization: True` has two traps on this pack
(handoff §3): its 1e-4 clamp is in PHYSICAL units and would weight PRECT ~0.0008x,
and uncapped it hands Z3_l17 ~9518x. makani also accepts an explicit weight list
(loss.py:160-164), used AS-IS -- so compute the tendency weighting offline, with a
DIMENSIONLESS cap, and write it where a reviewer can read it:

    w_c = clip(sigma_c / delta_c, 1, W_max);   w <- w / sum(w)

Normalized to sum 1 because makani's `constant` is ones/C (base_loss.py): equal
r_c across channels reproduces `constant` exactly, so the arm differs from the
baseline in the weighting and nothing else. ⚠ The list must be NESTED, [[...]]:
loss.py asserts on chw.shape[1]. temp_diff_normalization stays False.

This PRODUCES a candidate; it enables nothing. It is a loss change => jesswan's
sign-off before any arm is quoted (CLAUDE.md, division of labour).

    python make_capped_weights.py --pack <root> --config e3sm_alldata_nosoil.yaml \
        --root-key e3sm_alldata_nosoil --w-max 30 --out <dir>
      -> CAPPED_WEIGHTS_OK ... ; writes capped_weights_w<W>.yaml + .csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


def capped_weights(sigma, delta, w_max: float) -> np.ndarray:
    sigma, delta = np.asarray(sigma, float), np.asarray(delta, float)
    if np.any(delta <= 0) or not np.all(np.isfinite(delta)):
        raise ValueError("DEGENERATE_DELTA: a channel has a zero or non-finite tendency std")
    w = np.clip(sigma / delta, 1.0, w_max)
    return w / w.sum()


def main() -> int:
    import yaml

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pack", required=True, type=Path)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--root-key", required=True)
    p.add_argument("--w-max", type=float, default=30.0)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()

    pack_names = json.loads((a.pack / "metadata" / "data.json").read_text())["coords"]["channel"]
    sigma = np.load(a.pack / "stats" / "global_stds.npy").ravel()
    delta = np.load(a.pack / "stats" / "time_diff_stds.npy").ravel()
    names = yaml.safe_load(a.config.read_text())[a.root_key]["channel_names"]
    missing = [n for n in names if n not in pack_names]
    if missing:
        print(f"ERROR CHANNEL_ORDER_MISMATCH: {missing} not in the pack")
        return 2
    idx = [pack_names.index(n) for n in names]      # the config's (subset) order
    w = capped_weights(sigma[idx], delta[idx], a.w_max)

    a.out.mkdir(parents=True, exist_ok=True)
    tag = f"w{a.w_max:g}"
    with (a.out / f"capped_weights_{tag}.csv").open("w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["channel", "sigma_c", "delta_c", "sigma_over_delta", "weight", "weight_x_C"])
        for n, i, wi in zip(names, idx, w):
            wr.writerow([n, f"{sigma[i]:.6g}", f"{delta[i]:.6g}", f"{sigma[i] / delta[i]:.6g}",
                         f"{wi:.6g}", f"{wi * len(w):.4f}"])
    snippet = ("    # capped tendency weights, W_max=%g, from make_capped_weights.py -- NOT ENABLED\n"
               "    # (loss change: needs sign-off). Order = channel_names. Nested on purpose.\n"
               "    losses:\n    -   type: \"l2\"\n        channel_weights: [[%s]]\n"
               "        temp_diff_normalization: !!bool False\n"
               "        parameters:\n            squared: !!bool True\n"
               % (a.w_max, ", ".join(f"{x:.6g}" for x in w)))
    (a.out / f"capped_weights_{tag}.yaml").write_text(snippet)
    rel = w * len(w)
    top = int(np.argmax(w))
    n_cap = int(np.sum(sigma[idx] / delta[idx] >= a.w_max))
    print(f"{len(w)} channels; relative weight (x C) {rel.min():.3f}..{rel.max():.3f}, "
          f"span {rel.max() / rel.min():.1f}x; top {names[top]}; {n_cap} channels at the cap; "
          f"PRECT x C = {rel[names.index('PRECT')]:.3f}" if "PRECT" in names else "")
    print(f"CAPPED_WEIGHTS_OK out={a.out}/capped_weights_{tag}.yaml")
    return 0


if __name__ == "__main__":
    sys.exit(main())
