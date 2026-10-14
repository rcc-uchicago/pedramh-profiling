"""Re-take a K=56 read-out on a channel SUBSET, from the per-channel curves already scored.

Port F (`polaris_makani_ace2_ports_handoff.md` §6a) trains a 99-channel model; every
existing baseline is a median over 101 channels, and a median moves with the channel
set alone. This re-takes the old baseline on the common channels BEFORE any post-F
number exists, so the comparison cannot be tuned after the fact.

It is a re-take, not a re-score: `k56_metrics.h5` (score_rollout_nc.py) holds the
`(K, C)` mean-over-IC curves, and the read-out is `readout_from_curves` +
`classify_regime` -- imported, not re-implemented, so the rule cannot drift.

Self-check first: the FULL channel set must reproduce the published
`k56_readout.json` to 1e-9, or the h5 is not what the JSON was computed from and
the restricted number would mean nothing.

    python restrict_readout.py --scores-dir <.../scores> --drop SOILWATER_10CM TSOI_10CM
      -> RESTRICT_READOUT_OK ... ; writes k56_readout_common<N>.json beside the inputs
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_rollout_nc import classify_regime, readout_from_curves  # noqa: E402

READOUT_KEYS = ("nrmse126", "nrmse336", "vr126", "vr336", "acc126", "acc336",
                "slope_early_per_h", "slope_late_per_h", "r_slope", "max_channel_nrmse336")


def restrict(lead_h, chan, nrmse, vr, acc, drop):
    """Read-out over `chan` minus `drop`. Raises on a name not in `chan`."""
    chan = [str(c) for c in chan]
    unknown = [d for d in drop if d not in chan]
    if unknown:
        raise ValueError(f"UNKNOWN_CHANNEL: {unknown} not in the scored channel list")
    keep = [i for i, c in enumerate(chan) if c not in set(drop)]
    r = readout_from_curves(lead_h, nrmse[:, keep], vr[:, keep], acc[:, keep])
    return r, classify_regime(r), [chan[i] for i in keep]


def max_abs_diff(a: dict, b: dict) -> float:
    d = 0.0
    for k in READOUT_KEYS:
        x, y = a.get(k), b.get(k)
        if x is None or y is None:
            continue
        if math.isnan(x) and math.isnan(y):
            continue
        d = max(d, abs(x - y))
    return d


def main() -> int:
    import h5py

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scores-dir", required=True, type=Path,
                   help="dir with k56_metrics.h5 and k56_readout.json (score_rollout_nc.py)")
    p.add_argument("--drop", nargs="+", required=True)
    args = p.parse_args()

    with h5py.File(args.scores_dir / "k56_metrics.h5", "r") as f:
        lead_h = f["lead_hours"][()]
        chan = [c.decode() if isinstance(c, bytes) else str(c) for c in f["channel"][()]]
        nrmse, vr, acc = (f[f"{m}_mean"][()] for m in ("nrmse", "vr", "acc"))
    published = json.loads((args.scores_dir / "k56_readout.json").read_text())

    full, _, _ = restrict(lead_h, chan, nrmse, vr, acc, [])
    diff = max_abs_diff(full, published["readout"])
    if diff > 1e-9:
        print(f"ERROR READOUT_NOT_REPRODUCED: full-set re-take differs from k56_readout.json "
              f"by {diff:.3g} -- the h5 is not what the JSON was computed from")
        return 2

    sub, verdict, kept = restrict(lead_h, chan, nrmse, vr, acc, args.drop)
    out = args.scores_dir / f"k56_readout_common{len(kept)}.json"
    out.write_text(json.dumps(
        {"readout": sub, "verdict": verdict, "dropped": list(args.drop),
         "n_channels": len(kept), "full_set_readout": full,
         "source": str(args.scores_dir / "k56_metrics.h5"),
         "provenance": published.get("provenance", ""),
         "prereg": published.get("prereg", "")},
        indent=2, sort_keys=True) + "\n")
    print(f"full {len(chan)} reproduced (max|diff| {diff:.1e}); restricted to {len(kept)}:")
    for k in ("nrmse126", "nrmse336", "vr336", "acc126", "acc336", "max_channel_nrmse336"):
        print(f"  {k:22s} {full[k]:.4f} -> {sub[k]:.4f}")
    print(f"  verdict {published['verdict']['branch']} -> {verdict['branch']}")
    print(f"RESTRICT_READOUT_OK n={len(kept)} out={out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
