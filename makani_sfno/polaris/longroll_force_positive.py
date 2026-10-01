"""Port A: jesswan's long rollout, with ACE2-style `force_positive_names` clamping.

polaris_makani_ace2_ports_handoff.md §2. The harness is NOT re-implemented: this
imports `sfno_inference_e3sm.longroll` from jesswan's checkout, unmodified, and
swaps its `build_wrapper` for one whose forward clamps the named output channels
at physical 0 (z = -mean/std). The clamped prediction is what the loop records
AND what `append_history` feeds back -- ACE2's corrector semantics.

The reference is her job 7603141 (IC 2044-10-01 00Z, --max-steps 1460, fp32, base
prod1n_b32_sgdr): TREFHT global mean went unphysical near step 469 and the
prediction was NON-FINITE at step 628. The `none` arm must reproduce that first.

PRE-REGISTERED 2026-09-23 (operator asked for A; before any number):
  control  arm `none` must diverge at 628 +/- 5 steps, else the environment has
           drifted and the other arms are not comparable (CONTROL_MISMATCH).
  read-out per clamped arm, D = divergence step (1461 = ran clean):
           CLAMP_FIXES       D > 1460 or D >= 2 x 628
           CLAMP_NO_EFFECT   |D - 628| <= 0.10 x 628  -> reservoir-drift chain is NOT
                                                          the cause; retract corrector §4
           CLAMP_PARTIAL     anything else
  ⚠ Which channels are non-negative by physics is jesswan's call. These lists are a
  DIAGNOSTIC choice; a result is not a shippable fix until she confirms them.

    python longroll_force_positive.py --arm moist -- <longroll args...>
      -> FORCE_POSITIVE_ARM_OK arm=<a> diverged_at=<D> ...
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

JESSWAN_SRC = Path("/eagle/projects/lighthouse-uchicago/members/jesswan/"
                   "pedramh-profiling/makani_sfno/src")
REF_DIVERGED_AT = 628
ARMS = {
    "none": [],
    "soil": ["SOILWATER_10CM"],
    "moist": ["SOILWATER_10CM", "TMQ", "RHREFHT", "PRECT"] + [f"RELHUM_l{i:02d}" for i in range(18)],
}


def classify(d: int, ref: int = REF_DIVERGED_AT) -> str:
    if d > 1460 or d >= 2 * ref:
        return "CLAMP_FIXES"
    if abs(d - ref) <= 0.10 * ref:
        return "CLAMP_NO_EFFECT"
    return "CLAMP_PARTIAL"


class ClampWrapper:
    """Callable stand-in for the makani wrapper: forward, then clamp named channels."""

    def __init__(self, wrapper, idx, floor_z):
        self._w, self.idx, self.floor_z = wrapper, idx, floor_z
        self.n_clamped = 0

    def __call__(self, x):
        import torch
        pred = self._w(x)
        if not self.idx:
            return pred
        sub = pred[:, self.idx]
        fz = self.floor_z.to(pred.dtype)
        self.n_clamped += int((sub < fz).sum())
        out = pred.clone()
        out[:, self.idx] = torch.maximum(sub, fz)
        return out

    def __getattr__(self, name):
        return getattr(self._w, name)


def main() -> int:
    import numpy as np

    argv = sys.argv[1:]
    if "--" not in argv or argv[0] != "--arm":
        print("usage: longroll_force_positive.py --arm {none,soil,moist} -- <longroll args>")
        return 2
    arm = argv[1]
    names = ARMS[arm]
    sys.path.append(str(JESSWAN_SRC))      # AFTER ours: our sfno_training wins
    from sfno_inference_e3sm import longroll as L

    holder = {}
    orig = L.build_wrapper

    def build_wrapper(params, ckpt, device):
        import torch
        w = orig(params, ckpt, device)
        mean = np.load(params.global_means_path).reshape(-1)
        std = np.load(params.global_stds_path).reshape(-1)
        idx = [L.ALL_101.index(n) for n in names]
        floor = torch.tensor(-mean[idx] / std[idx], dtype=torch.float32,
                             device=device).reshape(1, -1, 1, 1)
        L.logger.info("force_positive arm=%s: %s", arm, names)
        holder["w"] = ClampWrapper(w, idx, floor)
        return holder["w"]

    L.build_wrapper = build_wrapper
    sys.argv = [sys.argv[0]] + argv[argv.index("--") + 1:]
    args = L.parse_args()
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    rc = L.run(args)

    import xarray as xr
    ds = xr.open_zarr(str(args.out_root / f"{args.run_name}__6hourly_subset.zarr"))
    d = int(ds.attrs.get("diverged_at_step", 0)) or 1461
    lat = np.deg2rad(ds["lat"].values.astype(float))
    w = np.cos(lat)[:, None] / np.cos(lat).sum() / ds.sizes["lon"]
    traj = {}
    for v in ("TREFHT", "SOILWATER_10CM"):
        if v in ds:
            a = ds[v].isel(member=0).values
            traj[v] = [float(np.nansum(a[k] * w)) for k in range(a.shape[0])]
    t0 = traj["TREFHT"][0]
    onset = next((k + 1 for k, x in enumerate(traj["TREFHT"]) if not np.isfinite(x) or abs(x - t0) > 50),
                 None)
    sw = traj.get("SOILWATER_10CM", [])
    out = {"arm": arm, "names": names, "diverged_at": d, "trefht_onset_step": onset,
           "n_clamped_cells": holder["w"].n_clamped, "longroll_rc": rc,
           "verdict": "CONTROL" if arm == "none" else classify(d),
           "soilwater_global_mean": {str(k + 1): sw[k] for k in range(0, len(sw), 50)},
           "soilwater_min_step_mean": float(np.nanmin(sw)) if sw else None}
    (args.out_root / f"force_positive_{arm}.json").write_text(json.dumps(out, indent=2) + "\n")
    if arm == "none" and abs(d - REF_DIVERGED_AT) > 5:
        print(f"ERROR CONTROL_MISMATCH diverged_at={d} vs reference {REF_DIVERGED_AT}")
        return 3
    print(f"FORCE_POSITIVE_ARM_OK arm={arm} diverged_at={d} onset={onset} "
          f"clamped_cells={holder['w'].n_clamped} verdict={out['verdict']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
