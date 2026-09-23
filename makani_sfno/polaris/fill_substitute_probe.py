"""What does the BASE 101-channel model tolerate in place of its soil inputs?

Peer review of the port F surgical transfer, 2026-09-23. Slicing input columns
8,9 out of the encoder is EXACTLY feeding the base model z = 0 there, at every
pixel. But SOILWATER_10CM / TSOI_10CM are bimodal: over the fill region (62-72 %
of cells) the input the base always saw is the fill constant, not z = 0. So a
surgical-transfer FAIL could mean "z = 0 is out of distribution", not "the
transfer does not work". This probe separates the two with no training:

    base     inputs untouched                            (the reference)
    z0       channels 8,9 := 0 in z-space                (== the sliced model, step 0)
    zfill    channels 8,9 := their fill constant, z-space (0 kg/m2, 270 K)
    tmean    channels 8,9 := their per-pixel time mean   (stats/time_means.npy)

Why a loader substitution and not a folded checkpoint: makani's encoder first
layer has a bias (a constant input folds into it), but EncoderDecoder's final
layer and big_skip's residual_transform are both bias=False, so the residual
path's constant W_r[:, 8:10] @ c has nowhere to go. Only c = 0 is exactly
representable by slicing; zfill and tmean would need the loader to keep feeding
two frozen channels (the "keep 100 inputs, drop 2 outputs" design).

Metric: single-step lat-weighted MSE in normalized units over the 99 KEPT output
channels, same ICs for every variant. Read out as a ratio to `base`.

PRE-REGISTERED before any number (2026-09-23), per variant:
    STRONG  ratio <= 1.25   WEAK  ratio <= 2.0   FAIL  otherwise
`z0`'s ratio predicts job 7646690's step-0 validation loss.

    python fill_substitute_probe.py --run-dir <run> --ckpt <best_ckpt_mp0.tar> \
        --data <pack>/valid --pack <pack> --n-samples 64 --out <dir>
      -> FILL_SUBSTITUTE_PROBE_OK ...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

DROP = (8, 9)                      # SOILWATER_10CM, TSOI_10CM in pack order
FILL_PHYS = {8: 0.0, 9: 270.0}     # convert_e3sm_to_makani_alldata.py NaN fills
STRONG, WEAK = 1.25, 2.0


def tier(ratio: float) -> str:
    if not np.isfinite(ratio):
        return "FAIL"
    return "STRONG" if ratio <= STRONG else ("WEAK" if ratio <= WEAK else "FAIL")


def main() -> int:
    import torch

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--ckpt", required=True, type=Path)
    p.add_argument("--data", required=True, type=Path, help="split dir, e.g. <pack>/valid")
    p.add_argument("--pack", required=True, type=Path, help="pack root (stats/, metadata/)")
    p.add_argument("--n-samples", type=int, default=64)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()

    from sfno_eval import metrics as M
    from sfno_inference.checkpoint_loader import build_wrapper_from_checkpoint, load_eval_params
    from sfno_training.trainer.plasim_trainer import _plasim_get_dataloader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ep = load_eval_params(a.run_dir, K=1)
    names = list(ep.channel_names)
    if [names[i] for i in DROP] != ["SOILWATER_10CM", "TSOI_10CM"]:
        print(f"ERROR CHANNEL_ORDER_MISMATCH: positions {DROP} are {[names[i] for i in DROP]}")
        return 2
    wrapper = build_wrapper_from_checkpoint(ep, a.ckpt, device=device)
    wrapper.eval()
    pre = wrapper.preprocessor
    _, ds, _ = _plasim_get_dataloader(ep, str(a.data), device, mode="eval")

    in_bias = np.asarray(ds.in_bias, np.float64).reshape(-1)
    in_scale = np.asarray(ds.in_scale, np.float64).reshape(-1)
    tm = np.load(a.pack / "stats" / "time_means.npy")[0]            # (C, H, W) physical
    subs = {
        "base": None,
        "z0": {c: 0.0 for c in DROP},
        "zfill": {c: (FILL_PHYS[c] - in_bias[c]) / in_scale[c] for c in DROP},
        "tmean": {c: torch.from_numpy(((tm[c] - in_bias[c]) / in_scale[c]).astype(np.float32))
                  for c in DROP},
    }
    keep = [c for c in range(len(names)) if c not in DROP]
    idxs = np.linspace(0, len(ds) - 1, a.n_samples).round().astype(int).tolist()

    sums = {k: 0.0 for k in subs}
    w = None
    for i in idxs:
        inp_state, tar, inp_forc, tar_forc = (torch.as_tensor(t) for t in ds[i])
        for name, sub in subs.items():
            x = inp_state.clone()
            for c, v in (sub or {}).items():
                x[:, c] = v
            gdata = tuple(t.unsqueeze(0).to(device) for t in (x, tar, inp_forc, tar_forc))
            with torch.inference_mode():
                inp, tar_z = pre.cache_unpredicted_features(*gdata)
                inp = pre.flatten_history(inp)
                pred = wrapper(inp).float()
                truth = pre.flatten_history(tar_z[:, :1]).float()
            if w is None:
                w = M.lat_weights(pred.shape[-2], "equiangular").to(pred.device, torch.float32)
                w = (w / w.mean()).reshape(1, 1, -1, 1)
            err = ((pred - truth)[:, keep] ** 2 * w).mean().item()
            sums[name] += err
    mse = {k: v / len(idxs) for k, v in sums.items()}
    out = {k: {"mse99": mse[k], "ratio": mse[k] / mse["base"], "tier": tier(mse[k] / mse["base"])}
           for k in subs}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "fill_substitute_probe.json").write_text(json.dumps(
        {"n_samples": len(idxs), "data": str(a.data), "ckpt": str(a.ckpt),
         "tiers": {"strong": STRONG, "weak": WEAK}, "result": out}, indent=2) + "\n")
    for k, r in out.items():
        print(f"{k:6s} mse99 {r['mse99']:.5f}  ratio {r['ratio']:.3f}  {r['tier']}")
    print(f"FILL_SUBSTITUTE_PROBE_OK n={len(idxs)} z0_tier={out['z0']['tier']} "
          f"out={a.out}/fill_substitute_probe.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
