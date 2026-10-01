#!/usr/bin/env python3
"""ace2_rollout.py — configs and readout for the ACE2 5-year rollout test.

Two arms (ai2's Hugging Face checkpoints), eight start dates each, one start date per
GPU (fme inference, no targets):
    era5_ai2     allenai/ACE2-ERA5  on our staged ERA5 (Oct 1,5,..,29 2001)
    eamv3_ai2    allenai/ACE2-EAMv3 driven by our E3SM-SRM archive
                 (build_eamv3_inputs.py; Oct 1,5,..,29 2044 = makani protocol frames)

Subcommands
    config   write one fme inference YAML for (arm, start index, steps, out dir)
    probe    read the prep job's probes: steps/s per arm, EAMv3 orientation verdict
    readout  per-run stability: first non-finite month, global-mean surface-pressure
             drift (hPa) from the monthly means fme writes

PASS tokens: ACE2_PROBE_OK, ACE2_5YR_READOUT_OK
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

M = "/lus/eagle/projects/lighthouse-uchicago/members/mehta5"
ERA5_DIR = "/eagle/projects/lighthouse-uchicago/ace2/ace_training"
ERA5_FILE = os.path.join(ERA5_DIR, "merged_ACE2_ERA5_final.nc")
ERA5_STARTS = [f"2001-10-{d:02d}T00:00:00" for d in range(1, 30, 4)]

# Per-arm variable names for the same physical quantities.
NAMES = {
    "era5": {"ps": "PRESsfc", "ts": "surface_temperature", "t7": "air_temperature_7",
             "q7": "specific_total_water_7", "u2": "eastward_wind_2"},
    "eamv3": {"ps": "PS", "ts": "TS", "t7": "T_7", "q7": "specific_total_water_7", "u2": "U_2"},
}
ARMS = ("era5_ai2", "eamv3_ai2")


def arm_family(arm: str) -> str:
    return "eamv3" if arm.startswith("eamv3") else "era5"


def write_config(a) -> None:
    fam = arm_family(a.arm)
    if fam == "era5":
        ic = {"path": ERA5_FILE, "start_indices": {"times": [ERA5_STARTS[a.start]]}}
        forcing = {"dataset": {"data_path": ERA5_DIR}, "num_data_workers": 2}
    else:
        ic = {"path": os.path.join(a.inputs, "ic", "ic.nc"), "start_indices": {"list": [a.start]}}
        forcing = {"dataset": {"data_path": os.path.join(a.inputs, "forcing")},
                   "num_data_workers": 2}
    names = list(NAMES[fam].values())
    cfg = {
        "experiment_dir": a.out,
        "n_forward_steps": a.steps,
        "forward_steps_in_memory": 50,
        "checkpoint_path": a.ckpt,
        "logging": {"log_to_screen": True, "log_to_wandb": False, "log_to_file": True,
                    "project": "ace2-5yr-rollout"},
        "initial_condition": ic,
        "forcing_loader": forcing,
        "allow_incompatible_dataset": bool(a.allow_incompatible),
        "data_writer": {"save_prediction_files": bool(a.save_predictions),
                        "save_monthly_files": not a.save_predictions,
                        "names": [NAMES[fam]["ps"]] if a.save_predictions else names},
        "aggregator": {"log_zonal_mean_images": False, "log_nino34_index": False},
    }
    import yaml
    os.makedirs(a.out, exist_ok=True)
    with open(a.yaml, "w") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)
    print(f"wrote {a.yaml}")


def _steps_per_second(log_path: str, n_steps: int) -> tuple[float | None, float | None]:
    """(stepping rate excluding init + final flush, init seconds) from fme's timer lines.

    fme's own "Total steps per second" divides by the whole run INCLUDING checkpoint and
    dataset initialization, which dominates a short probe; GlobalTimer.log_durations
    prints "<name> duration: Xs" for inference, inference/initialization and
    inference/final_writer_flush (fme/core/timing.py:201-203).
    """
    if not os.path.exists(log_path):
        return None, None
    dur = {}
    for line in open(log_path, errors="ignore"):
        m = re.search(r"(inference(?:/\w+)?) duration: ([0-9.]+)s", line)
        if m:
            dur[m.group(1)] = float(m.group(2))
    if "inference" not in dur:
        return None, None
    init = dur.get("inference/initialization", 0.0)
    stepping = dur["inference"] - init - dur.get("inference/final_writer_flush", 0.0)
    return (n_steps / stepping if stepping > 0 else None), init


def _area_weights(lat):
    import numpy as np
    w = np.cos(np.deg2rad(np.asarray(lat, dtype="f8")))
    return w / w.sum()


def probe(a) -> int:
    """steps/s per arm; EAMv3 orientation from 1..8-step PS error vs archive truth."""
    import numpy as np
    import xarray as xr
    res = {"steps_per_second": {}, "init_seconds": {}, "steps": {}, "orientation": None}
    ok = True
    budget = a.budget_min * 60.0
    for arm in ARMS:
        sps, init = _steps_per_second(os.path.join(a.root, f"probe_{arm}", "inference_out.log"),
                                      a.probe_steps)
        res["steps_per_second"][arm], res["init_seconds"][arm] = sps, init
        ok &= sps is not None
        if sps is None:
            print(f"probe {arm}: NO TIMING")
            continue
        # whole years that fit the walltime budget after init (+20% for the monthly writer)
        fit = (budget - (init or 0.0)) * sps / 1.2
        steps = min(a.target_steps, int(fit // 1460) * 1460)
        res["steps"][arm] = steps
        print(f"probe {arm}: {sps:.2f} steps/s, init {init:.0f}s -> {a.target_steps} steps "
              f"needs {a.target_steps / sps / 60 + (init or 0) / 60:.1f} min; plan {steps} steps"
              + ("" if steps == a.target_steps else "  TRUNCATED"))
        ok &= steps >= 1460
    rmse = {}
    for orient in ("asc", "flip"):
        pred_files = glob.glob(os.path.join(a.root, f"probe_orient_{orient}", "*prediction*.nc"))
        truth = os.path.join(a.root, f"inputs_{orient}", "truth", "truth_ps.nc")
        if not pred_files or not os.path.exists(truth):
            print(f"probe orientation {orient}: MISSING pred={pred_files} truth={truth}")
            ok = False
            continue
        p = xr.open_dataset(pred_files[0])["PS"].squeeze()
        t = xr.open_dataset(truth)["PS"].values
        pv = p.values.reshape(-1, t.shape[-2], t.shape[-1])
        if len(pv) == len(t) - 1:      # writer omitted the IC: prediction k is truth k+1
            t = t[1:]
        n = min(len(pv), len(t))
        w = _area_weights(xr.open_dataset(truth)["lat"].values)[:, None] / t.shape[-1]
        err = [float(np.sqrt((((pv[i] - t[i]) ** 2) * w).sum())) for i in range(n)]
        rmse[orient] = err
        print(f"probe orientation {orient}: PS rmse by step (Pa) = {[round(e, 1) for e in err]}")
    if len(rmse) == 2:
        # compare the last common step (the IC itself, step 0, is error-free by construction)
        k = min(len(rmse["asc"]), len(rmse["flip"])) - 1
        res["orientation"] = "asc" if rmse["asc"][k] < rmse["flip"][k] else "flip"
        res["orientation_rmse_last"] = {o: rmse[o][k] for o in rmse}
        print(f"ORIENTATION={res['orientation']} (rmse at step {k}: asc {rmse['asc'][k]:.1f}, "
              f"flip {rmse['flip'][k]:.1f} Pa)")
    with open(os.path.join(a.root, "probe.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    if ok and res["orientation"]:
        print("ACE2_PROBE_OK")
        return 0
    print("ERROR PROBE_INCOMPLETE")
    return 1


def readout(a) -> int:
    import numpy as np
    import xarray as xr
    rows = []
    for run in sorted(p for p in glob.glob(os.path.join(a.root, "run_*")) if os.path.isdir(p)):
        arm, start = re.match(r".*run_(\w+?)_s(\d+)$", run).groups()
        fam = arm_family(arm)
        files = glob.glob(os.path.join(run, "monthly_mean_predictions.nc"))
        if not files:
            rows.append({"arm": arm, "start": int(start), "status": "NO_OUTPUT"})
            continue
        ds = xr.open_dataset(files[0])
        lat_name = "lat" if "lat" in ds.dims else [d for d in ds.dims if "lat" in d][0]
        w = xr.DataArray(_area_weights(ds[lat_name].values), dims=[lat_name])
        out = {"arm": arm, "start": int(start)}
        for key in ("ps", "ts", "t7", "q7"):
            v = ds[NAMES[fam][key]]
            g = (v * w).sum(lat_name).mean([d for d in v.dims if "lon" in d]).squeeze()
            g = np.atleast_1d(g.values.astype("f8"))
            out[f"{key}_series"] = g.tolist()
        ps = np.asarray(out["ps_series"])
        finite = np.isfinite(ps)
        out["n_months"] = int(len(ps))
        out["first_nonfinite_month"] = None if finite.all() else int(np.argmin(finite))
        good = ps[finite]
        out["ps_drift_hpa"] = float((good[-1] - good[0]) / 100.0) if len(good) > 1 else None
        out["status"] = "FINITE" if finite.all() else "NONFINITE"
        rows.append(out)
    with open(os.path.join(a.root, "readout.json"), "w") as fh:
        json.dump(rows, fh, indent=1)
    print(f"{'arm':10s} {'start':>5s} {'status':>10s} {'months':>6s} {'1st_NaN_mo':>10s} {'dPS_hPa':>8s}")
    for r in rows:
        print(f"{r['arm']:10s} {r['start']:5d} {r['status']:>10s} {r.get('n_months', 0):6d} "
              f"{str(r.get('first_nonfinite_month')):>10s} "
              f"{(r.get('ps_drift_hpa') if r.get('ps_drift_hpa') is not None else float('nan')):8.2f}")
    expect = len(ARMS) * 8
    if len(rows) == expect and all(r["status"] != "NO_OUTPUT" for r in rows):
        print(f"ACE2_5YR_READOUT_OK runs={len(rows)}")
        return 0
    print(f"ERROR READOUT_INCOMPLETE runs={len(rows)}/{expect}")
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("config")
    c.add_argument("--arm", choices=ARMS, required=True)
    c.add_argument("--start", type=int, required=True, help="0..7")
    c.add_argument("--steps", type=int, required=True)
    c.add_argument("--ckpt", required=True)
    c.add_argument("--inputs", help="build_eamv3_inputs.py --out dir (eamv3 arm)")
    c.add_argument("--out", required=True)
    c.add_argument("--yaml", required=True)
    c.add_argument("--save-predictions", action="store_true", help="probe: per-step PS only")
    c.add_argument("--allow-incompatible", action="store_true")
    p = sub.add_parser("probe")
    p.add_argument("--root", required=True)
    p.add_argument("--probe-steps", type=int, default=120)
    p.add_argument("--target-steps", type=int, default=7300)
    p.add_argument("--budget-min", type=float, default=50.0,
                   help="rollout job's usable minutes per run (1 h debug-scaling walltime)")
    r = sub.add_parser("readout")
    r.add_argument("--root", required=True)
    a = ap.parse_args()
    return {"config": lambda: (write_config(a), 0)[1], "probe": lambda: probe(a),
            "readout": lambda: readout(a)}[a.cmd]()


if __name__ == "__main__":
    sys.exit(main())
