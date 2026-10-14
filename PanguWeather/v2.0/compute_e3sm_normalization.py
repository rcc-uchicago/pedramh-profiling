#!/usr/bin/env python
"""Compute E3SM normalization statistics that AGREE with the mask fills.

Replaces the shipped ``data_2015-2050_{mean,std_corr}.nc``, whose soil-temperature
entry was computed under a 0-fill while training fills 270 — leaving that channel
at 0.12 normalized spread instead of 1.0 and ~67x under-weighted in the loss.

**The rule this tool enforces:** the statistics must describe *exactly the array
the model receives*, i.e. the field AFTER the mask fill is substituted. Computing
them over the raw valid data instead (the natural-looking near-miss) gives soil
temperature a spread of 0.62 — better, still wrong.

Two stages, because the expensive one only ever needs running once:

``--stage moments``
    One streaming pass over the archive accumulating, per variable and per level,
    the count/sum/sum-of-squares of the **valid (non-NaN)** cells plus the masked
    cell count. Written to JSON. ~2 TB of reads.

``--stage nc``
    Derives mean/std for ANY fill from that JSON, analytically and exactly::

        mean  = (sum + n_masked * f) / n_total
        E[x2] = (sumsq + n_masked * f^2) / n_total
        std   = sqrt(E[x2] - mean^2)

    So changing a fill value is instant and needs no re-read. Emits the same
    schema as the shipped files (per-variable arrays over ``Z_2`` for upper-air,
    scalars for the rest), so they drop straight in.

Numerics: sums are accumulated in float64 about a per-variable shift constant
(the first file's valid mean). Naive sum-of-squares over ~3e9 cells of a field
like Z3 (up to 37,250 m) overflows float64's exact-integer range and silently
loses precision; shifting keeps the accumulands small.

Usage::

    # once, on a compute node (~2 TB of reads)
    python compute_e3sm_normalization.py --stage moments \
        --h5-dir $E3SM_ROOT/h5/plev_data --years 2015 2050 \
        --config config/E3SM_SFNO_H5_POLARIS.yaml \
        --out moments_2015-2050.json

    # then, instantly, for whichever fills you want
    python compute_e3sm_normalization.py --stage nc \
        --moments moments_2015-2050.json --out-dir . \
        --fill SST=-1.8 --fill TSOI_10CM=270
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np
import xarray as xr

# Upper-air names are level-suffixed in the h5; everything else is a flat key.
UPPER_AIR = ["T", "U", "V", "Z3", "RELHUM", "CLDICE", "CLDLIQ", "CLOUD"]
FLAT = [
    "TREFHT", "U10", "RHREFHT", "PS", "PSL", "TMQ",
    "SOILWATER_10CM", "TSOI_10CM",
    "FSNTOA", "FSNT", "PRECT",
    "PCT_GLACIER", "PFTDATA_MASK", "PCT_NATVEG", "TOPO",
    "SST", "ICE", "sol_in",
]
# Only a TRULY constant channel gets its std replaced — that is what the shipped
# "_std_corr" corrected (16 zero-std cloud levels in the raw file).
#
# ⚠ Do NOT raise this to something like 1e-6 "for safety": PRECT's real std is
# ~8.3e-8 (it is in m/s), so a 1e-6 floor would clobber a legitimate value to 1.0
# and silently destroy precipitation normalization. Small std != degenerate std.
STD_ZERO = 1e-30
# Reported (not altered) when a channel's spread is negligible against its own
# magnitude — informative, since such a channel carries almost no signal.
STD_REL_WARN = 1e-6


def parse_levels(config: Path) -> list[float]:
    text = config.read_text()
    key = "sigma_levels" if re.search(r"^\s*use_sigma_levels:\s*True", text, re.M) else "levels"
    m = re.search(rf"^\s*{key}:\s*(\[.*?\])", text, re.M | re.S)
    if not m:
        raise KeyError(f"no {key} in {config}")
    return [float(x) for x in ast.literal_eval(m.group(1))]


def parse_mask_fill(config: Path) -> dict[str, float]:
    m = re.search(r"^\s*mask_fill:\s*\{(.*?)\}", config.read_text(), re.M | re.S)
    return {k: float(v) for k, v in re.findall(r"'([^']+)':\s*([-\d.eE]+)", m.group(1))} if m else {}


def _level_key(group, prefix: str, level: float) -> str:
    exact = f"{prefix}_{level}"
    if exact in group:
        return exact
    best, best_d = None, float("inf")
    for k in group:
        if k.startswith(prefix + "_"):
            try:
                d = abs(float(k[len(prefix) + 1:]) - level)
            except ValueError:
                continue
            if d < best_d:
                best, best_d = k, d
    if best is None:
        raise KeyError(f"{prefix} @ {level}")
    return best


def _accumulate_one(path: str, levels: list[float], shifts: dict) -> dict:
    """Per-file partial moments about the per-variable shift constants."""
    out: dict[str, list] = {}
    with h5py.File(path, "r") as f:
        g = f["input"]
        for var in FLAT:
            if var not in g:
                continue
            a = np.asarray(g[var][:], dtype="float64")
            out[var] = [_moments(a, shifts[var])]
        for var in UPPER_AIR:
            per_level = []
            for lev in levels:
                try:
                    key = _level_key(g, var, lev)
                except KeyError:
                    per_level.append(None)
                    continue
                a = np.asarray(g[key][:], dtype="float64")
                per_level.append(_moments(a, shifts[var]))
            if any(p is not None for p in per_level):
                out[var] = per_level
    return out


def _moments(a: np.ndarray, shift: float) -> list:
    """(n_total, n_valid, sum(x-shift), sum((x-shift)^2)) over valid cells."""
    m = ~np.isnan(a)
    v = a[m] - shift
    return [int(a.size), int(m.sum()), float(v.sum()), float((v * v).sum())]


def _merge(dst: dict, src: dict) -> None:
    for var, levels in src.items():
        if var not in dst:
            dst[var] = [None if p is None else list(p) for p in levels]
            continue
        for i, p in enumerate(levels):
            if p is None:
                continue
            if dst[var][i] is None:
                dst[var][i] = list(p)
            else:
                for j in range(4):
                    dst[var][i][j] += p[j]


def stage_moments(a: argparse.Namespace) -> int:
    levels = parse_levels(a.config)
    lo, hi = a.years
    files = sorted(
        str(p) for p in a.h5_dir.iterdir()
        if re.match(r"^\d{4}_\d{4}\.h5$", p.name) and lo <= int(p.name[:4]) < hi
    )
    if a.limit:
        files = files[:: max(1, len(files) // a.limit)][:a.limit]
    if not files:
        print(f"ERROR NO_FILES in {a.h5_dir} for [{lo}, {hi})")
        return 1
    print(f"moments over {len(files)} files, years [{lo}, {hi}), {len(levels)} levels", flush=True)

    # Shift constants: the first file's valid mean per variable. Only needs to be
    # roughly right — it exists to keep the squared accumulands small.
    shifts: dict[str, float] = {}
    with h5py.File(files[0], "r") as f:
        g = f["input"]
        for var in FLAT:
            if var in g:
                x = np.asarray(g[var][:], dtype="float64")
                shifts[var] = float(np.nanmean(x)) if np.isfinite(np.nanmean(x)) else 0.0
        for var in UPPER_AIR:
            try:
                x = np.asarray(g[_level_key(g, var, levels[len(levels) // 2])][:], dtype="float64")
                shifts[var] = float(np.nanmean(x))
            except KeyError:
                shifts[var] = 0.0

    n_workers = a.workers or min(32, (os.cpu_count() or 8))
    acc: dict = {}
    done = 0
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = {ex.submit(_accumulate_one, p, levels, shifts): p for p in files}
        for fut in as_completed(futs):
            _merge(acc, fut.result())
            done += 1
            if done % 2000 == 0 or done == len(files):
                print(f"  {done}/{len(files)}", flush=True)

    a.out.write_text(json.dumps({
        "levels": levels, "shifts": shifts, "n_files": len(files),
        "years": [lo, hi], "source": str(a.h5_dir), "moments": acc,
    }))
    print(f"MOMENTS_OK {a.out}  ({len(acc)} variables)")
    return 0


def stage_nc(a: argparse.Namespace) -> int:
    blob = json.loads(a.moments.read_text())
    levels, shifts, acc = blob["levels"], blob["shifts"], blob["moments"]

    fills = dict(parse_mask_fill(a.config)) if a.config else {}
    for kv in a.fill or []:
        k, _, v = kv.partition("=")
        fills[k.strip()] = float(v)

    mean_vars, std_vars, warnings = {}, {}, []
    for var, per_level in acc.items():
        shift = shifts[var]
        mus, sds = [], []
        for p in per_level:
            if p is None:
                mus.append(np.nan); sds.append(np.nan); continue
            n_tot, n_val, s, ss = p
            n_msk = n_tot - n_val
            if n_msk and var not in fills:
                warnings.append(f"{var}: {100*n_msk/n_tot:.2f}% masked but NO fill given -> using 0.0")
            f = float(fills.get(var, 0.0)) - shift          # fill, in shifted space
            mu = (s + n_msk * f) / n_tot
            e2 = (ss + n_msk * f * f) / n_tot
            var_ = max(e2 - mu * mu, 0.0)
            sd = math.sqrt(var_)
            mu_phys = mu + shift
            if sd <= STD_ZERO:
                warnings.append(f"{var}: std is ZERO (constant channel) -> set to 1.0")
                sd = 1.0
            elif abs(mu_phys) > 0 and sd / abs(mu_phys) < STD_REL_WARN:
                # Kept as-is: a small std can be entirely correct (PRECT ~8.3e-8).
                warnings.append(f"{var}: std {sd:.3e} is {sd/abs(mu_phys):.1e} of |mean| — "
                                f"near-constant channel, KEPT (not floored)")
            mus.append(mu + shift)                           # back to physical units
            sds.append(sd)
        if len(per_level) == 1:
            mean_vars[var] = xr.DataArray(np.float32(mus[0]))
            std_vars[var] = xr.DataArray(np.float32(sds[0]))
        else:
            mean_vars[var] = xr.DataArray(np.asarray(mus, "float32"), dims=("Z_2",))
            std_vars[var] = xr.DataArray(np.asarray(sds, "float32"), dims=("Z_2",))

    coords = {"Z_2": np.asarray(levels, "float64")}
    attrs = {"fills_used": json.dumps({k: v for k, v in sorted(fills.items())}),
             "source": blob["source"], "n_files": blob["n_files"],
             "years": f"{blob['years'][0]}-{blob['years'][1]}",
             "note": "stats computed over the FILLED field (fill applied before moments)"}
    a.out_dir.mkdir(parents=True, exist_ok=True)
    tag = a.tag or f"{blob['years'][0]}-{blob['years'][1]}"
    mp = a.out_dir / f"data_{tag}_mean.nc"
    sp = a.out_dir / f"data_{tag}_std_corr.nc"
    xr.Dataset(mean_vars, coords=coords, attrs=attrs).to_netcdf(mp)
    xr.Dataset(std_vars, coords=coords, attrs=attrs).to_netcdf(sp)

    for w in dict.fromkeys(warnings):
        print(f"  WARN {w}")
    print(f"  fills used: {attrs['fills_used']}")
    for v in ("SST", "TSOI_10CM"):
        if v in mean_vars:
            print(f"  {v:10s} mean {float(mean_vars[v]):10.4f}  std {float(std_vars[v]):10.4f}")
    print(f"NORM_NC_OK {mp.name} + {sp.name}  ({len(mean_vars)} variables, Z_2={len(levels)})")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--stage", required=True, choices=["moments", "nc"])
    p.add_argument("--h5-dir", type=Path)
    p.add_argument("--years", type=int, nargs=2, default=[2015, 2050], metavar=("LO", "HI"))
    p.add_argument("--config", type=Path, help="PanguWeather YAML: supplies levels + mask_fill.")
    p.add_argument("--out", type=Path, help="moments JSON (stage=moments)")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--limit", type=int, default=0, help="Subsample to N files (testing only).")
    p.add_argument("--moments", type=Path, help="moments JSON (stage=nc)")
    p.add_argument("--out-dir", type=Path, default=Path("."))
    p.add_argument("--fill", action="append", metavar="VAR=VALUE",
                   help="Override/add a fill, e.g. --fill SST=-1.8. Repeatable.")
    p.add_argument("--tag", help="Filename tag; defaults to the year range.")
    a = p.parse_args(argv)

    if a.stage == "moments":
        for req in ("h5_dir", "config", "out"):
            if getattr(a, req) is None:
                p.error(f"--{req.replace('_','-')} is required for --stage moments")
        return stage_moments(a)
    if a.moments is None:
        p.error("--moments is required for --stage nc")
    return stage_nc(a)


if __name__ == "__main__":
    sys.exit(main())
