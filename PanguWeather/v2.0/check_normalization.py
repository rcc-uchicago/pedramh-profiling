#!/usr/bin/env python
"""Gate a PanguWeather normalization stats pair against the archive it normalizes.

Catches the whole "fill and stats disagree" defect class in one invariant:

    for every channel, normalizing the FILLED data must give mean ~0, std ~1.

That is not a style preference — it is what the stats are *for*. When a channel
comes out at std 0.12 (as TSOI_10CM does today), its squared error is ~67x
smaller than its siblings' in an element-mean MSE loss, so the objective barely
sees it; and as an input it arrives nearly flat.

Three checks per variable:

1. **scale**    normalized std within [--std-tol] of 1.0
2. **centre**   normalized |mean| within [--mean-tol] of 0.0
3. **fill sanity**  the fill value lies inside the valid data's range. A fill
   outside it (SST=270 on a degC field measuring [-1.8, 33.6]) is a units bug
   even when the stats agree with it — agreement makes it survivable, not right.

Check 3 is reported separately because a stats pair can pass 1 and 2 while the
fill is still nonphysical: that is exactly the SST situation, and it is why
check 3 exists.

Usage::

    python check_normalization.py \
      --mean  data_2015-2050_mean.nc \
      --std   data_2015-2050_std_corr.nc \
      --h5-dir $E3SM_ROOT/h5/plev_data \
      --config config/E3SM_SFNO_H5_POLARIS.yaml

PASS prints ``NORMALIZATION_OK``; any failure prints a greppable ``ERROR``.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

import h5py
import numpy as np
import xarray as xr

# The 108-field contract. Upper-air vars are level-suffixed in the h5.
UPPER_AIR = ["T", "U", "V", "Z3", "RELHUM"]
FLAT = (
    ["TREFHT", "U10", "RHREFHT", "PS", "PSL", "TMQ"]        # surface
    + ["SOILWATER_10CM", "TSOI_10CM"]                        # land
    + ["FSNTOA", "FSNT", "PRECT"]                            # diagnostic
    + ["PCT_GLACIER", "PFTDATA_MASK", "PCT_NATVEG", "TOPO"]  # constant boundary
    + ["SST", "ICE", "sol_in"]                               # varying boundary
)

# Spread across the archive; the default is deliberately not all-consecutive so a
# seasonal artefact cannot masquerade as a clean result.
DEFAULT_SAMPLES = [
    (2015, 0), (2015, 365), (2015, 730), (2015, 1095),
    (2020, 200), (2030, 800), (2040, 500), (2049, 1200),
]


def parse_mask_fill(config: Path) -> dict[str, float]:
    text = config.read_text()
    m = re.search(r"^\s*mask_fill:\s*\{(.*?)\}", text, re.M | re.S)
    if not m:
        raise KeyError(f"no mask_fill in {config}")
    return {k: float(v) for k, v in re.findall(r"'([^']+)':\s*([-\d.eE]+)", m.group(1))}


def parse_levels(config: Path) -> list[float]:
    text = config.read_text()
    key = "sigma_levels" if re.search(r"^\s*use_sigma_levels:\s*True", text, re.M) else "levels"
    m = re.search(rf"^\s*{key}:\s*(\[.*?\])", text, re.M | re.S)
    return ast.literal_eval(m.group(1))


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


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mean", type=Path, required=True)
    p.add_argument("--std", type=Path, required=True)
    p.add_argument("--h5-dir", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True, help="A PanguWeather E3SM YAML (for mask_fill + levels).")
    p.add_argument("--std-tol", type=float, default=0.10, help="Allowed |std - 1|.")
    p.add_argument("--mean-tol", type=float, default=0.10, help="Allowed |mean|.")
    p.add_argument("--upper-air-level", type=int, default=-1, help="Which level index to spot-check for upper-air vars.")
    a = p.parse_args(argv)

    fills = parse_mask_fill(a.config)
    levels = parse_levels(a.config)
    mean_ds = xr.open_dataset(a.mean, decode_times=False)
    std_ds = xr.open_dataset(a.std, decode_times=False)

    rows, fill_warnings = [], []
    for var in FLAT + UPPER_AIR:
        stack, valid_lo, valid_hi = [], np.inf, -np.inf
        for year, idx in DEFAULT_SAMPLES:
            path = a.h5_dir / f"{year}_{idx:04d}.h5"
            with h5py.File(path, "r") as f:
                g = f["input"]
                if var in UPPER_AIR:
                    lev = levels[a.upper_air_level]
                    arr = np.asarray(g[_level_key(g, var, lev)][:], dtype="float64")
                else:
                    arr = np.asarray(g[var][:], dtype="float64")
            v = arr[~np.isnan(arr)]
            if v.size:
                valid_lo, valid_hi = min(valid_lo, v.min()), max(valid_hi, v.max())
            stack.append(arr)
        raw = np.stack(stack)

        fill = fills.get(var)
        if fill is not None:
            if np.isnan(raw).any() and not (valid_lo <= fill <= valid_hi):
                fill_warnings.append((var, fill, valid_lo, valid_hi))
            raw = np.where(np.isnan(raw), fill, raw)
        elif np.isnan(raw).any():
            fill_warnings.append((var, None, valid_lo, valid_hi))

        mu = mean_ds[var].values
        sd = std_ds[var].values
        if mu.ndim:                                   # level-resolved stats
            mu, sd = float(mu[a.upper_air_level]), float(sd[a.upper_air_level])
        else:
            mu, sd = float(mu), float(sd)
        z = (raw - mu) / sd
        rows.append((var, float(z.mean()), float(z.std()), fill))

    bad = [r for r in rows if abs(r[2] - 1.0) > a.std_tol or abs(r[1]) > a.mean_tol]

    print(f"=== normalization check: {a.mean.name} / {a.std.name} ===")
    print(f"{'variable':16s} {'fill':>10} {'norm mean':>11} {'norm std':>10}   status")
    for var, zm, zs, fill in rows:
        ok = abs(zs - 1.0) <= a.std_tol and abs(zm) <= a.mean_tol
        fs = "—" if fill is None else f"{fill:g}"
        print(f"{var:16s} {fs:>10} {zm:11.4f} {zs:10.4f}   {'ok' if ok else 'FAIL'}")

    print()
    if fill_warnings:
        print("⚠ fill values outside the valid data range (units bug, even if the stats agree):")
        for var, fill, lo, hi in fill_warnings:
            got = "NO FILL DEFINED" if fill is None else f"fill {fill:g}"
            print(f"    {var:16s} {got:20s} valid range [{lo:.3f}, {hi:.3f}]")
        print()

    if bad:
        for var, zm, zs, _ in bad:
            print(f"ERROR NORMALIZATION_MISMATCH {var}: normalized mean {zm:.4f}, std {zs:.4f} "
                  f"(want 0 +/- {a.mean_tol}, 1 +/- {a.std_tol})")
            print(f"    -> the stats were computed under a DIFFERENT fill than the config applies.")
        return 1
    if fill_warnings:
        print("NORMALIZATION_OK (scale + centre) — but see the fill warnings above.")
        return 0
    print(f"NORMALIZATION_OK {len(rows)}/{len(rows)} variables")
    return 0


if __name__ == "__main__":
    sys.exit(main())
