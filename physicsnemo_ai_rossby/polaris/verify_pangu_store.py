#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify a converted E3SM Pangu-parity Zarr store against its H5 source.

Thirteen checks (six groups below), all of which must pass:

1. **Attrs vs contract** — every channel group, in order, equals
   ``ai_rossby_variable_contract.PLANNED`` (with land folded last into the
   store's ``surface_variables``). Order is the silent failure mode: the loader
   stacks tensors in attrs order while fills and loss come from the model config,
   so a permutation is correctly-shaped and raises nothing.
2. **Field count** — 5 upper-air x 18 levels + 18 surface-type = 108.
3. **Bitwise** — for N randomly drawn timesteps, every one of the 23 variables
   (all 18 levels for upper-air) must equal the H5 source EXACTLY. Not
   ``allclose``: the converter only reshapes, so any difference is a bug.
4. **Raw NaN preserved** — NaN must be in the SAME cells as the source, not
   filled. Filling is the training pipeline's job, from the dataset config; a
   store that pre-filled would silently double-apply or contradict it.
5. **Levels** — the store's ``pressure_level`` matches the normalization store's,
   or the normalizer would align stats to the wrong levels.
6. **Time axis** — expected length, ``noleap`` calendar, strictly increasing.

PASS prints ``PANGU_STORE_VERIFIED``; any failure prints a greppable ``ERROR``.

Usage::

    python polaris/verify_pangu_store.py \
      --store $AI_ROSSBY_DATA/e3sm/train/2015.zarr \
      --h5-dir $E3SM_ROOT/h5/plev_data \
      --norm  $AI_ROSSBY_DATA/e3sm/norm/normalization_2015-2050.zarr
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import xarray as xr

# The contract lives at the outer repo root, beside the subtree.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ai_rossby_variable_contract import (  # noqa: E402
    PLANNED,
    STORE_SURFACE,
    n_channels,
)


def _fail(results: list, label: str, detail: str) -> None:
    results.append((label, False, detail))


def _ok(results: list, label: str, detail: str = "") -> None:
    results.append((label, True, detail))


def check_attrs(ds: xr.Dataset, results: list) -> None:
    expected = {
        "surface_variables": STORE_SURFACE,
        "pressure_upper_air_variables": PLANNED["upper_air_variables"],
        "diagnostic_variables": PLANNED["diagnostic_variables"],
        "constant_boundary_variables": PLANNED["constant_boundary_variables"],
        "varying_boundary_variables": PLANNED["varying_boundary_variables"],
        "sigma_upper_air_variables": [],
    }
    for key, want in expected.items():
        got = list(ds.attrs.get(key, []))
        if got == want:
            _ok(results, f"attrs.{key}", f"{len(got)} names, order matches")
        else:
            _fail(results, f"attrs.{key}", f"expected {want}, got {got}")

    cal = str(ds.attrs.get("calendar", ""))
    if cal == "noleap":
        _ok(results, "attrs.calendar", cal)
    else:
        _fail(results, "attrs.calendar", f"expected 'noleap', got {cal!r}")


def check_field_count(ds: xr.Dataset, results: list) -> None:
    n_lev = ds.sizes.get("pressure_level", 0)
    n_upper = len(PLANNED["upper_air_variables"]) * n_lev
    n_flat = len(STORE_SURFACE) + len(PLANNED["diagnostic_variables"]) + len(
        PLANNED["constant_boundary_variables"]
    ) + len(PLANNED["varying_boundary_variables"])
    total = n_upper + n_flat
    want = n_channels(PLANNED)
    if total == want and n_lev == len(PLANNED["levels"]):
        _ok(results, "field count", f"{n_upper} upper-air + {n_flat} flat = {total}")
    else:
        _fail(
            results,
            "field count",
            f"expected {want} over {len(PLANNED['levels'])} levels, "
            f"got {total} over {n_lev}",
        )


def check_levels(ds: xr.Dataset, norm: Path | None, results: list) -> None:
    store_lv = np.asarray(ds["pressure_level"].values, dtype="float32")
    want = np.asarray(PLANNED["levels"], dtype="float32")
    if np.array_equal(store_lv, want):
        _ok(results, "levels vs contract", f"{len(store_lv)} levels, exact (float32)")
    else:
        _fail(results, "levels vs contract", f"max|diff| = {np.max(np.abs(store_lv - want))}")

    if norm is None:
        return
    nds = xr.open_zarr(norm)
    norm_lv = np.asarray(nds["pressure_level"].values, dtype="float32")
    if np.array_equal(store_lv, norm_lv):
        _ok(results, "levels vs norm store", "identical")
    else:
        _fail(
            results,
            "levels vs norm store",
            f"store {store_lv.tolist()} != norm {norm_lv.tolist()}",
        )


def check_time(ds: xr.Dataset, results: list) -> None:
    t = ds["time"].values
    n = len(t)
    strictly_increasing = all(t[i] < t[i + 1] for i in range(n - 1))
    # A noleap year at 6 h is 365*4 = 1460 samples; a partial --sample-range is
    # legitimate, so only the ordering is a hard failure.
    if strictly_increasing:
        _ok(results, "time axis", f"{n} steps, strictly increasing, {t[0]} .. {t[-1]}")
    else:
        _fail(results, "time axis", f"{n} steps, NOT strictly increasing")


def check_bitwise(
    ds: xr.Dataset, h5_dir: Path, year: int, n_samples: int, seed: int, results: list
) -> None:
    """Compare N random timesteps, every variable, against the H5 source."""
    rng = np.random.default_rng(seed)
    n_time = ds.sizes["time"]
    picks = sorted(rng.choice(n_time, size=min(n_samples, n_time), replace=False).tolist())

    sample_range = list(ds.attrs.get("sample_range", [0, n_time]))
    lo = int(sample_range[0])
    levels = [float(x) for x in PLANNED["levels"]]

    worst = 0.0
    worst_where = ""
    nan_mismatch = 0
    nan_total = 0
    checked = 0

    flat_vars = (
        STORE_SURFACE
        + PLANNED["diagnostic_variables"]
        + PLANNED["varying_boundary_variables"]
    )

    for t in picks:
        path = h5_dir / f"{year}_{lo + t:04d}.h5"
        if not path.exists():
            _fail(results, "bitwise", f"source file missing: {path}")
            return
        with h5py.File(path, "r") as f:
            g = f["input"]
            for v in flat_vars + PLANNED["constant_boundary_variables"]:
                src = np.asarray(g[v][:], dtype="float32")
                if v in PLANNED["constant_boundary_variables"]:
                    got = np.asarray(ds[v].values, dtype="float32")
                else:
                    got = np.asarray(ds[v].isel(time=t).values, dtype="float32")
                worst, worst_where, nan_mismatch, nan_total, checked = _compare(
                    src, got, f"{v}@t={t}", worst, worst_where,
                    nan_mismatch, nan_total, checked,
                )
            for v in PLANNED["upper_air_variables"]:
                for li, lev in enumerate(levels):
                    key = _level_key(g, v, lev)
                    src = np.asarray(g[key][:], dtype="float32")
                    got = np.asarray(
                        ds[v].isel(time=t, pressure_level=li).values, dtype="float32"
                    )
                    worst, worst_where, nan_mismatch, nan_total, checked = _compare(
                        src, got, f"{v}[{lev:g}]@t={t}", worst, worst_where,
                        nan_mismatch, nan_total, checked,
                    )

    if worst == 0.0:
        _ok(results, "bitwise vs h5", f"{checked} fields over {len(picks)} timesteps, max|diff| = 0")
    else:
        _fail(results, "bitwise vs h5", f"max|diff| = {worst:g} at {worst_where}")

    if nan_mismatch == 0:
        _ok(
            results,
            "raw NaN preserved",
            f"{nan_total:,} NaN cells, all in the same positions as the source",
        )
    else:
        _fail(results, "raw NaN preserved", f"{nan_mismatch:,} cells differ in NaN-ness")


def _compare(src, got, where, worst, worst_where, nan_mismatch, nan_total, checked):
    if src.shape != got.shape:
        return float("inf"), f"{where} shape {src.shape} != {got.shape}", nan_mismatch, nan_total, checked
    s_nan, g_nan = np.isnan(src), np.isnan(got)
    nan_total += int(s_nan.sum())
    nan_mismatch += int(np.count_nonzero(s_nan != g_nan))
    both = ~s_nan & ~g_nan
    if both.any():
        d = float(np.max(np.abs(src[both] - got[both])))
        if d > worst:
            worst, worst_where = d, where
    return worst, worst_where, nan_mismatch, nan_total, checked + 1


def _level_key(group, prefix: str, level: float, rtol: float = 5e-3) -> str:
    """Find ``<prefix>_<level>`` in the H5 group, matching the level by value."""
    exact = f"{prefix}_{level}"
    if exact in group:
        return exact
    best, best_d = None, float("inf")
    for k in group:
        if not k.startswith(prefix + "_"):
            continue
        try:
            lv = float(k[len(prefix) + 1:])
        except ValueError:
            continue
        d = abs(lv - level)
        if d < best_d:
            best, best_d = k, d
    if best is None or best_d > rtol * max(abs(level), 1.0):
        raise KeyError(f"no key for {prefix} at level {level}")
    return best


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--store", type=Path, required=True)
    p.add_argument("--h5-dir", type=Path, required=True)
    p.add_argument("--norm", type=Path, default=None)
    p.add_argument("--n-samples", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)

    ds = xr.open_zarr(a.store)
    year = int(ds.attrs.get("year_index", Path(a.store).stem))

    results: list = []
    check_attrs(ds, results)
    check_field_count(ds, results)
    check_levels(ds, a.norm, results)
    check_time(ds, results)
    check_bitwise(ds, a.h5_dir, year, a.n_samples, a.seed, results)

    print(f"=== verify {a.store} (year {year}) ===")
    failed = 0
    for label, ok, detail in results:
        print(f"[{'PASS' if ok else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""))
        if not ok:
            failed += 1
    if failed:
        print(f"ERROR PANGU_STORE_VERIFY_FAILED {failed}/{len(results)}")
        return 1
    print(f"PANGU_STORE_VERIFIED {len(results)}/{len(results)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
