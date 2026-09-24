#!/usr/bin/env python3
"""Bitwise comparison of two climate-driver member NetCDFs (flag-off equivalence gate).

    python polaris/compare_member_nc.py ref.nc new.nc

Compares every numeric variable present in ``ref`` (NaN == NaN). Prints one line
per differing variable with max |diff| and the first differing index, then
``MEMBER_NC_EQUIV_OK vars=<n>`` or ``ERROR MEMBER_NC_DIFFER n=<k>``.
"""
from __future__ import annotations

import sys

import netCDF4
import numpy as np


def compare(ref_path, new_path) -> list[str]:
    diffs = []
    with netCDF4.Dataset(ref_path) as a, netCDF4.Dataset(new_path) as b:
        for name, va in a.variables.items():
            if va.dtype == str or va.dtype.kind in "OSU":
                continue
            if name not in b.variables:
                diffs.append(f"{name}: missing in new")
                continue
            x = np.ma.filled(va[:], np.nan).astype(np.float64) if va.dtype.kind == "f" else np.asarray(va[:])
            vb = b.variables[name]
            y = np.ma.filled(vb[:], np.nan).astype(np.float64) if vb.dtype.kind == "f" else np.asarray(vb[:])
            if x.shape != y.shape:
                diffs.append(f"{name}: shape {x.shape} vs {y.shape}")
                continue
            same = (x == y) | (np.isnan(x) & np.isnan(y)) if x.dtype.kind == "f" else (x == y)
            if not np.all(same):
                idx = np.unravel_index(int(np.argmin(same)), x.shape)
                d = np.nanmax(np.abs(x - y)) if x.dtype.kind == "f" else "n/a"
                diffs.append(f"{name}: {int((~same).sum())} differ, max|d|={d}, first at {idx}")
    return diffs


def main(argv=None) -> int:
    argv = argv or sys.argv[1:]
    diffs = compare(argv[0], argv[1])
    for d in diffs:
        print(f"DIFF {d}")
    if diffs:
        print(f"ERROR MEMBER_NC_DIFFER n={len(diffs)}")
        return 1
    with netCDF4.Dataset(argv[0]) as a:
        n = sum(1 for v in a.variables.values() if not (v.dtype == str or v.dtype.kind in "OSU"))
    print(f"MEMBER_NC_EQUIV_OK vars={n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
