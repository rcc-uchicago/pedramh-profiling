#!/usr/bin/env python
"""Cut a channel subset out of a build_true_climatology.py output, by name.

A 99-channel run (F: no SOILWATER_10CM/TSOI_10CM) needs a 99-channel truth for
score_climate_fidelity.py, which requires member and truth channels to match
exactly. Rebuilding from the F run dir reads the pack through a non-contiguous
channel index and ran out of debug walltime (7720814: 4000/7299 samples in 1 h).
Every field in the truth file is per-channel -- time_mean is a per-channel time
average, global_means/global_stds are per-channel -- so selecting rows by name
from the 101-channel truth equals a 99-channel build, provided both runs use the
same stats files (checked for nf4_prod_b16_r1 vs f_nosoil_2n_b32_e23_scratch).

Usage:
    subset_true_climatology.py --true true_101.npz --names-from member_FSEMA00.nc --out true_99.npz
    subset_true_climatology.py --selftest

PASS tokens: SUBSET_TRUE_SELFTEST_OK, SUBSET_TRUE_CLIMATOLOGY_OK n=<channels> out=<path>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PER_CHANNEL = ("time_mean", "global_means", "global_stds")
CARRIED = ("lat", "lon", "years", "n_samples")


def subset(true: dict, names: list[str]) -> dict:
    """Rows of ``true``'s per-channel fields for ``names``, in that order."""
    import numpy as np

    have = [str(c) for c in true["channel_names"]]
    missing = [n for n in names if n not in have]
    if missing:
        raise ValueError(f"SUBSET_MISSING_CHANNEL: {len(missing)} not in the truth file, "
                         f"first {missing[:3]}")
    if len(set(names)) != len(names):
        raise ValueError("SUBSET_DUPLICATE_CHANNEL: requested names repeat")
    idx = np.array([have.index(n) for n in names])
    out = {k: np.asarray(true[k])[idx] for k in PER_CHANNEL}
    out["channel_names"] = np.array(names, dtype=object)
    for k in CARRIED:
        if k in true:
            out[k] = np.asarray(true[k])
    return out


def _selftest() -> None:
    import numpy as np

    rng = np.random.default_rng(0)
    names = [f"c{i}" for i in range(5)]
    true = dict(time_mean=rng.normal(size=(5, 3, 4)), global_means=rng.normal(size=5),
                global_stds=rng.uniform(1, 2, size=5), channel_names=np.array(names, dtype=object),
                lat=np.linspace(-60, 60, 3), lon=np.arange(4.0), years=np.array([2045]),
                n_samples=7)
    want = ["c3", "c0", "c4"]
    got = subset(true, want)
    assert [str(c) for c in got["channel_names"]] == want
    for k, src in (("time_mean", [3, 0, 4]), ("global_means", [3, 0, 4]), ("global_stds", [3, 0, 4])):
        assert np.array_equal(got[k], true[k][src]), k
    assert np.array_equal(got["lat"], true["lat"]) and int(got["n_samples"]) == 7
    for bad in (["c9"], ["c1", "c1"]):
        try:
            subset(true, bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted bad names {bad}")
    print("SUBSET_TRUE_SELFTEST_OK")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--true", type=Path, help="superset build_true_climatology.py output")
    p.add_argument("--names-from", type=Path, help="member_*.nc whose 'channel' variable gives names and order")
    p.add_argument("--out", type=Path)
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args(argv)
    if args.selftest:
        _selftest()
        return 0
    if not (args.true and args.names_from and args.out):
        p.error("--true, --names-from and --out are required unless --selftest")
    import numpy as np
    import netCDF4

    with netCDF4.Dataset(args.names_from, "r") as ds:
        names = [str(c) for c in ds["channel"][:]]
    true = dict(np.load(args.true, allow_pickle=True))
    out = subset(true, names)
    out["subset_of"] = np.array(str(args.true))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **out)
    print(f"SUBSET_TRUE_CLIMATOLOGY_OK n={len(names)} out={args.out}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001 -- one greppable line, then the traceback
        print(f"ERROR SUBSET_TRUE_CLIMATOLOGY_FAILED {type(exc).__name__}: {str(exc).splitlines()[0][:300]}")
        raise
