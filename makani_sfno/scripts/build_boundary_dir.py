#!/usr/bin/env python3
"""build_boundary_dir.py — Reshape emulator_adaptor output into the SFNO
dataloader's per-variable boundary layout.

Input
-----

    {input-root}/sim{NN}/boundary.{YYYY:04d}.nc

as produced by src/emulator_adaptor/adaptor.py (one bundled file per
(sim, year), containing sst/rsdt/sic on a 6h time axis).

Output
------

    {output-dir}/sst_masked_6h.nc           # non-leap (shorter-year) concat
    {output-dir}/rsdt_masked_6h.nc
    {output-dir}/sic_masked_6h.nc
    {output-dir}/sst_masked_6h_leap.nc      # leap (longer-year) concat, emitted only if present
    {output-dir}/rsdt_masked_6h_leap.nc
    {output-dir}/sic_masked_6h_leap.nc

This is the "downstream reshape step" that docs/emulator_adaptor_audit.md:181
explicitly leaves out of scope for the adaptor. It ships here as a standalone
script rather than as a mode on emulator_adaptor/adaptor.py so the adaptor's
audited convention-translation contract stays clean.

Grouping policy
---------------

Leap/non-leap classification is by **timestep count per year**, not calendar
flag. PlaSim's output cadence and calendar are configurable; the only
invariant that matters for the dataloader is that all years in a single
concat have the same time-axis length. The majority-count group gets
`_masked_6h.nc`; the minority-count group (if any) gets `_masked_6h_leap.nc`.
More than two distinct counts is an error.
"""

import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import xarray as xr

logger = logging.getLogger("build_boundary_dir")

VARIABLES = ("sst", "rsdt", "sic")


def enumerate_year_files(sims, year_start, year_end, input_root):
    files = []
    for sim in sims:
        for year in range(year_start, year_end + 1):
            p = input_root / f"sim{sim:02d}" / f"boundary.{year:04d}.nc"
            files.append((sim, year, p))
    return files


def group_by_length(files):
    groups = defaultdict(list)
    for sim, year, path in files:
        if not path.exists():
            raise FileNotFoundError(f"missing input: {path}")
        with xr.open_dataset(path, decode_times=False) as ds:
            n = ds.sizes["time"]
        groups[n].append((sim, year, path))
    return dict(groups)


def assign_group_names(groups):
    """Return {group_length: filename_suffix} mapping.

    Majority-length → ""
    Minority-length → "_leap"
    """
    if len(groups) > 2:
        raise RuntimeError(
            f"more than 2 distinct time-axis lengths across input files: {sorted(groups)}. "
            f"The SFNO per-variable layout only supports a 2-way split (non-leap/leap)."
        )
    if len(groups) == 1:
        only = next(iter(groups))
        return {only: ""}
    a, b = sorted(groups, key=lambda k: len(groups[k]), reverse=True)
    return {a: "", b: "_leap"}


def concat_group(year_files, var):
    pieces = []
    for sim, year, path in sorted(year_files, key=lambda t: (t[0], t[1])):
        ds = xr.open_dataset(path, decode_times=False)
        da = ds[var].astype(np.float32)
        pieces.append(da)
    return xr.concat(pieces, dim="time")


def write_variable(output_dir, var, suffix, da, source_attrs):
    out_path = output_dir / f"{var}_masked_6h{suffix}.nc"
    tmp_path = out_path.with_suffix(".nc.tmp")

    ds_out = da.to_dataset(name=var)
    # Preserve the adaptor's provenance attrs on the file level.
    for k, v in source_attrs.items():
        ds_out.attrs[k] = v
    ds_out.attrs["reshape_source"] = "scripts/build_boundary_dir.py"
    ds_out.attrs["reshape_variant"] = suffix.lstrip("_") or "non_leap"

    encoding = {var: {"dtype": "float32", "_FillValue": np.float32("nan")}}
    ds_out.to_netcdf(tmp_path, encoding=encoding)
    tmp_path.replace(out_path)
    logger.info("wrote %s (%d timesteps)", out_path, da.sizes["time"])


def process(args):
    files = enumerate_year_files(args.sims, args.years[0], args.years[1], args.input_root)
    logger.info("enumerated %d (sim, year) pairs", len(files))

    groups = group_by_length(files)
    logger.info("time-length groups: %s", {n: len(v) for n, v in groups.items()})

    suffixes = assign_group_names(groups)

    # Pull provenance attrs from the first input file (adaptor stamps rsdt_method etc.).
    first_path = files[0][2]
    with xr.open_dataset(first_path, decode_times=False) as ds0:
        source_attrs = dict(ds0.attrs)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for length, suffix in suffixes.items():
        year_files = groups[length]
        for var in VARIABLES:
            logger.info("building %s (suffix=%r) from %d years", var, suffix, len(year_files))
            da = concat_group(year_files, var)
            write_variable(args.output_dir, var, suffix, da, source_attrs)


def _parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Reshape emulator_adaptor per-(sim, year) boundary NetCDFs into the "
            "SFNO dataloader's per-variable layout (sst/rsdt/sic × {non-leap, leap})."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--sims", required=True, type=int, nargs="+", metavar="SIM",
                   help="Sim numbers to include (matches sim{NN}/ subdirs under --input-root).")
    p.add_argument("--years", required=True, type=int, nargs=2, metavar=("START", "END"),
                   help="Inclusive year range.")
    p.add_argument("--input-root", required=True, type=Path,
                   help="Adaptor output root, containing sim{NN}/boundary.{YYYY:04d}.nc.")
    p.add_argument("--output-dir", required=True, type=Path,
                   help="Destination boundaries/ dir (dataset-scoped, co-located with the H5 bundle).")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main():
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    if args.years[0] > args.years[1]:
        sys.exit(f"error: --years START ({args.years[0]}) must be <= END ({args.years[1]})")
    process(args)


if __name__ == "__main__":
    main()
