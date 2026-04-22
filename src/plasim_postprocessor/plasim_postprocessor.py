#!/usr/bin/env python3
"""plasim_postprocessor.py — single-purpose burn7-based PlaSim post-processor.

Converts raw PlaSim binary output (MOST.NNNN, one file per simulated year) into
per-sim-year NetCDF files at native 6-hourly cadence. Produces one audited
variable set (the SFNO emulator training contract), locked by
docs/audit_snapshots/manifest.txt.

Toolchain
---------
burn7  — derives every variable (sigma extraction + pressure interpolation of zg)
CDO    — plumbing only: merging burn7 outputs and computing pr_6h via runsum

Output contract (locked by docs/audit_snapshots/manifest.txt)
-------------------------------------------------------------
3D atmosphere (10 sigma levels): ta, ua, va, hus, zg
3D atmosphere (13 pressure levels): zg_plev @ [50,100,150,200,250,300,400,500,
                                                600,700,850,925,1000] hPa
Near-surface:        tas
Surface state:       ts, ps, psl, clt, pl (log_surface_pressure, = ln(ps in Pa))
Land:                mrso
Static boundary:     lsm, z0, sg
Radiation / energy:  rss, rls, rst, rlut, rsut, hfss, hfls
                     (PlaSim sign convention: positive = into receiver,
                      negative = leaving receiver)
Precipitation:       pr (burn7), pr_6h (CDO runsum,6)
Sea ice (conditional, gated by --with-sea-ice): sic

Breaking change (2026-04-21)
----------------------------
Pressure-level geopotential height used to be emitted as `zg`. It is now
emitted as `zg_plev`. The canonical `zg` variable is the new sigma-level
geopotential height (10 midpoints, co-located with ta/ua/va/hus). Consumers
that read pressure-level zg by name (SFNO diagnostic configs, NetCDF→H5
PRESSURE_LEVEL_VARS, etc.) need to update to `zg_plev`.

Output layout
-------------
{output-root}/sim{NN}/MOST.{YYYY}.nc — one file per sim-year, native cadence.

Module-load contract (Stampede3, mirrored in submit.slurm)
----------------------------------------------------------
module load intel/24.0 cdo netcdf python/3.12.11
export LD_LIBRARY_PATH=/opt/apps/intel24/netcdf/4.9.2/x86_64/lib:/home1/09979/awikner/netcdf-4.2/lib:$LD_LIBRARY_PATH
"""

import argparse
import logging
import subprocess
import sys
import tempfile
from pathlib import Path

logger = logging.getLogger("plasim_postprocessor")

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_BURN7 = SCRIPT_DIR / "burn7" / "Stampede3" / "burn7"

SIGMA_CODES: list[str] = [
    # 3D atmosphere (burn7 emits on 10 sigma levels per MODLEV)
    # zg on sigma is produced by the patched burn7 (midpoint hydrostatic
    # integration) and is co-located with ta/ua/va/hus.
    "ta", "ua", "va", "hus", "zg",
    # Near-surface
    "tas",
    # Surface state (pl = log_surface_pressure, emitted 2D despite the
    # sigma namelist because burn7 marks code 152 as twod=1)
    "ts", "ps", "psl", "clt", "pl",
    # Land
    "mrso",
    # Radiation + surface heat fluxes
    "rss", "rls", "rst", "rlut", "rsut", "hfss", "hfls",
    # Static boundary
    "lsm", "z0", "sg",
    # Precipitation source (pr_6h is derived from this by CDO runsum)
    "pr",
]
PRESSURE_LEVELS: list[int] = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
ACCUMULATE_PRECIP_HOURS: list[int] = [6]
SEA_ICE_CODE: str = "sic"


def enumerate_tasks(sims, year_start, year_end):
    return [(sim, year) for sim in sims for year in range(year_start, year_end + 1)]


def _write_sigma_namelist(path: Path, with_sea_ice: bool) -> None:
    code_vars = list(SIGMA_CODES)
    if with_sea_ice:
        code_vars.append(SEA_ICE_CODE)
    path.write_text(
        f"code={','.join(code_vars)}\n"
        f"MODLEV=10,9,8,7,6,5,4,3,2,1,0\n"
        f"vtype=sigma,htype=g,mean=0,netcdf=1\n"
    )


def _write_zg_namelist(path: Path) -> None:
    hpa = ",".join(str(x) for x in PRESSURE_LEVELS)
    path.write_text(
        f"code=zg\n"
        f"hpa={hpa}\n"
        f"vtype=p,htype=g,mean=0,netcdf=1\n"
    )


def _run_burn7(binary: Path, namelist: Path, input_file: Path, output_file: Path) -> None:
    with namelist.open("r") as nlin:
        result = subprocess.run(
            [str(binary), str(input_file), str(output_file)],
            stdin=nlin, capture_output=True, text=True,
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"burn7 failed (rc={result.returncode})\n"
            f"  binary   : {binary}\n"
            f"  namelist : {namelist.read_text().strip()}\n"
            f"  input    : {input_file}\n"
            f"  stderr   : {result.stderr.strip()}"
        )


def _run_cdo(args: list[str]) -> None:
    cmd = ["cdo", "-s", "-O", *args]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"CDO failed (rc={result.returncode}): {' '.join(cmd)}\n"
            f"  stderr: {result.stderr.strip()}"
        )


def process_one(sim: int, year: int, opts) -> None:
    input_path = Path(opts.input_root) / f"sim{sim}" / f"MOST.{year:04d}"
    output_path = Path(opts.output_root) / f"sim{sim}" / f"MOST.{year:04d}.nc"

    if not input_path.exists():
        logger.warning("[sim%s/%04d] skipping (input missing: %s)", sim, year, input_path)
        return

    if output_path.exists() and not opts.overwrite:
        logger.info("[sim%s/%04d] skipping %s (exists; pass --overwrite to force)",
                    sim, year, output_path)
        return

    if opts.dry_run:
        logger.info("[sim%s/%04d] dry-run: would process %s -> %s",
                    sim, year, input_path, output_path)
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tdname:
        td = Path(tdname)
        sigma_nl = td / "sigma.nl"
        zg_nl = td / "zg.nl"
        sigma_nc = td / "sigma.nc"           # contains ta/ua/va/hus/zg(sigma)/pl/...
        zg_raw = td / "zg_plev_raw.nc"       # burn7 emits pressure-zg as "zg"
        zg_renamed = td / "zg_plev.nc"       # renamed to "zg_plev" before merge
        current = td / "merged.nc"

        _write_sigma_namelist(sigma_nl, opts.with_sea_ice)
        logger.info("[sim%s/%04d] burn7 sigma (%s)", sim, year, sigma_nl.read_text().split('\n')[0])
        _run_burn7(opts.burn7_binary, sigma_nl, input_path, sigma_nc)

        _write_zg_namelist(zg_nl)
        logger.info("[sim%s/%04d] burn7 zg_plev @ %s hPa", sim, year, PRESSURE_LEVELS)
        _run_burn7(opts.burn7_binary, zg_nl, input_path, zg_raw)
        # Rename pressure-level zg to zg_plev before merging with sigma output
        # (which already contains sigma-level zg). Avoids a NetCDF variable-name
        # collision; the two live on distinct vertical dims (lev vs lev_2).
        _run_cdo(["chname,zg,zg_plev", str(zg_raw), str(zg_renamed)])
        _run_cdo(["merge", str(sigma_nc), str(zg_renamed), str(current)])

        for hours in ACCUMULATE_PRECIP_HOURS:
            pr_only = td / f"pr_only_{hours}h.nc"
            pr_sum = td / f"pr_sum_{hours}h.nc"
            pr_renamed = td / f"pr_renamed_{hours}h.nc"
            next_merged = td / f"merged_after_{hours}h.nc"
            logger.info("[sim%s/%04d] CDO accumulate pr_%dh", sim, year, hours)
            _run_cdo(["selname,pr", str(current), str(pr_only)])
            _run_cdo([f"runsum,{hours}", str(pr_only), str(pr_sum)])
            _run_cdo([f"chname,pr,pr_{hours}h", str(pr_sum), str(pr_renamed)])
            _run_cdo(["merge", str(current), str(pr_renamed), str(next_merged)])
            current = next_merged

        current.replace(output_path)

    logger.info("[sim%s/%04d] wrote %s", sim, year, output_path)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Single-purpose burn7-based PlaSim post-processor "
                    "for the SFNO emulator training variable set.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--sims", required=True, type=int, nargs="+", metavar="SIM",
                   help="Simulation numbers, e.g. 30 31 32.")
    p.add_argument("--years", required=True, type=int, nargs=2, metavar=("START", "END"),
                   help="Inclusive year range, e.g. 1 100.")
    p.add_argument("--input-root", default=None, type=Path,
                   help="Required for processing. Root containing sim{NN}/MOST.{YYYY:04d}. "
                        "Not needed with --count-tasks.")
    p.add_argument("--output-root", default=None, type=Path,
                   help="Required for processing. Root for {output-root}/sim{NN}/MOST.{YYYY:04d}.nc. "
                        "Not needed with --count-tasks.")
    p.add_argument("--with-sea-ice", action=argparse.BooleanOptionalAction, default=False,
                   help="Include sic (code 210, sea_ice_cover) in the output. "
                        "Default off — enabling it on a sim whose PlaSim run had sea ice "
                        "disabled will cause burn7 to fail. The operator must know whether "
                        "the source sims have sea ice; this script cannot infer it.")
    p.add_argument("--burn7-binary", default=DEFAULT_BURN7, type=Path,
                   help=f"Path to burn7 binary (default: {DEFAULT_BURN7}).")
    p.add_argument("--task-index", type=int, default=None,
                   help="0-based index into the (sim, year) task list (for SLURM array dispatch).")
    p.add_argument("--count-tasks", action="store_true",
                   help="Print the number of (sim, year) pairs and exit. "
                        "Does not require --input-root/--output-root.")
    p.add_argument("--overwrite", action="store_true",
                   help="Force re-write of existing output files.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print actions without executing.")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    year_start, year_end = args.years
    if year_start > year_end:
        sys.exit(f"error: --years START ({year_start}) must be <= END ({year_end})")

    tasks = enumerate_tasks(args.sims, year_start, year_end)

    if args.count_tasks:
        print(len(tasks))
        return

    missing = [name for name, val in (("--input-root", args.input_root),
                                       ("--output-root", args.output_root)) if val is None]
    if missing:
        sys.exit(f"error: {', '.join(missing)} required unless --count-tasks is set")

    if not args.burn7_binary.exists():
        sys.exit(f"error: burn7 binary not found at {args.burn7_binary}")

    if args.task_index is not None:
        if not (0 <= args.task_index < len(tasks)):
            sys.exit(f"error: --task-index {args.task_index} out of range [0, {len(tasks)})")
        tasks = [tasks[args.task_index]]

    for sim, year in tasks:
        process_one(sim, year, args)


if __name__ == "__main__":
    main()
