#!/usr/bin/env python3
"""adaptor.py — Emulator varying-boundary adaptor for PlaSim output.

Reads the per-sim-year NetCDF produced by src/plasim_postprocessor/plasim_postprocessor.py
(run with --with-sea-ice) and emits the SFNO emulator's varying-boundary
tuple (sst, rsdt, sic) as a single per-(sim, year) file:

    {output-root}/sim{NN}/boundary.{YYYY:04d}.nc

This is an emulator-contract translation layer — it is NOT part of the
postprocess pipeline. sst and rsdt are convention-dependent derivations,
not native PlaSim fields. Leap-year splitting / filename reshaping into
the emulator's {var}_masked_6h{,_leap}.nc layout is a separate downstream
concern.

Variables emitted
-----------------

sst — sea surface temperature (K)
    Ocean-masked surface temperature with a freezing-seawater clamp over
    ice-covered ocean:
        ocean = (lsm < 1e-6)                          strict land-mask zero
        icy   = (sic > SIC_THRESHOLD)                 majority-ice (ERA5/CMIP convention)
        sst = ts                        where ocean & ~icy
        sst = FREEZING_SEAWATER_K       where ocean &  icy
        sst = NaN                       elsewhere

rsdt — TOA incoming shortwave flux (W m-2)
    Two methods (select with --rsdt-method):
      arithmetic  (default): rst - rsut (PlaSim's own TOA SW accounting).
      astronomical         : analytic 6h-mean integration of the insolation
                             formula, using declination + distance-factor
                             approximations. Numpy-only, no external ephemeris.

sic — sea ice area fraction (1)
    Pass-through from the postprocessor, clipped to [0, 1] defensively
    (PlaSim can emit marginally-out-of-range values at cell edges).

All three are written as float32 with NaN `_FillValue`.
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger("emulator_adaptor")

FREEZING_SEAWATER_K: float = 271.35
SIC_THRESHOLD: float = 0.5
LAND_EPSILON: float = 1e-6
DEFAULT_SOLAR_CONSTANT_W_M2: float = 1367.0
DEFAULT_ECCENTRICITY: float = 0.0167
DEFAULT_OBLIQUITY_DEG: float = 23.441
DEFAULT_WINDOW_HOURS: float = 6.0

SST_MIN_K: float = FREEZING_SEAWATER_K - 0.01
SST_MAX_K: float = 310.0
RSDT_GLOBAL_MEAN_TOLERANCE_ARITH: float = 0.01   # ±1 %
RSDT_GLOBAL_MEAN_TOLERANCE_ASTRO: float = 0.005  # ±0.5 %


# ---------------------------------------------------------------------------
# sst construction
# ---------------------------------------------------------------------------
def compute_sst(ds: xr.Dataset) -> xr.DataArray:
    for req in ("ts", "lsm", "sic"):
        if req not in ds.data_vars:
            raise RuntimeError(
                f"adaptor requires '{req}' in postprocess output; rerun the "
                f"postprocessor with --with-sea-ice (for sic) and ensure sim "
                f"source emits ts/lsm."
            )

    ocean = ds["lsm"] < LAND_EPSILON
    icy = ds["sic"] > SIC_THRESHOLD
    sst = xr.where(
        ocean & ~icy, ds["ts"],
        xr.where(ocean & icy, FREEZING_SEAWATER_K, np.nan),
    )
    sst.attrs = {
        "units": "K",
        "long_name": "sea_surface_temperature",
        "standard_name": "sea_surface_temperature",
    }
    return sst


# ---------------------------------------------------------------------------
# rsdt — arithmetic path
# ---------------------------------------------------------------------------
def compute_rsdt_arithmetic(ds: xr.Dataset) -> xr.DataArray:
    for req in ("rst", "rsut"):
        if req not in ds.data_vars:
            raise RuntimeError(
                f"--rsdt-method arithmetic requires '{req}' in postprocess output."
            )
    rsdt = ds["rst"] - ds["rsut"]
    rsdt.attrs = {
        "units": "W m-2",
        "long_name": "toa_incident_shortwave_flux",
        "standard_name": "toa_incoming_shortwave_flux",
    }
    return rsdt


# ---------------------------------------------------------------------------
# rsdt — astronomical path (6h-mean analytic integral)
# ---------------------------------------------------------------------------
def _decode_doy_hour(time_da: xr.DataArray) -> tuple[np.ndarray, np.ndarray]:
    """Return (doy, utc_hour) arrays of shape (T,) from a time coord.

    Handles both cftime objects (non-standard calendars, e.g. proleptic_gregorian
    emitted by burn7) and numpy.datetime64 (pandas-decoded).
    """
    vals = time_da.values
    if len(vals) == 0:
        return np.empty(0), np.empty(0)
    first = vals[0]
    if hasattr(first, "timetuple"):
        doy = np.array([t.timetuple().tm_yday for t in vals], dtype=np.float64)
        hour = np.array(
            [t.hour + t.minute / 60.0 + t.second / 3600.0 for t in vals],
            dtype=np.float64,
        )
    else:
        idx = pd.to_datetime(vals)
        doy = idx.dayofyear.values.astype(np.float64)
        hour = (idx.hour + idx.minute / 60.0 + idx.second / 3600.0).values.astype(np.float64)
    return doy, hour


def compute_rsdt_astronomical(ds: xr.Dataset, opts) -> xr.DataArray:
    """Compute 6h-mean TOA shortwave via analytic integration.

    Per-cell, per-time expression:
        rsdt(t, lat, lon) = S0 * (a/r)**2 * <max(0, cos(zenith))>_{[t, t+dt]}

    The inner time-average is the analytic integral over the daylit portion
    of the window. Polar day / polar night are handled explicitly.

    Time-window convention: each timestamp t is treated as the START of a
    6h window [t, t+dt]. If PlaSim's output encodes end-of-window (common
    for accumulated fluxes), the per-cell diff vs the arithmetic path
    (computed in audit) will show a consistent zonal offset and the
    convention can be revisited.
    """
    doy, hour = _decode_doy_hour(ds["time"])
    n_t = len(doy)
    if n_t == 0:
        raise RuntimeError("astronomical rsdt: empty time dimension")

    dt_h = float(opts.window_hours)
    obliquity_rad = np.deg2rad(float(opts.obliquity_deg))

    # Per-time, shape (T,)
    dec = obliquity_rad * np.sin(2 * np.pi * (doy - 80.0) / 365.25)
    dist_factor = 1.0 + float(opts.eccentricity) * np.cos(2 * np.pi * (doy - 4.0) / 365.25)

    # lat, lon in radians
    lat_rad = np.deg2rad(ds["lat"].values).astype(np.float64)  # (lat,)
    lon_rad = np.deg2rad(ds["lon"].values).astype(np.float64)  # (lon,)
    n_lat = lat_rad.size
    n_lon = lon_rad.size

    # Hour angles at window endpoints, shape (T, lon) after broadcast.
    # h(t, lon) = (utc_hour(t) - 12) * pi/12 + lon_rad
    h_start = (hour - 12.0) * np.pi / 12.0                    # (T,)
    h_end = (hour + dt_h - 12.0) * np.pi / 12.0               # (T,)
    h1 = h_start[:, None] + lon_rad[None, :]                  # (T, lon)
    h2 = h_end[:, None] + lon_rad[None, :]                    # (T, lon)

    # Sunrise/sunset hour angle per (T, lat)
    cos_h0 = -np.tan(lat_rad)[None, :] * np.tan(dec)[:, None]  # (T, lat)
    polar_night = cos_h0 > 1.0
    polar_day = cos_h0 < -1.0
    h0 = np.arccos(np.clip(cos_h0, -1.0, 1.0))                # (T, lat), 0 at polar night, pi at polar day

    # Broadcast to (T, lat, lon) for the clip-to-daylit step
    h1_full = h1[:, None, :]  # (T, 1, lon) broadcasts with (T, lat, 1)
    h2_full = h2[:, None, :]
    h0_lat = h0[:, :, None]   # (T, lat, 1)

    h1_lit = np.clip(h1_full, -h0_lat, h0_lat)  # (T, lat, lon)
    h2_lit = np.clip(h2_full, -h0_lat, h0_lat)

    sin_lat = np.sin(lat_rad)[None, :, None]
    cos_lat = np.cos(lat_rad)[None, :, None]
    sin_dec = np.sin(dec)[:, None, None]
    cos_dec = np.cos(dec)[:, None, None]

    # Analytic integral of (sin(lat)sin(dec) + cos(lat)cos(dec)cos(h)) dh
    integ_day = (
        sin_lat * sin_dec * (h2_lit - h1_lit)
        + cos_lat * cos_dec * (np.sin(h2_lit) - np.sin(h1_lit))
    )
    # Polar day: integrate over the full window (no clipping to [-h0, h0])
    integ_polar_day = (
        sin_lat * sin_dec * (h2_full - h1_full)
        + cos_lat * cos_dec * (np.sin(h2_full) - np.sin(h1_full))
    )
    # Polar night: integrand = 0 over the whole window
    #   (h1_lit == h2_lit == +/-h0 clipped at boundaries anyway, so integ_day
    #    already evaluates to 0; we still mask explicitly for clarity.)
    polar_day_full = np.broadcast_to(polar_day[:, :, None], integ_day.shape)
    polar_night_full = np.broadcast_to(polar_night[:, :, None], integ_day.shape)
    integ = np.where(polar_day_full, integ_polar_day, integ_day)
    integ = np.where(polar_night_full, 0.0, integ)

    dh_rad = dt_h * np.pi / 12.0  # window length in radians (6h → pi/2)
    rsdt = float(opts.solar_constant) * (dist_factor[:, None, None] ** 2) * integ / dh_rad
    rsdt = np.maximum(rsdt, 0.0)  # defensive against float noise

    return xr.DataArray(
        rsdt.astype(np.float32),
        dims=("time", "lat", "lon"),
        coords={"time": ds["time"], "lat": ds["lat"], "lon": ds["lon"]},
        attrs={
            "units": "W m-2",
            "long_name": "toa_incident_shortwave_flux",
            "standard_name": "toa_incoming_shortwave_flux",
        },
    )


# ---------------------------------------------------------------------------
# Validation on write
# ---------------------------------------------------------------------------
def _area_weights(lat_da: xr.DataArray) -> xr.DataArray:
    w = np.cos(np.deg2rad(lat_da))
    return w / w.sum()


def _validate(out: xr.Dataset, lsm: xr.DataArray, opts) -> None:
    # sst hard checks (ignore NaN for range; use numpy nan-aware reductions)
    sst_vals = out["sst"].values
    ocean_mask = np.isfinite(sst_vals)
    if not ocean_mask.any():
        raise RuntimeError("sst validation: no finite values present (all NaN).")
    smin = float(np.nanmin(sst_vals))
    smax = float(np.nanmax(sst_vals))
    logger.info("  sst range over ocean: [%.2f, %.2f] K (bounds [%.2f, %.2f])",
                smin, smax, SST_MIN_K, SST_MAX_K)
    if smin < SST_MIN_K:
        raise RuntimeError(f"sst min {smin:.3f} K < {SST_MIN_K}")
    if smax > SST_MAX_K:
        raise RuntimeError(f"sst max {smax:.3f} K > {SST_MAX_K}")

    # Mask application: NaN fraction of sst should match global land fraction
    lsm0 = lsm.isel(time=0) if "time" in lsm.dims else lsm
    land_frac = float((lsm0 >= LAND_EPSILON).sum() / lsm0.size)
    sst_nan_frac = float(np.isnan(sst_vals).sum() / sst_vals.size)
    one_cell_tol = 1.0 / float(lsm0.size)
    diff = abs(sst_nan_frac - land_frac)
    logger.info(
        "  sst NaN fraction: %.6f  |  lsm land fraction: %.6f  |  |diff|=%.3e  (tol 1 cell = %.3e)",
        sst_nan_frac, land_frac, diff, one_cell_tol,
    )
    if diff > one_cell_tol * 1.5:
        raise RuntimeError(
            f"sst NaN fraction {sst_nan_frac} does not match lsm land "
            f"fraction {land_frac} to within one grid cell ({one_cell_tol:.3e})"
        )

    # rsdt area-weighted annual mean soft check
    w = _area_weights(out["lat"])
    global_mean = float(
        (out["rsdt"] * w).sum(dim="lat").mean(dim="lon").mean(dim="time")
    )
    expected = float(opts.solar_constant) / 4.0
    rel = abs(global_mean - expected) / expected
    tol = (RSDT_GLOBAL_MEAN_TOLERANCE_ASTRO if opts.rsdt_method == "astronomical"
           else RSDT_GLOBAL_MEAN_TOLERANCE_ARITH)
    logger.info("  rsdt global mean: %.3f W m-2  (expected %.3f, |rel|=%.4f, tol=%.4f)",
                global_mean, expected, rel, tol)
    if rel > tol:
        logger.warning(
            "rsdt global mean %s deviates from solar_constant/4 (%s) by %.3f%% "
            "(tol %.3f%%) — soft check only, not failing.",
            global_mean, expected, rel * 100, tol * 100,
        )

    # sic pass-through range
    smin_sic = float(out["sic"].min())
    smax_sic = float(out["sic"].max())
    logger.info("  sic range after clip: [%.4f, %.4f]", smin_sic, smax_sic)
    if smin_sic < 0.0 or smax_sic > 1.0:
        raise RuntimeError(f"sic out of [0,1] after clip: [{smin_sic}, {smax_sic}]")


# ---------------------------------------------------------------------------
# Per-task processing
# ---------------------------------------------------------------------------
def enumerate_tasks(sims, year_start, year_end):
    return [(sim, year) for sim in sims for year in range(year_start, year_end + 1)]


def process_one(sim: int, year: int, opts) -> None:
    input_path = Path(opts.input_root) / f"sim{sim}" / f"MOST.{year:04d}.nc"
    output_path = Path(opts.output_root) / f"sim{sim}" / f"boundary.{year:04d}.nc"

    if not input_path.exists():
        logger.warning("[sim%s/%04d] skipping (input missing: %s)", sim, year, input_path)
        return

    if output_path.exists() and not opts.overwrite:
        logger.info("[sim%s/%04d] skipping %s (exists; pass --overwrite to force)",
                    sim, year, output_path)
        return

    if opts.dry_run:
        logger.info("[sim%s/%04d] dry-run: would produce %s", sim, year, output_path)
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("[sim%s/%04d] reading %s", sim, year, input_path)
    with xr.open_dataset(input_path) as ds:
        sst = compute_sst(ds)
        if opts.rsdt_method == "arithmetic":
            rsdt = compute_rsdt_arithmetic(ds)
        else:
            rsdt = compute_rsdt_astronomical(ds, opts)
        sic_out = ds["sic"].clip(min=0.0, max=1.0).astype(np.float32)
        sic_out.attrs = {
            "units": "1",
            "long_name": "sea_ice_area_fraction",
            "standard_name": "sea_ice_area_fraction",
        }

        out = xr.Dataset({"sst": sst.astype(np.float32),
                          "rsdt": rsdt.astype(np.float32),
                          "sic": sic_out})
        out.attrs["rsdt_method"] = opts.rsdt_method
        if opts.rsdt_method == "astronomical":
            out.attrs["solar_constant_W_m2"] = float(opts.solar_constant)
            out.attrs["eccentricity"] = float(opts.eccentricity)
            out.attrs["obliquity_deg"] = float(opts.obliquity_deg)
            out.attrs["window_hours"] = float(opts.window_hours)

        logger.info("[sim%s/%04d] validating", sim, year)
        _validate(out, ds["lsm"], opts)

        fill = np.float32("nan")
        encoding = {v: {"dtype": "float32", "_FillValue": fill} for v in ("sst", "rsdt", "sic")}

        tmp_path = output_path.with_suffix(".nc.tmp")
        out.to_netcdf(tmp_path, encoding=encoding)
        tmp_path.replace(output_path)

    logger.info("[sim%s/%04d] wrote %s", sim, year, output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Emulator varying-boundary adaptor: postprocess NetCDF → "
                    "{sst, rsdt, sic} per-(sim, year) NetCDF.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--sims", required=True, type=int, nargs="+", metavar="SIM")
    p.add_argument("--years", required=True, type=int, nargs=2, metavar=("START", "END"))
    p.add_argument("--input-root", default=None, type=Path,
                   help="Postprocess output root (containing sim{NN}/MOST.{YYYY:04d}.nc). "
                        "Not required with --count-tasks.")
    p.add_argument("--output-root", default=None, type=Path,
                   help="Boundary output root (writes sim{NN}/boundary.{YYYY:04d}.nc). "
                        "Not required with --count-tasks.")
    p.add_argument("--rsdt-method", choices=("arithmetic", "astronomical"),
                   default="arithmetic",
                   help="arithmetic (default): rsdt = rst - rsut from PlaSim's own "
                        "TOA SW accounting. astronomical: analytic 6h-mean integration.")
    p.add_argument("--solar-constant", type=float, default=DEFAULT_SOLAR_CONSTANT_W_M2,
                   help="Solar constant in W m-2 (astronomical only).")
    p.add_argument("--eccentricity", type=float, default=DEFAULT_ECCENTRICITY,
                   help="Orbital eccentricity (astronomical only).")
    p.add_argument("--obliquity-deg", type=float, default=DEFAULT_OBLIQUITY_DEG,
                   help="Axial tilt in degrees (astronomical only).")
    p.add_argument("--window-hours", type=float, default=DEFAULT_WINDOW_HOURS,
                   help="Output cadence in hours (astronomical only).")
    p.add_argument("--task-index", type=int, default=None,
                   help="0-based index into the (sim, year) task list (for SLURM array dispatch).")
    p.add_argument("--count-tasks", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
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

    if args.task_index is not None:
        if not (0 <= args.task_index < len(tasks)):
            sys.exit(f"error: --task-index {args.task_index} out of range [0, {len(tasks)})")
        tasks = [tasks[args.task_index]]

    for sim, year in tasks:
        process_one(sim, year, args)


if __name__ == "__main__":
    main()
