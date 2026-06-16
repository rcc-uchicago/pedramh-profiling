"""F.H — Convert long_inference year-bounded NetCDF -> AI-RES 53-channel scorer NetCDF.

Phase F.H of the group-code training track plan v5.

Input: a NetCDF written by ``long_inference.py``'s ``convert_ensemble_to_xarray``
(``/work2/.../PanguWeather/v2.0/long_inference.py:1051-1188``). Schema:
  - Filename:  ``{save_basename}_member{run_iter:03}_y{current_year:04}.nc``
  - dims:      ensemble_idx, time (1460 non-leap / 1464 leap), plev=10, lev=10, lat, lon
  - vars:
    pl, tas (surface)        dims=[ensemble_idx, time, lat, lon]
    pr_6h (diagnostic)       dims=[ensemble_idx, time, lat, lon]
    zg (plev upper-air)      dims=[ensemble_idx, time, plev, lat, lon]
    ta, ua, va, hus (sigma)  dims=[ensemble_idx, time, lev, lat, lon]
  - time coord: xr.date_range(year_jan1, next_year_jan1, freq='6h', inclusive='left')
                so time[0] = IC (lead 0), time[k] = prediction at lead k*6h.

Output: NetCDF in scripts/score_nwp.py-compatible schema (mirrors
``src/sfno_inference/nc_writer.py``).

Default ``--max-output-leads = 60`` (= 360h, scorecard horizon). v10 truth has
1455 frames (~363 days), so leads 1..60 are always within truth horizon.
Year 128 is unblocked.

Channel mapping (v10 ordering):
  state[0]=pl, state[1]=tas
  state[2..11]   <- ta[k, sigma_i in TOA->surface order]    (10 sigma levels)
  state[12..21]  <- ua[k, ...]
  state[22..31]  <- va[k, ...]
  state[32..41]  <- hus[k, ...]
  state[42..51]  <- zg[k, plev_j in 200..1000 hPa order]    (10 pressure levels)
  diagnostic[52] <- pr_6h[k]
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import h5py
import numpy as np
import xarray as xr

logger = logging.getLogger("convert_group_long_inference_to_aires_nc")

V10_STATE_NAMES = (
    ["pl", "tas"]
    + [f"ta{i+1}" for i in range(10)]
    + [f"ua{i+1}" for i in range(10)]
    + [f"va{i+1}" for i in range(10)]
    + [f"hus{i+1}" for i in range(10)]
    + [f"zg{p}" for p in (200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)]
)
V10_DIAG_NAMES = ["pr_6h"]
V10_CHANNEL_NAMES = V10_STATE_NAMES + V10_DIAG_NAMES   # 53


def _read_v10_truth_and_ic(
    test_h5: Path, ic_global_idx: int, K: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, str]:
    """Return (init_state[52, H, W], truth[K, 53, H, W], lat, lon, time_at_ic, anchor)."""
    with h5py.File(test_h5, "r") as f:
        n_t = int(f["fields_state"].shape[0])
        if ic_global_idx + K >= n_t:
            raise ValueError(
                f"v10 test {test_h5} has only {n_t} timesteps; can't read lead "
                f"K={K} from ic_global_idx={ic_global_idx}. Truth horizon = {n_t - 1 - ic_global_idx}."
            )
        init_state = np.asarray(f["fields_state"][ic_global_idx], dtype=np.float32)  # (52, H, W)
        truth = np.empty((K, 53, *init_state.shape[1:]), dtype=np.float32)
        for k in range(K):
            tgt = ic_global_idx + (k + 1)
            truth[k, :52] = np.asarray(f["fields_state"][tgt], dtype=np.float32)
            truth[k, 52]  = np.asarray(f["fields_diagnostic"][tgt, 0], dtype=np.float32)
        lat = np.asarray(f["lat"][:], dtype=np.float64)
        lon = np.asarray(f["lon"][:], dtype=np.float64)
        time_plasim_at_ic = float(f["time_plasim"][ic_global_idx])
        plasim_time_units = f.attrs.get("plasim_time_units", b"")
        if isinstance(plasim_time_units, bytes):
            plasim_time_units = plasim_time_units.decode()
        m = re.search(r"days since (\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})", plasim_time_units)
        file_anchor = m.group(1) if m else "0001-01-01 00:00:00"
    return init_state, truth, lat, lon, time_plasim_at_ic, file_anchor


def _build_prediction_from_long_inference(
    long_inf_ds: xr.Dataset, K: int, H: int, W: int, *, ensemble_idx: int = 0,
) -> np.ndarray:
    """Pack long_inference per-variable arrays into the v10 53-channel layout for leads 1..K.

    The long_inference NetCDF may have dims [ensemble_idx, time, ...] or just
    [time, ...] for single-member output. time[0] is IC and time[k] for k>=1
    is the prediction at lead k*6h. We extract leads 1..K.
    """
    pred = np.empty((K, 53, H, W), dtype=np.float32)

    def _member_values(name: str) -> np.ndarray:
        da = long_inf_ds[name]
        if "ensemble_idx" in da.dims:
            da = da.isel(ensemble_idx=ensemble_idx)
        return da.values

    # Each variable: select ensemble_idx and time slice [1, K+1).
    pl = _member_values("pl")        # (time, lat, lon)
    tas = _member_values("tas")
    pr = _member_values("pr_6h")     # (time, lat, lon)
    ta = _member_values("ta")        # (time, lev, lat, lon)
    ua = _member_values("ua")
    va = _member_values("va")
    hus = _member_values("hus")
    zg = _member_values("zg")        # (time, plev, lat, lon)

    if pl.shape[0] < K + 1:
        raise ValueError(
            f"long_inference NetCDF has only {pl.shape[0]} time entries; "
            f"need at least K+1={K+1} (lead 0 = IC + leads 1..K). "
            f"Either reduce --max-output-leads or check that long_inference completed the year."
        )

    for k in range(K):
        t = k + 1   # IC at idx 0; prediction at lead k*6h is at idx k+1=k+1 (... wait)
        # long_inference stores: output_surface[time_step_in_year+1] = inv_transform(...)
        # so output[1] = first prediction (lead 6h), output[K] = K-th prediction (lead K*6h).
        # We want lead 1..K -> indices 1..K in long_inf_ds.
        pred[k, 0]      = pl[t]
        pred[k, 1]      = tas[t]
        pred[k, 2:12]   = ta[t, :10]
        pred[k, 12:22]  = ua[t, :10]
        pred[k, 22:32]  = va[t, :10]
        pred[k, 32:42]  = hus[t, :10]
        pred[k, 42:52]  = zg[t, :10]
        pred[k, 52]     = pr[t]
    return pred


def _assert_finite(name: str, arr: np.ndarray) -> None:
    if not np.isfinite(arr).all():
        nan_count = int(np.isnan(arr).sum())
        inf_count = int(np.isinf(arr).sum())
        raise RuntimeError(
            f"{name}: {nan_count} NaN, {inf_count} Inf — long_inference produced "
            f"non-finite output. Check ckpt + boundary forcing files."
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--long-inference-nc", required=True, type=Path,
                        help="Year-bounded long_inference NetCDF "
                             "(e.g. long_rollout_year121_member000_y0121.nc).")
    parser.add_argument("--test-h5", required=True, type=Path,
                        help="v10 test holdout (e.g. .../test/MOST.0121.h5). "
                             "Provides truth + init_state.")
    parser.add_argument("--ic-global-idx", type=int, default=0,
                        help="Index in the v10 test h5 corresponding to the IC (default 0).")
    parser.add_argument("--max-output-leads", type=int, default=60,
                        help="Number of leads to write (default 60 = 360h scorecard horizon). "
                             "Override to see the full year (1459 / 1463), but the scorer will "
                             "see NaN truth beyond v10's 1454-frame horizon.")
    parser.add_argument("--out", required=True, type=Path,
                        help="Output 53-channel scorer-compatible NetCDF.")
    parser.add_argument("--ckpt-path", default="", type=str)
    parser.add_argument("--eval-sha7", default="phaseF", type=str)
    parser.add_argument("--data-sha7", default="v10_zgplev", type=str)
    parser.add_argument("--train-sha7", default="group_full", type=str)
    parser.add_argument("--run-tag", default="phaseF_full", type=str)
    parser.add_argument("--rollout-mode", default="nwp", type=str)
    parser.add_argument("--ensemble-idx", type=int, default=0,
                        help="Which ensemble member to use (default 0).")
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING"))
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    long_inf_ds = xr.open_dataset(args.long_inference_nc, decode_times=False)
    n_time = int(long_inf_ds["pl"].sizes["time"])
    H = int(long_inf_ds["pl"].sizes["lat"])
    W = int(long_inf_ds["pl"].sizes["lon"])
    K = min(args.max_output_leads, n_time - 1)
    if K < args.max_output_leads:
        logger.warning("Requested K=%d leads but long_inference NetCDF has only %d time entries "
                       "(K+1 max = %d). Truncating to K=%d.",
                       args.max_output_leads, n_time, n_time - 1, K)
    logger.info("long_inference: time=%d, grid=(%d, %d), K=%d", n_time, H, W, K)

    init_state, truth, lat, lon, time_plasim_at_ic, file_anchor = _read_v10_truth_and_ic(
        args.test_h5, args.ic_global_idx, K
    )
    _assert_finite("v10 init_state", init_state)
    _assert_finite("v10 truth (leads 1..K)", truth)

    pred = _build_prediction_from_long_inference(long_inf_ds, K, H, W,
                                                 ensemble_idx=args.ensemble_idx)
    _assert_finite("long_inference prediction (leads 1..K)", pred)

    # Build scorer-compatible Dataset, mirroring src/sfno_inference/nc_writer.py.
    init_time_iso = file_anchor.replace(" ", "T")
    init_time_dt = np.datetime64(init_time_iso, "s") + np.timedelta64(
        int(round(time_plasim_at_ic * 86400)), "s"
    )
    lead_time = np.arange(1, K + 1, dtype=np.int64) * 6
    channel_ic = list(V10_CHANNEL_NAMES[:52])

    ds = xr.Dataset(
        data_vars=dict(
            prediction=(("init_time", "lead_time", "channel", "lat", "lon"),
                        pred[np.newaxis, ...]),
            truth=(("init_time", "lead_time", "channel", "lat", "lon"),
                   truth[np.newaxis, ...]),
            init_state=(("init_time", "channel_ic", "lat", "lon"),
                        init_state[np.newaxis, ...]),
        ),
        coords=dict(
            init_time=("init_time", np.array([init_time_dt])),
            lead_time=("lead_time", lead_time),
            channel=("channel", list(V10_CHANNEL_NAMES)),
            channel_ic=("channel_ic", channel_ic),
            lat=("lat", lat),
            lon=("lon", lon),
        ),
        attrs=dict(
            ckpt_path=args.ckpt_path,
            eval_sha7=args.eval_sha7,
            data_sha7=args.data_sha7,
            train_sha7=args.train_sha7,
            run_tag=args.run_tag,
            ic_file=str(args.test_h5),
            ic_sample_idx=int(args.ic_global_idx),
            ic_global_idx=int(args.ic_global_idx),
            file_anchor=file_anchor,
            time_plasim_at_ic=float(time_plasim_at_ic),
            rollout_mode=args.rollout_mode,
            K=int(K),
            dt_hours=6,
            source_long_inference_nc=str(args.long_inference_nc),
            converter="convert_group_long_inference_to_aires_nc",
        ),
    )
    ds["lead_time"].attrs["units"] = "hours"
    ds["lead_time"].attrs["description"] = "lead time offset from init_time"
    ds["lat"].attrs["units"] = "degrees_north"
    ds["lon"].attrs["units"] = "degrees_east"
    ds["prediction"].attrs["units"] = "physical (de-z-scored)"
    ds["truth"].attrs["units"] = "physical (de-z-scored)"
    ds["init_state"].attrs["units"] = "physical (de-z-scored)"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    encoding = {v: {"zlib": True, "complevel": 4} for v in ("prediction", "truth", "init_state")}
    ds.to_netcdf(args.out, encoding=encoding, format="NETCDF4")
    long_inf_ds.close()
    logger.info("Wrote %s (%d KB, K=%d leads)", args.out, args.out.stat().st_size // 1024, K)
    return 0


if __name__ == "__main__":
    sys.exit(main())
