"""Convert score-function rollout NetCDF -> AI-RES 53-channel scorer-compatible NetCDF.

Phase E.2 of the group-code training track plan v5.

Input: a NetCDF written by ``GroupEmulator.save_rollout_netcdf`` (see
``src/sfno_training_group/score_function/group_emulator.py``). Schema:
  - time     coord (T = K+1 entries; t=0 is the IC)
  - diag_time coord (K entries; pr_6h is between-step accumulation)
  - sigma    coord (10 sigma levels for ta/ua/va/hus)
  - lev      coord (10 zg pressure levels in Pa)
  - vars: pl, tas (time, lat, lon); ta, ua, va, hus (time, sigma, lat, lon);
          zg (time, lev, lat, lon); pr_6h (diag_time, lat, lon).

Output: NetCDF in the schema produced by ``src/sfno_inference/nc_writer.py``:
  - dims init_time=1, lead_time=K, channel=53, channel_ic=52, lat, lon
  - data_vars: prediction, truth, init_state
  - coords: init_time (datetime64), lead_time (hours int64), channel (53 names),
            channel_ic (52 names), lat, lon
  - attrs: ckpt_path, eval_sha7, data_sha7, train_sha7, run_tag, ic_file,
           ic_sample_idx, ic_global_idx, file_anchor, time_plasim_at_ic,
           rollout_mode, K, dt_hours.

Truth + init_state come from the v10 test holdout h5 (e.g.
``MOST.0121.h5``) — same source the existing scorer expects, ensuring the
converter's output is bit-compatible with ``score_nwp.py``.

Channel mapping (v10 ordering):
  state[0]=pl, state[1]=tas
  state[2..11]   <- ta[k+1, sigma_i in TOA->surface order]
  state[12..21]  <- ua[k+1, ...]
  state[22..31]  <- va[k+1, ...]
  state[32..41]  <- hus[k+1, ...]
  state[42..51]  <- zg[k+1, plev_j in 200..1000 hPa order]
  diagnostic[52] <- pr_6h[k]   (rollout's diag_time has K entries; aligned to lead k+1)
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

logger = logging.getLogger("convert_group_inference_to_aires_nc")

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
    test_h5: Path, ic_global_idx: int, K: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return (init_state[52, H, W], truth[K, 53, H, W]) from a v10 test h5."""
    with h5py.File(test_h5, "r") as f:
        n_t = int(f["fields_state"].shape[0])
        if ic_global_idx + K >= n_t:
            raise ValueError(
                f"v10 test {test_h5} has only {n_t} timesteps; can't read lead "
                f"K={K} from ic_global_idx={ic_global_idx}."
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


def _build_prediction(rollout_ds: xr.Dataset, K: int, H: int, W: int) -> np.ndarray:
    """Pack the wrapper's per-variable arrays into the 53-channel layout."""
    pred = np.empty((K, 53, H, W), dtype=np.float32)
    # rollout time has K+1 entries; t=0 is IC, t=k+1 corresponds to lead k+1*6h.
    pl = rollout_ds["pl"].values         # (T, H, W)
    tas = rollout_ds["tas"].values
    ta = rollout_ds["ta"].values         # (T, sigma=10, H, W)
    ua = rollout_ds["ua"].values
    va = rollout_ds["va"].values
    hus = rollout_ds["hus"].values
    zg = rollout_ds["zg"].values         # (T, lev=10, H, W)
    pr = rollout_ds["pr_6h"].values      # (diag_time=K, H, W)

    for k in range(K):
        t = k + 1
        pred[k, 0] = pl[t]
        pred[k, 1] = tas[t]
        pred[k, 2:12]   = ta[t, :10]
        pred[k, 12:22]  = ua[t, :10]
        pred[k, 22:32]  = va[t, :10]
        pred[k, 32:42]  = hus[t, :10]
        pred[k, 42:52]  = zg[t, :10]
        pred[k, 52]     = pr[k]
    return pred


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-nc", required=True, type=Path,
                        help="Wrapper output (e.g. score_fn_rollout.nc).")
    parser.add_argument("--test-h5", required=True, type=Path,
                        help="v10 test holdout (e.g. .../test/MOST.0121.h5).")
    parser.add_argument("--ic-global-idx", type=int, default=0,
                        help="Index in the v10 test h5 corresponding to the IC.")
    parser.add_argument("--out", required=True, type=Path,
                        help="Output 53-channel scorer-compatible NetCDF.")
    parser.add_argument("--ckpt-path", default="", type=str)
    parser.add_argument("--eval-sha7", default="phase1", type=str)
    parser.add_argument("--data-sha7", default="v10", type=str)
    parser.add_argument("--train-sha7", default="group_smoke", type=str)
    parser.add_argument("--run-tag", default="phase1_smoke", type=str)
    parser.add_argument("--rollout-mode", default="nwp", type=str)
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING"))
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    rollout_ds = xr.open_dataset(args.rollout_nc, decode_times=False)
    K = int(rollout_ds["pr_6h"].shape[0])
    H, W = rollout_ds["pl"].shape[-2:]
    logger.info("rollout: K=%d, grid=(%d, %d), state=%s",
                K, H, W, rollout_ds.attrs.get("loaded_state_kind", "?"))

    init_state, truth, lat, lon, time_plasim_at_ic, file_anchor = _read_v10_truth_and_ic(
        args.test_h5, args.ic_global_idx, K
    )
    pred = _build_prediction(rollout_ds, K, H, W)

    # Build the scorer-compatible Dataset, mirroring src/sfno_inference/nc_writer.py.
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
            source_rollout_nc=str(args.rollout_nc),
            loaded_state_kind=str(rollout_ds.attrs.get("loaded_state_kind", "?")),
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
    rollout_ds.close()
    logger.info("Wrote %s (%d KB)", args.out, args.out.stat().st_size // 1024)
    return 0


if __name__ == "__main__":
    sys.exit(main())
