"""nc_writer — physical-units NetCDF for one rollout result.

Implements docs/sfno_eval_plan.md §B.4. The schema:

    dims:
      init_time   = 1
      lead_time   = K              # K predictions at leads {1..K} × 6 h
      channel     = 53             # 52 state + 1 diagnostic (pr_6h)
      channel_ic  = 52             # IC has no diagnostic
      lat         = H              # 64 for the 64x128 grid
      lon         = W              # 128

    coords:
      init_time   = absolute datetime (parsed from h5 attr)
      lead_time   = np.arange(1, K+1) * 6  hours
      channel     = list of 53 channel names from config
      channel_ic  = channel[:52]
      lat         = legendre-gauss latitudes (read from training metadata)
      lon         = equiangular longitudes (read from training metadata)

    variables:
      prediction(init_time, lead_time, channel, lat, lon)   — physical units
      truth(init_time, lead_time, channel, lat, lon)         — physical units
      init_state(init_time, channel_ic, lat, lon)            — physical units

    global_attrs:
      ckpt_path, eval_sha7, data_sha7, train_sha7, run_tag,
      ic_file, ic_sample_idx, ic_global_idx, file_anchor,
      time_plasim_at_ic, rollout_mode, K, dt_hours
"""
from __future__ import annotations

import re
from datetime import timedelta
from pathlib import Path
from typing import Sequence

import numpy as np
import xarray as xr


_ANCHOR_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})\s+(\d{2}):(\d{2}):(\d{2})")


def _parse_anchor_to_datetime64(anchor: str):
    """Parse '0YYY-08-01 00:00:00' to a numpy datetime64.

    PlaSim uses proleptic-Gregorian dates with year < 1000. NumPy
    datetime64 supports the proleptic Gregorian calendar but requires a
    valid ISO date string. The leading-zero year format from the h5
    files (``"0126-08-01 ..."``) is accepted by ``np.datetime64`` as
    long as we keep the 4-digit zero-padded year. We normalise to
    ISO 8601 with a 'T' separator.
    """
    m = _ANCHOR_RE.match(anchor)
    if m is None:
        raise ValueError(f"unparseable anchor: {anchor!r}")
    Y, M, D, h, mi, s = m.groups()
    iso = f"{Y}-{M}-{D}T{h}:{mi}:{s}"
    return np.datetime64(iso, "s")


def write_rollout_nc(
    out_path,
    *,
    result,
    channel_names: Sequence[str],
    lat: Sequence[float] | np.ndarray,
    lon: Sequence[float] | np.ndarray,
    ckpt_path: str,
    eval_sha7: str,
    data_sha7: str,
    train_sha7: str,
    run_tag: str,
    rollout_mode: str = "nwp",
    dt_hours: int = 6,
) -> Path:
    """Write one ``RolloutResult`` to NetCDF in physical units.

    Returns the resolved output path.
    """
    from sfno_inference.rollout_driver import RolloutResult

    if not isinstance(result, RolloutResult):
        raise TypeError(f"result must be RolloutResult, got {type(result).__name__}")

    K = result.K
    pred = result.prediction.numpy()      # (K, 53, H, W)
    truth = result.truth.numpy()           # (K, 53, H, W)
    init_state = result.init_state.numpy() # (52, H, W)

    n_chan = pred.shape[1]
    n_chan_ic = init_state.shape[0]
    H, W = pred.shape[-2], pred.shape[-1]

    if len(channel_names) != n_chan:
        raise ValueError(
            f"len(channel_names)={len(channel_names)} but predictions have {n_chan} channels"
        )
    if len(lat) != H or len(lon) != W:
        raise ValueError(
            f"lat/lon shape ({len(lat)}, {len(lon)}) does not match prediction grid ({H}, {W})"
        )

    init_time_np = _parse_anchor_to_datetime64(result.file_anchor)
    init_time = init_time_np + np.timedelta64(int(round(result.time_plasim_at_ic * 86400)), "s")

    lead_time = np.arange(1, K + 1, dtype=np.int64) * dt_hours  # hours; integer

    # Channel-IC coord: states only (drops the 53rd diagnostic name).
    channel_ic = list(channel_names[:n_chan_ic])

    data_vars = {
        "prediction": (
            ("init_time", "lead_time", "channel", "lat", "lon"),
            pred[np.newaxis, ...],   # add init_time axis of length 1
        ),
        "truth": (
            ("init_time", "lead_time", "channel", "lat", "lon"),
            truth[np.newaxis, ...],
        ),
        "init_state": (
            ("init_time", "channel_ic", "lat", "lon"),
            init_state[np.newaxis, ...],
        ),
    }
    if result.truth_sic is not None:
        truth_sic = result.truth_sic.numpy().astype(np.float32, copy=False)
        data_vars["truth_sic"] = (
            ("init_time", "lead_time", "lat", "lon"),
            truth_sic[np.newaxis, ...],
        )

    ds = xr.Dataset(
        data_vars=data_vars,
        coords=dict(
            init_time=("init_time", np.array([init_time])),
            lead_time=("lead_time", lead_time),
            channel=("channel", list(channel_names)),
            channel_ic=("channel_ic", channel_ic),
            lat=("lat", np.asarray(lat, dtype=np.float64)),
            lon=("lon", np.asarray(lon, dtype=np.float64)),
        ),
        attrs=dict(
            ckpt_path=str(ckpt_path),
            eval_sha7=str(eval_sha7),
            data_sha7=str(data_sha7),
            train_sha7=str(train_sha7),
            run_tag=str(run_tag),
            ic_file=str(result.ic_file),
            ic_sample_idx=int(result.ic_sample_idx),
            ic_global_idx=int(result.ic_global_idx),
            file_anchor=str(result.file_anchor),
            time_plasim_at_ic=float(result.time_plasim_at_ic),
            rollout_mode=str(rollout_mode),
            K=int(K),
            dt_hours=int(dt_hours),
        ),
    )

    # Variable-level attrs.
    # Note: we deliberately do NOT use "hours since <reference>" as the
    # units string because xarray's CF-conventions decoder would try to
    # interpret lead_time as an absolute calendar coordinate. lead_time
    # is a relative offset; storing it as a plain integer ``hours`` is
    # both correct and round-trippable.
    ds["lead_time"].attrs["units"] = "hours"
    ds["lead_time"].attrs["description"] = "lead time offset from init_time"
    ds["lat"].attrs["units"] = "degrees_north"
    ds["lon"].attrs["units"] = "degrees_east"
    ds["prediction"].attrs["units"] = "physical (de-z-scored)"
    ds["truth"].attrs["units"] = "physical (de-z-scored)"
    ds["init_state"].attrs["units"] = "physical (de-z-scored)"
    if "truth_sic" in ds.variables:
        ds["truth_sic"].attrs["units"] = "fraction"
        ds["truth_sic"].attrs["description"] = (
            "Truth sea-ice fraction at each lead; NaN over land. "
            "Downstream tas_no_ice mask uses sic >= 0.15 to drop sea-ice cells."
        )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # zlib compression keeps each NetCDF roughly the size advertised in
    # §4 layout (~92 MB per NWP IC; ~2.36 GB per climate IC).
    base_vars = ("prediction", "truth", "init_state")
    encoded_vars = base_vars + (("truth_sic",) if "truth_sic" in ds.variables else ())
    encoding = {v: {"zlib": True, "complevel": 4} for v in encoded_vars}
    ds.to_netcdf(out_path, encoding=encoding, format="NETCDF4")
    return out_path
