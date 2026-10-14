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


def _resolve_init_time(result):
    """Return ``(init_time_value, labelling)`` for the ``init_time`` coordinate.

    Two labelling schemes, chosen by whether the pack carries a time anchor —
    ``h5.attrs["plasim_time_units"]``, read by
    ``rollout_driver._resolve_ic_provenance``:

    * **calendar** — anchor present (the PLaSim packs). Unchanged behaviour:
      ``anchor + time_plasim_at_ic`` as a ``datetime64[s]``.
    * **step_index** — anchor absent. The coordinate becomes the IC's global
      sample index, an integer.

    Why not date everything: the E3SM converter never writes that attribute
    (``convert_e3sm_to_makani_alldata.py`` writes ``source_root``/``year``/
    ``converter``/``lat_order``/``sst_units``/``level_naming`` and no anchor),
    and the pack is on a **noleap 365-day** calendar with a *split-cumulative*
    day count. Measured, not assumed: ``.pack_logs/test.log`` reports ``T=1460``
    for 2048, a leap year — 365x4, not 366x4. Adding those days onto a
    proleptic-Gregorian ``datetime64`` therefore drifts **one day per leap year
    crossed**; inside the two-year test split, 2049's dates would already be
    wrong by one. Fabricating a plausible-but-wrong date is worse than not
    dating the file.

    This has **zero effect on the model or on any metric** — lead time is a
    relative offset and is unchanged. It is the label on the output file, and
    lead-offset is the natural coordinate for a lagged ensemble anyway.
    → docs/2026-09-10_e3sm_inference_port_scope.md §2.3 (decision E).
    """
    anchor = str(getattr(result, "file_anchor", "") or "").strip()
    if not anchor:
        return np.int64(result.ic_global_idx), "step_index"
    # A present-but-malformed anchor is a packing bug, not a calendar choice:
    # let _parse_anchor_to_datetime64 raise rather than silently downgrading.
    init_time_np = _parse_anchor_to_datetime64(anchor)
    offset_s = int(round(result.time_plasim_at_ic * 86400))
    return init_time_np + np.timedelta64(offset_s, "s"), "calendar"


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

    # Orientation guard: the PlaSim grid is descending (North-first, row 0 =
    # +87.86 deg N). The data rows are written in that order, so the lat
    # coordinate MUST be strictly descending or the file is hemisphere-
    # mislabeled. Fail loud rather than silently corrupt coordinate-aware
    # downstream scoring.
    lat_arr = np.asarray(lat, dtype=np.float64)
    if H > 1 and not np.all(np.diff(lat_arr) < 0):
        raise ValueError(
            f"lat must be strictly descending (data is North-first); got "
            f"lat[0]={lat_arr[0]:.3f} lat[-1]={lat_arr[-1]:.3f}. "
            f"Refusing to write a mislabeled grid."
        )

    init_time, init_time_labelling = _resolve_init_time(result)

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
            lat=("lat", lat_arr),
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
            # Which scheme init_time is on, so a consumer never has to guess
            # whether a number is a date or an index. → _resolve_init_time.
            init_time_labelling=str(init_time_labelling),
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
    if init_time_labelling == "step_index":
        ds["init_time"].attrs["units"] = "1"
        ds["init_time"].attrs["description"] = (
            "IC global sample index -- NOT a date. The source pack carries no "
            "time anchor and is on a noleap calendar, so a proleptic-Gregorian "
            "date would drift a day per leap year crossed."
        )
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
