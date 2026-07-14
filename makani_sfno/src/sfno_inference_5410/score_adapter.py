"""Adapter: 5410 raw NetCDF + Derecho per-timestep h5 truth → score_nwp inference NC.

Per docs/2026-05-08_sfno_5410_scoring_plan.md (v4.4). Mirrors
``src/sfno_inference/nc_writer.py:113-…`` schema EXACTLY so that the
existing ``scripts/score_nwp.py`` runs unchanged on the adapted
output.

Inputs:
  * Raw 5410 NetCDF at ``upstream_raw/Y{Y}_s{s:04d}_member000_y{Y:04d}.nc``
    with vars ``pl, tas, pr_6h`` (time, lat, lon); ``ta, ua, va, hus``
    (time, lev=10, lat, lon); ``zg`` (time, plev=10, lat, lon).
    ``time=61`` = IC at index 0 + 60 forecast leads.
  * Derecho per-timestep truth at
    ``{truth_h5_dir}/{Y}_{ssss:04d}.h5`` with HDF5 group ``input``
    containing per-channel arrays ``(64, 128)``. Naming:
    ``pl, tas, pr_6h``, ``ta_<sigma>``×10, ``ua_<sigma>``×10,
    ``va_<sigma>``×10, ``hus_<sigma>``×10, ``zg_<plev>.0``×13
    (we slice the 10 plevs the climatology uses).

Outputs:
  * Adapted NetCDF at ``out_nc_path`` with the canonical schema:
    ``prediction[init_time=1, lead_time, channel=53, lat, lon]``,
    ``truth[init_time=1, lead_time, channel=53, lat, lon]``,
    ``init_state[init_time=1, channel_ic=52, lat, lon]``.
    Integer-hour ``lead_time = [6, 12, …, 360]``. Anchored datetime
    metadata: ``file_anchor = f"{Y:04d}-01-01 00:00:00"``,
    ``time_plasim_at_ic = s * 0.25`` (days), ``ic_file = f"{Y:04d}.h5"``,
    ``truth_h5_file = f"{Y}_{s:04d}.h5"``.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Mapping

import cftime
import h5py
import numpy as np
import xarray as xr


# --- Channel canon (53 channels in climatology order) -----------------
# Verified live 2026-05-08 against
# /scratch/.../sim52/baselines/climatology_proleptic_5410.nc::channel.
_SIGMA_LEVELS = (
    "0.03830000013113022", "0.11910000443458557", "0.21085000783205032",
    "0.3168500065803528",  "0.4368000030517578",  "0.5668000280857086",
    "0.6993500888347626",  "0.8233500719070435",  "0.9240999817848206",
    "0.983299970626831",
)
_PLEVS_HPA = (200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)
# 5410 raw NC plev coord values (Pa) in their natural order.
_PLEVS_PA = tuple(p * 100 for p in _PLEVS_HPA)


def canonical_channel_names() -> list[str]:
    """The 53 channels in climatology order: pl, tas, ta1-10, ua1-10,
    va1-10, hus1-10, zg{200..1000}, pr_6h."""
    names = ["pl", "tas"]
    for var in ("ta", "ua", "va", "hus"):
        names.extend(f"{var}{k+1}" for k in range(10))
    names.extend(f"zg{p}" for p in _PLEVS_HPA)
    names.append("pr_6h")
    assert len(names) == 53, f"channel count {len(names)} != 53"
    return names


def _h5_key_for_channel(name: str) -> str:
    """Map climatology channel name → key inside Derecho h5 ``input/`` group.

    pl, tas, pr_6h are direct keys.
    ta{1..10}, ua{...}, va{...}, hus{...} → ``<var>_<sigma_str>`` with
        sigma_str matching the float-formatted lev coord (e.g. "0.0383...").
    zg{200..1000} → ``zg_<plev_pa>.0`` (e.g. "zg_50000.0").
    """
    if name in ("pl", "tas", "pr_6h"):
        return name
    for var in ("ta", "ua", "va", "hus"):
        if name.startswith(var) and name[len(var):].isdigit():
            k = int(name[len(var):])
            if 1 <= k <= 10:
                return f"{var}_{_SIGMA_LEVELS[k - 1]}"
    if name.startswith("zg"):
        try:
            hpa = int(name[2:])
        except ValueError:
            pass
        else:
            if hpa in _PLEVS_HPA:
                return f"zg_{hpa * 100}.0"
    raise ValueError(f"no Derecho h5 key mapping for channel {name!r}")


def _read_raw_channel(raw_ds: xr.Dataset, name: str, t_index: int | slice) -> np.ndarray:
    """Read one channel from the 5410 raw NetCDF at the given time index/slice.

    Returns a numpy array shaped ``(..., lat, lon)`` (extra leading dim
    if t_index is a slice).
    """
    if name in ("pl", "tas", "pr_6h"):
        return raw_ds[name].isel(time=t_index).values
    for var in ("ta", "ua", "va", "hus"):
        if name.startswith(var) and name[len(var):].isdigit():
            k = int(name[len(var):])
            if 1 <= k <= 10:
                return raw_ds[var].isel(time=t_index, lev=k - 1).values
    if name.startswith("zg"):
        hpa = int(name[2:])
        if hpa in _PLEVS_HPA:
            return raw_ds["zg"].isel(time=t_index).sel(plev=hpa * 100).values
    raise ValueError(f"no raw NC slicing rule for channel {name!r}")


def _read_h5_channel(truth_input: h5py.Group, name: str) -> np.ndarray:
    """Read one (lat, lon) slice from the Derecho h5 ``input`` group."""
    return truth_input[_h5_key_for_channel(name)][:]


def _stack_53_channels_from_h5(truth_h5_path: Path,
                                channel_names: list[str]) -> np.ndarray:
    """Open a Derecho per-timestep h5 and return ``(53, H, W)`` in canonical order."""
    with h5py.File(truth_h5_path, "r") as f:
        inp = f["input"]
        return np.stack(
            [_read_h5_channel(inp, name) for name in channel_names], axis=0
        ).astype(np.float32)


def _stack_53_channels_from_raw(raw_ds: xr.Dataset, t_index: int,
                                 channel_names: list[str]) -> np.ndarray:
    """Read a single time step from the 5410 raw NC into ``(53, H, W)``."""
    return np.stack(
        [_read_raw_channel(raw_ds, name, t_index) for name in channel_names],
        axis=0,
    ).astype(np.float32)


def adapt_5410_ic_to_score_nwp(
    *,
    raw_nc_path: Path,
    truth_h5_dir: Path,
    Y: int,
    s: int,
    K: int,
    out_nc_path: Path,
    ckpt_path: str = "",
    eval_sha7: str = "",
    data_sha7: str = "",
    train_sha7: str = "",
    run_tag: str = "",
) -> None:
    """Convert one 5410 raw NetCDF + truth + IC state → score_nwp inference NC.

    Schema mirrors ``src/sfno_inference/nc_writer.py:113-…`` so the
    existing ``scripts/score_nwp.py`` runs unchanged.
    """
    raw_nc_path = Path(raw_nc_path)
    truth_h5_dir = Path(truth_h5_dir)
    out_nc_path = Path(out_nc_path)
    out_nc_path.parent.mkdir(parents=True, exist_ok=True)

    channel_names = canonical_channel_names()
    n_chan = len(channel_names)              # 53
    n_chan_ic = n_chan - 1                   # 52 (drops pr_6h)

    # --- prediction: 5410 raw NC at time[1:K+1] ---------------------
    with xr.open_dataset(raw_nc_path, decode_times=False) as raw_ds:
        if raw_ds.sizes["time"] < K + 1:
            raise ValueError(
                f"raw NC {raw_nc_path} has time={raw_ds.sizes['time']}, "
                f"need at least K+1={K + 1}"
            )
        H = int(raw_ds.sizes["lat"])
        W = int(raw_ds.sizes["lon"])
        lat = np.asarray(raw_ds["lat"].values, dtype=np.float64)
        lon = np.asarray(raw_ds["lon"].values, dtype=np.float64)

        # Build prediction[K, 53, H, W] by per-channel slicing of times 1..K.
        prediction = np.empty((K, n_chan, H, W), dtype=np.float32)
        for k_lead in range(K):
            t_idx = k_lead + 1   # raw time=0 is IC; first forecast is time=1
            prediction[k_lead] = _stack_53_channels_from_raw(
                raw_ds, t_idx, channel_names,
            )

    # --- truth + truth_sic: Derecho per-timestep h5 at samples s+1..s+K
    truth = np.empty((K, n_chan, H, W), dtype=np.float32)
    truth_sic = np.empty((K, H, W), dtype=np.float32)
    for k_lead in range(K):
        s_target = s + (k_lead + 1)   # lead_time = (k_lead + 1) * 6 h
        truth_h5 = truth_h5_dir / f"{Y}_{s_target:04d}.h5"
        if not truth_h5.is_file():
            raise FileNotFoundError(
                f"missing truth h5 for Y={Y} sample s+{k_lead + 1}={s_target}: "
                f"{truth_h5}"
            )
        truth[k_lead] = _stack_53_channels_from_h5(truth_h5, channel_names)
        with h5py.File(truth_h5, "r") as f:
            truth_sic[k_lead] = np.asarray(f["input/sic"][...], dtype=np.float32)

    # --- init_state: Derecho h5 at sample s, 52 state channels only --
    ic_h5 = truth_h5_dir / f"{Y}_{s:04d}.h5"
    if not ic_h5.is_file():
        raise FileNotFoundError(f"missing IC h5 for Y={Y} s={s}: {ic_h5}")
    init_state = _stack_53_channels_from_h5(ic_h5, channel_names)[:n_chan_ic]
    # init_state.shape == (52, H, W) — drops pr_6h at index 52.

    # --- coords ------------------------------------------------------
    # init_time uses cftime so years < 1582 round-trip correctly.
    init_dt = cftime.DatetimeProlepticGregorian(
        Y, 1, 1, 0, has_year_zero=True,
    ) + dt.timedelta(hours=s * 6)
    init_time_arr = np.array([init_dt], dtype=object)
    lead_time = np.arange(1, K + 1, dtype=np.int64) * 6   # [6, 12, ..., 360]

    channel_ic = channel_names[:n_chan_ic]

    out_ds = xr.Dataset(
        data_vars=dict(
            prediction=(
                ("init_time", "lead_time", "channel", "lat", "lon"),
                prediction[np.newaxis, ...],
            ),
            truth=(
                ("init_time", "lead_time", "channel", "lat", "lon"),
                truth[np.newaxis, ...],
            ),
            init_state=(
                ("init_time", "channel_ic", "lat", "lon"),
                init_state[np.newaxis, ...],
            ),
            truth_sic=(
                ("init_time", "lead_time", "lat", "lon"),
                truth_sic[np.newaxis, ...],
            ),
        ),
        coords=dict(
            init_time=("init_time", init_time_arr),
            lead_time=("lead_time", lead_time),
            channel=("channel", list(channel_names)),
            channel_ic=("channel_ic", list(channel_ic)),
            lat=("lat", lat),
            lon=("lon", lon),
        ),
        attrs=dict(
            ckpt_path=str(ckpt_path),
            eval_sha7=str(eval_sha7),
            data_sha7=str(data_sha7),
            train_sha7=str(train_sha7),
            run_tag=str(run_tag),
            # ic_file uses XXXX.h5 form so score_nwp.py:139's
            # ``replace("MOST.","").replace(".h5","")`` yields "0121"
            # (clean year) not "121_0000".
            ic_file=f"{Y:04d}.h5",
            truth_h5_file=f"{Y}_{s:04d}.h5",   # provenance only
            ic_sample_idx=int(s),
            ic_global_idx=int(s),
            # file_anchor MUST match score_nwp.py:92 regex
            # (YYYY-MM-DD HH:MM:SS).
            file_anchor=f"{Y:04d}-01-01 00:00:00",
            time_plasim_at_ic=float(s) * 0.25,   # days; 6h step → 0.25 d
            rollout_mode="nwp",
            K=int(K),
            dt_hours=6,
        ),
    )

    # Var-level attrs (mirror nc_writer.py).
    out_ds["lead_time"].attrs["units"] = "hours"
    out_ds["lead_time"].attrs["description"] = "lead time offset from init_time"
    out_ds["lat"].attrs["units"] = "degrees_north"
    out_ds["lon"].attrs["units"] = "degrees_east"
    out_ds["prediction"].attrs["units"] = "physical (group conventions)"
    out_ds["truth"].attrs["units"] = "physical (group conventions)"
    out_ds["init_state"].attrs["units"] = "physical (group conventions)"
    out_ds["truth_sic"].attrs["units"] = "fraction"
    out_ds["truth_sic"].attrs["description"] = (
        "Truth sea-ice fraction at each lead; NaN over land. "
        "Downstream tas_no_ice mask uses sic >= 0.15 to drop sea-ice cells."
    )

    out_ds.to_netcdf(out_nc_path)


__all__ = (
    "canonical_channel_names",
    "adapt_5410_ic_to_score_nwp",
)
