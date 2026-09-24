"""climate_driver — constant-memory, cross-file, multi-year SFNO rollout.

Sibling of :mod:`sfno_inference.rollout_driver` (which it does NOT modify).
``rollout_one_ic`` fetches the whole K-frame block, keeps every prediction and
``cat``s them: ~6 K-frame copies on the GPU, OOM at K=200 on a 40 GB A100 (probe
7648967). A 5-year run from Oct 2044 is 7,667 steps. This driver:

  - carries **one state**, and streams forcing in chunks of ``chunk_len`` frames;
  - crosses year files (the dataset's filename order is one continuous axis);
  - reduces to climate statistics on the fly (float64, z-space) and writes no
    per-step fields.

Design reference: ``polaris_makani_streaming_driver_handoff.md`` §3.

**The step body is ``rollout_one_ic``'s, unchanged** (cache → flatten →
``wrapper(inpt)`` under the same autocast → ``append_history(inpt, pred, i)``).
That is what keeps 107→101, the PRECT strip before feedback, and bf16 identical
to training (CLAUDE.md #1). The only difference is *where the forcing block comes
from*: one chunk at a time instead of one ``dataset[idx]``. Two stock
``Preprocessor2D`` traps make that difference dangerous (handoff §2a):

  1. ``append_history(x1, x2, step)`` copies target forcing ``step`` into the input
     only ``if step < unpredicted_tar_eval.shape[1]`` (stock ``preprocessor.py:219``).
     An overrunning chunk-local index **silently keeps stale forcing**.
     → :func:`_check_local_step` on every step.
  2. ``cache_unpredicted_features(x, y, xz=None, yz)`` **sets the input forcing to
     None** (``:386-389``). → at each chunk boundary re-cache with ``xz`` = the
     current input forcing, and :func:`_check_forcing_cache` afterwards.

Accumulating in z-space is exact: de-normalisation is affine per channel, so
means de-normalise at the end. The stability monitor reports in units of the
run's ``global_stds`` (σ) and the global mean in physical units.

Channel-agnostic: every count comes from ``eval_params`` and the stats files.
"""
from __future__ import annotations

import bisect
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# noleap calendar (E3SM: 365 days x 4 = 1460 frames/year; NOT PlaSim's 1455/1459)
# ---------------------------------------------------------------------------

FRAMES_PER_YEAR = 1460
NOLEAP_MONTH_DAYS = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
PACK_SPLITS = ("train", "valid", "test")
STABILITY_THRESHOLDS = (3.0, 10.0)
STABILITY_METRICS = ("std_sigma", "anom_rms_sigma")


class ClimateDriverError(RuntimeError):
    """A precondition the driver refuses to run without."""


class ForcingIndexError(ClimateDriverError):
    """Chunk-local forcing index outside the cached block (handoff trap §2a.1)."""


class ForcingCacheError(ClimateDriverError):
    """Preprocessor forcing cache missing or mis-shaped (handoff trap §2a.2)."""


def month_edges(frames_per_year: int = FRAMES_PER_YEAR) -> list[int]:
    """Frame index of each noleap month start, plus the year end (13 entries)."""
    if frames_per_year % 365:
        raise ClimateDriverError(
            f"frames_per_year={frames_per_year} is not a whole number of frames per "
            "noleap day (365 d)")
    per_day = frames_per_year // 365
    edges = [0]
    for d in NOLEAP_MONTH_DAYS:
        edges.append(edges[-1] + d * per_day)
    return edges


def month_of_frame(frame: int, frames_per_year: int = FRAMES_PER_YEAR) -> int:
    """1-based noleap month of a frame within its year."""
    if not 0 <= frame < frames_per_year:
        raise ValueError(f"frame {frame} outside [0, {frames_per_year})")
    return bisect.bisect_right(month_edges(frames_per_year), frame)


def valid_time(start_year: int, start_frame: int, step: int,
               frames_per_year: int = FRAMES_PER_YEAR) -> tuple[int, int]:
    """(year, frame) of lead ``step`` (step 0 = the IC; step k = k·6 h later)."""
    a = start_frame + step
    return start_year + a // frames_per_year, a % frames_per_year


def step_of(start: tuple[int, int], target: tuple[int, int],
            frames_per_year: int = FRAMES_PER_YEAR) -> int:
    """Lead index at which ``target=(year, frame)`` is reached from ``start``."""
    return (target[0] - start[0]) * frames_per_year + target[1] - start[1]


def years_needed(start_year: int, start_frame: int, n_steps: int,
                 frames_per_year: int = FRAMES_PER_YEAR) -> list[int]:
    """Every year file touched from the IC through lead ``n_steps``."""
    last_year, _ = valid_time(start_year, start_frame, n_steps, frames_per_year)
    return list(range(start_year, last_year + 1))


# ---------------------------------------------------------------------------
# year files as one axis
# ---------------------------------------------------------------------------

def locate_year_files(pack: Path, years: Sequence[int],
                      splits: Sequence[str] = PACK_SPLITS) -> dict[int, Path]:
    """Find ``<pack>/<split>/<year>.h5`` for each year. Refuses a gap or a duplicate."""
    pack = Path(pack)
    found: dict[int, Path] = {}
    for y in years:
        hits = [pack / s / f"{y}.h5" for s in splits if (pack / s / f"{y}.h5").is_file()]
        if not hits:
            raise ClimateDriverError(
                f"YEAR_GAP: no {y}.h5 under {pack}/{{{','.join(splits)}}}; refusing to "
                "roll across a missing year")
        if len(hits) > 1:
            raise ClimateDriverError(f"YEAR_DUPLICATE: {y}.h5 in more than one split: {hits}")
        found[y] = hits[0]
    return found


def build_year_dir(year_files: dict[int, Path], dest: Path) -> Path:
    """Make ``dest`` a directory of **symlinks only**, ``YYYY.h5`` → the pack file.

    ``PlasimForcingDataset`` treats the files of a directory in filename order as
    one continuous axis, so this is how a rollout crosses 2044 → 2045 → …
    Refuses to touch a real file in ``dest``, and refuses stray ``.h5`` entries
    (a stale symlink to another year would silently join the axis).
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    wanted = {f"{y}.h5": Path(p).resolve() for y, p in year_files.items()}
    for entry in dest.iterdir():
        if entry.suffix != ".h5":
            continue
        if not entry.is_symlink():
            raise ClimateDriverError(f"YEAR_DIR_NOT_SYMLINK: {entry} is a real file; refusing")
        if entry.name not in wanted:
            raise ClimateDriverError(
                f"YEAR_DIR_STRAY: {entry} is not one of {sorted(wanted)}; remove it")
        if entry.resolve() != wanted[entry.name]:
            raise ClimateDriverError(
                f"YEAR_DIR_MISMATCH: {entry} -> {entry.resolve()}, expected {wanted[entry.name]}")
    for name, target in wanted.items():
        link = dest / name
        if not link.is_symlink():
            os.symlink(target, link)
    return dest


def file_years(dataset) -> list[int]:
    """Year of each dataset file, parsed from its ``YYYY.h5`` name."""
    try:
        return [int(Path(p).stem) for p in dataset.files_paths]
    except ValueError as exc:
        raise ClimateDriverError(
            f"dataset files are not YYYY.h5: {[Path(p).name for p in dataset.files_paths]}"
        ) from exc


def check_year_axis(dataset, frames_per_year: int = FRAMES_PER_YEAR) -> list[int]:
    """Refuse anything but consecutive years of exactly ``frames_per_year`` frames."""
    years = file_years(dataset)
    if years != list(range(years[0], years[0] + len(years))):
        raise ClimateDriverError(f"YEAR_GAP: dataset years are not consecutive: {years}")
    bad = [(y, n) for y, n in zip(years, dataset.n_samples_file) if n != frames_per_year]
    if bad:
        raise ClimateDriverError(
            f"FRAMES_PER_YEAR: expected {frames_per_year} frames per file, got {bad}")
    if int(getattr(dataset, "dt", 1)) != 1:
        raise ClimateDriverError(f"dataset.dt={dataset.dt}; the driver assumes one frame per step")
    return years


def timestamp_axis_mode(dataset) -> dict:
    """Do the files' own timestamps continue across files, or did the dataset
    synthesise a continuous axis (``plasim_forcing_dataset.py:133`` warning)?"""
    import h5py

    raw = []
    for p in dataset.files_paths:
        with h5py.File(p, "r") as f:
            raw.append(np.asarray(f[dataset.dataset_path].dims[0]["timestamp"][...],
                                  dtype=np.int64))
    cat = np.concatenate(raw)
    diffs = np.diff(cat)
    step = int(diffs[0]) if diffs.size else 0
    bad = np.flatnonzero(diffs != step)
    return {
        "mode": "continuous" if bad.size == 0 else "synthesized",
        "step_seconds": step,
        "first_bad_concat_index": int(bad[0]) if bad.size else -1,
        "file_first_last": [[int(r[0]), int(r[-1])] for r in raw],
    }


# ---------------------------------------------------------------------------
# forcing chunks — the one thing that differs from rollout_one_ic
# ---------------------------------------------------------------------------

def read_forcing_chunk(dataset, ic_global_idx: int, first_lead: int, n: int) -> torch.Tensor:
    """Forcing at leads ``first_lead .. first_lead+n-1``, as ``(n, C_f, H, W)`` float32.

    The same three operations, in the same order, as
    ``PlasimForcingDataset.get_sample_at_index`` (``:337-346``): ``_read_forcing``,
    normalise with the dataset's forcing stats, ``as_tensor`` + ``grid_converter``.
    Bitwise identical to the corresponding slice of ``tar_forcing`` by construction.
    """
    arr = dataset._read_forcing(ic_global_idx, first_lead, first_lead + n)
    arr = (arr - dataset.forcing_bias) / dataset.forcing_scale
    return dataset.grid_converter(torch.as_tensor(arr, dtype=torch.float32))


def _forcing_step_index(i: int) -> int:
    """Chunk-local index passed to ``append_history``.

    The identity. It is a function only so the §5.2 test can seed an off-by-one
    here and show both the guard and the streaming-vs-block comparison go red.
    """
    return i


def _boundary_input_forcing(preprocessor):
    """Input forcing to re-cache at a chunk boundary: the current one (trap §2a.2).

    Seam for the §5.3 test, which seeds ``None`` and a stale block here.
    """
    return preprocessor.get_unpredicted_features()[0]


def _check_local_step(local: int, preprocessor) -> None:
    n = int(preprocessor.unpredicted_tar_eval.shape[1])
    if not 0 <= local < n:
        raise ForcingIndexError(
            f"FORCING_INDEX_OUT_OF_RANGE: chunk-local step {local} not in [0, {n}); "
            "stock append_history would silently keep the previous step's forcing")


def _check_forcing_cache(preprocessor, n: int, n_forcing: int) -> None:
    if getattr(preprocessor, "training", False):
        raise ForcingCacheError("preprocessor is in training mode; eval caches would be ignored")
    u_in = preprocessor.unpredicted_inp_eval
    u_tar = preprocessor.unpredicted_tar_eval
    if u_in is None or u_tar is None:
        raise ForcingCacheError(
            "FORCING_CACHE_WIPED: input or target forcing is None after re-cache "
            "(cache_unpredicted_features with xz=None)")
    if u_in.shape[1] != 1 or u_in.shape[2] != n_forcing:
        raise ForcingCacheError(f"input forcing cache shape {tuple(u_in.shape)}; "
                                f"expected (1, 1, {n_forcing}, H, W)")
    if u_tar.shape[1] != n or u_tar.shape[2] != n_forcing:
        raise ForcingCacheError(f"target forcing cache shape {tuple(u_tar.shape)}; "
                                f"expected (1, {n}, {n_forcing}, H, W)")


# ---------------------------------------------------------------------------
# the streaming loop
# ---------------------------------------------------------------------------

@dataclass
class StreamInfo:
    n_steps_run: int
    stopped_early: bool
    feedback_dtype: str


def stream_rollout(
    *,
    wrapper,
    dataset,
    ic_global_idx: int,
    n_steps: int,
    chunk_len: int,
    eval_params,
    device,
    on_step: Callable[[int, torch.Tensor], bool],
    assert_contract: bool = True,
) -> StreamInfo:
    """Roll ``n_steps`` leads from ``ic_global_idx`` holding one state.

    ``on_step(k, pred)`` gets lead ``k`` (1-based) and the fp32 prediction
    ``(1, N_out, H, W)`` in z-space — exactly the tensor ``rollout_one_ic`` stashes
    for lead ``k``. Returning ``False`` stops the rollout before feedback.

    ``dataset`` must have ``n_future == 0`` (``valid_autoreg_steps=0``): the IC
    fetch then reads one target frame, not a K-frame block.
    """
    if dataset.n_future != 0:
        raise ClimateDriverError(
            f"dataset.n_future={dataset.n_future}; build it with valid_autoreg_steps=0")
    if chunk_len < 1 or n_steps < 1:
        raise ValueError(f"chunk_len={chunk_len}, n_steps={n_steps}: both must be >= 1")
    if ic_global_idx + n_steps >= dataset.n_samples_total:
        raise ClimateDriverError(
            f"lead {n_steps} from global idx {ic_global_idx} needs frame "
            f"{ic_global_idx + n_steps}, dataset has {dataset.n_samples_total}")

    preprocessor = wrapper.preprocessor  # the wrapper's own instance, as rollout_one_ic
    n_state = int(eval_params.n_state_channels)
    n_out = int(eval_params.N_out_channels)
    n_forcing = int(eval_params.n_forcing_channels)

    inp_state, _tar, inp_forcing, _tar_forcing = dataset[ic_global_idx]
    H, W = inp_state.shape[-2], inp_state.shape[-1]

    autocast_enabled = bool(eval_params.amp_enabled) and (torch.device(device).type == "cuda")
    autocast_dtype = eval_params.amp_dtype if autocast_enabled else torch.float32

    feedback_dtype = ""
    inpt = None
    done = 0
    while done < n_steps:
        n = min(chunk_len, n_steps - done)
        yz = read_forcing_chunk(dataset, ic_global_idx, done + 1, n).unsqueeze(0).to(device)
        if done == 0:
            x = inp_state.unsqueeze(0).to(device)
            xz = inp_forcing.unsqueeze(0).to(device)
        else:
            x = inpt
            xz = _boundary_input_forcing(preprocessor)
        inp, _ = preprocessor.cache_unpredicted_features(x, None, xz, yz)
        _check_forcing_cache(preprocessor, n, n_forcing)
        if done == 0:
            inpt = preprocessor.flatten_history(inp)
            if assert_contract:
                assert inpt.shape == (1, n_state, H, W), (
                    f"flattened input shape {tuple(inpt.shape)} != (1, {n_state}, {H}, {W})")

        # === step body: rollout_one_ic (rollout_driver.py:221-249), unchanged ===
        for i in range(n):
            local = _forcing_step_index(i)
            _check_local_step(local, preprocessor)
            k = done + i + 1

            with torch.inference_mode(), torch.amp.autocast(
                device_type=torch.device(device).type,
                enabled=autocast_enabled,
                dtype=autocast_dtype,
            ):
                pred = wrapper(inpt)

            if assert_contract:
                assert pred.shape == (1, n_out, H, W), (
                    f"step {k}: pred shape {tuple(pred.shape)} != (1, {n_out}, {H}, {W})")
            if not feedback_dtype:
                # Logged, not changed: G2 must match rollout_one_ic as it is (§2b).
                feedback_dtype = str(pred.dtype)

            if not on_step(k, pred.detach().to(torch.float32).clone()):
                return StreamInfo(k, True, feedback_dtype)

            inpt = preprocessor.append_history(inpt, pred, local)
            if assert_contract:
                assert inpt.shape == (1, n_state, H, W), (
                    f"step {k}: post-append_history inpt shape "
                    f"{tuple(inpt.shape)} != (1, {n_state}, {H}, {W})")
        done += n

    return StreamInfo(done, False, feedback_dtype)


# ---------------------------------------------------------------------------
# reductions + stability monitor
# ---------------------------------------------------------------------------

def load_stats_f64(path, n_out: int, name: str) -> np.ndarray:
    """Per-channel stats as float64 ``(n_out,)``. Refuses a channel-count mismatch."""
    a = np.load(path).astype(np.float64)
    if a.size != n_out or a.shape not in ((n_out,), (1, n_out, 1, 1)):
        raise ClimateDriverError(
            f"STATS_CHANNEL_MISMATCH: {name} {path} has shape {a.shape}; "
            f"model has N_out_channels={n_out}")
    return a.reshape(n_out)


def load_time_means_z(path, mean: np.ndarray, std: np.ndarray, H: int, W: int) -> np.ndarray:
    """Training-period time mean ``(C, H, W)`` in z-space (float64)."""
    tm = np.load(path).astype(np.float64)
    if tm.ndim == 4 and tm.shape[0] == 1:
        tm = tm[0]
    if tm.shape != (mean.size, H, W):
        raise ClimateDriverError(
            f"STATS_CHANNEL_MISMATCH: time_means {path} has shape {tm.shape}; "
            f"expected ({mean.size}, {H}, {W})")
    return (tm - mean[:, None, None]) / std[:, None, None]


@dataclass
class MonthRecord:
    year: int
    month: int
    count: int
    first_step: int
    mean_z: np.ndarray  # (C, H, W) float64


class ClimateReducer:
    """Float64 on-the-fly reductions, z-space, on ``device``.

    - whole-window time mean over leads ``>= score_start_step``;
    - current-month sum + count, handed to ``month_sink`` at each month edge;
    - per-lead area-weighted global mean, spatial std and RMS anomaly vs
      ``time_means_z``, for **every** lead including spin-up;
    - optional zonal mean per lead.

    ``update`` returns False on the first non-finite prediction and records it;
    that lead contributes to nothing.
    """

    def __init__(self, *, n_steps: int, n_out: int, H: int, W: int,
                 lat_weights: np.ndarray, time_means_z: np.ndarray | None,
                 score_start_step: int, device,
                 month_sink: Callable[[MonthRecord], None] | None = None,
                 zonal: bool = False):
        f64 = dict(dtype=torch.float64, device=device)
        self.n_steps, self.n_out = n_steps, n_out
        self.score_start_step = score_start_step
        self.w = torch.as_tensor(np.asarray(lat_weights, dtype=np.float64), **f64)
        if self.w.shape != (H,):
            raise ClimateDriverError(f"lat weights shape {tuple(self.w.shape)} != ({H},)")
        self.tm_z = None if time_means_z is None else torch.as_tensor(time_means_z, **f64)
        self.month_sink = month_sink

        self.global_mean_z = torch.full((n_steps, n_out), float("nan"), **f64)
        self.std_sigma = torch.full((n_steps, n_out), float("nan"), **f64)
        self.anom_rms_sigma = torch.full((n_steps, n_out), float("nan"), **f64)
        self.zonal_mean_z = (torch.full((n_steps, n_out, H), float("nan"),
                                        dtype=torch.float32, device=device) if zonal else None)
        self.time_sum = torch.zeros((n_out, H, W), **f64)
        self.time_count = 0
        self.month_sum = torch.zeros((n_out, H, W), **f64)
        self.month_key: tuple[int, int] | None = None
        self.month_count = 0
        self.month_first_step = -1
        self.n_months = 0

        self.last_step = 0
        self.truncated_at_step = -1
        self.truncated_channel = -1

    def _area_mean(self, x: torch.Tensor) -> torch.Tensor:
        """(C, H, W) → (C,), equiangular cos-lat weights (sum to 1)."""
        return x.mean(dim=-1) @ self.w

    def update(self, k: int, pred: torch.Tensor, year: int, month: int) -> bool:
        p = pred.reshape(pred.shape[-3:]).to(torch.float64)       # (C, H, W)
        finite = torch.isfinite(p)
        if not bool(finite.all()):
            self.truncated_at_step = k
            self.truncated_channel = int(torch.nonzero(~finite.flatten(1).all(dim=1))[0])
            return False
        i = k - 1
        zonal = p.mean(dim=-1)                                     # (C, H)
        gm = zonal @ self.w                                        # (C,)
        self.global_mean_z[i] = gm
        self.std_sigma[i] = torch.sqrt(self._area_mean((p - gm[:, None, None]) ** 2))
        if self.tm_z is not None:
            self.anom_rms_sigma[i] = torch.sqrt(self._area_mean((p - self.tm_z) ** 2))
        if self.zonal_mean_z is not None:
            self.zonal_mean_z[i] = zonal.to(torch.float32)
        self.last_step = k

        if k >= self.score_start_step:
            self.time_sum += p
            self.time_count += 1
            if self.month_key != (year, month):
                self.flush_month()
                self.month_key = (year, month)
                self.month_first_step = k
            self.month_sum += p
            self.month_count += 1
        return True

    def flush_month(self) -> None:
        if self.month_count == 0:
            return
        rec = MonthRecord(self.month_key[0], self.month_key[1], self.month_count,
                          self.month_first_step,
                          (self.month_sum / self.month_count).cpu().numpy())
        if self.month_sink is not None:
            self.month_sink(rec)
        self.n_months += 1
        self.month_sum.zero_()
        self.month_count = 0

    def time_mean_z(self) -> np.ndarray | None:
        if self.time_count == 0:
            return None
        return (self.time_sum / self.time_count).cpu().numpy()

    def series(self) -> dict[str, np.ndarray]:
        n = self.last_step
        out = {"global_mean_z": self.global_mean_z[:n].cpu().numpy(),
               "std_sigma": self.std_sigma[:n].cpu().numpy(),
               "anom_rms_sigma": self.anom_rms_sigma[:n].cpu().numpy()}
        if self.zonal_mean_z is not None:
            out["zonal_mean_z"] = self.zonal_mean_z[:n].cpu().numpy()
        return out


def first_exceedance(series: np.ndarray, threshold: float) -> np.ndarray:
    """Per column, the 1-based lead of the first value ``> threshold``; -1 if never."""
    over = np.nan_to_num(series, nan=-np.inf) > threshold          # (T, C)
    hit = over.any(axis=0)
    return np.where(hit, over.argmax(axis=0) + 1, -1).astype(np.int32)


def stability_summary(series: dict[str, np.ndarray],
                      thresholds: Sequence[float] = STABILITY_THRESHOLDS) -> dict:
    """``first_bad_step[metric][threshold][c]`` and the channel-median crossing lead."""
    first_bad = np.full((len(STABILITY_METRICS), len(thresholds),
                         series["std_sigma"].shape[1]), -1, dtype=np.int32)
    median_cross = np.full((len(STABILITY_METRICS), len(thresholds)), -1, dtype=np.int32)
    for m, name in enumerate(STABILITY_METRICS):
        s = series[name]
        if s.shape[0] == 0 or np.all(np.isnan(s)):
            continue
        med = np.nanmedian(s, axis=1)[:, None]
        for t, thr in enumerate(thresholds):
            first_bad[m, t] = first_exceedance(s, thr)
            median_cross[m, t] = first_exceedance(med, thr)[0]
    return {"first_bad_step": first_bad, "median_cross_step": median_cross}


# ---------------------------------------------------------------------------
# NetCDF output (written incrementally: monthly means and snapshots as they come)
# ---------------------------------------------------------------------------

class MemberWriter:
    """One NetCDF per member. Opened as ``<out>.partial``, renamed on close."""

    def __init__(self, out_path: Path, *, channel_names: Sequence[str],
                 lat: Sequence[float], lon: Sequence[float],
                 out_bias: np.ndarray, out_scale: np.ndarray):
        import netCDF4

        self.out_path = Path(out_path)
        self.partial = self.out_path.with_name(self.out_path.name + ".partial")
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.bias, self.scale = out_bias, out_scale
        C, H, W = len(channel_names), len(lat), len(lon)
        ds = netCDF4.Dataset(self.partial, "w", format="NETCDF4")
        ds.createDimension("channel", C)
        ds.createDimension("lat", H)
        ds.createDimension("lon", W)
        ds.createDimension("step", None)
        ds.createDimension("month", None)
        ds.createDimension("snapshot", None)
        ds.createDimension("metric", len(STABILITY_METRICS))
        ds.createDimension("threshold", len(STABILITY_THRESHOLDS))
        v = ds.createVariable("channel", str, ("channel",))
        v[:] = np.array(list(channel_names), dtype=object)
        ds.createVariable("lat", "f8", ("lat",))[:] = np.asarray(lat, dtype=np.float64)
        ds.createVariable("lon", "f8", ("lon",))[:] = np.asarray(lon, dtype=np.float64)
        v = ds.createVariable("metric", str, ("metric",))
        v[:] = np.array(STABILITY_METRICS, dtype=object)
        ds.createVariable("threshold", "f8", ("threshold",))[:] = np.array(STABILITY_THRESHOLDS)

        mm = ds.createVariable("monthly_mean", "f4", ("month", "channel", "lat", "lon"),
                               zlib=False, chunksizes=(1, 1, H, W))
        mm.units = "physical"
        mm.description = ("mean over the scored leads whose valid time falls in the "
                          "noleap month; partial months at either end carry their count")
        for name in ("month_year", "month_index", "month_count", "month_first_step"):
            ds.createVariable(name, "i4", ("month",))
        sv = ds.createVariable("snapshot", "f4", ("snapshot", "channel", "lat", "lon"),
                               chunksizes=(1, 1, H, W))
        sv.units = "physical"
        ds.createVariable("snapshot_step", "i4", ("snapshot",))
        ds.createVariable("snapshot_kind", str, ("snapshot",))
        self.ds = ds
        self.n_month = 0
        self.n_snap = 0

    def _phys(self, z: np.ndarray) -> np.ndarray:
        return z * self.scale[:, None, None] + self.bias[:, None, None]

    def write_month(self, rec: MonthRecord) -> None:
        i = self.n_month
        self.ds["monthly_mean"][i] = self._phys(rec.mean_z).astype(np.float32)
        self.ds["month_year"][i] = rec.year
        self.ds["month_index"][i] = rec.month
        self.ds["month_count"][i] = rec.count
        self.ds["month_first_step"][i] = rec.first_step
        self.n_month += 1
        self.ds.sync()

    def write_snapshot(self, step: int, pred_z: torch.Tensor, kind: str) -> None:
        z = pred_z.reshape(pred_z.shape[-3:]).to(torch.float64).cpu().numpy()
        i = self.n_snap
        self.ds["snapshot"][i] = self._phys(z).astype(np.float32)
        self.ds["snapshot_step"][i] = step
        self.ds["snapshot_kind"][i] = kind
        self.n_snap += 1

    def finish(self, *, series: dict, stability: dict, time_mean_z: np.ndarray | None,
               valid_year: np.ndarray, valid_frame: np.ndarray, score_start_step: int,
               attrs: dict) -> Path:
        ds = self.ds
        n = series["global_mean_z"].shape[0]
        ds.createVariable("step", "i4", ("step",))[:n] = np.arange(1, n + 1)
        ds.createVariable("valid_year", "i4", ("step",))[:n] = valid_year[:n]
        ds.createVariable("valid_frame", "i4", ("step",))[:n] = valid_frame[:n]
        ds.createVariable("scored", "i1", ("step",))[:n] = (
            np.arange(1, n + 1) >= score_start_step).astype(np.int8)
        gm = ds.createVariable("global_mean", "f8", ("step", "channel"))
        gm.units = "physical"
        gm.description = "area-weighted (equiangular cos-lat) global mean, every lead incl. spin-up"
        gm[:n] = series["global_mean_z"] * self.scale + self.bias
        for name in STABILITY_METRICS:
            v = ds.createVariable(name, "f8", ("step", "channel"))
            v.units = "global_stds (sigma)"
            v[:n] = series[name]
        ds["std_sigma"].description = "area-weighted spatial std of the prediction / global_stds"
        ds["anom_rms_sigma"].description = ("area-weighted RMS of (prediction - time_means) "
                                            "/ global_stds")
        if "zonal_mean_z" in series:
            zm = ds.createVariable("zonal_mean", "f4", ("step", "channel", "lat"))
            zm.units = "physical"
            zm[:n] = (series["zonal_mean_z"] * self.scale[:, None]
                      + self.bias[:, None]).astype(np.float32)
        v = ds.createVariable("first_bad_step", "i4", ("metric", "threshold", "channel"))
        v.description = "1-based lead of the first value > threshold; -1 = never"
        v[:] = stability["first_bad_step"]
        v = ds.createVariable("median_cross_step", "i4", ("metric", "threshold"))
        v.description = "1-based lead at which the channel median first exceeds threshold; -1 = never"
        v[:] = stability["median_cross_step"]
        tm = ds.createVariable("time_mean", "f8", ("channel", "lat", "lon"))
        tm.units = "physical"
        tm.description = "mean over all scored leads (float64 accumulation)"
        if time_mean_z is not None:
            tm[:] = self._phys(time_mean_z)
        for key, val in attrs.items():
            if isinstance(val, (dict, list, tuple)):
                val = json.dumps(val)
            elif isinstance(val, bool):
                val = int(val)
            elif val is None:
                val = ""
            ds.setncattr(key, val)
        ds.close()
        os.replace(self.partial, self.out_path)
        return self.out_path

    def abort(self) -> None:
        try:
            self.ds.setncattr("status", "aborted")
            self.ds.close()
        except Exception:  # noqa: BLE001 -- best effort on the error path
            pass


# ---------------------------------------------------------------------------
# one member, end to end
# ---------------------------------------------------------------------------

@dataclass
class MemberResult:
    out_path: Path
    n_steps_run: int
    truncated_at_step: int
    truncated_channel: str
    handoffs: list
    s_per_step: float
    feedback_dtype: str
    stability: dict = field(repr=False)


def _direct_forcing(path: str, local_idx: int, dataset) -> torch.Tensor:
    """Forcing frame read straight from the file (not via the dataset's index map)."""
    import h5py

    sx, ex, sy, ey = dataset._crop_bounds()
    with h5py.File(os.path.realpath(path), "r") as f:
        arr = f[dataset.forcing_dataset_path][local_idx:local_idx + 1, :, sx:ex, sy:ey]
    arr = (arr - dataset.forcing_bias) / dataset.forcing_scale
    return dataset.grid_converter(torch.as_tensor(arr, dtype=torch.float32))


def run_member(
    *,
    wrapper,
    dataset,
    eval_params,
    device,
    start: tuple[int, int],
    n_steps: int,
    score_start: tuple[int, int],
    chunk_len: int,
    out_path: Path,
    channel_names: Sequence[str],
    lat: Sequence[float],
    lon: Sequence[float],
    time_means_path=None,
    snapshot_every: int = 0,
    zonal: bool = False,
    frames_per_year: int = FRAMES_PER_YEAR,
    provenance: dict | None = None,
    assert_contract: bool = True,
) -> MemberResult:
    """Roll one member from ``start=(year, frame)`` for ``n_steps`` leads; write its NetCDF."""
    years = check_year_axis(dataset, frames_per_year)
    n_out = int(eval_params.N_out_channels)
    if len(channel_names) != n_out:
        raise ClimateDriverError(
            f"STATS_CHANNEL_MISMATCH: {len(channel_names)} channel names, N_out={n_out}")
    mean = load_stats_f64(eval_params.global_means_path, n_out, "global_means")
    std = load_stats_f64(eval_params.global_stds_path, n_out, "global_stds")
    H, W = len(lat), len(lon)
    tm_z = (load_time_means_z(time_means_path, mean, std, H, W)
            if time_means_path else None)

    if start[0] not in years:
        raise ClimateDriverError(f"start year {start[0]} not in dataset years {years}")
    fidx0 = years.index(start[0])
    ic = int(dataset.file_offsets[fidx0]) + int(start[1])
    if tuple(dataset._get_indices(ic)) != (fidx0, start[1]):
        raise ClimateDriverError(f"IC index map: global {ic} -> {dataset._get_indices(ic)}, "
                                 f"expected {(fidx0, start[1])}")
    score_start_step = step_of(start, score_start, frames_per_year)
    if score_start_step < 1:
        raise ClimateDriverError(f"score_start {score_start} is not after start {start}")

    from sfno_ensemble.scores import equiangular_weights

    writer = MemberWriter(out_path, channel_names=channel_names, lat=lat, lon=lon,
                          out_bias=mean, out_scale=std)
    reducer = ClimateReducer(n_steps=n_steps, n_out=n_out, H=H, W=W,
                             lat_weights=equiangular_weights(H), time_means_z=tm_z,
                             score_start_step=score_start_step, device=device,
                             month_sink=writer.write_month, zonal=zonal)
    valid_year = np.full(n_steps, -1, dtype=np.int32)
    valid_frame = np.full(n_steps, -1, dtype=np.int32)
    handoffs: list[dict] = []
    pending: list[dict] = []
    state = {"prev_fidx": fidx0, "last_good": None}
    preprocessor = wrapper.preprocessor

    def on_step(k: int, pred: torch.Tensor) -> bool:
        fidx, lidx = dataset._get_indices(ic + k)
        year = years[fidx]
        if (year, lidx) != valid_time(start[0], start[1], k, frames_per_year):
            raise ClimateDriverError(f"lead {k}: dataset maps to {(year, lidx)}, calendar says "
                                     f"{valid_time(start[0], start[1], k, frames_per_year)}")
        # The input forcing now in the cache is that of lead k-1's valid time; verify
        # it against a direct read of the new file on the lead after a hand-off.
        for h in pending:
            u = preprocessor.unpredicted_inp_eval.reshape(-1, *pred.shape[-2:])
            ref = _direct_forcing(dataset.files_paths[h["to_file_idx"]], 0, dataset)
            h["forcing_direct_match"] = bool(torch.equal(u.cpu(), ref.reshape(u.shape)))
            h["verified_at_step"] = k
        pending.clear()
        if fidx != state["prev_fidx"]:
            h = {"step": k, "from": Path(dataset.files_paths[state["prev_fidx"]]).name,
                 "to": Path(dataset.files_paths[fidx]).name, "to_local_idx": int(lidx),
                 "to_file_idx": int(fidx), "forcing_direct_match": None}
            handoffs.append(h)
            pending.append(h)
            state["prev_fidx"] = fidx
        valid_year[k - 1], valid_frame[k - 1] = year, lidx

        if not reducer.update(k, pred, year, month_of_frame(lidx, frames_per_year)):
            if state["last_good"] is not None:
                writer.write_snapshot(k - 1, state["last_good"], "last_finite")
            writer.write_snapshot(k, pred, "first_nonfinite")
            return False
        if snapshot_every and k % snapshot_every == 0:
            writer.write_snapshot(k, pred, "periodic")
        state["last_good"] = pred
        return True

    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time()
    try:
        info = stream_rollout(wrapper=wrapper, dataset=dataset, ic_global_idx=ic,
                              n_steps=n_steps, chunk_len=chunk_len, eval_params=eval_params,
                              device=device, on_step=on_step, assert_contract=assert_contract)
    except BaseException:
        writer.abort()
        raise
    elapsed = time.time() - t0
    reducer.flush_month()
    series = reducer.series()
    stability = stability_summary(series)
    peak = (torch.cuda.max_memory_allocated(device) / 2**30
            if torch.device(device).type == "cuda" else 0.0)
    trunc_ch = (channel_names[reducer.truncated_channel]
                if reducer.truncated_at_step > 0 else "")
    for h in handoffs:
        h.pop("to_file_idx", None)

    attrs = dict(provenance or {})
    attrs.update(
        start_year=start[0], start_frame=start[1], ic_global_idx=ic,
        n_steps_requested=n_steps, n_steps_run=reducer.last_step,
        score_start_year=score_start[0], score_start_frame=score_start[1],
        score_start_step=score_start_step, scored_steps=reducer.time_count,
        n_months_written=writer.n_month,
        chunk_len=chunk_len, frames_per_year=frames_per_year, calendar="noleap",
        truncated_at_step=reducer.truncated_at_step, truncated_channel=trunc_ch,
        feedback_dtype=info.feedback_dtype, amp_dtype=str(eval_params.amp_dtype),
        amp_enabled=bool(eval_params.amp_enabled),
        year_files=[[Path(p).name, os.path.realpath(p)] for p in dataset.files_paths],
        handoffs=handoffs, time_means_path=str(time_means_path or ""),
        s_per_step=elapsed / max(info.n_steps_run, 1), wall_seconds=elapsed,
        peak_gpu_mem_gib=peak,
        stability_normaliser="run global_stds (sigma); anom_rms vs time_means_path",
        status="truncated" if reducer.truncated_at_step > 0 else "complete",
    )
    writer.finish(series=series, stability=stability, time_mean_z=reducer.time_mean_z(),
                  valid_year=valid_year, valid_frame=valid_frame,
                  score_start_step=score_start_step, attrs=attrs)
    return MemberResult(out_path=Path(out_path), n_steps_run=reducer.last_step,
                        truncated_at_step=reducer.truncated_at_step,
                        truncated_channel=trunc_ch, handoffs=handoffs,
                        s_per_step=attrs["s_per_step"], feedback_dtype=info.feedback_dtype,
                        stability=stability)
