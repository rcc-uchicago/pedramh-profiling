#!/usr/bin/env python3
"""validate.py — five-mode validation for the v10 packager output.

Modes (per docs/plasim_zg_plev_migration_plan.md v7 §3.7):

* ``files``       — per-file structural checks + cross-file monotonicity
                    + v10-specific attrs (zg_source_var, zg_pressure_levels_hpa,
                    postprocessor_git_sha). Does *not* require ``stats/``.
* ``stats``       — checks the six .npy shapes / dtypes / std-epsilon and the
                    saved zg500 mean (defense-in-depth vs the inline
                    ``_audit_zg500_inline`` in ``stats.py``).
* ``smoke``       — synthetic-fixture pytest (CI-runnable).
* ``smoke-live``  — live-data preflight against the actual ``--output-root``.
                    Loads ``metadata/data.json`` + the rendered yaml, instantiates
                    the patched Makani loader, runs a 3-step rollout. Compute-node only.
* ``full``        — files → stats → smoke → smoke-live, in order.

Legacy ``structural`` and ``makani_smoke`` modes remain as
deprecation-warned aliases for ``full`` and ``smoke`` respectively.

CLI
---
validate.py --output-root {root}
            --mode {files,stats,smoke,smoke-live,full}
            [--config-name plasim_sim52_zgplev_baseline]
            [--yaml-config PATH]
            [--epsilon 1e-6]
            [--allow-unknown-postproc-sha]
            [-v]
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import warnings
from pathlib import Path

import h5py
import numpy as np

from plasim_makani_packager.channels import (
    DIAGNOSTIC_CHANNELS,
    FORCING_CHANNELS,
    STATE_CHANNELS,
    TARGET_CHANNELS,
    ZG_PLEV_HPA,
)
from plasim_makani_packager.packager import STEP_SECONDS, ZG_SOURCE_VAR
from plasim_makani_packager.stats import (
    MIN_STD_EPSILON,
    STATIC_FORCING_NAMES,
    ZG500_AUDIT_RANGE_M,
)

logger = logging.getLogger("plasim_makani_packager.validate")

REQUIRED_FILE_ATTRS: tuple[str, ...] = (
    "plasim_time_units",
    "plasim_calendar",
    "rsdt_method",
    "sst_land_fill_K",
)
# v10 file_attrs added by §3.3 — checked separately by _validate_v10_attrs
# so the legacy required-attr loop stays focused on the v9-locked attrs.
V10_REQUIRED_ATTRS: tuple[str, ...] = (
    "zg_source_var",
    "zg_pressure_levels_hpa",
    "postprocessor_git_sha",
)
REQUIRED_DATASETS: tuple[str, ...] = (
    "fields_state",
    "fields_diagnostic",
    "forcing",
    "timestamp",
    "time_plasim",
    "channel_state",
    "channel_diagnostic",
    "channel_forcing",
    "lat",
    "lon",
)
SPLITS: tuple[str, ...] = ("train", "valid", "test")

DEFAULT_CONFIG_NAME: str = "plasim_sim52_zgplev_baseline"

_PLACEHOLDER_RE = re.compile(r"\{\{[^}]+\}\}")


class ValidationError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Per-file structural checks
# ---------------------------------------------------------------------------
def _bytes_to_str(arr: np.ndarray) -> list[str]:
    return [x.decode("ascii") if isinstance(x, bytes) else str(x) for x in arr]


def _check_dim_scales(dset: h5py.Dataset, scales: tuple[h5py.Dataset, ...]) -> None:
    if dset.ndim != 4:
        raise ValidationError(f"{dset.name}: expected 4D, got {dset.ndim}D")
    for axis, expected in enumerate(scales):
        attached = list(dset.dims[axis].items())
        if not attached:
            raise ValidationError(
                f"{dset.name}: no dim scale attached at axis {axis} "
                f"(expected '{expected.name}')"
            )
        if expected.name not in {d[1].name for d in attached}:
            raise ValidationError(
                f"{dset.name}: dim scale '{expected.name}' not attached at axis "
                f"{axis} (found {[d[1].name for d in attached]})"
            )


def _validate_file(path: Path, year: int) -> int:
    """Return T (number of time steps) of the validated file."""
    with h5py.File(path, "r") as f:
        for name in REQUIRED_DATASETS:
            if name not in f:
                raise ValidationError(f"{path.name}: missing dataset '{name}'")
        for attr in REQUIRED_FILE_ATTRS:
            if attr not in f.attrs:
                raise ValidationError(
                    f"{path.name}: missing file attr '{attr}'"
                )

        state = f["fields_state"]
        diag = f["fields_diagnostic"]
        forcing = f["forcing"]
        ts = f["timestamp"]
        tp = f["time_plasim"]

        T, C_s, H, W = state.shape
        if C_s != 52 or state.dtype != np.float32:
            raise ValidationError(
                f"{path.name}: fields_state shape/dtype mismatch "
                f"{state.shape} {state.dtype}"
            )
        if diag.shape != (T, 1, H, W) or diag.dtype != np.float32:
            raise ValidationError(
                f"{path.name}: fields_diagnostic shape/dtype mismatch "
                f"{diag.shape} {diag.dtype}"
            )
        if forcing.shape != (T, 6, H, W) or forcing.dtype != np.float32:
            raise ValidationError(
                f"{path.name}: forcing shape/dtype mismatch "
                f"{forcing.shape} {forcing.dtype}"
            )
        if (H, W) != (64, 128):
            raise ValidationError(
                f"{path.name}: expected 64x128 grid, got {H}x{W}"
            )
        if ts.shape != (T,) or ts.dtype != np.int64:
            raise ValidationError(
                f"{path.name}: timestamp shape/dtype {ts.shape} {ts.dtype}"
            )
        if tp.shape != (T,) or tp.dtype != np.float64:
            raise ValidationError(
                f"{path.name}: time_plasim shape/dtype {tp.shape} {tp.dtype}"
            )

        cs = _bytes_to_str(f["channel_state"][...])
        cd = _bytes_to_str(f["channel_diagnostic"][...])
        cf = _bytes_to_str(f["channel_forcing"][...])
        if cs != STATE_CHANNELS:
            raise ValidationError(
                f"{path.name}: channel_state != expected master list"
            )
        if cd != DIAGNOSTIC_CHANNELS:
            raise ValidationError(
                f"{path.name}: channel_diagnostic != expected master list"
            )
        if cf != FORCING_CHANNELS:
            raise ValidationError(
                f"{path.name}: channel_forcing != expected master list"
            )

        # Dim scales on all 4D datasets
        ts_scale = f["timestamp"]
        state_scale = f["channel_state"]
        diag_scale = f["channel_diagnostic"]
        force_scale = f["channel_forcing"]
        lat_scale = f["lat"]
        lon_scale = f["lon"]
        _check_dim_scales(
            state, (ts_scale, state_scale, lat_scale, lon_scale)
        )
        _check_dim_scales(
            diag, (ts_scale, diag_scale, lat_scale, lon_scale)
        )
        _check_dim_scales(
            forcing, (ts_scale, force_scale, lat_scale, lon_scale)
        )

        # Finite content (stream in small chunks to avoid loading full tensor)
        for dset, label in ((state, "fields_state"), (diag, "fields_diagnostic"), (forcing, "forcing")):
            arr = dset[...]
            if not np.isfinite(arr).all():
                bad = np.argwhere(~np.isfinite(arr))[:5]
                raise ValidationError(
                    f"{path.name}: non-finite values in {label}, first indices: "
                    f"{bad.tolist()}"
                )

        # Timestamp within-file: monotonic, diff == STEP_SECONDS
        ts_vals = ts[...]
        dts = np.diff(ts_vals)
        if not np.all(dts == STEP_SECONDS):
            uniq = np.unique(dts).tolist()
            raise ValidationError(
                f"{path.name}: timestamp diffs not all {STEP_SECONDS}: {uniq}"
            )

        # PlaSim time: strictly monotonic
        tp_vals = tp[...]
        if not np.all(np.diff(tp_vals) > 0):
            raise ValidationError(
                f"{path.name}: time_plasim is not strictly increasing"
            )

        attrs = {k: f.attrs[k] for k in REQUIRED_FILE_ATTRS}
        rsdt_method = attrs["rsdt_method"]
        if isinstance(rsdt_method, bytes):
            rsdt_method = rsdt_method.decode()
        if rsdt_method != "astronomical":
            raise ValidationError(
                f"{path.name}: rsdt_method='{rsdt_method}', expected 'astronomical'"
            )
        sst_fill = float(attrs["sst_land_fill_K"])
        if not (200.0 < sst_fill < 320.0):
            raise ValidationError(
                f"{path.name}: sst_land_fill_K={sst_fill} out of plausible range"
            )

        return int(T)


def _validate_v10_attrs(path: Path, *, allow_unknown_postproc_sha: bool = False) -> None:
    """v10 file_attrs check (§3.7 _validate_v10_attrs):

    - ``zg_source_var`` == ``"zg_plev"``
    - ``zg_pressure_levels_hpa`` == ``ZG_PLEV_HPA``
    - ``postprocessor_git_sha`` non-empty string; in production ≠ ``"unknown"``
      unless ``allow_unknown_postproc_sha`` is set.
    """
    with h5py.File(path, "r") as f:
        for attr in V10_REQUIRED_ATTRS:
            if attr not in f.attrs:
                raise ValidationError(
                    f"{path.name}: missing v10 file attr '{attr}'"
                )

        zsv = f.attrs["zg_source_var"]
        if isinstance(zsv, bytes):
            zsv = zsv.decode()
        if str(zsv) != ZG_SOURCE_VAR:
            raise ValidationError(
                f"{path.name}: zg_source_var='{zsv}', expected '{ZG_SOURCE_VAR}'"
            )

        zpl = np.array(f.attrs["zg_pressure_levels_hpa"]).astype(int).tolist()
        if zpl != list(ZG_PLEV_HPA):
            raise ValidationError(
                f"{path.name}: zg_pressure_levels_hpa={zpl}, expected {list(ZG_PLEV_HPA)}"
            )

        sha = f.attrs["postprocessor_git_sha"]
        if isinstance(sha, bytes):
            sha = sha.decode()
        sha = str(sha)
        if not sha:
            raise ValidationError(
                f"{path.name}: postprocessor_git_sha is empty"
            )
        if sha == "unknown" and not allow_unknown_postproc_sha:
            raise ValidationError(
                f"{path.name}: postprocessor_git_sha='unknown' (production gate). "
                f"Pass --allow-unknown-postproc-sha to relax for tests / dry runs."
            )


# ---------------------------------------------------------------------------
# Cross-file monotonicity (within each split independently)
# ---------------------------------------------------------------------------
def _assert_cross_file_monotonic(files_in_order: list[Path]) -> None:
    """Check that /timestamp is monotonic + uniform-dT within the given file
    sequence. Makani's MultifilesDataset expects this per `files_pattern`
    (i.e. per split directory), NOT across splits; train/valid/test are
    independent datasets from Makani's perspective, so each starts fresh."""
    last = None
    for path in files_in_order:
        with h5py.File(path, "r") as f:
            first = int(f["timestamp"][0])
            final = int(f["timestamp"][-1])
        if last is not None:
            if first <= last:
                raise ValidationError(
                    f"monotonicity broken at {path.name}: first={first} "
                    f"<= previous_last={last}"
                )
            if first - last != STEP_SECONDS:
                raise ValidationError(
                    f"uniform-dT broken at {path.name}: first - previous_last "
                    f"= {first - last} != {STEP_SECONDS}"
                )
        last = final


def _split_files_in_year_order(output_root: Path) -> dict[str, list[Path]]:
    """{split -> [files in year order]} for each of train/valid/test."""
    out: dict[str, list[Path]] = {}
    for split in SPLITS:
        pairs: list[tuple[int, Path]] = []
        for p in (output_root / split).glob("MOST.*.h5"):
            year = int(p.stem.split(".")[1])
            pairs.append((year, p))
        pairs.sort(key=lambda p: p[0])
        out[split] = [p for _, p in pairs]
    return out


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------
def _validate_stats(output_root: Path, epsilon: float) -> None:
    stats = output_root / "stats"
    expected = {
        "global_means.npy": (1, 53, 1, 1),
        "global_stds.npy": (1, 53, 1, 1),
        "time_means.npy": (1, 53, 64, 128),
        "forcing_global_means.npy": (1, 6, 1, 1),
        "forcing_global_stds.npy": (1, 6, 1, 1),
        "forcing_time_means.npy": (1, 6, 64, 128),
    }
    for name, shape in expected.items():
        path = stats / name
        if not path.exists():
            raise ValidationError(f"missing stats file: {path}")
        arr = np.load(path)
        if arr.shape != shape:
            raise ValidationError(
                f"{name}: shape {arr.shape}, expected {shape}"
            )
        if arr.dtype != np.float32:
            raise ValidationError(
                f"{name}: dtype {arr.dtype}, expected float32"
            )
        if not np.isfinite(arr).all():
            raise ValidationError(f"{name}: non-finite values present")

    std_tgt = np.load(stats / "global_stds.npy").ravel()
    for i, name in enumerate(TARGET_CHANNELS):
        if std_tgt[i] < epsilon:
            raise ValidationError(
                f"global_stds[{name}]={std_tgt[i]:.3e} < {epsilon:.1e}"
            )
    std_frc = np.load(stats / "forcing_global_stds.npy").ravel()
    for i, name in enumerate(FORCING_CHANNELS):
        if std_frc[i] < epsilon:
            raise ValidationError(
                f"forcing_global_stds[{name}]={std_frc[i]:.3e} < {epsilon:.1e}"
            )

    # Static forcing channels: time_means across space should match the
    # per-cell value from any single training file (spatial variability is
    # fine; temporal variability must be zero).
    forcing_time_means = np.load(stats / "forcing_time_means.npy")
    for i, name in enumerate(FORCING_CHANNELS):
        if name not in STATIC_FORCING_NAMES:
            continue
        # Confirm: choose any training file and check the first time step
        # matches the time-averaged value for this channel.
        train_files = sorted((output_root / "train").glob("MOST.*.h5"))
        if not train_files:
            continue
        with h5py.File(train_files[0], "r") as f:
            sample = f["forcing"][0, i]
        if not np.allclose(
            forcing_time_means[0, i], sample, atol=1e-5, rtol=0.0
        ):
            raise ValidationError(
                f"static forcing '{name}': time_means vs first-sample diff > 1e-5"
            )


def _validate_zg500_saved_mean(output_root: Path) -> None:
    """Defense-in-depth re-check of the saved zg500 mean (§3.7).

    ``stats.py``'s ``_audit_zg500_inline`` runs against ``mean_tgt`` before
    any .npy is written. This re-runs the same range check against the
    file on disk, in case the .npy was edited or arrived from elsewhere.
    """
    means_path = output_root / "stats" / "global_means.npy"
    if not means_path.exists():
        raise ValidationError(f"missing {means_path}")
    arr = np.load(means_path).reshape(-1)
    try:
        idx = TARGET_CHANNELS.index("zg500")
    except ValueError as e:
        raise ValidationError(
            "TARGET_CHANNELS does not contain 'zg500' — this build is not v10."
        ) from e
    val = float(arr[idx])
    lo, hi = ZG500_AUDIT_RANGE_M
    if not (lo <= val <= hi):
        raise ValidationError(
            f"saved global_means[zg500]={val:.2f} m outside "
            f"audit band [{lo:.0f}, {hi:.0f}] m. "
            f"(stats.py inline audit should have caught this — "
            f"the saved .npy may have been hand-edited.)"
        )


# ---------------------------------------------------------------------------
# YAML placeholder check (used by smoke-live)
# ---------------------------------------------------------------------------
def _assert_no_yaml_placeholders(yaml_path: Path) -> None:
    """Fail fast if a yaml passed to smoke-live still has ``{{...}}`` markers.

    Mirrors ``scripts/preflight.py:311-326``. Catches the case where an
    operator passes an unrendered trainer-side template via
    ``--yaml-config`` — which would otherwise surface deep inside the
    dataloader as a FileNotFoundError on a path like
    ``{{OUTPUT_ROOT}}/train`` (Codex v6 high finding).
    """
    text = yaml_path.read_text()
    leftover = sorted(set(_PLACEHOLDER_RE.findall(text)))
    if leftover:
        raise ValidationError(
            f"yaml {yaml_path} still contains placeholder(s) {leftover[:5]}; "
            f"render with `sed -e 's|{{{{OUTPUT_ROOT}}}}|...|g' "
            f"-e 's|{{{{EXP_DIR}}}}|...|g'` (see "
            f"src/sfno_training/submit_full.slurm:60-62) before passing to "
            f"--yaml-config."
        )


# ---------------------------------------------------------------------------
# Mode entry points
# ---------------------------------------------------------------------------
def run_files(output_root: Path, *, allow_unknown_postproc_sha: bool = False) -> None:
    split_files = _split_files_in_year_order(output_root)
    total = sum(len(v) for v in split_files.values())
    if total == 0:
        raise ValidationError(
            f"no MOST.*.h5 files under {output_root}/{{train,valid,test}}"
        )

    for split, files in split_files.items():
        for path in files:
            year = int(path.stem.split(".")[1])
            T = _validate_file(path, year)
            _validate_v10_attrs(
                path, allow_unknown_postproc_sha=allow_unknown_postproc_sha
            )
            logger.info("  ok  %s  (T=%d)", path, T)
        if files:
            _assert_cross_file_monotonic(files)
            logger.info(
                "%s: %d files, uniform-dT across file boundaries ✓",
                split,
                len(files),
            )


def run_stats(output_root: Path, epsilon: float) -> None:
    _validate_stats(output_root, epsilon)
    _validate_zg500_saved_mean(output_root)
    logger.info("stats ok ✓")


def run_smoke_synthetic() -> int:
    """Synthetic-fixture pytest (Phase 4b regression).

    Independent of ``--output-root``; exercises the patched Makani loader
    + preprocessor + wrappers on a synthetic fixture to catch wrapper-patch
    regressions. CI-runnable.
    """
    import pytest  # imported lazily: structural mode doesn't need pytest / torch

    repo_root = Path(__file__).resolve().parents[2]
    test_path = (
        repo_root
        / "tests"
        / "plasim_makani_packager"
        / "test_multifile_loader_smoke.py"
    )
    if not test_path.exists():
        raise ValidationError(f"smoke test module not found: {test_path}")
    return pytest.main(["-x", "-q", str(test_path)])


def run_smoke_live(
    output_root: Path,
    config_name: str,
    n_steps: int = 3,
    *,
    yaml_config_override: Path | None = None,
) -> None:
    """Live-data preflight: actually load the new dataset and roll forward.

    Distinct from ``run_smoke_synthetic`` (the pytest synthetic-fixture
    regression). This is the real gate for "the new dataset will train" —
    it touches the real ``metadata/data.json``, the real yaml, and the
    real H5 files, and exercises the full PlasimForcingDataset →
    PlasimSingleStepWrapper path against them.

    Path resolution (per plan §3.7):

      1. If ``yaml_config_override`` is set, use it directly. The override
         must already be a *rendered* yaml (placeholders substituted) —
         enforced by ``_assert_no_yaml_placeholders``. P4 only renders the
         baseline yaml from the packager-side template; trainer-side
         templates (smoke / tiny / short / full) live in
         ``src/sfno_training/config/`` with ``{{OUTPUT_ROOT}}`` /
         ``{{EXP_DIR}}`` placeholders and must be rendered by the caller
         (e.g. via the ``sed`` invocation
         ``submit_full.slurm:60-62`` uses) before being passed here
         (Codex v6 high finding).
      2. Else default to ``{output_root}/config/{config_name}.yaml`` —
         the packager-rendered baseline; concrete paths already
         substituted.
    """
    cfg_path = yaml_config_override or (
        output_root / "config" / f"{config_name}.yaml"
    )
    meta_path = output_root / "metadata" / "data.json"
    if not cfg_path.exists() or not meta_path.exists():
        raise ValidationError(
            f"smoke-live requires {cfg_path} and {meta_path} to exist; "
            f"run --mode files / stats first, plus metadata.py rendering "
            f"the baseline yaml into {output_root}/config/ (P4 in plan §5)."
        )
    _assert_no_yaml_placeholders(cfg_path)

    # Delegate to scripts/preflight.py — the canonical live-preflight path
    # that the submit_zgplev_*.slurm scripts already use. This avoids
    # duplicating PlasimTrainer construction here (and the prior inline
    # implementation was based on an older PlasimSingleStepWrapper API
    # that no longer exists; moving the source of truth into preflight.py
    # keeps the gate aligned with what training actually runs).
    import subprocess
    import sys

    repo_root = Path(__file__).resolve().parents[2]
    preflight = repo_root / "scripts" / "preflight.py"
    if not preflight.exists():
        raise ValidationError(
            f"smoke-live delegates to {preflight} but it is missing"
        )

    cmd = [
        sys.executable,
        str(preflight),
        "--yaml_config",
        str(cfg_path),
        "--config",
        config_name,
        "--amp-mode",
        "bf16",
    ]
    logger.info("  smoke-live: %s", " ".join(cmd))
    rc = subprocess.run(cmd, check=False).returncode
    if rc != 0:
        raise ValidationError(
            f"smoke-live preflight (delegated to scripts/preflight.py) "
            f"exited rc={rc}"
        )
    logger.info("  smoke-live ok (n_steps=%d via preflight contract dry-run)", n_steps)


def run_full(
    output_root: Path,
    epsilon: float,
    config_name: str,
    *,
    yaml_config_override: Path | None = None,
    allow_unknown_postproc_sha: bool = False,
) -> int:
    run_files(output_root, allow_unknown_postproc_sha=allow_unknown_postproc_sha)
    run_stats(output_root, epsilon)
    rc = run_smoke_synthetic()
    if rc != 0:
        return rc
    run_smoke_live(
        output_root, config_name, yaml_config_override=yaml_config_override
    )
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
_VALID_MODES = ("files", "stats", "smoke", "smoke-live", "full")
_LEGACY_MODES = {"structural": "full", "makani_smoke": "smoke"}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--output-root", required=True, type=Path)
    p.add_argument(
        "--mode",
        choices=(*_VALID_MODES, *_LEGACY_MODES),
        default="full",
    )
    p.add_argument(
        "--config-name",
        type=str,
        default=DEFAULT_CONFIG_NAME,
        help="YAML top-level config key (used by smoke-live / full).",
    )
    p.add_argument(
        "--yaml-config",
        type=Path,
        default=None,
        help="Path to a rendered (no {{...}} placeholders) yaml. Overrides "
        "the default {output-root}/config/{config-name}.yaml convention.",
    )
    p.add_argument("--epsilon", type=float, default=MIN_STD_EPSILON)
    p.add_argument(
        "--allow-unknown-postproc-sha",
        action="store_true",
        help="Relax the postprocessor_git_sha != 'unknown' production gate; "
        "use only for test / dry-run packaging.",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    mode = args.mode
    if mode in _LEGACY_MODES:
        new_mode = _LEGACY_MODES[mode]
        warnings.warn(
            f"--mode {mode} is deprecated; use --mode {new_mode} instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        mode = new_mode

    try:
        if mode == "files":
            run_files(
                args.output_root,
                allow_unknown_postproc_sha=args.allow_unknown_postproc_sha,
            )
        elif mode == "stats":
            run_stats(args.output_root, args.epsilon)
        elif mode == "smoke":
            rc = run_smoke_synthetic()
            if rc != 0:
                sys.exit(f"pytest smoke returned {rc}")
        elif mode == "smoke-live":
            run_smoke_live(
                args.output_root,
                args.config_name,
                yaml_config_override=args.yaml_config,
            )
        elif mode == "full":
            rc = run_full(
                args.output_root,
                args.epsilon,
                args.config_name,
                yaml_config_override=args.yaml_config,
                allow_unknown_postproc_sha=args.allow_unknown_postproc_sha,
            )
            if rc != 0:
                sys.exit(f"pytest smoke returned {rc}")
        else:  # pragma: no cover — argparse choices already constrain this
            sys.exit(f"unknown mode: {mode}")
    except ValidationError as e:
        sys.exit(f"validation failed: {e}")


if __name__ == "__main__":
    main()
