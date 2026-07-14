"""Per-Y yaml override + checkpoint symlink shim (§B.1, §3 P-2).

Reads the upstream yaml at
``/work2/.../v2.0/config/SFNO_PLASIM_H5_DERECHO_5410.yaml`` and writes
**8 per-IC-year copies** to ``<config_dir>/SFNO_PLASIM_H5_DERECHO_5410_stampede3_Y<Y>.yaml``.

For each Y ∈ {121..128}, given an explicit forecast horizon ``K`` (in
6-hour lead steps):
  * Path overrides: `data_dir`, `bias_data_dir`, `climatology_file`,
    `load_exp_dir` are remapped from `/glade/...` to Stampede3 mounts.
  * `exp_dir` is set to the eval run's inference dir (drives upstream's
    checkpoint-discovery globstr — §3 P-2).
  * `val_year_start = Y, val_year_end = Y + 1` — required by the
    boundary-loader contract at `data_loader_multifiles.py:948-960`
    (§B.0).
  * `leap_year = 52, no_leap_year = 51` — blocking-forecast boundary
    template convention recovered from Derecho provenance on 2026-05-09.
    Target/truth years are still `Y`; only prescribed boundary forcing
    is read from the 51/52 template years.
  * `ensemble_inference_hours = (K + 1) * 6` — IC-dataset preload knob
    (`data_loader_multifiles.py:831`, ensemble branch).
  * `prediction_duration_days = (K + 1) * 6 / 24` — BCS rollout span
    (`data_loader_multifiles.py:818-823`, single_ic branch). REQUIRED
    for sub-year rollouts; without it `long_rollout_years = 0` and the
    BCS date range collapses.
  * `save_forecasts: True`, `log_to_wandb: False` are forced.

Output buffer holds ``K + 1`` rows (IC at index 0, forecast leads at
indices 1..K). Adapter slicing of ``time[1:K+1]`` extracts the K
scored forecast leads.

Also assembles the per-Y single-file checkpoint symlink shim
(§3 P-2): ``<exp_dir>/SFNO_PLASIM_H5_DERECHO_5410_stampede3_Y<Y>/5410/checkpoints/ckpt_epoch_50.tar``
→ ``/work2/.../v2.0/results/SFNO/5410/checkpoints/ckpt_epoch_50.tar``.
"""
from __future__ import annotations

from pathlib import Path

from ruamel.yaml import YAML


# ---- Pinned paths (verified 2026-05-07) -----------------------------------
UPSTREAM_REPO = Path(
    "/work2/11114/zhixingliu/stampede3/SFNO_Climate_Emulator/artifacts/derecho_glade/PanguWeather/v2.0"
)
UPSTREAM_YAML_PATH = UPSTREAM_REPO / "config" / "SFNO_PLASIM_H5_DERECHO_5410.yaml"
UPSTREAM_CKPT_PATH = (
    UPSTREAM_REPO
    / "results"
    / "SFNO"
    / "5410"
    / "checkpoints"
    / "ckpt_epoch_50.tar"
)
UPSTREAM_LOAD_EXP_DIR = UPSTREAM_REPO / "results"

# Stampede3 data tree mirroring `/glade/derecho/scratch/awikner/PLASIM/...`.
STAMPEDE3_DATA_ROOT = Path(
    "/scratch/11114/zhixingliu/SFNO_Climate_Emulator/data/derecho_glade/sim52"
)
STAMPEDE3_DATA_DIR = STAMPEDE3_DATA_ROOT / "h5" / "sigma_data"
STAMPEDE3_BIAS_DIR = STAMPEDE3_DATA_ROOT / "bias"
STAMPEDE3_CLIM_NC = STAMPEDE3_DATA_ROOT / "sigma_data" / "climatology.nc"

TEST_YEARS = tuple(range(121, 129))
LEAP_YEARS = (124, 128)
BOUNDARY_NO_LEAP_YEAR = 51
BOUNDARY_LEAP_YEAR = 52
RUN_NUM = "5410"
CKPT_FILENAME = "ckpt_epoch_50.tar"


def _raw_steps_for_K(K: int) -> int:
    """Return the upstream output buffer size (raw 6-hour rows) for ``K`` forecast leads.

    Output buffer holds ``K + 1`` rows: IC at index 0, forecast leads at
    indices 1..K. Upstream ``long_inference.py:562, 836`` sizes the
    buffer as ``(final_datetime - init_datetime) / 6h``, so callers must
    set ``final_datetime = init_datetime + (K + 1) * 6h``.

    Rejects ``bool`` explicitly because ``isinstance(True, int)`` is True
    in Python.
    """
    if isinstance(K, bool) or not isinstance(K, int) or K < 1:
        raise ValueError(
            f"K must be a positive int (not bool), got {K!r} ({type(K).__name__})"
        )
    return K + 1


def _horizon_hours_for_K(K: int) -> int:
    """Return the rollout horizon in hours: ``(K + 1) * 6``.

    This drives both ``ensemble_inference_hours`` (IC-dataset preload)
    and ``prediction_duration_days`` (BCS rollout span). Capped at 8784
    (one leap year) — multi-year rollouts are out of scope here.
    """
    h = _raw_steps_for_K(K) * 6
    if h > 8784:
        raise ValueError(
            f"K={K} → {h}h exceeds one-year cap (8784); multi-year rollouts not supported here"
        )
    return h

# Top-level YAML key passed to upstream as ``--config``. The upstream
# yaml has anchors {base_config, PLASIM, SFNO, modified_1}; SFNO is the
# correct architecture section. Per-Y val_year_start overrides are
# applied to the SFNO section in-place by ``_override_section`` below
# so YParams loads them when ``--config=SFNO`` is selected.
CONFIG_SECTION = "SFNO"


def _yaml_name_for_year(Y: int) -> str:
    return f"SFNO_PLASIM_H5_DERECHO_5410_stampede3_Y{Y}.yaml"


def config_basename_for_year(Y: int) -> str:
    """Return the upstream ``--config`` value for year ``Y``.

    Always returns ``"SFNO"`` (the YAML top-level key, not the file
    basename). Per-Y differentiation lives in `--yaml_config` and the
    SFNO section's own `val_year_start`/`val_year_end` fields. Y is
    accepted but unused, retained for caller-side symmetry.
    """
    return CONFIG_SECTION


def _override_section(
    section: dict,
    *,
    Y: int,
    K: int,
    exp_dir: Path,
) -> None:
    """Apply per-Y path/value overrides in-place to one yaml section.

    ``K`` is the forecast-leads horizon (required, no default — preflight
    requires K to be explicit at every call site). Sets both
    ``ensemble_inference_hours = (K+1)*6`` (IC-dataset preload) and
    ``prediction_duration_days = (K+1)*6/24`` (BCS rollout span).
    """
    # Validates K (raises ValueError on bool / non-int / <1 / >8784h).
    horizon_hours = _horizon_hours_for_K(K)
    if "data_dir" in section:
        section["data_dir"] = str(STAMPEDE3_DATA_DIR)
    if "bias_data_dir" in section:
        section["bias_data_dir"] = str(STAMPEDE3_BIAS_DIR)
    if "climatology_file" in section:
        section["climatology_file"] = str(STAMPEDE3_CLIM_NC)
    if "load_exp_dir" in section:
        section["load_exp_dir"] = str(UPSTREAM_LOAD_EXP_DIR)
    # exp_dir drives upstream checkpoint discovery — point it at the
    # per-run inference tree where the symlink shim lives.
    if "exp_dir" in section:
        section["exp_dir"] = str(exp_dir)
    if "val_year_start" in section:
        section["val_year_start"] = int(Y)
    if "val_year_end" in section:
        section["val_year_end"] = int(Y) + 1
    # Upstream's `data_loader_multifiles.py:931-934` builds the
    # varying-boundary h5 path from a template year (`self.leap_year` /
    # `self.no_leap_year`). Derecho blocking provenance (2026-05-09)
    # shows the trusted blocking path uses year 51 for non-leap target
    # years and year 52 for leap target years, e.g. Y121 s0 step 1 reads
    # `51_0000.h5`. Preserve those values separately so the in-process
    # orchestrator can reapply them after every per-IC reconfigure call.
    section["boundary_leap_year"] = BOUNDARY_LEAP_YEAR
    section["boundary_no_leap_year"] = BOUNDARY_NO_LEAP_YEAR
    section["leap_year"] = BOUNDARY_LEAP_YEAR
    section["no_leap_year"] = BOUNDARY_NO_LEAP_YEAR
    if "log_to_wandb" in section:
        section["log_to_wandb"] = False
    section["save_forecasts"] = True
    # IC-dataset preload knob (`data_loader_multifiles.py:483, 597, 831`,
    # ensemble branch). Set on every section so SFNO's explicit copy
    # overrides PLASIM's via YAML merge-resolution at load time.
    section["ensemble_inference_hours"] = horizon_hours
    # BCS rollout span (`data_loader_multifiles.py:818-823`, single_ic
    # branch). The BCS data loader is constructed with single_ic=True at
    # `long_inference.py:202` and reads this key to bound the iteration
    # count: end_date = start_date + timedelta(days=prediction_duration_days).
    # Without this key the loader falls back to long_rollout_years which
    # is 0 for sub-year rollouts → empty/wrong date range.
    section["prediction_duration_days"] = horizon_hours / 24.0
    # Length-1 placeholder. Upstream `data_loader_multifiles.py:829`
    # reads only `len(self.params.save_basenames)` to size a date_range
    # array. Each long_inference.py invocation processes exactly one IC
    # (single-IC invariant enforced in scripts/eval_inference_5410.py),
    # so the length is invariably 1. Values are unused on this path.
    section["save_basenames"] = ["_unused_len1"]
    # Force deterministic NWP rollout (per 2026-05-08 user decision).
    # `long_inference.py:204` constructs `Perturber` only if
    # `epsilon_factor > 0`; both perturber call sites at :560/:832 are
    # independently gated by `epsilon_factor > 0` at :558/:830. Setting
    # 0.0 disables all perturbation paths and removes any need for
    # `perturbation_type` (which the upstream yaml does not define).
    section["epsilon_factor"] = 0.0


def build_per_y_yaml(
    Y: int,
    config_dir: Path,
    exp_dir: Path,
    *,
    K: int,
    src_yaml: Path = UPSTREAM_YAML_PATH,
) -> Path:
    """Write the per-Y override yaml; return its path.

    ``K`` is the forecast-leads horizon (required keyword-only — callers
    MUST supply K explicitly so it propagates to every yaml). Output
    yaml has ``ensemble_inference_hours = (K+1)*6`` and
    ``prediction_duration_days = (K+1)*6/24``.

    The output is a copy of the upstream yaml with path-fields remapped
    to Stampede3, `val_year_start/end` pinned to ``(Y, Y+1)``, and
    forecast/wandb flags forced.
    """
    if Y not in TEST_YEARS:
        raise ValueError(f"Y must be one of {TEST_YEARS}, got {Y!r}")

    yaml = YAML()
    yaml.preserve_quotes = True
    with open(src_yaml, "r") as f:
        doc = yaml.load(f)

    for section_name, section in doc.items():
        if isinstance(section, dict):
            _override_section(section, Y=Y, K=K, exp_dir=exp_dir)

    config_dir = Path(config_dir)
    config_dir.mkdir(parents=True, exist_ok=True)
    out_path = config_dir / _yaml_name_for_year(Y)
    with open(out_path, "w") as f:
        yaml.dump(doc, f)
    return out_path


def ckpt_shim_path(exp_dir: Path, Y: int | None = None) -> Path:
    """Return the §3 P-2 symlink shim path.

    Upstream's ``long_inference.py`` computes::

        expDir = os.path.join(params.exp_dir, args.config, str(run_num))

    so with ``--config=SFNO`` and ``run_num=5410`` the shim must live
    at ``<exp_dir>/SFNO/5410/checkpoints/ckpt_epoch_50.tar`` (one shim
    serves all 8 Y values; the same checkpoint backs all of them).
    The ``Y`` argument is retained for caller-side symmetry but unused.
    """
    return (
        Path(exp_dir)
        / CONFIG_SECTION
        / RUN_NUM
        / "checkpoints"
        / CKPT_FILENAME
    )


def build_ckpt_symlink_shim(
    Y: int,
    exp_dir: Path,
    *,
    target: Path = UPSTREAM_CKPT_PATH,
    overwrite: bool = False,
) -> Path:
    """Create the per-Y single-file ckpt symlink shim.

    Returns the shim path. Asserts post-conditions per §3 P-2:
      * `os.path.islink(shim)` is True;
      * `os.path.realpath(shim)` equals the absolute upstream
        `ckpt_epoch_50.tar` path.

    If ``overwrite`` is False and the shim already exists pointing at
    the right target, leaves it alone (idempotent). If it exists but
    points elsewhere, raises FileExistsError.
    """
    import os

    if not Path(target).is_file():
        raise FileNotFoundError(
            f"upstream checkpoint missing: {target}; cannot build shim for Y={Y}"
        )

    shim = ckpt_shim_path(exp_dir, Y)
    shim.parent.mkdir(parents=True, exist_ok=True)

    if shim.exists() or shim.is_symlink():
        existing = os.path.realpath(shim)
        if existing == str(Path(target).resolve()):
            return shim
        if overwrite:
            shim.unlink()
        else:
            raise FileExistsError(
                f"shim {shim} exists but points at {existing}, "
                f"not {target}; pass overwrite=True to replace"
            )

    shim.symlink_to(str(Path(target).resolve()))

    # Post-conditions per §3 P-2.
    if not shim.is_symlink():
        raise RuntimeError(f"shim creation failed: {shim} is not a symlink")
    realpath = os.path.realpath(shim)
    expected = str(Path(target).resolve())
    if realpath != expected:
        raise RuntimeError(
            f"shim {shim} resolves to {realpath}, not {expected}"
        )
    return shim


def build_all(config_dir: Path, exp_dir: Path, *, K: int) -> dict[int, dict[str, Path]]:
    """Build per-Y yaml + ckpt shim for every year in ``TEST_YEARS``.

    ``K`` is the forecast-leads horizon (required keyword-only).

    Returns ``{Y: {"yaml": Path, "shim": Path}}``.
    """
    out = {}
    for Y in TEST_YEARS:
        out[Y] = {
            "yaml": build_per_y_yaml(Y, config_dir, exp_dir, K=K),
            "shim": build_ckpt_symlink_shim(Y, exp_dir),
        }
    return out
