"""Hydrate ``YParams`` with every field upstream ``long_inference.py main()``
injects between argparse and ``Stepper(...)`` construction.

The in-process orchestrator (``scripts/eval_inference_5410.py``) builds the
Stepper once and reuses it across all 96 ICs. Upstream's ``main()`` does
significant param injection before ``Stepper(...)`` is called — see
``/work2/.../v2.0/long_inference.py`` lines 1252-1445. The new orchestrator
must reproduce the upstream setup surface, with one 5410-specific
correction: the single-IC BCS loader must use the same boundary phase as
the model's training/validation autoregression path. The standalone
``long_inference.py`` path previously hard-coded ``nc_bc_offset = 18``
for a different long-run convention; for the 5410 NWP eval that shifts
the first model step from IC-time boundary forcing to ``init + 18h`` and
corrupts the rollout.

Three helpers split the param surface by lifetime:

  * :func:`hydrate_static_params` — mutations that don't depend on a
    specific IC or year. Called once per orchestrator invocation, before
    ``Stepper(...)``.
  * :func:`set_per_y_params` — fields that change per target year
    (val_year_start, val_year_end) plus the boundary template years
    (leap_year, no_leap_year). Called when crossing a Y boundary.
  * :func:`set_per_ic_params` — fields that change per IC (init_datetime,
    final_datetime, init_nc_filepaths, init_nc_timestep_offset,
    save_basename, output_dir). Called every IC.

Plus a static-architecture invariant check across the 8 per-Y yamls:

  * :func:`assert_yamls_share_static_arch` — proves that the model can be
    built from any one params object and reused across all Y values
    (i.e., that swapping the per-Y yaml-loaded params for a different Y
    only changes the year-dependent fields, not the architecture/ckpt
    config).
"""
from __future__ import annotations

import datetime as dt
import os
import sys
from pathlib import Path
from typing import Any

import cftime
import xarray as xr


# Pinned static values upstream main() injects (constants, not env-derived).
#
# Boundary phase fix: validation autoregression uses boundary fields at
# the current step. The NWP eval must match validation, so the corrected
# offset is 0.
_NC_BC_OFFSET = 0
_RUN_ITER_DEFAULT = 1
_WORLD_SIZE_DEFAULT = 1
_BATCH_SIZE_DEFAULT = 1  # = world_size for single-rank
_LOCAL_RANK_DEFAULT = 0
_ENABLE_AMP = True  # long_inference.py:1375
_RESUMING = True  # long_inference.py:1427

# Fields whose presence/values define a "fully hydrated for the static
# architecture phase" params. Used by tests/test_upstream_hydration.py.
_STATIC_HYDRATION_ATTRS = (
    "run_iter", "has_diagnostic", "num_ensemble_members",
    "ensemble_members_per_pred", "nc_bc_offset",
    "world_size", "batch_size", "local_rank", "enable_amp",
    "experiment_dir", "checkpoint_dir", "best_checkpoint_path",
    "latest_checkpoint_path", "checkpoint_path_globstr", "resuming",
    "log_to_wandb", "log_to_screen",
)
_PER_Y_HYDRATION_ATTRS = (
    "val_year_start", "val_year_end", "leap_year", "no_leap_year",
)
_PER_IC_HYDRATION_ATTRS = (
    "init_datetime", "final_datetime", "init_nc_filepaths",
    "init_nc_timestep_offset", "save_basename", "output_dir",
)

# Fields that must match across all 8 per-Y yamls. Anything outside this
# allowlist + the per-Y allowlist is unexpected and would be flagged by
# `assert_yamls_share_static_arch` as drift between yamls.
_PER_Y_FIELDS = frozenset(_PER_Y_HYDRATION_ATTRS)


def _add_upstream_to_path(upstream_repo: Path) -> None:
    """Insert upstream repo on sys.path so ``utils.YParams`` resolves."""
    p = str(upstream_repo)
    if p not in sys.path:
        sys.path.insert(0, p)


def hydrate_static_params(
    yaml_path: Path,
    K: int,
    *,
    upstream_repo: Path,
    run_num: str = "5410",
    config_section: str = "SFNO",
    world_rank: int = 0,
    local_rank: int = 0,
):
    """Load YParams + apply every static-phase mutation upstream main() does.

    Mirrors ``long_inference.py:1252-1436`` for the **static** phase only
    — i.e., all fields that don't depend on a specific IC or year. Per-IC
    fields (init_datetime, final_datetime, init_nc_filepaths,
    init_nc_timestep_offset, save_basename, output_dir) are set by
    :func:`set_per_ic_params`. Per-Y fields (val_year_start, val_year_end,
    leap_year, no_leap_year) come from the loaded yaml directly and can
    be re-mutated per Y crossing via :func:`set_per_y_params`.

    Pinned values: ``nc_bc_offset == 0``, ``world_size == 1``,
    ``batch_size == 1``, ``enable_amp == True``, ``log_to_wandb == False``,
    ``resuming == True``, ``run_iter == 1``.

    Parameters
    ----------
    yaml_path
        Path to one of the 8 per-Y yamls. The static fields are
        identical across all 8 yamls (cross-checked by
        :func:`assert_yamls_share_static_arch`), so passing yaml_paths[0]
        is sufficient for the static phase.
    K
        Forecast-leads horizon. Currently unused by the static phase but
        accepted so callers don't accidentally drop it; future hydration
        bumps may pin K-derived fields here.
    upstream_repo
        ``/work2/.../v2.0`` so ``utils.YParams`` resolves.
    run_num, config_section
        Forwarded to upstream's experiment_dir construction at
        long_inference.py:1394.
    world_rank, local_rank
        Single-rank inference uses 0/0.
    """
    upstream_repo = Path(upstream_repo)
    _add_upstream_to_path(upstream_repo)
    from utils.YParams import YParams  # type: ignore

    if not yaml_path.is_file():
        raise FileNotFoundError(f"yaml not found: {yaml_path}")

    # WANDB_MODE is set as an env var, not a params field, in upstream.
    os.environ.setdefault("WANDB_MODE", "offline")

    params = YParams(str(yaml_path), config_section)

    # --- one-time injections from upstream main() (line ranges below). ---

    # 1262: run_iter (CLI arg, default 1).
    params['run_iter'] = _RUN_ITER_DEFAULT

    # 1263-1269: has_diagnostic from diagnostic_variables length.
    if hasattr(params, 'diagnostic_variables') and len(params.diagnostic_variables) > 0:
        params['has_diagnostic'] = True
    else:
        params['has_diagnostic'] = False

    # 1273-1274: num_ensemble_members default 1.
    if not hasattr(params, 'num_ensemble_members'):
        params['num_ensemble_members'] = 1

    # Boundary phase fix: see _NC_BC_OFFSET above.
    params['nc_bc_offset'] = _NC_BC_OFFSET

    # 1281-1282: ensemble_members_per_pred default = num_ensemble_members.
    if not hasattr(params, 'ensemble_members_per_pred'):
        params['ensemble_members_per_pred'] = params.num_ensemble_members

    # 1284-1294: world_size + batch_size. Single-rank inference.
    params['world_size'] = _WORLD_SIZE_DEFAULT
    params['batch_size'] = _BATCH_SIZE_DEFAULT

    # 1372: local_rank.
    params['local_rank'] = local_rank

    # 1375: enable_amp pinned True regardless of CLI arg.
    params['enable_amp'] = _ENABLE_AMP

    # 1392-1404: experiment_dir + checkpoint paths.
    expDir = os.path.join(params.exp_dir, config_section, str(run_num))
    if world_rank == 0 and not os.path.isdir(expDir):
        os.makedirs(expDir, exist_ok=True)
    params['experiment_dir'] = os.path.abspath(expDir)
    params['checkpoint_dir'] = os.path.join(expDir, 'checkpoints')
    params['best_checkpoint_path'] = os.path.join(
        params['checkpoint_dir'], 'best_ckpt.tar',
    )
    params['latest_checkpoint_path'] = os.path.join(
        params['checkpoint_dir'], 'ckpt_latest.tar',
    )
    params['checkpoint_path_globstr'] = os.path.join(
        params['checkpoint_dir'], 'ckpt_epoch_*.tar',
    )

    # 1407-1424: ckpt resolution priority best > latest > globstr.
    if os.path.isfile(params.best_checkpoint_path):
        ckpt_path = params.best_checkpoint_path
    elif os.path.isfile(params.latest_checkpoint_path):
        ckpt_path = params.latest_checkpoint_path
    else:
        import glob
        from natsort import natsorted
        cks = natsorted([
            f for f in glob.glob(params.checkpoint_path_globstr) if os.path.isfile(f)
        ])
        if not cks:
            raise FileNotFoundError(
                f"no checkpoint found at {params.best_checkpoint_path}, "
                f"{params.latest_checkpoint_path}, or "
                f"{params.checkpoint_path_globstr}"
            )
        ckpt_path = cks[-1]
    params['best_checkpoint_path'] = ckpt_path

    # 1427: resuming.
    params['resuming'] = _RESUMING

    # 1435-1436: log flags.
    params['log_to_wandb'] = False
    if not hasattr(params, 'log_to_screen'):
        params['log_to_screen'] = True
    params['log_to_screen'] = (world_rank == 0) and bool(params.log_to_screen)

    return params


def set_per_y_params(params, *, Y: int) -> None:
    """Mutate target-year and boundary-template fields.

    These fields drive the boundary-loader contract:
      * val_year_start = Y, val_year_end = Y + 1 (data_loader_multifiles.py:948)
      * leap_year/no_leap_year select the prescribed-boundary template
        years (data_loader_multifiles.py:931-934). If the yaml carries
        boundary_leap_year/boundary_no_leap_year, use those; otherwise
        fall back to the historical Y-as-template behavior.
    """
    if not isinstance(Y, int) or isinstance(Y, bool):
        raise ValueError(f"Y must be int (not bool), got {Y!r}")
    if Y < 1:
        raise ValueError(f"Y must be positive, got {Y}")
    boundary_leap_year = int(getattr(params, 'boundary_leap_year', Y))
    boundary_no_leap_year = int(getattr(params, 'boundary_no_leap_year', Y))
    params['val_year_start'] = Y
    params['val_year_end'] = Y + 1
    params['leap_year'] = boundary_leap_year
    params['no_leap_year'] = boundary_no_leap_year


def set_per_ic_params(
    params,
    *,
    init_datetime,
    final_datetime,
    init_nc_filepaths,
    save_basename: str,
    output_dir,
) -> None:
    """Mutate the per-IC fields. Called every IC, before reconfigure_for_ic.

    Recomputes ``init_nc_timestep_offset`` by opening each IC NC file and
    looking up ``init_datetime`` in its time index — same as upstream
    main() lines 1340-1344. The new ``Stepper.reconfigure_for_ic`` (LP-004)
    validates this offset against the current files and fails loud if a
    caller bypasses this helper.
    """
    if not isinstance(save_basename, str) or not save_basename:
        raise ValueError(
            f"save_basename must be non-empty str, got {save_basename!r}"
        )
    paths = (
        list(init_nc_filepaths)
        if isinstance(init_nc_filepaths, (list, tuple))
        else [init_nc_filepaths]
    )
    if len(paths) != 1:
        raise ValueError(
            f"single-IC invariant: init_nc_filepaths must be length 1, "
            f"got {len(paths)}"
        )
    paths = [str(p) for p in paths]

    # Cftime-normalize datetimes (mirrors long_inference.py:1334-1338, 1362-1366).
    # ``datetime_class_from_calendar`` is exported from upstream's
    # data_loader_multifiles module — see long_inference.py:8 for the
    # canonical import.
    from utils.data_loader_multifiles import datetime_class_from_calendar  # type: ignore[import-not-found]
    _dt_cls = datetime_class_from_calendar(params.calendar)
    init_dt = _dt_cls(
        init_datetime.year, init_datetime.month, init_datetime.day,
        hour=init_datetime.hour, has_year_zero=params.has_year_zero,
    )
    final_dt = _dt_cls(
        final_datetime.year, final_datetime.month, final_datetime.day,
        hour=final_datetime.hour, has_year_zero=params.has_year_zero,
    )

    params['init_datetime'] = init_dt
    params['final_datetime'] = final_dt
    params['init_nc_filepaths'] = paths
    params['save_basename'] = save_basename
    params['output_dir'] = str(output_dir)

    # Recompute init_nc_timestep_offset by opening each IC NC.
    offsets = []
    for ic_path in paths:
        with xr.open_dataset(ic_path, engine='netcdf4') as ds:
            offsets.append(int(ds.get_index("time").get_loc(init_dt)))
    params['init_nc_timestep_offset'] = offsets


def assert_yamls_share_static_arch(yaml_paths: list[Path]) -> None:
    """Cross-yaml invariant: every architecture/checkpoint/normalization/
    precision field matches across all 8 yamls.

    Only ``val_year_start``, ``val_year_end``, ``leap_year``, ``no_leap_year``
    may differ. Anything else differing means swapping the per-Y yaml
    would silently change the model architecture, which would invalidate
    the "build Stepper once, loop 96 ICs" plan.

    Loads each yaml as a plain dict (via ruamel) so we don't need
    ``utils.YParams`` here — this preflight runs without the upstream
    repo on sys.path.
    """
    from ruamel.yaml import YAML

    if len(yaml_paths) < 2:
        # Single yaml is trivially self-consistent.
        return

    yaml_paths = [Path(p) for p in yaml_paths]
    yaml = YAML()
    yaml.preserve_quotes = True

    docs = []
    for yp in yaml_paths:
        if not yp.is_file():
            raise FileNotFoundError(f"yaml not found: {yp}")
        with open(yp, "r") as f:
            docs.append(yaml.load(f))

    # Pull the SFNO section from each (architecture lives there).
    sfno_sections = []
    for yp, doc in zip(yaml_paths, docs):
        if "SFNO" not in doc:
            raise ValueError(f"yaml {yp} missing SFNO section")
        sfno_sections.append(doc["SFNO"])

    ref_yp, ref = yaml_paths[0], sfno_sections[0]
    ref_keys = set(ref.keys())

    for yp, sec in zip(yaml_paths[1:], sfno_sections[1:]):
        sec_keys = set(sec.keys())
        if sec_keys != ref_keys:
            missing = ref_keys - sec_keys
            extra = sec_keys - ref_keys
            raise ValueError(
                f"yaml {yp} key set differs from {ref_yp}: "
                f"missing {sorted(missing)}, extra {sorted(extra)}"
            )
        for key in ref_keys:
            if key in _PER_Y_FIELDS:
                continue  # per-Y fields legitimately differ
            ref_val = ref[key]
            sec_val = sec[key]
            if ref_val != sec_val:
                raise ValueError(
                    f"yaml {yp}: SFNO.{key} == {sec_val!r} but "
                    f"{ref_yp}: SFNO.{key} == {ref_val!r}. "
                    f"All non-(val_year_start, val_year_end, leap_year, "
                    f"no_leap_year) fields must match across the 8 per-Y "
                    f"yamls — otherwise swapping the per-Y yaml would "
                    f"change the model architecture or normalization."
                )


__all__ = (
    "hydrate_static_params",
    "set_per_y_params",
    "set_per_ic_params",
    "assert_yamls_share_static_arch",
    "_STATIC_HYDRATION_ATTRS",
    "_PER_Y_HYDRATION_ATTRS",
    "_PER_IC_HYDRATION_ATTRS",
)
