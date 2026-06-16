"""User-facing inference yaml override for SFNO-5410.

Companion to ``stampede3_yaml_override.py``. The eval-track override
pins ``epsilon_factor=0``, ``save_basenames=["_unused_len1"]``, and
``ensemble_inference_hours=8760/8784`` because the 96-IC sim52 sweep
runs deterministic, single-IC, full-year rollouts. This module is the
clean, configurable variant for an external user who:

  * brings her own initial-condition NetCDF,
  * wants an arbitrary forecast horizon (in days),
  * may want deterministic OR perturbation-ensemble rollouts.

It reuses the path constants + checkpoint-shim builder from
``stampede3_yaml_override.py`` so there's a single source of truth for
where the 5410 boundary tree, climatology, and checkpoint live on
Stampede3.

Upstream constraints worth knowing
----------------------------------
1. ``long_inference.py:1318`` parses ``--init_datetime`` /
   ``--final_datetime`` as ``"%Y-%m-%d_%H:%M:%S"`` with
   ``calendar=params.calendar`` (yaml-fixed to ``proleptic_gregorian``)
   and ``has_year_zero=True``.
2. ``long_inference.py:558, 830`` gate perturbation application on
   ``params.init_datetime.year == 1`` *in addition to*
   ``epsilon_factor > 0``. So for a perturbation-ensemble rollout to
   actually inject noise, the IC's absolute year must be 0001. The
   day-of-year + sub-day position are preserved; only the absolute
   year is relabelled. Boundary-forcing lookup is template-year-driven
   (see point 3) so this relabelling is safe.
3. ``data_loader_multifiles.py:931-934`` builds the boundary h5 path
   as ``<boundary_data_dir>/<template_year>_<idx>.h5`` where
   ``template_year`` is ``params.leap_year`` if the *data* year is leap
   (366-day) and ``params.no_leap_year`` otherwise. Both values must
   point at a year whose h5 tree actually exists on disk.
4. ``num_ensemble_members`` (yaml) controls the rollout fan-out;
   ``len(save_basenames)`` plays no role in 5410 single-IC inference
   (the loader reads ``len(save_basenames)`` only to size a date_range
   array, see ``data_loader_multifiles.py:829``).
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Optional

import cftime
from ruamel.yaml import YAML

from sfno_inference_5410.stampede3_yaml_override import (
    CONFIG_SECTION,
    LEAP_YEARS,
    STAMPEDE3_BIAS_DIR,
    STAMPEDE3_CLIM_NC,
    STAMPEDE3_DATA_DIR,
    UPSTREAM_CKPT_PATH,
    UPSTREAM_LOAD_EXP_DIR,
    UPSTREAM_YAML_PATH,
    build_ckpt_symlink_shim,
    ckpt_shim_path,
)


VALID_PERTURBATION_TYPES = ("gaussian_noise", "gaussian_noise_n_minus_1", "perlin_noise")
DEFAULT_BOUNDARY_TEMPLATE_YEAR = 121
ENSEMBLE_FORCED_INIT_YEAR = 1


def _is_leap_proleptic(year: int) -> bool:
    """Proleptic-gregorian leap rule (every 4, except every 100, except every 400)."""
    return (year % 4 == 0) and (year % 100 != 0 or year % 400 == 0)


def datetime_str(t: cftime.DatetimeProlepticGregorian) -> str:
    return t.strftime("%Y-%m-%d_%H:%M:%S")


def derive_init_final(
    init_year: int,
    init_month: int,
    init_day: int,
    init_hour: int,
    horizon_days: int,
) -> tuple[cftime.DatetimeProlepticGregorian, cftime.DatetimeProlepticGregorian]:
    """Compute (init_dt, final_dt) given the IC clock + horizon in days.

    The model emits 6-hourly outputs; ``horizon_days`` need not be an
    integer number of years. ``final_dt`` is the *exclusive* endpoint
    used by ``xr.cftime_range(..., inclusive='left')`` so the last
    saved step is at ``final_dt - 6h``.
    """
    if horizon_days <= 0:
        raise ValueError(f"horizon_days must be positive, got {horizon_days}")
    init_dt = cftime.DatetimeProlepticGregorian(
        init_year, init_month, init_day, init_hour, has_year_zero=True
    )
    final_dt = init_dt + dt.timedelta(hours=int(horizon_days * 24))
    return init_dt, final_dt


def relabel_for_ensemble(
    init_dt: cftime.DatetimeProlepticGregorian,
) -> cftime.DatetimeProlepticGregorian:
    """Shift the IC's absolute year to 0001 (perturbation gate workaround).

    Day-of-year and sub-day position are preserved. Caller must
    relabel the IC NetCDF's ``time`` coord to match (see
    ``run_sfno_5410_inference.py``'s ``--relabel-ic-to-year-1`` path).
    """
    return cftime.DatetimeProlepticGregorian(
        ENSEMBLE_FORCED_INIT_YEAR,
        init_dt.month,
        init_dt.day,
        init_dt.hour,
        init_dt.minute,
        init_dt.second,
        has_year_zero=True,
    )


def _override_section_user(
    section: dict,
    *,
    init_year: int,
    final_year: int,
    boundary_template_year: int,
    horizon_hours: int,
    epsilon_factor: float,
    perturbation_type: Optional[str],
    num_ensemble_members: int,
    save_basename: str,
    boundary_data_dir: Path,
    bias_data_dir: Path,
    climatology_file: Path,
    exp_dir: Path,
) -> None:
    """Apply user-inference overrides in-place to one yaml section.

    Diverges from ``stampede3_yaml_override._override_section`` in
    three deliberate ways:
      * does NOT pin ``epsilon_factor`` — caller chooses;
      * sets ``save_basenames=[save_basename]`` (length-1 still,
        but a real name not a placeholder);
      * sets ``ensemble_inference_hours = horizon_hours`` — so the
        boundary forcing array is sized to the user's horizon, not
        a fixed sim52 year.
    """
    if "data_dir" in section:
        section["data_dir"] = str(boundary_data_dir)
    if "bias_data_dir" in section:
        section["bias_data_dir"] = str(bias_data_dir)
    if "climatology_file" in section:
        section["climatology_file"] = str(climatology_file)
    if "load_exp_dir" in section:
        section["load_exp_dir"] = str(UPSTREAM_LOAD_EXP_DIR)
    if "exp_dir" in section:
        section["exp_dir"] = str(exp_dir)

    section["val_year_start"] = int(init_year)
    section["val_year_end"] = int(final_year) + (1 if final_year == init_year else 0)

    section["leap_year"] = int(boundary_template_year)
    section["no_leap_year"] = int(boundary_template_year)

    if "log_to_wandb" in section:
        section["log_to_wandb"] = False
    section["save_forecasts"] = True
    section["ensemble_inference_hours"] = int(horizon_hours)
    section["save_basenames"] = [save_basename]
    section["epsilon_factor"] = float(epsilon_factor)
    section["num_ensemble_members"] = int(num_ensemble_members)
    section["ensemble_members_per_pred"] = int(num_ensemble_members)
    if epsilon_factor > 0:
        if perturbation_type is None:
            raise ValueError(
                "perturbation_type is required when epsilon_factor > 0; "
                f"valid values: {VALID_PERTURBATION_TYPES}"
            )
        if perturbation_type not in VALID_PERTURBATION_TYPES:
            raise ValueError(
                f"perturbation_type={perturbation_type!r} not in "
                f"{VALID_PERTURBATION_TYPES}"
            )
        section["perturbation_type"] = perturbation_type


def build_user_yaml(
    *,
    out_dir: Path,
    exp_dir: Path,
    init_year: int,
    final_year: int,
    boundary_template_year: int,
    horizon_hours: int,
    epsilon_factor: float,
    num_ensemble_members: int,
    save_basename: str,
    perturbation_type: Optional[str] = None,
    boundary_data_dir: Path = STAMPEDE3_DATA_DIR,
    bias_data_dir: Path = STAMPEDE3_BIAS_DIR,
    climatology_file: Path = STAMPEDE3_CLIM_NC,
    src_yaml: Path = UPSTREAM_YAML_PATH,
) -> Path:
    """Write a per-run user-inference yaml; return its path.

    Writes to ``<out_dir>/SFNO_PLASIM_H5_DERECHO_5410_user.yaml``.
    The output is a copy of the upstream yaml with path-fields remapped
    to Stampede3 (or to the user's BYO boundary tree), the inference
    knobs set per the caller's choice, and forecast/wandb flags forced.
    """
    if not src_yaml.is_file():
        raise FileNotFoundError(f"upstream yaml not found: {src_yaml}")
    if num_ensemble_members < 1:
        raise ValueError(f"num_ensemble_members must be >=1, got {num_ensemble_members}")
    if (
        boundary_template_year in LEAP_YEARS
        and not _is_leap_proleptic(init_year)
    ):
        raise ValueError(
            f"boundary_template_year={boundary_template_year} is leap (sim52) "
            f"but init_year={init_year} is not leap (proleptic_gregorian). "
            f"The h5 tree has 1464 indices but the rollout will reach at most "
            f"1460. Pick a non-leap template (e.g. 121) or a leap init year."
        )
    if (
        boundary_template_year not in LEAP_YEARS
        and _is_leap_proleptic(init_year)
        and (final_year > init_year or horizon_hours > 1460 * 6)
    ):
        raise ValueError(
            f"init_year={init_year} is leap (366 days) but template "
            f"boundary_template_year={boundary_template_year} is non-leap "
            f"(only 1460 h5 indices). A rollout extending past day 365 "
            f"will read past end-of-tree. Pick a leap template (124 or 128) "
            f"or shorten the horizon."
        )

    yaml = YAML()
    yaml.preserve_quotes = True
    with open(src_yaml, "r") as f:
        doc = yaml.load(f)

    for _, section in doc.items():
        if isinstance(section, dict):
            _override_section_user(
                section,
                init_year=init_year,
                final_year=final_year,
                boundary_template_year=boundary_template_year,
                horizon_hours=horizon_hours,
                epsilon_factor=epsilon_factor,
                perturbation_type=perturbation_type,
                num_ensemble_members=num_ensemble_members,
                save_basename=save_basename,
                boundary_data_dir=Path(boundary_data_dir),
                bias_data_dir=Path(bias_data_dir),
                climatology_file=Path(climatology_file),
                exp_dir=Path(exp_dir),
            )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "SFNO_PLASIM_H5_DERECHO_5410_user.yaml"
    with open(out_path, "w") as f:
        yaml.dump(doc, f)
    return out_path


__all__ = (
    "VALID_PERTURBATION_TYPES",
    "DEFAULT_BOUNDARY_TEMPLATE_YEAR",
    "ENSEMBLE_FORCED_INIT_YEAR",
    "build_user_yaml",
    "build_ckpt_symlink_shim",
    "ckpt_shim_path",
    "datetime_str",
    "derive_init_final",
    "relabel_for_ensemble",
    "UPSTREAM_CKPT_PATH",
)
