"""sfno_inference_5410 — Stampede3 adapter around upstream PanguWeather/v2.0.

Implements the group-SFNO-5410 evaluation track (Phase 1 NWP only)
specified in ``docs/2026-05-06_group_sfno_5410_eval_plan.md``. The
upstream inference engine is `long_inference.py` from the v2.0 source
tree at ``/work2/.../PanguWeather/v2.0/``; this package supplies the
Stampede3-specific glue (path-overridden yaml + symlink shim, IC offsets,
IC-source dispatcher, output adapter, channel map).

Modules:
  - ``ic_offsets`` — 12 ICs/year × monthly stride 122 (§A.2).
  - ``ic_source`` — ``resolve_ic_nc_path(Y, s, run_root)`` reads
    ``<run_root>/inference/ic_source.json`` (written by §3 P-7 gate)
    and dispatches to one of three IC-NetCDF sources.
  - ``stampede3_yaml_override`` — emit per-Y Stampede3-pathed copy of the
    upstream yaml + assemble the per-Y single-file ckpt symlink shim
    (§3 P-2).
"""
from sfno_inference_5410.ic_offsets import nwp_ic_offsets_5410
from sfno_inference_5410.ic_source import resolve_ic_nc_path
from sfno_inference_5410.stampede3_yaml_override import (
    build_per_y_yaml,
    build_ckpt_symlink_shim,
    TEST_YEARS,
    UPSTREAM_CKPT_PATH,
    UPSTREAM_YAML_PATH,
)

__all__ = [
    "nwp_ic_offsets_5410",
    "resolve_ic_nc_path",
    "build_per_y_yaml",
    "build_ckpt_symlink_shim",
    "TEST_YEARS",
    "UPSTREAM_CKPT_PATH",
    "UPSTREAM_YAML_PATH",
]
