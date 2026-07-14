"""Climatology coord-rename compat shim for `score_nwp.py`.

The 5410 group climatology
(``/scratch/.../sim52/baselines/climatology_proleptic_5410.nc``) uses
``time_of_year`` as its day-of-year dim. The own-track ``score_nwp.py``
hardcodes ``ds["doy"]`` at line 79. This helper writes a
single-rename copy of the climatology to a destination path so
``score_nwp.py`` can run unchanged.

Per docs/2026-05-08_sfno_5410_scoring_plan.md (v4.4) §"Climatology
coord rename" (Codex round-1 blocker fix #3).
"""
from __future__ import annotations

from pathlib import Path


def write_compat_clim(src: Path, dst: Path) -> None:
    """Rename ``time_of_year → doy`` in the 5410 climatology and write to dst.

    Idempotent if the input already has ``doy`` as a dim — in that case
    we just symlink (no rename needed). Always writes to ``dst`` (or
    creates a symlink) so the downstream ``score_nwp.py --clim-nc``
    invocation has a single canonical path.

    Parameters
    ----------
    src
        Source climatology NetCDF. Must have either ``time_of_year`` or
        ``doy`` as a dim of length 366.
    dst
        Destination path. Parent dirs are created. If ``dst`` already
        exists it is overwritten (or replaced if it was a symlink).
    """
    import xarray as xr

    src = Path(src)
    dst = Path(dst)
    if not src.is_file():
        raise ValueError(f"climatology source not found: {src}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()

    with xr.open_dataset(src) as ds:
        if "doy" in ds.dims:
            # Already in the canonical form score_nwp.py expects — symlink.
            dst.symlink_to(src.resolve())
            return
        if "time_of_year" not in ds.dims:
            raise ValueError(
                f"climatology {src} has neither 'doy' nor 'time_of_year' "
                f"dim; available dims: {dict(ds.sizes)}"
            )
        out = ds.rename({"time_of_year": "doy"})
    # Write outside the open() context to avoid file-handle conflicts.
    out.to_netcdf(dst)


__all__ = ("write_compat_clim",)
