"""IC-NetCDF source dispatcher (§B.2, §3 P-7).

`resolve_ic_nc_path(Y, s, run_root)` reads
`<run_root>/inference/ic_source.json` (written once by the §3 P-7 gate
when it picks a contingency) and returns the path of the IC NetCDF for
sample ``s`` of year ``Y``.

Three sources:
  - ``plev_data`` — per-year `plev_data/<Y>_gaussian.nc` (transferred);
    same file for all 12 ICs of a year, indexed by `init_datetime`.
  - ``sigma_data_transferred`` — per-year `sigma_data/<Y>_gaussian.nc`
    after a contingency-C-A Globus transfer.
  - ``ic_nc_built_from_h5`` — per-IC single-timestep NetCDF built from
    the per-timestep h5 files (contingency C-B).
"""
from __future__ import annotations

import json
from pathlib import Path


_PLEV_DATA_DIR = Path(
    "/scratch/11114/zhixingliu/SFNO_Climate_Emulator/data/derecho_glade/sim52/plev_data"
)
_SIGMA_DATA_DIR = Path(
    "/scratch/11114/zhixingliu/SFNO_Climate_Emulator/data/derecho_glade/sim52/sigma_data"
)
_VALID_SOURCES = ("plev_data", "sigma_data_transferred", "ic_nc_built_from_h5")


def resolve_ic_nc_path(Y: int, s: int, run_root: Path) -> Path:
    """Return the IC NetCDF path for ``(Y, s)`` based on `ic_source.json`.

    Raises
    ------
    FileNotFoundError
        If `<run_root>/inference/ic_source.json` is absent.
    ValueError
        If the JSON's `ic_source` field is not one of the three known
        values.
    """
    cfg_path = Path(run_root) / "inference" / "ic_source.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(
            f"missing {cfg_path} — §3 P-7 gate must run first to pin "
            "the IC source"
        )
    cfg = json.loads(cfg_path.read_text())
    src = cfg.get("ic_source")
    if src == "plev_data":
        return _PLEV_DATA_DIR / f"{Y}_gaussian.nc"
    if src == "sigma_data_transferred":
        return _SIGMA_DATA_DIR / f"{Y}_gaussian.nc"
    if src == "ic_nc_built_from_h5":
        return Path(run_root) / "inference" / "ic_nc" / f"{Y}_{s:04d}.nc"
    raise ValueError(
        f"unknown ic_source: {src!r}; expected one of {_VALID_SOURCES}"
    )


def write_ic_source_json(
    run_root: Path,
    ic_source: str,
    *,
    gate_pass_sha256: str,
    extra: dict | None = None,
) -> Path:
    """Persist the gate decision to ``<run_root>/inference/ic_source.json``.

    Returns the path written. Called by the §3 P-7 gate.
    """
    if ic_source not in _VALID_SOURCES:
        raise ValueError(
            f"ic_source must be one of {_VALID_SOURCES}, got {ic_source!r}"
        )
    import datetime as _dt
    payload = {
        "ic_source": ic_source,
        "resolved_at": _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "gate_pass_sha256": gate_pass_sha256,
    }
    if extra:
        payload.update(extra)
    out_dir = Path(run_root) / "inference"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ic_source.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    return out_path
