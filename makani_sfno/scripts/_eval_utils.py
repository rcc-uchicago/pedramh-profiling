"""Shared helpers for the NWP scoring + report scripts.

Per docs/plasim_zg_plev_migration_plan.md (v7) §3.10. v9 emulator outputs
carry sigma-level ``zg5`` as the Z500 proxy; v10 emulator outputs carry
literal ``zg500``. This module centralises the resolution of the actual
channel-name list (read from the inference NetCDFs) and the
"which channel is Z500?" detection so ``score_nwp.py`` and
``render_eval_report.py`` agree.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence


# Preferred order — try v10 (literal Z500) first, fall back to v9 sigma proxy.
Z500_PREFERRED: tuple[str, ...] = ("zg500", "zg5")


def resolve_channel_names(
    inference_glob: Path,
    *,
    metadata_json_override: Path | None = None,
) -> list[str]:
    """Resolve the channel-name list for adaptive Z500 detection.

    Priority (per plan §3.10):

      1. Explicit ``metadata_json_override`` — operator escape hatch.
      2. First inference NetCDF matched by ``inference_glob``'s
         ``channel`` coord. All other NetCDFs in the same glob must
         agree (hard-fail if any disagree — that would mean the eval
         mixed v9 and v10 outputs).
    """
    if metadata_json_override is not None:
        return list(
            json.loads(metadata_json_override.read_text())["coords"]["channel"]
        )

    import xarray as xr  # lazy: not needed for the override path

    nc_files = sorted(inference_glob.parent.glob(inference_glob.name))
    if not nc_files:
        raise RuntimeError(f"no inference NetCDFs at {inference_glob}")
    with xr.open_dataset(nc_files[0]) as ds0:
        names0 = list(ds0["channel"].values.astype(str))
    for p in nc_files[1:]:
        with xr.open_dataset(p) as ds:
            names = list(ds["channel"].values.astype(str))
        if names != names0:
            raise RuntimeError(
                f"channel-name disagreement: {nc_files[0].name} has "
                f"{names0[:3]}…, {p.name} has {names[:3]}…; eval cannot "
                f"mix v9 (zg5) and v10 (zg500) outputs."
            )
    return names0


def detect_z500_channel(channel_names: Sequence[str]) -> tuple[str, str]:
    """Return ``(channel_id, label)``.

    ``label`` is human-readable provenance — "Z500 (literal)" for v10,
    "Z500 (sigma proxy, v9)" for v9 — and gets surfaced in printed gate
    messages and report headers so the reader knows what the channel
    actually represents.
    """
    for name in Z500_PREFERRED:
        if name in channel_names:
            label = (
                "Z500 (literal)"
                if name == "zg500"
                else "Z500 (sigma proxy, v9)"
            )
            return name, label
    raise RuntimeError(
        f"no Z500 channel found in {list(channel_names)}; "
        f"expected one of {Z500_PREFERRED}"
    )


def bias_channels(channel_names: Sequence[str]) -> tuple[str, ...]:
    """The 5-channel bias-map list, with Z500 resolved adaptively."""
    z500, _ = detect_z500_channel(channel_names)
    return ("tas", "pr_6h", z500, "ua5", "ta5")
