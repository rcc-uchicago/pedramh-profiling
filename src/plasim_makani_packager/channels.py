"""Channel lists for the PlaSim → Makani three-dataset layout.

Locked by docs/plasim_zg_plev_migration_plan.md (v7). v10 contract:
zg sigma levels are replaced by ten pressure-level channels named
``zg{P}`` (P in hPa), in TOA → surface order. Order matters: consumers
read channels by index into the HDF5 channel dimension.

For v9 regeneration (sigma-level zg1..zg10), check out the
``plasim-makani-packager-v9-final`` git tag.
"""

from __future__ import annotations

# Sigma ordering convention: lev[0] = TOA, lev[9] = surface
# → taN where N=1..10, ta1 = TOA, ta10 = surface (verified against
# MOST.{YYYY}.nc lev coord: [0.0383 .. 0.9833]).
_SIGMA_VARS: tuple[str, ...] = ("ta", "ua", "va", "hus")
_SIGMA_LEVELS: int = 10


def _sigma_names(var: str) -> list[str]:
    return [f"{var}{i}" for i in range(1, _SIGMA_LEVELS + 1)]


# Pressure-level zg subset (v10). TOA → surface order, integer hPa.
# Selected per docs/plasim_zg_plev_migration_plan.md L1: drops 50/100
# (redundant given top sigma + sigma channels) and 1000 (below-ground
# extrapolation noise); includes 925 for boundary-layer coupling and
# 500 explicitly for blocking / Z500 skill.
ZG_PLEV_HPA: tuple[int, ...] = (
    150, 200, 250, 300, 400, 500, 600, 700, 850, 925,
)
assert len(ZG_PLEV_HPA) == 10


def _zg_plev_names() -> list[str]:
    return [f"zg{p}" for p in ZG_PLEV_HPA]


STATE_CHANNELS: list[str] = (
    ["pl", "tas"]
    + _sigma_names("ta")
    + _sigma_names("ua")
    + _sigma_names("va")
    + _sigma_names("hus")
    + _zg_plev_names()
)
assert len(STATE_CHANNELS) == 52
assert STATE_CHANNELS[42] == "zg150"
assert STATE_CHANNELS[47] == "zg500"
assert STATE_CHANNELS[51] == "zg925"

DIAGNOSTIC_CHANNELS: list[str] = ["pr_6h"]

# Forcing order: three static (lsm, sg, z0) first, then three varying.
# `sst`, `rsdt`, `sic` come from the emulator adaptor (boundary_astro).
FORCING_CHANNELS: list[str] = ["lsm", "sg", "z0", "sst", "rsdt", "sic"]

TARGET_CHANNELS: list[str] = STATE_CHANNELS + DIAGNOSTIC_CHANNELS
assert len(TARGET_CHANNELS) == 53
