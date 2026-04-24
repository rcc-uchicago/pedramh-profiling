"""Channel lists for the PlaSim → Makani three-dataset layout.

Locked by docs/plasim_makani_packager_plan.md (v9). Order matters:
consumers read these by index into the HDF5 channel dimension.
"""

from __future__ import annotations

# Sigma ordering convention: lev[0] = TOA, lev[9] = surface
# → taN where N=1..10, ta1 = TOA, ta10 = surface (verified against
# MOST.{YYYY}.nc lev coord: [0.0383 .. 0.9833]).
_SIGMA_VARS: tuple[str, ...] = ("ta", "ua", "va", "hus", "zg")
_SIGMA_LEVELS: int = 10


def _sigma_names(var: str) -> list[str]:
    return [f"{var}{i}" for i in range(1, _SIGMA_LEVELS + 1)]


STATE_CHANNELS: list[str] = (
    ["pl", "tas"]
    + _sigma_names("ta")
    + _sigma_names("ua")
    + _sigma_names("va")
    + _sigma_names("hus")
    + _sigma_names("zg")
)
assert len(STATE_CHANNELS) == 52

DIAGNOSTIC_CHANNELS: list[str] = ["pr_6h"]

# Forcing order: three static (lsm, sg, z0) first, then three varying.
# `sst`, `rsdt`, `sic` come from the emulator adaptor (boundary_astro).
FORCING_CHANNELS: list[str] = ["lsm", "sg", "z0", "sst", "rsdt", "sic"]

TARGET_CHANNELS: list[str] = STATE_CHANNELS + DIAGNOSTIC_CHANNELS
assert len(TARGET_CHANNELS) == 53
