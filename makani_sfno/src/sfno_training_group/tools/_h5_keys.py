"""Canonical group-format h5 key builder.

Vendored from scripts/infer_sfno5410_blocking_h100_packed.py:113-127 to keep one
source of truth. Group's data_loader_multifiles._get_variable_list (lines 632-643)
constructs identical keys.
"""

from __future__ import annotations

# 10 PlaSim sigma levels, TOA -> surface, exact 16-decimal float repr group expects.
SIGMA_LEVELS: tuple[float, ...] = (
    0.03830000013113022,
    0.11910000443458557,
    0.21085000783205032,
    0.31685000658035278,
    0.43680000305175781,
    0.56680002808570862,
    0.69935008883476257,
    0.82335007190704346,
    0.92409998178482056,
    0.98329997062683105,
)

# 10 zg pressure levels in Pa (=hPa * 100), ascending.
PLEVS_PA: tuple[int, ...] = (
    20000, 25000, 30000, 40000, 50000,
    60000, 70000, 85000, 92500, 100000,
)

# v10 zg level names (hPa) in the same order as PLEVS_PA. zg200 == PLEVS_PA[0]/100.
ZG_HPA: tuple[int, ...] = tuple(p // 100 for p in PLEVS_PA)


def h5_key(var: str, level_i: int | None = None) -> str:
    """Return the per-(var, level) key string under /input/ in a group-format h5.

    Surface (pl, tas), diagnostic (pr_6h), and forcing (lsm, sg, z0, sst, rsdt, sic)
    use bare names. Sigma vars (ta, ua, va, hus) use ``f"{var}_{SIGMA_LEVELS[i]}"``
    with the raw float repr (no rounding). zg uses ``f"zg_{PLEVS_PA[i]}.0"``.
    """
    if var in ("pl", "tas", "pr_6h", "lsm", "sg", "z0", "sst", "rsdt", "sic"):
        return var
    if var in ("ta", "ua", "va", "hus"):
        if level_i is None:
            raise ValueError(f"{var} requires level_i in [0, 9]")
        return f"{var}_{SIGMA_LEVELS[level_i]}"
    if var == "zg":
        if level_i is None:
            raise ValueError("zg requires level_i in [0, 9]")
        return f"zg_{PLEVS_PA[level_i]}.0"
    raise ValueError(f"unknown h5 variable {var!r}")


def all_input_keys_for_smoke() -> list[str]:
    """Return the full 59-key list expected by GetDataset for our smoke YAML.

    Order: 50 upper-air per-level (ta_<sigma>... × 10, then ua, va, hus, zg)
    + 2 surface (pl, tas) + 1 diagnostic (pr_6h)
    + 6 forcing (lsm, sg, z0, sst, rsdt, sic) = 59.
    """
    keys: list[str] = []
    for var in ("ta", "ua", "va", "hus", "zg"):
        for i in range(10):
            keys.append(h5_key(var, i))
    keys.extend(["pl", "tas", "pr_6h", "lsm", "sg", "z0", "sst", "rsdt", "sic"])
    return keys
