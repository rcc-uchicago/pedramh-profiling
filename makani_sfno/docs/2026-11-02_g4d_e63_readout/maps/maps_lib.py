"""Shared pieces of the G4D map figures: variables and units, 500 hPa height, area weights, FDR.

numpy only and python-3.6 compatible, so test_maps_lib.py runs with the login node's system python3.
"""
import numpy as np

MAP_LEADS_H = (6, 30, 60, 270)                     # the operator's usual RMSE-map leads
CURVE_LEADS_H = (6, 12, 24, 30, 36, 48, 60, 72, 126, 168, 270, 336)
EXTRACT_LEADS_H = tuple(sorted(set(MAP_LEADS_H) | set(CURVE_LEADS_H)))
P500 = 50000.0                                     # Pa

# key, long name for titles, units label, factor from stored units to the plotted units.
# Stored units are E3SM's: K, m/s (PRECT is a rate in m/s), m. Only PRECT is converted.
VARS = (
    ("TREFHT", "Near-surface air temperature (TREFHT, 2 m)", "K", 1.0),
    ("PRECT", "Precipitation rate (PRECT)", "mm day$^{-1}$", 1000.0 * 86400.0),
    ("U10", "10 m wind speed (U10)", "m s$^{-1}$", 1.0),
    ("U_l17", "Zonal wind, lowest model level, ~13 m (U_l17)", "m s$^{-1}$", 1.0),
    ("Z3_l12", "Geopotential height on model level 12 (Z3_l12; ~492 hPa reference, not a true 500 hPa surface)",
     "m", 1.0),
)
# Z3_l12 stands in for Z500 (operator 2026-11-02): the E3SM source files carry no hyam/hybm, so a true 500 hPa
# surface cannot be built. Level 12's reference pressure is 492 hPa; over high terrain the hybrid level sits at lower
# pressure (higher up), so it departs from a 500 hPa map there. The interpolation helpers below are kept (and
# unit-tested) for when the coefficients exist.
VAR_KEYS = tuple(v[0] for v in VARS)
Z3_LEVELS = tuple("Z3_l%02d" % k for k in range(18))   # l00 = top ... l17 = lowest


def level_pressure(ps, hyam, hybm, p0):
    """Mid-level pressure (Pa) of E3SM's hybrid sigma-pressure coordinate: p_k = hyam_k * P0 + hybm_k * PS.

    ps (H, W) in Pa; hyam, hybm (L,) top -> bottom. Returns (L, H, W).
    """
    ps = np.asarray(ps, dtype=np.float64)
    return (np.asarray(hyam, dtype=np.float64)[:, None, None] * float(p0)
            + np.asarray(hybm, dtype=np.float64)[:, None, None] * ps[None])


def height_at_pressure(z, p, target=P500):
    """Interpolate heights z (L, H, W) on pressures p (L, H, W), p increasing with L, to `target` Pa.

    Linear in ln p between the two levels that bracket the target. Columns where the target lies
    below the lowest level (surface pressure under 500 hPa, i.e. high topography) or above the top
    level are NaN - no extrapolation.
    """
    z = np.asarray(z, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    below = p >= target                          # levels at or below the target surface
    kb = np.argmax(below, axis=0)                # first level with p >= target
    valid = below.any(axis=0) & (kb > 0)
    kb = np.where(valid, kb, 1)
    ka = kb - 1
    take = lambda a, k: np.take_along_axis(a, k[None], axis=0)[0]
    pa, pb, za, zb = take(p, ka), take(p, kb), take(z, ka), take(z, kb)
    w = (np.log(target) - np.log(pa)) / (np.log(pb) - np.log(pa))
    out = za + w * (zb - za)
    return np.where(valid, out, np.nan)


def z500_from_channels(fields, names, hyam, hybm, p0):
    """Z500 (H, W) from a (C, H, W) stack that holds PS and Z3_l00..Z3_l17 (any order, by name)."""
    idx = {n: i for i, n in enumerate(names)}
    ps = fields[idx["PS"]]
    z = np.stack([fields[idx[n]] for n in Z3_LEVELS])
    return height_at_pressure(z, level_pressure(ps, hyam, hybm, p0))


def lat_weights(lat):
    """cos(lat) weights summing to 1 over latitude rows (equiangular grid), as score_window_protocol."""
    w = np.cos(np.deg2rad(np.asarray(lat, dtype=np.float64)))
    return w / w.sum()


def area_mean(field, w):
    """Area-weighted mean over the last two axes (lat, lon), NaN cells dropped and weights renormalised."""
    f = np.asarray(field, dtype=np.float64)
    ww = np.broadcast_to(w[:, None], f.shape[-2:])
    ok = np.isfinite(f)
    num = np.where(ok, f, 0.0) * ww
    den = ok * ww
    return num.sum(axis=(-2, -1)) / den.sum(axis=(-2, -1))


def fdr_mask(p, alpha=0.10):
    """Benjamini-Hochberg: True where the null is rejected at false discovery rate `alpha`.

    alpha_FDR = 0.10 follows Wilks (2016, BAMS 97, 2263) for field significance on spatially correlated
    maps. NaN p-values are never rejected.
    """
    p = np.asarray(p, dtype=np.float64)
    flat = p.ravel()
    ok = np.isfinite(flat)
    pv = flat[ok]
    out = np.zeros(flat.shape, dtype=bool)
    if pv.size:
        order = np.argsort(pv)
        ranked = pv[order]
        thresh = alpha * np.arange(1, pv.size + 1) / pv.size
        passed = np.nonzero(ranked <= thresh)[0]
        if passed.size:
            cut = ranked[passed[-1]]
            sub = np.zeros(pv.size, dtype=bool)
            sub[pv <= cut] = True
            out[np.nonzero(ok)[0]] = sub
    return out.reshape(p.shape)


def roll_lon(field, lon):
    """Re-centre a (..., H, W) field from 0..360 to -180..180 east. Returns (field, lon)."""
    lon = np.asarray(lon, dtype=np.float64)
    lon_c = np.where(lon >= 180.0, lon - 360.0, lon)
    order = np.argsort(lon_c)
    return np.take(field, order, axis=-1), lon_c[order]
