"""Shared pieces of the five-year figures: calendar, frame alignment, binning, variables, bias statistics.

numpy only and python-3.6 compatible, so test_fiveyear_lib.py runs with the login node's system python3.

Calendar: E3SM noleap, 4 frames a day, 1460 frames a year (climate_driver.py:56-95). A member's step k has a
(valid_year, valid_frame); the truth files hold frames 1..1459 of each year (frame 0, Jan 1 00:00, is never a
target in the truth builder), so every series here is built on frames 1..1459 for both, i.e. on identical times.
"""
import numpy as np

FRAMES_PER_YEAR = 1460
NOLEAP_MONTH_DAYS = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
FIRST_MONTH = (2044, 10)                          # the runs start Oct 1-29 2044
YEARS = (2045, 2046, 2047, 2048, 2049)
SELECTION_WINDOW = (2045, 2047)                   # the protocol's scored window; 2048-2049 were never used to select
MONTHS = [(2044, m) for m in (10, 11, 12)] + [(y, m) for y in YEARS for m in range(1, 13)]
N_MONTHS = len(MONTHS)                            # 63

# key, long name, units label, factor from stored units to plotted units
SERIES_VARS = (
    ("TREFHT", "Near-surface air temperature (TREFHT, 2 m)", "K", 1.0),
    ("PRECT", "Precipitation rate (PRECT)", "mm day$^{-1}$", 1000.0 * 86400.0),
    ("U10", "10 m wind speed (U10)", "m s$^{-1}$", 1.0),
    ("U_l17", "Zonal wind, lowest model level, ~13 m (U_l17)", "m s$^{-1}$", 1.0),
    ("Z3_l12", "Geopotential height, model level 12 (Z3_l12, ~492 hPa)", "m", 1.0),
    ("PS", "Surface pressure (PS; total air mass)", "hPa", 0.01),
    ("TMQ", "Total precipitable water (TMQ)", "kg m$^{-2}$", 1.0),
)
SERIES_KEYS = tuple(v[0] for v in SERIES_VARS)
MAP_KEYS = ("TREFHT", "PRECT", "U10", "U_l17", "Z3_l12")      # the operator's map variables (maps_lib.VAR_KEYS)
HOVMOLLER_KEYS = ("TREFHT", "PRECT")
ZONAL_KEYS = SERIES_KEYS                          # what truth_series.py and the re-roll keep as zonal means


def month_edges():
    """Frame index of each noleap month start plus the year end (13 entries), as climate_driver.month_edges."""
    per_day = FRAMES_PER_YEAR // 365
    edges = [0]
    for d in NOLEAP_MONTH_DAYS:
        edges.append(edges[-1] + d * per_day)
    return edges


def month_of_frames(frames):
    """1-based noleap month of each frame (array), as climate_driver.month_of_frame (bisect_right on the edges)."""
    f = np.asarray(frames)
    if f.size and (f.min() < 0 or f.max() >= FRAMES_PER_YEAR):
        raise ValueError("frame outside [0, %d)" % FRAMES_PER_YEAR)
    return np.searchsorted(np.array(month_edges()), f, side="right")


def month_slot(years, months):
    """Index into MONTHS (0 = 2044-10); -1 for anything outside it."""
    s = (np.asarray(years) - FIRST_MONTH[0]) * 12 + np.asarray(months) - FIRST_MONTH[1]
    return np.where((s >= 0) & (s < N_MONTHS), s, -1)


def aligned(years, frames):
    """Steps that count: inside MONTHS and not frame 0 (absent from the truth)."""
    years, frames = np.asarray(years), np.asarray(frames)
    slot = month_slot(years, month_of_frames(frames))
    return (slot >= 0) & (frames != 0), slot


def bin_by_month(values, years, frames):
    """Monthly means of values (n_steps, ...) on MONTHS. Returns (means (N_MONTHS, ...) NaN where empty, counts)."""
    v = np.asarray(values, dtype=np.float64)
    keep, slot = aligned(years, frames)
    sums = np.zeros((N_MONTHS,) + v.shape[1:])
    counts = np.zeros(N_MONTHS, dtype=np.int64)
    np.add.at(sums, slot[keep], v[keep])
    np.add.at(counts, slot[keep], 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        means = sums / counts.reshape((-1,) + (1,) * (v.ndim - 1))
    means[counts == 0] = np.nan
    return means, counts


def year_means(values, years, frames, wanted=YEARS):
    """Mean of values (n_steps, ...) over each wanted year's frames 1..1459. Returns (means (len(wanted), ...), n)."""
    v = np.asarray(values, dtype=np.float64)
    years, frames = np.asarray(years), np.asarray(frames)
    out = np.full((len(wanted),) + v.shape[1:], np.nan)
    n = np.zeros(len(wanted), dtype=np.int64)
    for i, y in enumerate(wanted):
        sel = (years == y) & (frames != 0)
        n[i] = int(sel.sum())
        if n[i]:
            out[i] = v[sel].mean(axis=0)
    return out, n


def bias_stats(members, truth):
    """Ensemble-mean bias against a single E3SM realisation, with the members' spread as internal variability.

    members (n, ...), truth (...). Returns (bias, t, sd_scale, dof) where t = bias / (s * sqrt(1/n + 1)): the
    (1 + 1/n) inflation counts E3SM's own realisation as one more draw, exactly the test of the 2045-2047 bias maps.
    sd_scale = s * sqrt(1/n + 1); a 95% interval is bias +- t_{0.975, n-1} * sd_scale.
    """
    m = np.asarray(members, dtype=np.float64)
    n = m.shape[0]
    bias = m.mean(axis=0) - np.asarray(truth, dtype=np.float64)
    sd_scale = m.std(axis=0, ddof=1) * np.sqrt(1.0 / n + 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        t = bias / sd_scale
    return bias, t, sd_scale, n - 1
