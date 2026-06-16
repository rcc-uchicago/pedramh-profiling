"""IC offsets for the group-SFNO-5410 NWP evaluation (§A.2).

12 ICs/year on a monthly stride of 122 timesteps (≈ 30.5 days at 6-h
cadence), with K=60 lead-step rollout. Anchored at Jan 1 of year Y,
00:00:00 (proleptic_gregorian, has_year_zero=True) — see §A.1.

The expected offsets for both 1460 and 1464 timesteps (non-leap and
leap test years) are the same:

    [0, 122, 244, 366, 488, 610, 732, 854, 976, 1098, 1220, 1342]

since `s + K = 1342 + 60 = 1402 < 1460`. Pinned in `test_ic_offsets.py`.
"""
from __future__ import annotations


_DEFAULT_OFFSETS = (0, 122, 244, 366, 488, 610, 732, 854, 976, 1098, 1220, 1342)


def nwp_ic_offsets_5410(
    n_samples: int,
    K: int = 60,
    n_ic: int = 12,
    step: int = 122,
) -> list[int]:
    """Return ``n_ic`` IC sample indices on a monthly stride.

    Parameters
    ----------
    n_samples : int
        Number of 6-hour timesteps in the calendar year (1460 or 1464).
    K : int, default 60
        Forecast lead-step horizon. Used only to validate that the last
        IC's window fits within the year.
    n_ic : int, default 12
        Number of ICs (one per ~month).
    step : int, default 122
        IC stride in 6-hour timesteps.

    Returns
    -------
    list[int]
        Sample indices ``[0, step, 2*step, ..., (n_ic-1)*step]``.

    Raises
    ------
    ValueError
        If the last IC's lead-K window would run past ``n_samples``.
    """
    last_s = (n_ic - 1) * step
    if last_s + K >= n_samples:
        raise ValueError(
            f"IC schedule overruns year: last IC s={last_s}, K={K}, "
            f"last lead s+K={last_s + K} >= n_samples={n_samples}"
        )
    return [i * step for i in range(n_ic)]
