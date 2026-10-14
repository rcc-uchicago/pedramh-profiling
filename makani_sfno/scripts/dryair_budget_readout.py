"""Split a climate member's PS drift into its dry-air and water parts.

Question (polaris_makani_b_dryair_findings_handoff.md §2, §4.1): DRYAIR24 runs with
the dry-air fix on, which pins the global mean of ``PS - g·TMQ`` to each step's
input, yet its global-mean PS drifts ≈ −21 hPa over 5 years. Either

  H1  dry mass is flat and TMQ fell by ΔPS/g (≈ −214 kg m⁻², i.e. negative), or
  H2  dry mass itself leaks (the fix is undone between steps, e.g. by rounding
      of the fed-back state), and TMQ is roughly physical.

Reads the ``global_mean`` (step, channel) series that ``climate_driver`` writes for
every lead (same equiangular cos-lat weights as ``mass_fix.py``) plus the monthly
means, and prints one line per member. Measurement only: no model is run.

Usage:  python dryair_budget_readout.py <member.nc> ...      (or --selftest)
PASS = DRYAIR_BUDGET_OK n=<members>
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np

GRAVITY = 9.80665  # m s-2, mass_fix.GRAVITY


def area_weights(nlat: int) -> np.ndarray:
    """mass_fix.equiangular_weights, duplicated so this needs only numpy/netCDF4."""
    lat = np.deg2rad(90.0 - (np.arange(nlat) + 0.5) * (180.0 / nlat))
    w = np.cos(lat)
    return w / w.sum()


def readout(path: Path) -> dict:
    import netCDF4

    with netCDF4.Dataset(path) as ds:
        names = [str(c) for c in ds["channel"][:]]
        ips, iq = names.index("PS"), names.index("TMQ")
        gm = np.asarray(ds["global_mean"][:], dtype=np.float64)           # (n, C)
        ps, q = gm[:, ips], gm[:, iq]
        dry = ps - GRAVITY * q
        out = dict(member=path.stem.removeprefix("member_"), n=len(ps),
                   dps=(ps[-1] - ps[0]) / 100.0, ddry=(dry[-1] - dry[0]) / 100.0,
                   gdq=GRAVITY * (q[-1] - q[0]) / 100.0, q0=q[0], q1=q[-1],
                   qmin_gm=float(q.min()),
                   # largest one-step change in dry mass: ~0 if the fix holds every step
                   step_ddry=float(np.abs(np.diff(dry)).max() / 100.0) if len(dry) > 1 else 0.0)
        mm = ds["monthly_mean"]
        nm = mm.shape[0]
        if nm:
            w = area_weights(mm.shape[2])[:, None] / mm.shape[3]
            last = np.asarray(mm[nm - 1, iq], dtype=np.float64)
            mins = [float(np.asarray(mm[k, iq]).min()) for k in range(nm)]
            out.update(mm_qmin=min(mins), mm_negfrac=float((w * (last < 0)).sum()))
        else:
            out.update(mm_qmin=float("nan"), mm_negfrac=float("nan"))
    return out


def fmt(r: dict) -> str:
    return (f"{r['member']:<16} n={r['n']:>5} dPS={r['dps']:+8.2f} dDRY={r['ddry']:+8.2f} "
            f"g*dTMQ={r['gdq']:+8.2f} hPa | TMQ gm {r['q0']:6.2f}->{r['q1']:7.2f} "
            f"(min {r['qmin_gm']:7.2f}) kg/m2 | max|step dDRY|={r['step_ddry']:.3f} hPa | "
            f"monthly TMQ min={r['mm_qmin']:8.2f} lastmonth neg={100 * r['mm_negfrac']:5.1f}%")


def selftest() -> None:
    """Synthetic member: dry mass fixed, TMQ falls 10 kg/m2 => dPS = g*dTMQ, dDRY = 0."""
    import netCDF4

    n, H, W = 5, 4, 8
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "member_SELF00.nc"
        with netCDF4.Dataset(p, "w") as ds:
            for k, v in (("channel", 3), ("step", n), ("month", 1), ("lat", H), ("lon", W)):
                ds.createDimension(k, v)
            ds.createVariable("channel", str, ("channel",))[:] = np.array(["T", "PS", "TMQ"], dtype=object)
            q = np.linspace(25.0, 15.0, n)
            ps = 98000.0 + GRAVITY * (q - q[0])
            gm = np.stack([np.full(n, 280.0), ps, q], axis=1)
            ds.createVariable("global_mean", "f8", ("step", "channel"))[:] = gm
            mm = np.full((1, 3, H, W), 20.0, dtype=np.float32)
            mm[0, 2, 0, :] = -1.0                                   # one polar row negative
            ds.createVariable("monthly_mean", "f4", ("month", "channel", "lat", "lon"))[:] = mm
        r = readout(p)
    exp_frac = area_weights(H)[0]
    checks = {"dDRY": abs(r["ddry"]) < 1e-9, "g*dTMQ": abs(r["gdq"] - GRAVITY * -10 / 100) < 1e-9,
              "dPS": abs(r["dps"] - r["gdq"]) < 1e-9, "mm_qmin": r["mm_qmin"] == -1.0,
              "negfrac": abs(r["mm_negfrac"] - exp_frac) < 1e-12}
    bad = [k for k, ok in checks.items() if not ok]
    if bad:
        print(f"ERROR DRYAIR_BUDGET_SELFTEST failed: {bad} {r}")
        sys.exit(1)
    print("DRYAIR_BUDGET_SELFTEST_OK")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("nc", nargs="*", type=Path)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    n = 0
    for p in a.nc:
        try:
            print(fmt(readout(p)))
            n += 1
        except Exception as e:  # noqa: BLE001 -- one bad file must not hide the rest
            print(f"ERROR DRYAIR_BUDGET_READ {p.name}: {type(e).__name__}: {e}")
    print(f"DRYAIR_BUDGET_OK n={n}" if n == len(a.nc) and n else
          f"ERROR DRYAIR_BUDGET_INCOMPLETE read={n}/{len(a.nc)}")


if __name__ == "__main__":
    main()
