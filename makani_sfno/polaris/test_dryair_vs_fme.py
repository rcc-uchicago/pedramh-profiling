#!/usr/bin/env python
"""Numerical cross-check: makani ``DryAirFix`` vs ACE2's ``_force_conserve_dry_air``.

Both run on one synthetic state (the real 180x360 cell-centred grid, an 8-layer
hybrid sigma-pressure column, a +350 Pa global dry-air error plus noise):

* ACE2 (fme, run from its own venv): ``ps`` and ``specific_total_water_0..7``,
  its own ``LatLonOperations`` area mean and ``HybridSigmaPressureCoordinate``.
* makani (``mass_fix.py`` loaded by path; it imports only numpy/torch): ``PS``
  and ``TMQ``, with ``TMQ`` set to fme's own ``total_water_path`` so both see
  the same column water.

Checks (each prints one line; any failure prints ``ERROR <check>``):

1. same gravity constant; makani's ``equiangular_weights`` == fme's cos-lat
   area weights on this grid.
2. each implementation conserves the global-mean dry air of ITS OWN definition
   to < 1e-6 Pa (ACE2: q_k held fixed; makani: TMQ held fixed).
3. the predicted algebraic relation between them holds exactly:
   dPS_ace * (1 - sum_k db_k q_k) == dPS_makani   (< 1e-6 Pa).
4. the size of the difference that relation implies (reported, not gated).
5. a bf16 prediction through makani's fix still conserves to < 0.01 Pa.

PASS token: DRYAIR_VS_FME_OK
"""
from __future__ import annotations

import argparse
import importlib.util
import sys

import numpy as np
import torch

NLAT, NLON, B = 180, 360, 3
# 9 interfaces, top -> surface; p_k = ak + bk * ps
AK = [100.0, 3000.0, 8000.0, 12000.0, 12000.0, 9000.0, 5000.0, 1500.0, 0.0]
BK = [0.0, 0.0, 0.02, 0.1, 0.25, 0.45, 0.65, 0.85, 1.0]
Q_BASE = [1e-6, 5e-6, 2e-5, 1e-4, 1e-3, 3e-3, 6e-3, 1e-2]


def _load_mass_fix(path):
    spec = importlib.util.spec_from_file_location("makani_mass_fix", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fail(name, msg):
    print(f"ERROR DRYAIR_VS_FME_{name}: {msg}")
    return 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--mass-fix", required=True, help="path to makani's mass_fix.py")
    args = p.parse_args(argv)

    from fme.core.constants import GRAVITY as FME_G
    from fme.core.coordinates import HybridSigmaPressureCoordinate
    from fme.core.corrector.atmosphere import _force_conserve_dry_air
    from fme.core.gridded_ops import LatLonOperations
    from fme.core.metrics import spherical_area_weights

    mf = _load_mass_fix(args.mass_fix)
    f64 = torch.float64
    torch.manual_seed(0)
    bad = 0

    # --- 1. constants and weights --------------------------------------------
    lat = torch.tensor(90.0 - (np.arange(NLAT) + 0.5) * (180.0 / NLAT), dtype=f64)
    area = spherical_area_weights(lat, NLON).to(f64)
    w_fme = area.sum(dim=-1)
    w_mk = torch.as_tensor(mf.equiangular_weights(NLAT), dtype=f64)
    w_rel = float((w_mk / w_fme - 1).abs().max())
    print(f"1 gravity fme={FME_G} makani={mf.GRAVITY}  weights max rel diff={w_rel:.2e}")
    if mf.GRAVITY != FME_G:
        bad |= _fail("GRAVITY", f"{mf.GRAVITY} != {FME_G}")
    if w_rel > 1e-12:
        bad |= _fail("WEIGHTS", f"max rel diff {w_rel:.3e}")

    ops = LatLonOperations(area)
    vc = HybridSigmaPressureCoordinate(ak=torch.tensor(AK, dtype=f64), bk=torch.tensor(BK, dtype=f64))
    coslat = torch.cos(torch.deg2rad(lat))[:, None]

    def q_field(scale):
        q = torch.tensor(Q_BASE, dtype=f64) * (0.3 + coslat[..., None] ** 2)        # moist tropics
        q = q * (1 + scale * torch.randn(B, NLAT, NLON, len(Q_BASE), dtype=f64))
        return q.clamp(min=0.0)

    ps_in = 98500.0 + 1500.0 * torch.randn(B, NLAT, NLON, dtype=f64)
    q_in = q_field(0.2)
    ps_gen = ps_in + 200.0 * torch.randn(B, NLAT, NLON, dtype=f64) + 350.0
    q_gen = (q_in * (1 + 0.05 * torch.randn_like(q_in))).clamp(min=0.0)

    def as_ace(ps, q):
        d = {"PS": ps}
        d.update({f"specific_total_water_{k}": q[..., k] for k in range(q.shape[-1])})
        return d

    def gm(x):                                    # fme's own area mean, (B,)
        return ops.area_weighted_mean(x, keepdim=False)

    twp_in = vc.vertical_integral(q_in, ps_in)
    twp_gen = vc.vertical_integral(q_gen, ps_gen)
    dry_in = gm(ps_in - FME_G * twp_in)
    err = gm(ps_gen - FME_G * twp_gen) - dry_in
    print(f"  synthetic global dry-air error before fix: {err.min():.1f}..{err.max():.1f} Pa")

    # --- ACE2 ------------------------------------------------------------------
    ace = _force_conserve_dry_air(as_ace(ps_in, q_in), as_ace(ps_gen, q_gen),
                                  area_weighted_mean=ops.area_weighted_mean,
                                  vertical_coordinate=vc, precision=f64)
    ps_ace = ace["PS"]
    # --- makani ----------------------------------------------------------------
    mu = np.array([98500.0, 25.0]); sd = np.array([1200.0, 15.0])
    def z(ps, tmq):
        return torch.stack([(ps - mu[0]) / sd[0], (tmq - mu[1]) / sd[1]], dim=1)
    fix = mf.DryAirFix(channel_names=["PS", "TMQ"], global_means=mu, global_stds=sd, nlat=NLAT)
    out = fix(z(ps_in, twp_in), z(ps_gen, twp_gen))
    ps_mk = out[:, 0].to(f64) * sd[0] + mu[0]

    # --- 2. each conserves its own definition ---------------------------------
    r_ace = float((gm(ps_ace - FME_G * vc.vertical_integral(q_gen, ps_ace)) - dry_in).abs().max())
    r_mk = float((gm(ps_mk - FME_G * twp_gen) - dry_in).abs().max())
    print(f"2 residual global dry-air error: ACE2 {r_ace:.2e} Pa (q fixed)  makani {r_mk:.2e} Pa (TMQ fixed)")
    if r_ace > 1e-6:
        bad |= _fail("ACE_CONSERVE", f"{r_ace:.3e} Pa")
    if r_mk > 1e-6:
        bad |= _fail("MAKANI_CONSERVE", f"{r_mk:.3e} Pa")

    # --- 3. the exact relation -------------------------------------------------
    db = torch.tensor(BK, dtype=f64).diff()
    factor = 1 - (db * q_gen).sum(-1)
    d_ace, d_mk = ps_ace - ps_gen, ps_mk - ps_gen
    rel3 = float((d_ace * factor - d_mk).abs().max())
    print(f"3 max |dPS_ace*(1-sum db q) - dPS_makani| = {rel3:.2e} Pa")
    if rel3 > 1e-6:
        bad |= _fail("RELATION", f"{rel3:.3e} Pa")

    # --- 4. size of the difference (reported) ---------------------------------
    diff = (d_ace - d_mk).abs()
    r_mk_under_ace = float((gm(ps_mk - FME_G * vc.vertical_integral(q_gen, ps_mk)) - dry_in).abs().max())
    print(f"4 |dPS_ace - dPS_makani|: max {float(diff.max()):.3f} Pa, "
          f"max relative {float((diff / d_ace.abs()).max()):.2e} (= max sum db q {float((1 - factor).max()):.2e}); "
          f"makani's PS under ACE2's definition leaves {r_mk_under_ace:.3f} Pa")

    # --- 5. bf16 prediction -----------------------------------------------------
    out16 = fix(z(ps_in, twp_in), z(ps_gen, twp_gen).to(torch.bfloat16))
    ps16 = out16[:, 0].to(f64) * sd[0] + mu[0]
    tmq16 = z(ps_gen, twp_gen).to(torch.bfloat16)[:, 1].to(f64) * sd[1] + mu[1]
    r16 = float((gm(ps16 - FME_G * tmq16) - dry_in).abs().max())
    print(f"5 bf16 prediction -> {out16.dtype}; residual {r16:.2e} Pa")
    if out16.dtype != torch.float32 or r16 > 1e-2:
        bad |= _fail("BF16", f"dtype {out16.dtype}, residual {r16:.3e} Pa")

    if bad:
        return 1
    print("DRYAIR_VS_FME_OK")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001 -- one greppable line, then the traceback
        print(f"ERROR DRYAIR_VS_FME_FAILED {type(exc).__name__}: {str(exc).splitlines()[0][:300]}")
        raise
