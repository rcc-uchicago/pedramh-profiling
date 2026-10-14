#!/usr/bin/env python3
"""build_eamv3_inputs.py — drive ai2's pretrained ACE2-EAMv3 with our E3SM-SRM archive.

ACE2-EAMv3 reads 39 inputs on 8 hybrid layers. After the first step it only needs
FORCING from data, and all of that maps onto our archive:

    LANDFRAC  <- PFTDATA_MASK present (binary; the archive has no fractional coastlines)
    ICEFRAC   <- (1 - LANDFRAC) * ICE
    OCNFRAC   <- (1 - LANDFRAC) * (1 - ICE)      (E3SM: LANDFRAC + ICEFRAC + OCNFRAC = 1)
    PHIS      <- TOPO * g  (0 over ocean)
    SOLIN     <- sol_in
    TS        <- SST + 273.15 over ocean, TREFHT elsewhere (fme overwrites TS with this
                 where OCNFRAC >= 0.5 every step; the land part is only the IC)

The INITIAL STATE is an approximation, and only it:
    PS                      <- PS
    T_k, U_k, V_k           <- dp-weighted layer mean of our 18 levels over ACE layer k,
                               layer interfaces p = ak + bk*PS from the CHECKPOINT; our
                               level k sits at p = (lev_k/1000 hPa)*PS (sigma assumption:
                               the archive carries no hyam/hybm), linear in p between
                               levels, constant beyond the top/bottom level
    specific_total_water_k  <- layer mean of q + CLDLIQ + CLDICE, q from RELHUM (clipped
                               to [0,100] %) and T with a liquid/ice-blended qsat; no
                               rain water in the archive
    TS                      <- as above

Latitude: the archive stores rows SOUTH-FIRST (lat -89.5 .. 89.5; the makani converter
flips them). We keep that order; --flip writes a north-first variant so a probe can check
which orientation the checkpoint was trained on (polaris_ace2_rollout_prep.pbs).

Outputs (netCDF, cftime noleap, 6-hourly):
    <out>/ic/ic.nc                      the 8 start states (sample dim = time)
    <out>/forcing/forcing_<year>.nc     forcing from the first start frame to the end
    <out>/truth/truth_ps.nc             archive PS for the first start + 8 steps (probe)
    <out>/build_summary.json

    python build_eamv3_inputs.py --selftest
    python build_eamv3_inputs.py --ckpt <ace2_EAMv3_ckpt.tar> --out <dir> [--flip] [--short]
PASS = EAMV3_INPUTS_OK (or EAMV3_SELFTEST_OK)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

ARCHIVE = ("/eagle/lighthouse-uchicago/members/jesswan/AI4SRM/data/"
           "E3SMv3_SSP245AMIP_CTL_SST0051_REST0101/h5/plev_data")
H, W = 180, 360
STEPS_PER_YEAR = 1460                       # noleap, 6-hourly
G = 9.80616
# Same level keys, same order (TOA -> surface), as convert_e3sm_to_makani_alldata.py.
LEVEL_EXACT = [
    "4.714998332947841", "10.655023096474308", "19.235455601758737",
    "28.79458853709195", "50.11779996521295", "69.59908688413749",
    "96.46377266572703", "145.04282239200347", "200.99889546355382",
    "256.72368590525895", "302.21364012188303", "385.999023919911",
    "492.46857402252755", "608.6437744215842", "713.7046383204334",
    "849.6612491105952", "925.5197481473349", "998.4964394917621",
]
SIGMA = np.array([float(s) for s in LEVEL_EXACT]) / 1000.0   # level pressure / PS
N_LAYER = 8
N_SUB = 32          # dp-uniform samples per layer for the layer mean

# The makani 5-year protocol's start frames (CHANGELOG 2026-09-24, job 7649597).
START_YEAR = 2044
START_FRAMES = [1092 + 16 * i for i in range(8)]
END_YEAR = 2049


def frame_path(year: int, idx: int) -> str:
    return os.path.join(ARCHIVE, f"{year}_{idx:04d}.h5")


# ---- physics helpers (pure numpy; exercised by --selftest) -------------------------
def qsat(t: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Saturation specific humidity (kg/kg), liquid above 0 C, ice below -23 C, blended."""
    tc = t - 273.15
    es_liq = 611.2 * np.exp(17.67 * tc / (t - 29.65))
    es_ice = 611.2 * np.exp(22.46 * tc / (t - 0.53))
    w = np.clip((t - 250.16) / (273.15 - 250.16), 0.0, 1.0)
    es = w * es_liq + (1.0 - w) * es_ice
    es = np.minimum(es, 0.5 * p)            # guard the near-vacuum top levels
    return 0.622 * es / (p - 0.378 * es)


def layer_means(x: np.ndarray, ps: np.ndarray, ak: np.ndarray, bk: np.ndarray) -> np.ndarray:
    """dp-weighted mean of x (18, H, W) over each ACE layer -> (8, H, W).

    x is piecewise linear in sigma between the fixed knots SIGMA (sigma = p/PS), constant
    beyond them; layer k spans sigma in [ak_k/PS + bk_k, ak_{k+1}/PS + bk_{k+1}].
    Uniform-in-sigma samples within a column are uniform in p, i.e. mass-weighted.
    """
    out = np.empty((N_LAYER,) + ps.shape, dtype=np.float64)
    frac = (np.arange(N_SUB) + 0.5) / N_SUB
    for k in range(N_LAYER):
        s_top = ak[k] / ps + bk[k]
        s_bot = ak[k + 1] / ps + bk[k + 1]
        s = s_top[None] + (s_bot - s_top)[None] * frac[:, None, None]        # (N_SUB,H,W)
        s = np.clip(s, SIGMA[0], SIGMA[-1])
        j = np.clip(np.searchsorted(SIGMA, s, side="right") - 1, 0, len(SIGMA) - 2)
        w = (s - SIGMA[j]) / (SIGMA[j + 1] - SIGMA[j])
        rows, cols = np.indices(ps.shape)
        x0 = x[j, rows[None], cols[None]]
        x1 = x[j + 1, rows[None], cols[None]]
        out[k] = ((1.0 - w) * x0 + w * x1).mean(axis=0)
    return out


def selftest() -> int:
    rng = np.random.default_rng(0)
    ps = rng.uniform(55000.0, 103000.0, size=(4, 5))
    ak = np.array([0.0, 500, 2000, 5000, 9000, 12000, 8000, 3000, 0.0])
    bk = np.array([0.0, 0, 0, 0.05, 0.2, 0.4, 0.65, 0.85, 1.0])
    # 1) a constant profile has that constant as every layer mean
    c = np.full((18, 4, 5), 7.25)
    assert np.allclose(layer_means(c, ps, ak, bk), 7.25), "constant profile"
    # 2) a profile linear in sigma has the sigma-midpoint value inside the knot range
    lin = 3.0 + 2.0 * SIGMA[:, None, None] * np.ones((18, 4, 5))
    got = layer_means(lin, ps, ak, bk)
    for k in range(2, N_LAYER - 1):
        s_mid = 0.5 * (ak[k] / ps + bk[k] + ak[k + 1] / ps + bk[k + 1])
        inside = (ak[k] / ps + bk[k] >= SIGMA[0]) & (ak[k + 1] / ps + bk[k + 1] <= SIGMA[-1])
        assert np.allclose(got[k][inside], (3.0 + 2.0 * s_mid)[inside], rtol=1e-3), f"linear layer {k}"
    # 3) qsat: ~0.0107 kg/kg at 15 C / 1000 hPa (Bolton es = 17.05 hPa); monotone in T
    q15 = float(qsat(np.array(288.15), np.array(100000.0)))
    assert 0.0095 < q15 < 0.0110, f"qsat(15C) = {q15}"
    tt = np.linspace(200.0, 310.0, 200)
    assert np.all(np.diff(qsat(tt, np.full_like(tt, 80000.0))) > 0), "qsat monotone"
    # 4) orientation flip is an involution on a (C,H,W) stack
    a = rng.normal(size=(3, H, W))
    assert np.array_equal(a[:, ::-1, :][:, ::-1, :], a)
    print("EAMV3_SELFTEST_OK layer_means(const,linear) qsat(range,monotone) flip")
    return 0


# ---- archive reads ---------------------------------------------------------------
def _read_forcing_frame(args):
    import h5py
    year, idx = args
    with h5py.File(frame_path(year, idx), "r") as f:
        g = f["input"]
        return (g["ICE"][...], g["SST"][...], g["sol_in"][...], g["TREFHT"][...])


def _read_ic_frame(year: int, idx: int) -> dict:
    import h5py
    out = {}
    with h5py.File(frame_path(year, idx), "r") as f:
        g = f["input"]
        for v in ("T", "U", "V", "RELHUM", "CLDLIQ", "CLDICE"):
            out[v] = np.stack([g[f"{v}_{lev}"][...] for lev in LEVEL_EXACT]).astype(np.float64)
        for v in ("PS", "TREFHT", "SST", "ICE", "PFTDATA_MASK", "TOPO"):
            out[v] = g[v][...].astype(np.float64)
    return out


def read_vertical_coordinate(ckpt: str):
    import torch
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    vc = state["stepper"]["dataset_info"]["vertical_coordinate"]
    ak = np.asarray(vc["ak"].cpu().numpy(), dtype=np.float64)
    bk = np.asarray(vc["bk"].cpu().numpy(), dtype=np.float64)
    if ak.shape != (N_LAYER + 1,) or bk.shape != (N_LAYER + 1,):
        raise SystemExit(f"ERROR VERTICAL_COORD_SHAPE ak{ak.shape} bk{bk.shape}")
    return ak, bk


# ---- netCDF writers --------------------------------------------------------------
def _hours(year: int, idx: int) -> float:
    return ((year - START_YEAR) * STEPS_PER_YEAR + idx) * 6.0


def _new_nc(path: str, n_time: int | None):
    import netCDF4
    ds = netCDF4.Dataset(path, "w", format="NETCDF4")
    ds.createDimension("time", n_time)
    ds.createDimension("lat", H)
    ds.createDimension("lon", W)
    t = ds.createVariable("time", "f8", ("time",))
    t.units = f"hours since {START_YEAR}-01-01 00:00:00"
    t.calendar = "noleap"
    la = ds.createVariable("lat", "f8", ("lat",))
    lo = ds.createVariable("lon", "f8", ("lon",))
    la.units, lo.units = "degrees_north", "degrees_east"
    return ds, t, la, lo


def _coords(flip: bool):
    lat = np.arange(-89.5, 90.0, 1.0)
    return (lat[::-1] if flip else lat), np.arange(0.5, 360.0, 1.0)


def _orient(a: np.ndarray, flip: bool) -> np.ndarray:
    return a[..., ::-1, :] if flip else a


def static_fields(fr: dict) -> dict:
    land = np.isfinite(fr["PFTDATA_MASK"]).astype(np.float64)
    phis = np.where(np.isfinite(fr["TOPO"]), fr["TOPO"], 0.0) * G
    return {"LANDFRAC": land, "PHIS": phis}


def forcing_fields(ice, sst, solin, trefht, land) -> dict:
    ice = np.where(np.isfinite(ice), ice, 0.0)
    ocean = 1.0 - land
    ts = np.where((land < 0.5) & np.isfinite(sst), sst + 273.15, trefht)
    return {"ICEFRAC": ocean * ice, "OCNFRAC": ocean * (1.0 - ice), "SOLIN": solin, "TS": ts}


def initial_state(fr: dict, ak, bk, land) -> dict:
    ps = fr["PS"]
    p_lev = SIGMA[:, None, None] * ps[None]
    rh = np.clip(fr["RELHUM"], 0.0, 100.0) / 100.0
    q = rh * qsat(fr["T"], p_lev)
    tw = q + np.maximum(fr["CLDLIQ"], 0.0) + np.maximum(fr["CLDICE"], 0.0)
    out = {"PS": ps}
    for name, field in (("T", fr["T"]), ("U", fr["U"]), ("V", fr["V"]),
                        ("specific_total_water", tw)):
        lm = layer_means(field, ps, ak, bk)
        for k in range(N_LAYER):
            out[f"{name}_{k}"] = lm[k]
    sst = fr["SST"]
    out["TS"] = np.where((land < 0.5) & np.isfinite(sst), sst + 273.15, fr["TREFHT"])
    return out


def build(args) -> int:
    ak, bk = read_vertical_coordinate(args.ckpt)
    print(f"ak={ak.tolist()}\nbk={bk.tolist()}")
    for sub in ("ic", "forcing", "truth"):
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)
    lat, lon = _coords(args.flip)
    starts = START_FRAMES[:1] if args.short else START_FRAMES

    # -- initial states --
    frames = [_read_ic_frame(START_YEAR, i) for i in starts]
    stat = static_fields(frames[0])
    land = stat["LANDFRAC"]
    states = [initial_state(fr, ak, bk, land) for fr in frames]
    names = list(states[0].keys())
    ds, t, la, lo = _new_nc(os.path.join(args.out, "ic", "ic.nc"), len(starts))
    t[:] = [_hours(START_YEAR, i) for i in starts]
    la[:], lo[:] = lat, lon
    for n in names:
        v = ds.createVariable(n, "f4", ("time", "lat", "lon"))
        v[:] = _orient(np.stack([s[n] for s in states]).astype(np.float32), args.flip)
    ds.close()

    # -- forcing, year by year from the first start frame --
    last_year = START_YEAR if args.short else END_YEAR
    summary = {"ak": ak.tolist(), "bk": bk.tolist(), "flip": args.flip, "starts": starts,
               "forcing_files": [], "ic_names": names}
    with Pool(args.workers) as pool:
        for year in range(START_YEAR, last_year + 1):
            i0 = starts[0] if year == START_YEAR else 0
            i1 = min(i0 + 64, STEPS_PER_YEAR) if args.short else STEPS_PER_YEAR
            idxs = list(range(i0, i1))
            rows = pool.map(_read_forcing_frame, [(year, i) for i in idxs], chunksize=8)
            path = os.path.join(args.out, "forcing", f"forcing_{year}.nc")
            ds, t, la, lo = _new_nc(path, len(idxs))
            t[:] = [_hours(year, i) for i in idxs]
            la[:], lo[:] = lat, lon
            for n, a in stat.items():                         # time-invariant
                ds.createVariable(n, "f4", ("lat", "lon"))[:] = _orient(a.astype(np.float32), args.flip)
            for i in range(N_LAYER + 1):
                ds.createVariable(f"ak_{i}", "f8", ())[...] = ak[i]
                ds.createVariable(f"bk_{i}", "f8", ())[...] = bk[i]
            vars_ = {n: ds.createVariable(n, "f4", ("time", "lat", "lon"))
                     for n in ("ICEFRAC", "OCNFRAC", "SOLIN", "TS")}
            for j, (ice, sst, solin, trefht) in enumerate(rows):
                ff = forcing_fields(ice.astype(np.float64), sst.astype(np.float64),
                                    solin.astype(np.float64), trefht.astype(np.float64), land)
                for n, a in ff.items():
                    vars_[n][j] = _orient(a.astype(np.float32), args.flip)
            ds.close()
            summary["forcing_files"].append({"year": year, "frames": [i0, i1], "path": path})
            print(f"forcing {year}: frames {i0}..{i1 - 1} -> {path}", flush=True)

    # -- truth PS for the first start + 8 steps (orientation / sanity probe) --
    idxs = list(range(starts[0], starts[0] + 9))
    import h5py
    ps_truth = []
    for i in idxs:
        with h5py.File(frame_path(START_YEAR, i), "r") as f:
            ps_truth.append(f["input"]["PS"][...].astype(np.float32))
    ds, t, la, lo = _new_nc(os.path.join(args.out, "truth", "truth_ps.nc"), len(idxs))
    t[:] = [_hours(START_YEAR, i) for i in idxs]
    la[:], lo[:] = lat, lon
    ds.createVariable("PS", "f4", ("time", "lat", "lon"))[:] = _orient(np.stack(ps_truth), args.flip)
    ds.close()

    # -- sanity numbers to compare against the checkpoint's normalization means --
    wlat = np.cos(np.deg2rad(np.arange(-89.5, 90.0, 1.0)))[:, None] * np.ones((1, W))
    wlat /= wlat.sum()
    s0 = states[0]
    summary["sanity_area_mean_ic0"] = {n: float((s0[n] * wlat).sum()) for n in
                                       ("PS", "TS", "T_7", "specific_total_water_7", "U_2")}
    summary["sanity_plain_mean"] = {"LANDFRAC": float(land.mean()), "PHIS": float(stat["PHIS"].mean())}
    with open(os.path.join(args.out, "build_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    for n, a in s0.items():
        if not np.all(np.isfinite(a)):
            print(f"ERROR NONFINITE_IC {n}")
            return 1
    print(f"EAMV3_INPUTS_OK starts={len(starts)} flip={args.flip} short={args.short} out={args.out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--ckpt")
    ap.add_argument("--out")
    ap.add_argument("--flip", action="store_true", help="write north-first rows")
    ap.add_argument("--short", action="store_true", help="1 start, 64 forcing frames (probe)")
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not (a.ckpt and a.out):
        ap.error("--ckpt and --out are required unless --selftest")
    return build(a)


if __name__ == "__main__":
    sys.exit(main())
