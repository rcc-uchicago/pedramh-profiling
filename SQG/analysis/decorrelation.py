"""decorrelation.py -- how fast the SQG theta field (as the m=16 UViT sees it) decorrelates in time.

Question: online training samples a (theta(t), theta(t + 0.02)) pair every 0.02/100; how much new information does a
neighbouring pair carry? Measured here, read-only, from the stored trajectories:

  dense   : AllTrainData/trainingdata_mXX_*_save0.0025_pair0.02/simulation/t*.nc  (frames every 0.0025, 400 per file)
            -> lags 0.0025 .. 0.96 inside one file (short lags, including below the pair step 0.02)
  archive : trainingdata/simulation/trainingdata_mXX_*_save0.02.nc                   (frames every 0.02, 25000 per file)
            -> lags 0.02 .. 20 (long lags)

For a base frame at t and a lag tau, with x = theta[::16, ::16] (256^2, point subsampling exactly as the training data)
and the pair increment d(t) = x(t + 0.02) - x(t) (what the residual UViT predicts):
  theta     : spatial correlation C_x(tau) = corr(x(t), x(t + tau)), overall and per |k| band (256^2 grid)
  incr      : C_d(tau) = corr(d(t), d(t + tau)) -- redundancy of the TARGETS of two pairs tau apart
  full      : corr of the full 4096^2 theta (overall only), for reference
  relchange : ||x(t + tau) - x(t)|| / ||x(t) - mean||   and the same for d
Means are spatial means over the 2pi-periodic domain; correlations exclude the k = 0 mode. Averaged over windows
(several files / base times per member, several members).

Writes <out>/decorrelation_<tag>.csv (long format: source, lag_frames, lag_time, field, band, mean, std, n) and
<out>/decorrelation_<tag>_summary.json (e-folding and integral times, small-lag fit 1 - C = a tau^p extrapolated to
the online sampling interval 2e-4 = 0.02/100 and to 2e-3 = 0.02/10).

  python decorrelation.py --smoke          # 1 dense + 1 archive window, few lags
  python decorrelation.py --workers 24     # full run (sbatch decorrelation_midway.sh)
"""
import os

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")   # files may be open by the writers
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import csv
import glob
import json
import multiprocessing as mp
import re
import time

import h5py
import numpy as np

ROOT = "/project/pedramh/weidong/SQG/SQG_deterForcing"
DENSE_DIRS = f"{ROOT}/AllTrainData/trainingdata_m*_save0.0025_pair0.02"
ARCHIVE_GLOB = f"{ROOT}/trainingdata/simulation/trainingdata_m*_save0.02.nc"
M = 16                    # subsampling of the training data (theta[::16, ::16])
N = 4096
PAIR_T = 0.02             # training pair step
ONLINE_DT = PAIR_T / 100  # online sampling interval (STRIDE 40 x DT 5e-6)
SEED = 0

# lags in frames (dense frame = 0.0025, archive frame = 0.02); increments need frame + pair offset
DENSE_LAGS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 320, 384]
ARCHIVE_LAGS = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200, 250, 350, 500, 750, 1000]
SMOKE_DENSE_LAGS, SMOKE_ARCHIVE_LAGS = [1, 2, 8, 32], [1, 5, 50]

# |k| bands on the 256^2 grid (integer wavenumbers, L = 2 pi); top band reaches the coarse Nyquist 128 (aliased)
BANDS = [(1, 2), (2, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 182)]
n = N // M
_kx = np.arange(n // 2 + 1)[None, :]
_ky = np.fft.fftfreq(n, 1.0 / n)[:, None]
KR = np.sqrt(_kx ** 2 + _ky ** 2)
W = np.where((_kx == 0) | (_kx == n // 2), 1.0, 2.0) * np.ones_like(KR)   # rfft2 Hermitian weights
MASKS = {"all": KR > 0} | {f"k{lo}-{hi}": (KR >= lo) & (KR < hi) for lo, hi in BANDS}


def spec_corr(A, B):
    """Correlation per band from rfft2 coefficients (k = 0 excluded): {band: corr}."""
    cross, pa, pb = W * np.real(A * np.conj(B)), W * np.abs(A) ** 2, W * np.abs(B) ** 2
    return {b: float(cross[m].sum() / np.sqrt(pa[m].sum() * pb[m].sum())) for b, m in MASKS.items()}


def rel_change(a, b):
    a0 = a - a.mean()
    return float(np.linalg.norm(b - a) / np.linalg.norm(a0))


def full_corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float(np.dot(a.ravel(), b.ravel()) / np.sqrt(np.dot(a.ravel(), a.ravel()) * np.dot(b.ravel(), b.ravel())))


def run_window(task):
    """One base frame: read the needed frames, return rows (lag_frames, field, band, value)."""
    source, path, base, lags, pair_off = task
    need = sorted({0, pair_off} | {j for j in lags} | {j + pair_off for j in lags})
    t0 = time.time()
    x, full0, rows = {}, None, []
    with h5py.File(path, "r") as f:
        th = f["theta"]
        for j in need:
            fr = th[base + j].astype(np.float64)
            x[j] = fr[::M, ::M].copy()
            if j == 0:
                full0 = fr
            elif j in lags:
                rows.append((j, "full", "all", full_corr(full0, fr)))
            del fr
    X = {j: np.fft.rfft2(v) for j, v in x.items()}
    d = {j: x[j + pair_off] - x[j] for j in [0] + list(lags)}
    D = {j: np.fft.rfft2(v) for j, v in d.items()}
    rows.append((0, "theta_rms", "all", float((x[0] - x[0].mean()).std())))
    rows.append((0, "incr_rms", "all", float((d[0] - d[0].mean()).std())))
    for j in lags:
        for b, c in spec_corr(X[0], X[j]).items():
            rows.append((j, "theta", b, c))
        for b, c in spec_corr(D[0], D[j]).items():
            rows.append((j, "incr", b, c))
        rows.append((j, "theta_relchange", "all", rel_change(x[0], x[j])))
        rows.append((j, "incr_relchange", "all", rel_change(d[0], d[j])))
    return source, path, base, rows, len(need), time.time() - t0


def member_of(path):
    return int(re.search(r"_m(\d\d)_", path).group(1))


def dense_tasks(lags, files_per_member, rng):
    tasks = []
    for d in sorted(glob.glob(DENSE_DIRS)):
        files = sorted(glob.glob(f"{d}/simulation/t*.nc"))[:-2]   # the newest files may still be written
        ok = []
        for p in files:
            with h5py.File(p, "r") as f:
                if int(np.ravel(f.attrs.get("complete", [0]))[0]) == 1 and f["theta"].shape[0] == 400:
                    ok.append(p)
        if not ok:
            continue
        pick = np.unique(np.linspace(0, len(ok) - 1, files_per_member).round().astype(int))
        tasks += [("dense", ok[i], 0, lags, 8) for i in pick]     # pair offset 8 frames = 0.02
    return tasks


def archive_tasks(lags, windows_per_member, rng):
    tasks = []
    for p in sorted(glob.glob(ARCHIVE_GLOB)):
        with h5py.File(p, "r") as f:
            nrec = int(np.ravel(f.attrs["valid_records"])[0])
        span = max(lags) + 1
        bases = np.sort(rng.choice(nrec - span, windows_per_member, replace=False))
        tasks += [("archive", p, int(b), lags, 1) for b in bases]  # pair offset 1 frame = 0.02
    return tasks


def efold_time(t, c):
    """First tau with C < 1/e (log-linear interpolation), None if never reached."""
    for i in range(1, len(t)):
        if c[i] < np.exp(-1) <= c[i - 1]:
            l0, l1 = np.log(max(c[i - 1], 1e-12)), np.log(max(c[i], 1e-12))
            return float(t[i - 1] + (t[i] - t[i - 1]) * (-1 - l0) / (l1 - l0))
    return None


def integral_time(t, c):
    """int_0^{first zero} C dtau (trapezoid, C(0) = 1); also whether a zero crossing was reached."""
    t, c = np.concatenate([[0.0], t]), np.concatenate([[1.0], c])
    z = np.where(c <= 0)[0]
    stop = z[0] + 1 if len(z) else len(c)
    return float(np.trapz(c[:stop], t[:stop])), bool(len(z))


def small_lag_fit(t, c, tmax):
    """Fit 1 - C = a tau^p on tau <= tmax; returns a, p and 1 - C extrapolated to ONLINE_DT and 10 ONLINE_DT."""
    sel = (t <= tmax + 1e-12) & (c < 1)
    if sel.sum() < 2:
        return None
    p, loga = np.polyfit(np.log(t[sel]), np.log(1 - c[sel]), 1)
    f = lambda tau: float(np.exp(loga) * tau ** p)
    return {"a": float(np.exp(loga)), "p": float(p), "fit_lags": t[sel].tolist(),
            "one_minus_C_at_2e-4": f(ONLINE_DT), "one_minus_C_at_2e-3": f(10 * ONLINE_DT)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dense-files", type=int, default=6, help="dense files (base frames) per member")
    ap.add_argument("--archive-windows", type=int, default=4, help="archive base frames per member")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
    args = ap.parse_args()
    tag = "smoke" if args.smoke else "full"
    rng = np.random.default_rng(SEED)

    if args.smoke:
        tasks = dense_tasks(SMOKE_DENSE_LAGS, 1, rng)[:1] + archive_tasks(SMOKE_ARCHIVE_LAGS, 1, rng)[:1]
    else:
        tasks = dense_tasks(DENSE_LAGS, args.dense_files, rng) + archive_tasks(ARCHIVE_LAGS, args.archive_windows, rng)
    nd = sum(t[0] == "dense" for t in tasks)
    print(f"decorrelation [{tag}]: {nd} dense + {len(tasks) - nd} archive windows, {args.workers} workers", flush=True)

    vals = {}                                     # (source, lag_frames, field, band) -> list of values
    members = {"dense": set(), "archive": set()}
    t0 = time.time()
    with mp.get_context("fork").Pool(args.workers) as pool:
        for k, (source, path, base, rows, nframes, dt) in enumerate(pool.imap_unordered(run_window, tasks), 1):
            members[source].add(member_of(path))
            for j, field, band, v in rows:
                vals.setdefault((source, j, field, band), []).append(v)
            print(f"  [{k}/{len(tasks)}] {source} m{member_of(path):02d} {os.path.basename(path)[:24]} base {base}: "
                  f"{nframes} frames in {dt:.0f} s", flush=True)

    frame_t = {"dense": PAIR_T / 8, "archive": PAIR_T}
    os.makedirs(args.out, exist_ok=True)
    csv_path = os.path.join(args.out, f"decorrelation_{tag}.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["source", "lag_frames", "lag_time", "field", "band", "mean", "std", "n"])
        for (s, j, field, band), v in sorted(vals.items()):
            w.writerow([s, j, f"{j * frame_t[s]:.6g}", field, band, f"{np.mean(v):.8g}", f"{np.std(v):.3g}", len(v)])

    def curve(source, field, band="all"):
        js = sorted(j for (s, j, f, b) in vals if s == source and f == field and b == band and j > 0)
        return np.array([j * frame_t[source] for j in js]), np.array([np.mean(vals[(source, j, field, band)]) for j in js])

    summary = {"members": {k: sorted(v) for k, v in members.items()}, "online_dt": ONLINE_DT, "pair_t": PAIR_T,
               "theta_rms_m16": float(np.mean(vals.get(("dense", 0, "theta_rms", "all"), [np.nan]))),
               "incr_rms_m16": float(np.mean(vals.get(("dense", 0, "incr_rms", "all"), [np.nan])))}
    for field in ("theta", "incr", "full"):
        td, cd = curve("dense", field)
        ta, ca = curve("archive", field)
        # merged curve: dense below the archive's first lag, archive from there on (more windows, longer reach)
        t = np.concatenate([td[td < PAIR_T - 1e-12], ta]) if len(ta) else td
        c = np.concatenate([cd[td < PAIR_T - 1e-12], ca]) if len(ta) else cd
        order = np.argsort(t)
        t, c = t[order], c[order]
        tint, crossed = integral_time(t, c) if len(t) else (None, False)
        s = {"efold_time": efold_time(t, c) if len(t) else None, "integral_time": tint, "zero_crossed": crossed,
             "small_lag_fit": small_lag_fit(td, cd, 0.0101) if len(td) else None}
        for tau in (PAIR_T / 8, PAIR_T, 0.1, 1.0):
            hit = np.where(np.isclose(t, tau))[0]
            if len(hit):
                s[f"C_at_{tau:g}"] = float(c[hit[0]])
        if field in ("theta", "incr"):
            s["bands_C_at_0.02"] = {b: float(np.mean(vals[("archive", 1, field, b)])) for b in MASKS
                                    if ("archive", 1, field, b) in vals}
            s["bands_efold_time"] = {}
            for b in MASKS:
                tb_d, cb_d = curve("dense", field, b)
                tb_a, cb_a = curve("archive", field, b)
                tb = np.concatenate([tb_d[tb_d < PAIR_T - 1e-12], tb_a]) if len(tb_a) else tb_d
                cb = np.concatenate([cb_d[tb_d < PAIR_T - 1e-12], cb_a]) if len(tb_a) else cb_d
                o = np.argsort(tb)
                s["bands_efold_time"][b] = efold_time(tb[o], cb[o]) if len(tb) else None
        summary[field] = s
    with open(os.path.join(args.out, f"decorrelation_{tag}_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)

    th, inc = summary["theta"], summary["incr"]
    fit_t, fit_i = th["small_lag_fit"] or {}, inc["small_lag_fit"] or {}
    print(f"theta_m16: C(0.0025)={th.get('C_at_0.0025', float('nan')):.6f} C(0.02)={th.get('C_at_0.02', float('nan')):.5f} "
          f"e-fold={th['efold_time']} integral={th['integral_time']}")
    print(f"incr_m16 : C(0.0025)={inc.get('C_at_0.0025', float('nan')):.6f} C(0.02)={inc.get('C_at_0.02', float('nan')):.5f} "
          f"e-fold={inc['efold_time']} integral={inc['integral_time']}")
    print(f"small-lag fit 1-C=a*tau^p: theta p={fit_t.get('p')} 1-C(2e-4)={fit_t.get('one_minus_C_at_2e-4')}; "
          f"incr p={fit_i.get('p')} 1-C(2e-4)={fit_i.get('one_minus_C_at_2e-4')}")
    print(f"DECORR_OK {tag}: {len(tasks)} windows in {time.time() - t0:.0f} s -> {csv_path}")


if __name__ == "__main__":
    main()
