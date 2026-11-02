"""decorrelation.py -- how fast the SQG theta field decorrelates in time, on the full grid and the subsampled views.

Question: online training samples a (theta(t), theta(t + 0.02)) pair every 0.02/100; how much new information does a
neighbouring pair carry, and does the answer depend on the grid? Measured here, read-only, from stored trajectories:

  dense   : AllTrainData/trainingdata_mXX_*_save0.0025_pair0.02/simulation/t*.nc  (frames every 0.0025, 400 per file)
            -> lags 0.0025 .. 0.96 inside one file (short lags, including below the pair step 0.02)
  archive : trainingdata/simulation/trainingdata_mXX_*_save0.02.nc                   (frames every 0.02, 25000 per file)
            -> lags 0.02 .. 20 (long lags)

For every subsampling factor m in --factors (1 = full 4096^2 grid, 4 = 1024^2, 16 = 256^2 = the UViT training view;
point subsampling theta[::m, ::m] exactly as 0-coarse_grain.py, so scales above the coarse Nyquist alias), a base frame
at t and a lag tau, with x = theta[::m, ::m] and the pair increment d(t) = x(t + 0.02) - x(t) (the residual target):
  theta_m<m>          : spatial correlation C_x(tau) = corr(x(t), x(t + tau)), overall and per octave |k| band
  incr_m<m>           : C_d(tau) = corr(d(t), d(t + tau)) -- redundancy of the TARGETS of two pairs tau apart
  theta/incr_relchange_m<m> : ||x(t + tau) - x(t)|| / ||x(t) - mean|| and the same for d
  theta/incr_rms_m<m>, *_bandvar_m<m> : rms and band variance fractions of x and d (lag 0) -- where the variance sits
Correlations exclude k = 0; bands with no power (e.g. above the 2/3 dealiasing cutoff 1365 on the full grid) are
skipped. Frames are streamed lag by lag, so memory stays ~1 GB per worker even at 4096^2.

Writes <out>/decorrelation_<tag>.csv (long format: source, lag_frames, lag_time, field, band, mean, std, n) and
<out>/decorrelation_<tag>_summary.json (per factor: e-folding and integral times, small-lag fit 1 - C = a tau^p
extrapolated to the online sampling interval 2e-4 = 0.02/100 and to 2e-3 = 0.02/10, band C(0.02) and e-fold times).

  python decorrelation.py --smoke                          # 1 dense + 1 archive window, few lags
  python decorrelation.py --workers 24 --factors 1 4 16    # full run (sbatch decorrelation_midway.sh)
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
import scipy.fft as sfft

ROOT = "/project/pedramh/weidong/SQG/SQG_deterForcing"
DENSE_DIRS = f"{ROOT}/AllTrainData/trainingdata_m*_save0.0025_pair0.02"
ARCHIVE_GLOB = f"{ROOT}/trainingdata/simulation/trainingdata_m*_save0.02.nc"
N = 4096
PAIR_T = 0.02             # training pair step
ONLINE_DT = PAIR_T / 100  # online sampling interval (STRIDE 40 x DT 5e-6)
SEED = 0

# lags in frames (dense frame = 0.0025, archive frame = 0.02); increments need frame + pair offset
DENSE_LAGS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 320, 384]
ARCHIVE_LAGS = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200, 250, 350, 500, 750, 1000]
SMOKE_DENSE_LAGS, SMOKE_ARCHIVE_LAGS = [1, 2, 8, 32], [1, 5, 50]


class Grid:
    """rfft2 weights and octave |k| band masks for an n x n periodic grid (integer wavenumbers, L = 2 pi)."""

    def __init__(self, n):
        kx = np.arange(n // 2 + 1)[None, :]
        ky = np.fft.fftfreq(n, 1.0 / n)[:, None]
        kr = np.sqrt(kx ** 2 + ky ** 2)
        self.w = np.where((kx == 0) | (kx == n // 2), 1.0, 2.0) * np.ones_like(kr)   # Hermitian weights
        edges = [1]
        while edges[-1] < n // 2:
            edges.append(edges[-1] * 2)
        edges.append(int(np.ceil(n / 2 * np.sqrt(2))) + 1)          # corner band up to the diagonal Nyquist
        self.masks = {"all": kr > 0} | {f"k{lo}-{hi}": (kr >= lo) & (kr < hi) for lo, hi in zip(edges, edges[1:])}

    def corr(self, A, B):
        """{band: corr} from rfft2 coefficients; bands without power are omitted."""
        cross, pa, pb = self.w * np.real(A * np.conj(B)), self.w * np.abs(A) ** 2, self.w * np.abs(B) ** 2
        out = {}
        for b, m in self.masks.items():
            den = np.sqrt(pa[m].sum() * pb[m].sum())
            if den > 0:
                out[b] = float(cross[m].sum() / den)
        return out

    def bandvar(self, A):
        """Fraction of the (k != 0) variance in each band."""
        p = self.w * np.abs(A) ** 2
        tot = p[self.masks["all"]].sum()
        return {b: float(p[m].sum() / tot) for b, m in self.masks.items() if b != "all" and p[m].sum() > 0}


GRIDS = {}


def rel_change(a, b):
    return float(np.linalg.norm(b - a) / np.linalg.norm(a - a.mean()))


def run_window(task):
    """One base frame: stream the needed frames lag by lag, return rows (lag_frames, field, band, value)."""
    source, path, base, lags, po, factors = task
    t0, rows, nread = time.time(), [], 0
    for m in factors:
        GRIDS.setdefault(m, Grid(N // m))
    with h5py.File(path, "r") as f:
        th = f["theta"]
        cache = {}

        def frame(j):                       # keep the 4 most recently read frames (consecutive lags share frames)
            nonlocal nread
            if j not in cache:
                cache[j] = th[base + j].astype(np.float64)
                nread += 1
                while len(cache) > 4:       # dicts keep insertion order: drop the oldest, never the new one
                    del cache[next(iter(cache))]
            return cache[j]

        x0f, xpf = frame(0), frame(po)
        ref = {}
        for m in factors:
            g, x0, xp = GRIDS[m], x0f[::m, ::m], xpf[::m, ::m]
            d0 = xp - x0
            X0, D0 = sfft.rfft2(x0), sfft.rfft2(d0)
            ref[m] = (x0.copy(), d0, X0, D0)
            rows.append((0, f"theta_rms_m{m}", "all", float((x0 - x0.mean()).std())))
            rows.append((0, f"incr_rms_m{m}", "all", float((d0 - d0.mean()).std())))
            rows += [(0, f"theta_bandvar_m{m}", b, v) for b, v in g.bandvar(X0).items()]
            rows += [(0, f"incr_bandvar_m{m}", b, v) for b, v in g.bandvar(D0).items()]
        for j in lags:
            xjf, xjpf = frame(j), frame(j + po)
            for m in factors:
                g, (x0, d0, X0, D0) = GRIDS[m], ref[m]
                xj = xjf[::m, ::m]
                dj = xjpf[::m, ::m] - xj
                rows += [(j, f"theta_m{m}", b, c) for b, c in g.corr(X0, sfft.rfft2(xj)).items()]
                rows += [(j, f"incr_m{m}", b, c) for b, c in g.corr(D0, sfft.rfft2(dj)).items()]
                rows.append((j, f"theta_relchange_m{m}", "all", rel_change(x0, xj)))
                rows.append((j, f"incr_relchange_m{m}", "all", rel_change(d0, dj)))
    return source, path, base, rows, nread, time.time() - t0


def member_of(path):
    return int(re.search(r"_m(\d\d)_", path).group(1))


def dense_tasks(lags, files_per_member, factors):
    tasks = []
    for d in sorted(glob.glob(DENSE_DIRS)):
        files = sorted(glob.glob(f"{d}/simulation/t*.nc"))[:-2]   # the newest files may still be written
        if not files:
            continue
        pick = np.unique(np.linspace(0, len(files) - 1, files_per_member).round().astype(int))
        for i in pick:
            with h5py.File(files[i], "r") as f:
                if int(np.ravel(f.attrs.get("complete", [0]))[0]) == 1 and f["theta"].shape[0] == 400:
                    tasks.append(("dense", files[i], 0, lags, 8, factors))   # pair offset 8 frames = 0.02
    return tasks


def archive_tasks(lags, windows_per_member, rng, factors):
    tasks = []
    for p in sorted(glob.glob(ARCHIVE_GLOB)):
        with h5py.File(p, "r") as f:
            nrec = int(np.ravel(f.attrs["valid_records"])[0])
        bases = np.sort(rng.choice(nrec - max(lags) - 1, windows_per_member, replace=False))
        tasks += [("archive", p, int(b), lags, 1, factors) for b in bases]   # pair offset 1 frame = 0.02
    return tasks


def efold_time(t, c):
    """First tau with C < 1/e (log-linear interpolation from C(0) = 1), None if never reached."""
    t, c = np.concatenate([[0.0], t]), np.concatenate([[1.0], c])
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
    ap.add_argument("--factors", type=int, nargs="+", default=[1, 4, 16], help="subsampling factors (1 = 4096^2)")
    ap.add_argument("--dense-files", type=int, default=12, help="dense files (base frames) per member")
    ap.add_argument("--archive-windows", type=int, default=8, help="archive base frames per member")
    ap.add_argument("--tag", default=None, help="output tag (default: smoke / multigrid)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
    args = ap.parse_args()
    tag = args.tag or ("smoke" if args.smoke else "multigrid")
    rng = np.random.default_rng(SEED)
    fac = sorted(set(args.factors))

    if args.smoke:
        tasks = dense_tasks(SMOKE_DENSE_LAGS, 1, fac)[:1] + archive_tasks(SMOKE_ARCHIVE_LAGS, 1, rng, fac)[:1]
    else:
        tasks = dense_tasks(DENSE_LAGS, args.dense_files, fac) + archive_tasks(ARCHIVE_LAGS, args.archive_windows, rng, fac)
    nd = sum(t[0] == "dense" for t in tasks)
    print(f"decorrelation [{tag}]: factors {fac}, {nd} dense + {len(tasks) - nd} archive windows, "
          f"{args.workers} workers", flush=True)

    vals = {}                                     # (source, lag_frames, field, band) -> list of values
    members = {"dense": set(), "archive": set()}
    t0 = time.time()
    with mp.get_context("fork").Pool(args.workers) as pool:
        for k, (source, path, base, rows, nread, dt) in enumerate(pool.imap_unordered(run_window, tasks), 1):
            members[source].add(member_of(path))
            for j, field, band, v in rows:
                vals.setdefault((source, j, field, band), []).append(v)
            print(f"  [{k}/{len(tasks)}] {source} m{member_of(path):02d} {os.path.basename(path)[:24]} base {base}: "
                  f"{nread} frames in {dt:.0f} s", flush=True)

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

    def merged(field, band="all"):
        """dense below the archive's first lag (0.02), archive from there on (more windows, longer reach)."""
        td, cd = curve("dense", field, band)
        ta, ca = curve("archive", field, band)
        keep = td < PAIR_T - 1e-12
        t, c = np.concatenate([td[keep], ta]), np.concatenate([cd[keep], ca])
        o = np.argsort(t)
        return t[o], c[o], td, cd

    def mean0(field, band="all"):
        v = vals.get(("archive", 0, field, band)) or vals.get(("dense", 0, field, band))
        return float(np.mean(v)) if v else None

    summary = {"members": {k: sorted(v) for k, v in members.items()}, "online_dt": ONLINE_DT, "pair_t": PAIR_T,
               "factors": fac}
    for m in fac:
        bands = sorted({b for (_, _, f, b) in vals if f == f"theta_m{m}" and b != "all"},
                       key=lambda b: int(b[1:].split("-")[0]))
        sm = {"grid": N // m, "theta_rms": mean0(f"theta_rms_m{m}"), "incr_rms": mean0(f"incr_rms_m{m}")}
        for kind in ("theta", "incr"):
            field = f"{kind}_m{m}"
            t, c, td, cd = merged(field)
            tint, crossed = integral_time(t, c) if len(t) else (None, False)
            s = {"efold_time": efold_time(t, c) if len(t) else None, "integral_time": tint, "zero_crossed": crossed,
                 "small_lag_fit": small_lag_fit(td, cd, 0.0101) if len(td) else None}
            for tau in (PAIR_T / 8, PAIR_T, 0.1, 1.0):
                hit = np.where(np.isclose(t, tau))[0]
                if len(hit):
                    s[f"C_at_{tau:g}"] = float(c[hit[0]])
            s["bandvar"] = {b: mean0(f"{kind}_bandvar_m{m}", b) for b in bands}
            s["bands_C_at_0.0025"] = {b: float(np.mean(vals[("dense", 1, field, b)])) for b in bands
                                      if ("dense", 1, field, b) in vals}
            s["bands_C_at_0.02"] = {b: float(np.mean(vals[("archive", 1, field, b)])) for b in bands
                                    if ("archive", 1, field, b) in vals}
            s["bands_efold_time"] = {}
            for b in bands:
                tb, cb, _, _ = merged(field, b)
                s["bands_efold_time"][b] = efold_time(tb, cb) if len(tb) else None
            sm[kind] = s
        summary[f"m{m}"] = sm
    with open(os.path.join(args.out, f"decorrelation_{tag}_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)

    nan = float("nan")
    for m in fac:
        for kind in ("theta", "incr"):
            s = summary[f"m{m}"][kind]
            fit = s["small_lag_fit"] or {}
            print(f"m{m:<2d} ({N // m}^2) {kind:5s}: C(0.0025)={s.get('C_at_0.0025', nan):.5f} "
                  f"C(0.02)={s.get('C_at_0.02', nan):.5f} e-fold={s['efold_time']} "
                  f"fit p={fit.get('p', nan):.2f} 1-C(2e-4)={fit.get('one_minus_C_at_2e-4', nan):.2e}")
    print(f"DECORR_OK {tag}: {len(tasks)} windows in {time.time() - t0:.0f} s -> {csv_path}")


if __name__ == "__main__":
    main()
