"""tr_pairs.py -- gates A and B's data for tilerows_if25 (PREDICTIONS_gates_tilerows.md, written first): the same 16
archived states x 4 consecutive 0.02 pairs as the taxonomy job 7824121 (taxonomy.py), with the new arms run from
every pair start. One A100 per process; run_tr_ab_polaris.pbs starts 4, then gates_ab_tr.py.

  "$PY" tr_pairs.py --out <dir> --worker <w> --ckpt <member>:<npz> [--ckpt ...] --taxo <taxonomy_7824121 dir>
        --warp-path <dir> --nvmath-path <dir>

Per state (loaded as o1.2 does, taxonomy.py L114-124) and pair j, S_{j+1} = lean_u1's state 4,000 steps after S_j:
  lean_u1        regenerated (bitwise production); its sub / af frames at every lag are checked bit for bit against
                 job 7824121's stored pair (validity: the new arms start from exactly the stored pairs' states).
  tilerows_if25  o1.2's per-frame advance from o12run/tilerows_advance.py (2 float64 IF-AB3 switching steps, then
                 25-step frames of the tile-row step at h = 8e-6): the candidate.
  if32_25        the pure-JAX float32 IF-AB3 at the same h (reported).
  ref160         first pair of each state only: float64 IF-AB3 at 160 steps per frame (h = 1.25e-6, the time-step
                 probe's reference), 0.02 lag only (information: who is closer to the converged solution).
Frames as taxonomy.py: sub = float32(irfft2(theta_hat)[::16, ::16]) / 8 (256^2) and af = the alias-free coarse spectrum,
at LAGS (frames); tilerows_if25 / if32_25 at every lag, ref160 at the last. One npz per pair, named as the taxonomy
job's (pair_w<w>_s<si>_p<j>.npz) so gates_ab_tr.py can merge them. Prints TRAB_WORKER_DONE last.
"""
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import argparse                                                      # noqa: E402
import hashlib                                                       # noqa: E402
import json                                                          # noqa: E402
import math                                                          # noqa: E402
import site                                                          # noqa: E402
import sys                                                           # noqa: E402
import time                                                          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OPT = os.path.dirname(HERE)
ANALYSIS = os.path.dirname(OPT)
for p in (os.path.join(OPT, "seed1"), os.path.join(OPT, "delta"), os.path.join(OPT, "taxonomy"),
          os.path.join(OPT, "timestep"), os.path.join(OPT, "tilerows"), os.path.join(OPT, "o12run"), ANALYSIS):
    sys.path.insert(0, p)

THETA_SCALE = 8.0
NC = 256
HALF = NC // 2
LAGS = (1, 10, 25, 50, 100)                                          # frames; o1.2's pair = 100 (taxonomy.py)
REF_NSUB = 160


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, float):
        return x if math.isfinite(x) else repr(x)
    return x


def main():
    t_wall = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--worker", type=int, required=True)
    ap.add_argument("--ckpt", action="append", required=True)
    ap.add_argument("--taxo", required=True, help="job 7824121's raw dir (stored lean_u1 pairs)")
    ap.add_argument("--pairs", type=int, default=4)
    ap.add_argument("--warp-path", required=True)
    ap.add_argument("--nvmath-path", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    W = args.worker

    sys.path.insert(0, args.warp_path)
    site.addsitedir(args.nvmath_path)
    import warp as wp
    wp.config.enable_backward = False
    wp.init()
    import numpy as np
    import profile_solver as PS
    GD = PS.import_asqg()
    import jax
    import jax.numpy as jnp
    import lean_solver as LS
    import if_ab3 as IF
    import tilerows_advance as TA
    block = jax.block_until_ready
    N, DT, STRIDE, M = GD.N, GD.DT, PS.STRIDE, PS.M
    template = GD.sqg.init(GD.p, jax.random.key(GD.SEED_INITIAL_CONDITION))
    res = {"worker": W, "device_kind": jax.devices()[0].device_kind, "jax": jax.__version__, "lags_frames": LAGS,
           "ref_nsub": REF_NSUB, "states": [], "bitwise_vs_7824121": [], "npz": [], "errors": []}

    def config_hash(member):
        cfg = dict(GD.CONFIG, member=member, seed_initial_condition=member)
        return hashlib.sha256(json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def load(path, member):
        own, GD.CONFIG_HASH = GD.CONFIG_HASH, config_hash(member)
        try:
            return GD.load_checkpoint(template, path)
        finally:
            GD.CONFIG_HASH = own

    _, ncol = LS.preconditions(GD)
    cu1 = LS.build_consts(GD, ncol)
    u1_call = LS.make_advance(GD, STRIDE, ncol=ncol, unroll1=True)
    adv = {"lean_u1": lambda st: u1_call(st, cu1),
           "tilerows_if25": TA.make(GD, STRIDE, "tilerows_if25", rank=W, log=print, gate=False),
           "if32_25": TA.make(GD, STRIDE, "if32_25", rank=W, log=print, gate=False)}
    ladder = IF.Ladder(GD, ncol, cu1)

    def af_rows(a):
        return jnp.concatenate([a[:HALF, :HALF + 1], a[N - HALF:, :HALF + 1]], 0)

    coarse_sub = jax.jit(lambda st: jnp.fft.irfft2(st.state.qh[0], s=(N, N))[::M, ::M].astype(jnp.float32))
    af_scale = (NC * NC) / (N * N) / THETA_SCALE

    @jax.jit
    def coarse_af(st):
        c = af_rows(st.state.qh[0]) * af_scale
        return c.at[HALF, :].set(0).at[:, HALF].set(0)

    diag = jax.jit(GD.diagnostics)

    def spe_cfl(st):
        values, _, umax = jax.device_get(diag(st))
        return float(values[0]), float(umax) * GD.KMAX * DT

    def frames_of(st):
        return (np.asarray(coarse_sub(st), dtype=np.float32) / np.float32(THETA_SCALE), np.asarray(coarse_af(st)))

    def run_frames(f, s0):
        """sub / af at lag 0 and LAGS of 100 calls of f from s0, and the end state."""
        sub, afd = {}, {}
        sub[0], afd[0] = frames_of(s0)
        s = s0
        for i in range(1, LAGS[-1] + 1):
            s = f(s)
            if i in LAGS:
                sub[i], afd[i] = frames_of(s)
        return sub, afd, block(s)

    for si, spec in enumerate(args.ckpt):
        m, path = spec.split(":", 1)
        member = int(m)
        t0 = time.time()
        st, completed = load(path, member)
        st = block(st)
        if int(st._ablevel) != 2:
            raise SystemExit(f"ERROR TRAB_NOT_LEVEL2 {path}")
        spe, cfl = spe_cfl(st)
        info = {"member": member, "path": path, "t": float(GD.block_time(completed)), "spe": spe, "cfl_DT": cfl,
                "pairs": []}
        print(f"TRAB_STATE w={W} s={si} member={member} t={info['t']:.2f} SPE={spe:.6g} CFL_DT={cfl:.3f}", flush=True)
        s = st
        for j in range(args.pairs):
            tj = time.time()
            name = f"pair_w{W}_s{si}_p{j}.npz"
            sub_r, af_r, s_next = run_frames(adv["lean_u1"], s)
            stored = os.path.join(args.taxo, name)
            with np.load(stored) as z:
                bit = all(np.array_equal(sub_r[k].view(np.uint32), z[f"lean_u1__sub__{k}"].view(np.uint32))
                          and np.array_equal(af_r[k].view(np.uint64), z[f"lean_u1__af__{k}"].view(np.uint64))
                          for k in (0,) + LAGS)
            res["bitwise_vs_7824121"].append({"pair": name, "equal": bool(bit)})
            dump, secs = {}, {}
            for arm in ("tilerows_if25", "if32_25"):
                ta = time.time()
                sub_v, af_v, _ = run_frames(adv[arm], s)
                secs[arm] = round(time.time() - ta, 2)
                for k in LAGS:
                    dump[f"{arm}__sub__{k}"] = sub_v[k]
                    dump[f"{arm}__af__{k}"] = af_v[k]
            if j == 0:
                ta = time.time()
                end = block(ladder.run(s, "if", REF_NSUB))
                dump[f"ref160__sub__{LAGS[-1]}"], dump[f"ref160__af__{LAGS[-1]}"] = frames_of(end)
                secs["ref160"] = round(time.time() - ta, 2)
            np.savez(os.path.join(args.out, name), **dump)
            res["npz"].append(name)
            spe_j, cfl_j = spe_cfl(s)
            pj = {"j": j, "spe": spe_j, "cfl_DT": cfl_j, "lean_u1_bitwise_vs_7824121": bool(bit), "seconds": secs,
                  "pair_seconds": round(time.time() - tj, 1)}
            info["pairs"].append(pj)
            print(f"TRAB_PAIR w={W} s={si} p={j} lean_u1==7824121 {bit} arms_s={secs} ({pj['pair_seconds']:.0f} s)",
                  flush=True)
            s = s_next
        info["seconds"] = round(time.time() - t0, 1)
        res["states"].append(info)
        with open(os.path.join(args.out, f"worker_{W}.json"), "w") as fh:
            json.dump(jsonable(res), fh, indent=1)
    ok = all(b["equal"] for b in res["bitwise_vs_7824121"])
    res["wall_s"] = round(time.time() - t_wall, 1)
    with open(os.path.join(args.out, f"worker_{W}.json"), "w") as fh:
        json.dump(jsonable(res), fh, indent=1)
    print(f"TRAB_WORKER_DONE w={W} pairs={len(res['npz'])} lean_u1_bitwise_all={ok} wall={res['wall_s']:.0f}s",
          flush=True)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except BaseException as exc:
        first = str(exc).strip().splitlines()[0][:200] if str(exc).strip() else ""
        print(f"ERROR TRAB_CRASHED {type(exc).__name__}: {first}", flush=True)
        raise
