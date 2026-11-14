"""hybrid_probe.py -- the hybrid solver step (hybrid_step.py) against the JAX steps (PREDICTIONS_hybrid.md, written first).

  "$PY" hybrid_probe.py --out <dir> --ckpt <member>:<npz> --warp-path <dir> --nvmath-path <dir> [--deadline <epoch s>]

Run with cuFFT >= 11.3 preloaded (run_hybrid_polaris.pbs: CUDA 12.9.1's libcufft.so.11 = 11.4.1; LTO callbacks need it),
so every arm in this process, JAX's included, uses the same cuFFT.
  H0 env      versions; the cuFFT / nvJitLink actually mapped; cuFFT version >= 11300.
  H1 micro    a plain nvmath C2C 4096^2 (no callbacks) vs torch.fft.fft2, complex64 / complex128.
  H2 gates    from one archived state, n = 1, 40, 4,000 steps:
                hybrid32     vs lean_u1, next to lean32ek_fast vs lean_u1     (G32)
                hybrid32_if  vs if64 (if_ab3 'if' at DT), next to if32 vs if64 (GIF)
                hybrid64     vs lean_u1                                        (G64)
              rel = |theta_a - theta_b| / |theta_b| over the full complex128 state; at 4,000 steps also the band
              errors of the 0.02 delta on o1.2's sub and af frames (taxonomy bands), as timestep_probe.py.
  H3 timing   ms/step: hybrid32 / hybrid32_if / hybrid64 eager (400 steps) and hybrid32 replayed from a CUDA graph
              (48 captured steps); the per-op split of hybrid32 (CUDA events); the JAX steps lean_u1, lean32ek_fast,
              if32 (30 calls x 40 steps), same GPU.
Gates (pre-set): G32: rel(hybrid32, lean_u1) <= min(1e-4, 2 rel(lean32ek_fast, lean_u1)) at n = 1 and 40;
GIF: the same with if64 / if32; G64: rel(hybrid64, lean_u1) <= 1e-12 at n = 1. Tokens: HYB_ENV_OK, HYB_GATE32,
HYB_GATEIF, HYB_GATE64, HYB_BENCH, HYB_PROBE_DONE failures=N.
"""
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import argparse                                                      # noqa: E402
import hashlib                                                       # noqa: E402
import json                                                          # noqa: E402
import math                                                          # noqa: E402
import site                                                          # noqa: E402
import statistics                                                    # noqa: E402
import sys                                                           # noqa: E402
import time                                                          # noqa: E402
import traceback                                                     # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OPT = os.path.dirname(HERE)
ANALYSIS = os.path.dirname(OPT)
for p in (os.path.join(OPT, "seed1"), os.path.join(OPT, "delta"), os.path.join(OPT, "taxonomy"),
          os.path.join(OPT, "timestep"), HERE, ANALYSIS):
    sys.path.insert(0, p)

THETA_SCALE = 8.0
NC, HALF = 256, 128
BANDS = ((1, 2), (2, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 128), (128, 182))


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, float):
        return x if math.isfinite(x) else repr(x)
    return x


def mapped(substr):
    try:
        with open("/proc/self/maps") as f:
            return sorted({ln.split()[-1] for ln in f if substr in ln and "/" in ln})
    except OSError:
        return []


def main():
    t_wall = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--warp-path", required=True)
    ap.add_argument("--nvmath-path", required=True)
    ap.add_argument("--deadline", type=float, default=0.0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    R = {"errors": []}
    failures = 0

    def save():
        with open(os.path.join(args.out, "hybrid_probe.json"), "w") as fh:
            json.dump(jsonable(R), fh, indent=1, default=str)

    def err(where, e):
        R["errors"].append({"where": where, "error": f"{type(e).__name__}: {e}"[:2000], "tb": traceback.format_exc()[-2500:]})
        print(f"ERROR HYB_{where}: {type(e).__name__}: {str(e)[:300]}", flush=True)
        save()

    def late(margin=300):
        return bool(args.deadline) and time.time() > args.deadline - margin

    # ---- H0 env (torch first, as o1.2) ----------------------------------------------------------------------------
    import numpy as np
    import torch
    import profile_solver as PS
    GD = PS.import_asqg()                                           # jax + the A-SQG module (x64 on)
    import jax
    import jax.numpy as jnp
    from pyqg_jax import steppers as pst
    sys.path.insert(0, args.warp_path)
    site.addsitedir(args.nvmath_path)                                # processes the numba-cuda / cuda-bindings .pth
    import warp as wp
    wp.config.enable_backward = False
    wp.init()
    import numba
    import nvmath
    from nvmath.bindings import cufft as cb
    import hybrid_step as HS
    import lean_solver as LS
    import lean32_precise as LP
    import lean32_fast as LF
    import if_ab3 as IF
    block = jax.block_until_ready
    N, NK, DT, STRIDE, M = GD.N, GD.N // 2 + 1, GD.DT, PS.STRIDE, PS.M
    assert N == HS.N and STRIDE == 40 and N // M == NC
    env = {"torch": torch.__version__, "jax": jax.__version__, "warp": wp.__version__, "numba": numba.__version__,
           "nvmath": nvmath.__version__, "cufft_version": int(cb.get_version()), "libcufft": mapped("libcufft"),
           "libnvJitLink": mapped("libnvJitLink"), "libcudart": mapped("libcudart"), "gpu": torch.cuda.get_device_name(0),
           "ld_preload": os.environ.get("LD_PRELOAD", "")}
    R["env"] = env
    print("env " + json.dumps(env), flush=True)
    if env["cufft_version"] < 11300:
        print(f"ERROR HYB_CUFFT_TOO_OLD {env['cufft_version']} (LTO callbacks need >= 11300)", flush=True)
        R["failures"] = 1
        save()
        print("HYB_PROBE_DONE failures=1", flush=True)
        return
    print("HYB_ENV_OK", flush=True)
    save()

    def ev_time(stream, fn, reps, batches=5, warm=2):
        for _ in range(warm):
            fn()
        stream.synchronize()
        ts = []
        for _ in range(batches):
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record(stream)
            for _ in range(reps):
                fn()
            e1.record(stream)
            e1.synchronize()
            ts.append(e0.elapsed_time(e1) / reps)
        return {"med_ms": statistics.median(ts), "min_ms": min(ts), "max_ms": max(ts)}

    # ---- H1 micro: plain nvmath C2C vs torch.fft.fft2 -----------------------------------------------------------------
    micro = R["micro"] = {}
    try:
        s = torch.cuda.Stream()
        for cdt in (torch.complex64, torch.complex128):
            x = torch.randn(N, N, dtype=cdt, device="cuda")
            f = nvmath.fft.FFT(x, axes=[0, 1], options={"fft_type": "C2C", "inplace": True}, execution="cuda", stream=s)
            f.plan(stream=s)
            f.execute(direction="forward", stream=s)
            cb.set_stream(f.handle, s.cuda_stream)
            ptr = x.data_ptr()
            name = str(cdt).split(".")[-1]
            micro[f"nvmath_c2c_{name}"] = ev_time(s, lambda: cb.xt_exec(f.handle, ptr, ptr, HS.FWD), 20)
            with torch.cuda.stream(s):
                micro[f"torch_fft2_{name}"] = ev_time(s, lambda: torch.fft.fft2(x), 20)
            del f, x
        print("micro " + json.dumps({k: round(v["med_ms"], 4) for k, v in micro.items()}), flush=True)
    except Exception as e:                                           # noqa: BLE001
        err("MICRO", e)
    save()

    # ---- the state, the JAX arms, the hybrid objects ------------------------------------------------------------------
    template = GD.sqg.init(GD.p, jax.random.key(GD.SEED_INITIAL_CONDITION))

    def config_hash(member):
        cfg = dict(GD.CONFIG, member=member, seed_initial_condition=member)
        return hashlib.sha256(json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    m, path = args.ckpt.split(":", 1)
    own, GD.CONFIG_HASH = GD.CONFIG_HASH, config_hash(int(m))
    try:
        st0, completed = GD.load_checkpoint(template, path)
    finally:
        GD.CONFIG_HASH = own
    st0 = block(st0)
    assert int(st0._ablevel) == 2
    R["state"] = {"member": int(m), "path": path, "t": float(GD.block_time(completed))}
    th0 = np.asarray(st0.state.qh[0])
    tn0, to0 = np.asarray(st0._updates[0].qh[0]), np.asarray(st0._updates[1].qh[0])

    _, ncol = LS.preconditions(GD)
    base = LS.build_consts(GD, ncol)
    c32 = LP.tables(GD)
    lad = IF.Ladder(GD, ncol, base)
    c_dt = lad.consts(DT)
    jcalls = {}
    for n in (1, 40):
        jcalls[("lean_u1", n)] = (lambda f: (lambda s: f(s, base)))(LS.make_advance(GD, n, ncol=ncol, unroll1=True))
        jcalls[("lean32ek_fast", n)] = (lambda f: (lambda s: f(s, c32)))(
            LF.make_call(pst, N, NK, DT, n, mode="ek", u1=True, ring=3))
        jcalls[("if32", n)] = (lambda f: (lambda s: f(s, c32)))(IF.make_call_if32(pst, N, NK, DT, n))
        jcalls[("if64", n)] = (lambda f: (lambda s: f(s, c_dt)))(lad.adv("if", n))

    def run_jax(arm, n):
        call = jcalls[(arm, 1 if n == 1 else 40)]
        s = st0
        for _ in range(1 if n == 1 else n // 40):
            s = call(s)
        return block(s)

    # hybrid tables (compact)
    edec32 = HS.to_compact(np.asarray(c32["edec"]))
    f0p32 = np.concatenate([np.asarray(c32["f0bot"]), np.asarray(c32["f0top"])], 0).astype(np.complex64)
    s64 = np.asarray(base["s"])
    s64 = s64.reshape(s64.shape[-2:]) if s64.ndim == 3 else s64      # (1, N, NK) or (N, NK)
    kinv64 = HS.to_compact(np.broadcast_to(s64, (N, NK)))
    k32 = np.asarray(c32["kinv"])
    kinv32 = HS.to_compact(np.broadcast_to(k32.reshape(k32.shape[-2:]) if k32.ndim == 3 else k32, (N, NK)))
    decay64 = HS.to_compact(np.asarray(GD.decay))
    F0 = np.asarray(GD.F0_hat[0])
    rows = HS.ROW_IDX[HS.KM - HS.PATCH: HS.KM + HS.PATCH]
    f0p64 = np.ascontiguousarray(F0[rows, :HS.PATCH])
    outside = F0.copy()
    outside[rows, :HS.PATCH] = 0
    R["checks"] = {"f0_inside_patch": bool(not np.any(outside != 0)),
                   "f0_patch32_equals_c32": bool(np.array_equal(f0p32, f0p64.astype(np.complex64)))}
    if not R["checks"]["f0_inside_patch"]:
        raise SystemExit("ERROR HYB_FORCING_OUTSIDE_PATCH")
    hyb = {}
    try:
        k32k64 = HS._kernels()
        cbs32, cbs64 = HS.compile_callbacks("float32"), HS.compile_callbacks("float64")
        t32 = {"kinv": kinv32, "edec": edec32, "f0patch": f0p32}
        hyb["hybrid32"] = HS.Hybrid("float32", t32, DT, kernels=k32k64, callbacks=cbs32)
        hyb["hybrid32_if"] = HS.Hybrid("float32", t32, DT, if_mode=True, kernels=k32k64, callbacks=cbs32)
        hyb["hybrid64"] = HS.Hybrid("float64", {"kinv": kinv64, "decay": decay64, "f0patch": f0p64}, DT,
                                    kernels=k32k64, callbacks=cbs64)
        R["layouts"] = {k: {kk: str(vv) for kk, vv in h.layouts.items()} for k, h in hyb.items()}
    except Exception as e:                                           # noqa: BLE001
        err("HYBRID_BUILD", e)
        failures += 1

    def run_hyb(name, n):
        h = hyb[name]
        h.pack(th0, tn0, to0)
        h.run(n)
        return h.theta()

    # ---- frames and band errors (timestep_probe.py's) ----------------------------------------------------------------
    def af_rows(a):
        return np.concatenate([a[:HALF, :HALF + 1], a[N - HALF:, :HALF + 1]], 0)

    af_scale = (NC * NC) / (N * N) / THETA_SCALE

    def frames(th):
        sub = np.fft.irfft2(th, s=(N, N))[::M, ::M] / THETA_SCALE
        af = af_rows(th) * af_scale
        af[HALF, :] = 0
        af[:, HALF] = 0
        return np.fft.rfft2(sub), af

    ky = np.fft.fftfreq(NC, 1.0 / NC)
    kxa = np.arange(NC // 2 + 1)
    kap = np.sqrt(ky[:, None] ** 2 + kxa[None, :] ** 2)
    masks = [(kap >= a) & (kap < b) for a, b in BANDS]

    def band_err(x, ref, x0):
        return [float(np.sqrt(np.mean(np.abs(x[mk] - ref[mk]) ** 2)) / np.sqrt(np.mean(np.abs(ref[mk] - x0[mk]) ** 2)))
                for mk in masks]

    def rel(a, b):
        return float(np.linalg.norm(a - b) / np.linalg.norm(b))

    # ---- H2 gates ------------------------------------------------------------------------------------------------------
    gates = R["gates"] = {}
    f_start = frames(th0)
    for n in (1, 40, 4000):
        if late():
            gates[str(n)] = {"skipped": "deadline"}
            continue
        t0 = time.time()
        res = {}
        try:
            J = {arm: np.asarray(run_jax(arm, n).state.qh[0]) for arm in ("lean_u1", "lean32ek_fast", "if32", "if64")}
            H = {name: run_hyb(name, n) for name in hyb}
            res["rel"] = {
                "hybrid32_vs_u1": rel(H["hybrid32"], J["lean_u1"]), "fast_vs_u1": rel(J["lean32ek_fast"], J["lean_u1"]),
                "hybrid32_vs_fast": rel(H["hybrid32"], J["lean32ek_fast"]),
                "hybrid32if_vs_if64": rel(H["hybrid32_if"], J["if64"]), "if32_vs_if64": rel(J["if32"], J["if64"]),
                "hybrid64_vs_u1": rel(H["hybrid64"], J["lean_u1"]), "hybrid64_vs_fast": rel(H["hybrid64"], J["lean32ek_fast"]),
                "finite": {k: bool(np.all(np.isfinite(v))) for k, v in H.items()}}
            if n == 4000:
                Fr = {k: frames(v) for k, v in list(J.items()) + list(H.items())}
                be = {}
                for a, b in (("hybrid32", "lean_u1"), ("lean32ek_fast", "lean_u1"), ("hybrid64", "lean_u1"),
                             ("hybrid32_if", "if64"), ("if32", "if64")):
                    be[f"{a}_vs_{b}"] = {"sub": band_err(Fr[a][0], Fr[b][0], f_start[0]),
                                         "af": band_err(Fr[a][1], Fr[b][1], f_start[1])}
                res["bands"] = be
        except Exception as e:                                       # noqa: BLE001
            err(f"GATES_n{n}", e)
            failures += 1
        res["seconds"] = round(time.time() - t0, 1)
        gates[str(n)] = res
        if "rel" in res:
            print(f"HYB_REL n={n} " + " ".join(f"{k}={v:.3e}" for k, v in res["rel"].items() if k != "finite")
                  + f" finite={res['rel']['finite']}", flush=True)
        save()

    def gate(key_h, key_ref, n):
        r = gates.get(str(n), {}).get("rel")
        if not r:
            return None
        return bool(r[key_h] <= min(1e-4, 2 * r[key_ref]))

    g32 = [gate("hybrid32_vs_u1", "fast_vs_u1", n) for n in (1, 40)]
    gif = [gate("hybrid32if_vs_if64", "if32_vs_if64", n) for n in (1, 40)]
    r1 = gates.get("1", {}).get("rel")
    g64 = bool(r1["hybrid64_vs_u1"] <= 1e-12) if r1 else None
    R["verdict"] = {"G32": g32, "GIF": gif, "G64": g64}
    print(f"HYB_GATE32 pass={all(x is True for x in g32)} (n=1, 40: {g32})", flush=True)
    print(f"HYB_GATEIF pass={all(x is True for x in gif)} (n=1, 40: {gif})", flush=True)
    print(f"HYB_GATE64 pass={g64 is True}", flush=True)
    if "4000" in gates and "bands" in gates["4000"]:
        for k, v in gates["4000"]["bands"].items():
            print(f"HYB_BANDS {k:>24s} sub " + " ".join(f"{x:.2e}" for x in v["sub"]) + " | af "
                  + " ".join(f"{x:.2e}" for x in v["af"]), flush=True)
    save()

    # ---- H3 timing -----------------------------------------------------------------------------------------------------
    bench = R["bench_ms_per_step"] = {}
    try:
        for name, h in hyb.items():
            h.pack(th0, tn0, to0)
            bench[f"{name}_eager"] = ev_time(h.stream, lambda h=h: h.run(40), 10)["med_ms"] / 40
        h = hyb["hybrid32"]
        h.pack(th0, tn0, to0)
        h.run(3)
        h.stream.synchronize()
        try:
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=h.stream):
                h.run(48)                                            # a multiple of 3: the ring returns to its slots
            with torch.cuda.stream(h.stream):                        # replay() launches on the current stream
                bench["hybrid32_graph"] = ev_time(h.stream, g.replay, 5)["med_ms"] / 48
            bench["hybrid32_graph_finite"] = bool(np.all(np.isfinite(h.theta())))
        except Exception as e:                                       # noqa: BLE001
            err("GRAPH", e)
        # per-op split of hybrid32 (CUDA events between the 4 launches, 40 steps)
        h.pack(th0, tn0, to0)
        cbm, (hw, hq, hz), (pp, qi, qq) = h.cb, h.handles, h.ptr
        evs = [[torch.cuda.Event(enable_timing=True) for _ in range(5)] for _ in range(40)]
        for i in range(40):
            e = evs[i]
            e[0].record(h.stream)
            cbm.xt_exec(hw, pp, pp, HS.FWD)
            e[1].record(h.stream)
            cbm.xt_exec(hq, qi, qq, HS.INV)
            e[2].record(h.stream)
            cbm.xt_exec(hz, pp, pp, HS.FWD)
            e[3].record(h.stream)
            a, b, f = h.slot
            wp.launch(h.kernel, dim=(HS.CR, HS.CC), inputs=[h.wZ, h.wth, h.wlo, h.wT[a], h.wT[b], h.wdamp, h.wf0,
                                                           h.wT[f], h.c[0], h.c[1], h.c[2], 0], stream=h.ws)
            h.slot = [f, a, b]
            e[4].record(h.stream)
        h.stream.synchronize()
        names = ("c2c_w_to_p", "c2r_theta", "c2c_z_store_kept", "warp_update")
        bench["hybrid32_split_ms"] = {nm: statistics.median(evs[i][j].elapsed_time(evs[i][j + 1]) for i in range(5, 40))
                                      for j, nm in enumerate(names)}
        # JAX steps, same GPU
        for arm in ("lean_u1", "lean32ek_fast", "if32"):
            call = jcalls[(arm, 40)]
            x = st0
            for _ in range(3):
                x = call(x)
            block(x)
            ts = []
            for _ in range(3):
                t0 = time.perf_counter()
                for _ in range(30):
                    x = call(x)
                block(x)
                ts.append(time.perf_counter() - t0)
            bench[arm] = 1e3 * statistics.median(ts) / (30 * 40)
    except Exception as e:                                           # noqa: BLE001
        err("BENCH", e)
        failures += 1
    print("HYB_BENCH " + json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in bench.items()}),
          flush=True)
    R["failures"] = failures
    R["wall_s"] = round(time.time() - t_wall, 1)
    save()
    print(f"HYB_PROBE_DONE failures={failures}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except BaseException as exc:
        first = str(exc).strip().splitlines()[0][:200] if str(exc).strip() else ""
        print(f"ERROR HYB_CRASHED {type(exc).__name__}: {first}", flush=True)
        raise
