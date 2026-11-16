"""tilerows_probe.py -- tilerows_step.py against the JAX steps (PREDICTIONS_tilerows.md, written first).

  "$PY" tilerows_probe.py --out <dir> --ckpt <member>:<npz> --warp-path <dir> --nvmath-path <dir> [--deadline <s>]

Arms: tilerows32 (lean32ek numerics), tilerows32_if (IF-AB3), next to the JAX lean_u1, lean32ek_fast, if32, if64.
n = 1, 40, 4,000 steps from one archived state: rel = |theta_a - theta_b| / |theta_b| (full complex128 state) and, at
4,000 steps, the band errors of the 0.02 delta on sub / af frames. Gates (pre-set, the hybrid's G32 / GIF):
rel(tilerows32, lean_u1) <= min(1e-4, 2 rel(lean32ek_fast, lean_u1)) at n = 1, 40 (IF: if64 / if32).
Timing: ms/step eager (10 x 40 steps), a 48-step CUDA graph, the per-op split (CUDA events), and the JAX steps.
Tokens: TR_ENV_OK, TR_REL, TR_GATE, TR_BANDS, TR_BENCH, TR_PROBE_DONE failures=N.
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
        with open(os.path.join(args.out, "tilerows_probe.json"), "w") as fh:
            json.dump(jsonable(R), fh, indent=1, default=str)

    def err(where, e):
        R["errors"].append({"where": where, "error": f"{type(e).__name__}: {e}"[:2000], "tb": traceback.format_exc()[-2500:]})
        print(f"ERROR TR_{where}: {type(e).__name__}: {str(e)[:300]}", flush=True)
        save()

    def late(margin=300):
        return bool(args.deadline) and time.time() > args.deadline - margin

    import numpy as np
    import torch
    import profile_solver as PS
    GD = PS.import_asqg()
    import jax
    from pyqg_jax import steppers as pst
    sys.path.insert(0, args.warp_path)
    site.addsitedir(args.nvmath_path)
    import warp as wp
    wp.config.enable_backward = False
    wp.init()
    from nvmath.bindings import cufft as cb
    import tilerows_step as TS
    import lean_solver as LS
    import lean32_precise as LP
    import lean32_fast as LF
    import if_ab3 as IF
    block = jax.block_until_ready
    N, NK, DT, STRIDE, M = GD.N, GD.N // 2 + 1, GD.DT, PS.STRIDE, PS.M
    assert N == TS.N and STRIDE == 40
    R["env"] = {"torch": torch.__version__, "jax": jax.__version__, "warp": wp.__version__,
                "cufft_version": int(cb.get_version()), "gpu": torch.cuda.get_device_name(0),
                "ld_preload": os.environ.get("LD_PRELOAD", "")}
    print("TR_ENV_OK " + json.dumps(R["env"]), flush=True)

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
        return np.asarray(block(s).state.qh[0])

    k = np.asarray(c32["kinv"])
    k = k.reshape(k.shape[-2:]) if k.ndim == 3 else k
    tables = {"kinv": TS.to_compact(np.broadcast_to(k, (N, NK))), "edec": TS.to_compact(np.asarray(c32["edec"])),
              "f0patch": np.concatenate([np.asarray(c32["f0bot"]), np.asarray(c32["f0top"])], 0).astype(np.complex64)}
    tr = {}
    try:
        tr["tilerows32"] = TS.TileRows(tables, DT)
        tr["tilerows32_if"] = TS.TileRows(tables, DT, if_mode=True)
    except Exception as e:                                           # noqa: BLE001
        err("BUILD", e)
        failures += 1

    def run_tr(name, n):
        h = tr[name]
        h.pack(th0, tn0, to0)
        h.run(n)
        return h.theta()

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

    gates = R["gates"] = {}
    f_start = frames(th0)
    for n in (1, 40, 4000):
        if late():
            gates[str(n)] = {"skipped": "deadline"}
            continue
        res = {}
        try:
            J = {arm: run_jax(arm, n) for arm in ("lean_u1", "lean32ek_fast", "if32", "if64")}
            H = {name: run_tr(name, n) for name in tr}
            res["rel"] = {"tilerows32_vs_u1": rel(H["tilerows32"], J["lean_u1"]),
                          "fast_vs_u1": rel(J["lean32ek_fast"], J["lean_u1"]),
                          "tilerows32_vs_fast": rel(H["tilerows32"], J["lean32ek_fast"]),
                          "tilerows32if_vs_if64": rel(H["tilerows32_if"], J["if64"]),
                          "if32_vs_if64": rel(J["if32"], J["if64"]),
                          "finite": {k_: bool(np.all(np.isfinite(v))) for k_, v in H.items()}}
            if n == 4000:
                Fr = {k_: frames(v) for k_, v in list(J.items()) + list(H.items())}
                res["bands"] = {f"{a}_vs_{b}": {"sub": band_err(Fr[a][0], Fr[b][0], f_start[0]),
                                                 "af": band_err(Fr[a][1], Fr[b][1], f_start[1])}
                                for a, b in (("tilerows32", "lean_u1"), ("lean32ek_fast", "lean_u1"),
                                             ("tilerows32_if", "if64"), ("if32", "if64"))}
        except Exception as e:                                       # noqa: BLE001
            err(f"GATES_n{n}", e)
            failures += 1
        gates[str(n)] = res
        if "rel" in res:
            print(f"TR_REL n={n} " + " ".join(f"{k_}={v:.3e}" for k_, v in res["rel"].items() if k_ != "finite")
                  + f" finite={res['rel']['finite']}", flush=True)
        save()

    def gate(kh, kr, n):
        r = gates.get(str(n), {}).get("rel")
        return None if not r else bool(r[kh] <= min(1e-4, 2 * r[kr]))

    g32 = [gate("tilerows32_vs_u1", "fast_vs_u1", n) for n in (1, 40)]
    gif = [gate("tilerows32if_vs_if64", "if32_vs_if64", n) for n in (1, 40)]
    R["verdict"] = {"G32": g32, "GIF": gif}
    print(f"TR_GATE32 pass={all(x is True for x in g32)} ({g32})", flush=True)
    print(f"TR_GATEIF pass={all(x is True for x in gif)} ({gif})", flush=True)
    for k_, v in gates.get("4000", {}).get("bands", {}).items():
        print(f"TR_BANDS {k_:>26s} sub " + " ".join(f"{x:.2e}" for x in v["sub"]) + " | af "
              + " ".join(f"{x:.2e}" for x in v["af"]), flush=True)
    save()

    # ---- timing ------------------------------------------------------------------------------------------------------
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
        return statistics.median(ts)

    bench = R["bench_ms_per_step"] = {}
    try:
        for name, h in tr.items():
            h.pack(th0, tn0, to0)
            bench[f"{name}_eager"] = ev_time(h.stream, lambda h=h: h.run(40), 10) / 40
        h = tr["tilerows32"]
        h.pack(th0, tn0, to0)
        try:
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=h.stream):
                h.run(48)
            with torch.cuda.stream(h.stream):
                bench["tilerows32_graph"] = ev_time(h.stream, g.replay, 5) / 48
            bench["tilerows32_graph_finite"] = bool(np.all(np.isfinite(h.theta())))
        except Exception as e:                                       # noqa: BLE001
            err("GRAPH", e)
        h.pack(th0, tn0, to0)
        ops = h.ops()
        evs = [[torch.cuda.Event(enable_timing=True) for _ in range(len(ops) + 1)] for _ in range(40)]
        for i in range(40):
            evs[i][0].record(h.stream)
            for j, (_, op) in enumerate(ops):
                op()
                evs[i][j + 1].record(h.stream)
        h.stream.synchronize()
        bench["tilerows32_split_ms"] = {nm: statistics.median(evs[i][j].elapsed_time(evs[i][j + 1]) for i in range(5, 40))
                                        for j, (nm, _) in enumerate(ops)}
        for arm in ("lean32ek_fast", "if32", "lean_u1"):
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
    print("TR_BENCH " + json.dumps({k_: (round(v, 4) if isinstance(v, float) else v) for k_, v in bench.items()}),
          flush=True)
    R["failures"] = failures
    R["wall_s"] = round(time.time() - t_wall, 1)
    save()
    print(f"TR_PROBE_DONE failures={failures}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except BaseException as exc:
        first = str(exc).strip().splitlines()[0][:200] if str(exc).strip() else ""
        print(f"ERROR TR_CRASHED {type(exc).__name__}: {first}", flush=True)
        raise
