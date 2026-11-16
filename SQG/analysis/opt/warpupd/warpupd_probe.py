"""warpupd_probe.py -- Warp SIMT kernels inside the JAX lean32ek_fast step (warpupd.py; PREDICTIONS_warpupd.md first).

  "$PY" warpupd_probe.py --out <dir> --ckpt <member>:<npz> --warp-path <dir> [--deadline <epoch s>]
  "$PY" warpupd_probe.py --out <dir> --ckpt ... --warp-path ... --ncu <arm>     (one 40-step call in a profiler range)

Arms (40-step calls AB3State -> AB3State): the JAX baselines lean_u1 (float64), lean32ek_fast, if32 (lean32ek_fast +
IF-AB3), if64; and warpupd's xla_compact, xla_compact_if, warp_full, warp_compact, warp_compact_if, warp_simt,
warp_simt_if. From one archived state, n = 1, 40, 400 steps:
  rel = |theta_a - theta_b| / |theta_b| over the full complex128 state; bitwise vs lean32ek_fast (if32 for _if).
  Gate (pre-set, the hybrid's G32 / GIF): rel(arm, lean_u1) <= min(1e-4, 2 rel(lean32ek_fast, lean_u1)) at n = 1, 40
  (the _if arms: if64 / if32).
Timing: ms/step, 3 x 30 calls of 40 steps, every arm, same GPU. Tokens: WU_ENV_OK, WU_REL, WU_GATE, WU_BENCH,
WU_PROBE_DONE failures=N (ncu mode: WU_NCU_DONE).
"""
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import argparse                                                      # noqa: E402
import hashlib                                                       # noqa: E402
import json                                                          # noqa: E402
import math                                                          # noqa: E402
import statistics                                                    # noqa: E402
import sys                                                           # noqa: E402
import time                                                          # noqa: E402
import traceback                                                     # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OPT = os.path.dirname(HERE)
ANALYSIS = os.path.dirname(OPT)
for p in (os.path.join(OPT, "seed1"), os.path.join(OPT, "delta"), os.path.join(OPT, "taxonomy"),
          os.path.join(OPT, "timestep"), os.path.join(OPT, "warp"), HERE, ANALYSIS):
    sys.path.insert(0, p)

WARP_ARMS = ("xla_compact", "xla_compact_if", "warp_full", "warp_compact", "warp_compact_if", "warp_simt",
             "warp_simt_if")


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
    ap.add_argument("--deadline", type=float, default=0.0)
    ap.add_argument("--ncu", default="", help="profile one 40-step call of this arm (cudaProfilerStart/Stop)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    R = {"errors": []}
    failures = 0

    def save():
        name = f"warpupd_ncu_{args.ncu}.json" if args.ncu else "warpupd_probe.json"
        with open(os.path.join(args.out, name), "w") as fh:
            json.dump(jsonable(R), fh, indent=1, default=str)

    def err(where, e):
        R["errors"].append({"where": where, "error": f"{type(e).__name__}: {e}"[:2000], "tb": traceback.format_exc()[-2500:]})
        print(f"ERROR WU_{where}: {type(e).__name__}: {str(e)[:300]}", flush=True)
        save()

    def late(margin=300):
        return bool(args.deadline) and time.time() > args.deadline - margin

    import numpy as np
    import torch                                                     # noqa: F401  (o1.2's import order)
    import profile_solver as PS
    GD = PS.import_asqg()
    import jax
    from pyqg_jax import steppers as pst
    sys.path.insert(0, args.warp_path)
    import warp as wp
    wp.config.enable_backward = False
    wp.init()
    import warp_xla as WX
    import warpupd as WU
    import lean_solver as LS
    import lean32_precise as LP
    import lean32_fast as LF
    import if_ab3 as IF
    block = jax.block_until_ready
    N, NK, DT, STRIDE = GD.N, GD.N // 2 + 1, GD.DT, PS.STRIDE
    assert N == WU.N and STRIDE == 40
    R["env"] = {"jax": jax.__version__, "warp": wp.__version__, "gpu": jax.devices()[0].device_kind}
    print("WU_ENV_OK " + json.dumps(R["env"]), flush=True)

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

    _, ncol = LS.preconditions(GD)
    base = LS.build_consts(GD, ncol)
    c32 = WU.compact_tables(LP.tables(GD))
    lad = IF.Ladder(GD, ncol, base)
    c_dt = lad.consts(DT)

    def builders(arm, n):
        if arm == "lean_u1":
            f = LS.make_advance(GD, n, ncol=ncol, unroll1=True)
            return lambda s: f(s, base)
        if arm == "lean32ek_fast":
            f = LF.make_call(pst, N, NK, DT, n, mode="ek", u1=True, ring=3)
            return lambda s: f(s, c32)
        if arm == "if32":
            f = IF.make_call_if32(pst, N, NK, DT, n)
            return lambda s: f(s, c32)
        if arm == "if64":
            f = lad.adv("if", n)
            return lambda s: f(s, c_dt)
        variant, if_mode = (arm[:-3], True) if arm.endswith("_if") else (arm, False)
        f = WU.make_call(pst, variant, DT, n, if_mode=if_mode)
        return lambda s: f(s, c32)

    # ---- ncu mode: one 40-step call of one arm inside a profiler range --------------------------------------------
    if args.ncu:
        call = builders(args.ncu, 40)
        x = block(call(st0))
        x = block(call(x))
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
        x = block(call(x))
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
        R["ncu_arm"] = args.ncu
        R["finite"] = bool(np.all(np.isfinite(np.asarray(x.state.qh))))
        save()
        print(f"WU_NCU_DONE arm={args.ncu} finite={R['finite']} stats={WX.STATS}", flush=True)
        return

    arms = ["lean_u1", "lean32ek_fast", "if32", "if64"] + list(WARP_ARMS)
    calls = {}
    for arm in arms:
        for n in (1, 40):
            try:
                calls[(arm, n)] = builders(arm, n)
            except Exception as e:                                   # noqa: BLE001
                err(f"BUILD_{arm}_{n}", e)
                failures += 1

    def run(arm, n):
        call = calls[(arm, 1 if n == 1 else 40)]
        s = st0
        for _ in range(1 if n == 1 else n // 40):
            s = call(s)
        return np.asarray(block(s).state.qh[0])

    def rel(a, b):
        return float(np.linalg.norm(a - b) / np.linalg.norm(b))

    gates = R["gates"] = {}
    for n in (1, 40, 400):
        if late():
            gates[str(n)] = {"skipped": "deadline"}
            continue
        th = {}
        for arm in arms:
            try:
                th[arm] = run(arm, n)
            except Exception as e:                                   # noqa: BLE001
                err(f"RUN_{arm}_{n}", e)
                failures += 1
        res = {"rel_vs_ref": {}, "bitwise_vs_base": {}, "finite": {}}
        for arm in WARP_ARMS:
            if arm not in th:
                continue
            ref, base_arm = ("if64", "if32") if arm.endswith("_if") else ("lean_u1", "lean32ek_fast")
            res["rel_vs_ref"][arm] = rel(th[arm], th[ref])
            res["bitwise_vs_base"][arm] = bool(np.array_equal(th[arm].view(np.uint64), th[base_arm].view(np.uint64)))
            res["rel_vs_base"] = res.get("rel_vs_base", {})
            res["rel_vs_base"][arm] = rel(th[arm], th[base_arm])
            res["finite"][arm] = bool(np.all(np.isfinite(th[arm])))
        if "lean32ek_fast" in th and "lean_u1" in th:
            res["fast_vs_u1"] = rel(th["lean32ek_fast"], th["lean_u1"])
        if "if32" in th and "if64" in th:
            res["if32_vs_if64"] = rel(th["if32"], th["if64"])
        gates[str(n)] = res
        print(f"WU_REL n={n} fast_vs_u1={res.get('fast_vs_u1', float('nan')):.3e} "
              f"if32_vs_if64={res.get('if32_vs_if64', float('nan')):.3e} | "
              + " ".join(f"{a}={v:.3e}{'(bit)' if res['bitwise_vs_base'].get(a) else ''}"
                         for a, v in res["rel_vs_ref"].items()), flush=True)
        save()

    verdict = {}
    for arm in WARP_ARMS:
        ok = []
        for n in ("1", "40"):
            g = gates.get(n, {})
            refv = g.get("if32_vs_if64" if arm.endswith("_if") else "fast_vs_u1")
            v = g.get("rel_vs_ref", {}).get(arm)
            ok.append(None if v is None or refv is None else bool(v <= min(1e-4, 2 * refv)))
        verdict[arm] = ok
        print(f"WU_GATE {arm} pass={all(x is True for x in ok)} ({ok})", flush=True)
    R["verdict"] = verdict
    save()

    # ---- timing ------------------------------------------------------------------------------------------------------
    bench = R["bench_ms_per_step"] = {}
    for arm in arms:
        if late(120) or (arm, 40) not in calls:
            continue
        try:
            call = calls[(arm, 40)]
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
        except Exception as e:                                       # noqa: BLE001
            err(f"BENCH_{arm}", e)
    print("WU_BENCH " + " ".join(f"{k}={v:.4f}" for k, v in bench.items()), flush=True)
    R["bridge_stats"] = dict(WX.STATS)
    if WX.STATS["errors"]:
        failures += 1
        print(f"ERROR WU_BRIDGE_ERRORS {WX.STATS}", flush=True)
    R["failures"] = failures
    R["wall_s"] = round(time.time() - t_wall, 1)
    save()
    print(f"WU_PROBE_DONE failures={failures}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except BaseException as exc:
        first = str(exc).strip().splitlines()[0][:200] if str(exc).strip() else ""
        print(f"ERROR WU_CRASHED {type(exc).__name__}: {first}", flush=True)
        raise
