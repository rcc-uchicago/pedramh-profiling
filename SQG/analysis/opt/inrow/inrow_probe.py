"""inrow_probe.py -- inrow_step.InRow (in-tile C2R) against tilerows_step.TileRows and the JAX steps
(PREDICTIONS_inrow.md, written first).

  "$PY" inrow_probe.py --out <dir> --tag prod|cufft114 --ckpt <member>:<npz> --warp-path <dir> --nvmath-path <dir>
                       [--deadline <epoch s>]

Arms: tilerows32 / tilerows32_if (the current step), inrow32 / inrow32_if (deciding: theta_first, block 256),
inrow32_wfirst and inrow32_b512 (timing + rel only), next to the JAX lean_u1, lean32ek_fast, if32, if64.
Validity: THETA (in-tile q_u vs float64 N^2 irfft2, rms rel <= 1e-5; cuFFT C2R's beside it), HI_PRESERVED (the
out-of-place column pass leaves hi bit-identical), HI_ZERO_ROWS (hi stays 0 outside +-KM after 4,000 steps), finite.
Gates (tilerows probe's, unchanged): rel(arm, ref64) <= min(1e-4, 2 rel(twin32, ref64)) at n = 1, 40.
WORTH: valid, G32, GIF, and inrow32 eager <= 0.97 x tilerows32 eager (the deciding pass is --tag prod).
Tokens: IR_ENV_OK, IR_BUILD, IR_THETA, IR_REL, IR_GATE32, IR_GATEIF, IR_BANDS, IR_BENCH, IR_VERDICT,
IR_PROBE_DONE failures=N. Writes inrow_probe_<tag>.json.
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
          os.path.join(OPT, "timestep"), os.path.join(OPT, "tilerows"), HERE, ANALYSIS):
    sys.path.insert(0, p)
THETA_SCALE = 8.0
NC, HALF = 256, 128
BANDS = ((1, 2), (2, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 128), (128, 182))
THETA_TOL, WORTH_RATIO = 1e-5, 0.97
GATE_ARMS = ("tilerows32", "tilerows32_if", "inrow32", "inrow32_if")
VARIANTS = ("inrow32_wfirst", "inrow32_b512")


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
    ap.add_argument("--tag", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--warp-path", required=True)
    ap.add_argument("--nvmath-path", required=True)
    ap.add_argument("--deadline", type=float, default=0.0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    R = {"tag": args.tag, "errors": []}
    failures = 0

    def save():
        with open(os.path.join(args.out, f"inrow_probe_{args.tag}.json"), "w") as fh:
            json.dump(jsonable(R), fh, indent=1, default=str)

    def err(where, e):
        R["errors"].append({"where": where, "error": f"{type(e).__name__}: {e}"[:2000], "tb": traceback.format_exc()[-2500:]})
        print(f"ERROR IR_{where}: {type(e).__name__}: {str(e)[:300]}", flush=True)
        save()

    def late(margin=240):
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
    import inrow_step as IR
    import lean_solver as LS
    import lean32_precise as LP
    import lean32_fast as LF
    import if_ab3 as IF
    block = jax.block_until_ready
    N, NK, DT, STRIDE, M = GD.N, GD.N // 2 + 1, GD.DT, PS.STRIDE, PS.M
    assert N == TS.N == IR.N and STRIDE == 40
    R["env"] = {"torch": torch.__version__, "jax": jax.__version__, "warp": wp.__version__,
                "cufft_version": int(cb.get_version()), "gpu": torch.cuda.get_device_name(0),
                "ld_preload": os.environ.get("LD_PRELOAD", ""), "ccp": IR.CCP}
    print("IR_ENV_OK " + json.dumps(R["env"]), flush=True)

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

    # ---- build: construct, then one step (Warp compiles at first launch) ----------------------------------------
    specs = [("tilerows32", lambda: TS.TileRows(tables, DT)),
             ("tilerows32_if", lambda: TS.TileRows(tables, DT, if_mode=True)),
             ("inrow32", lambda: IR.InRow(tables, DT)),
             ("inrow32_if", lambda: IR.InRow(tables, DT, if_mode=True)),
             ("inrow32_wfirst", lambda: IR.InRow(tables, DT, order="w_first")),
             ("inrow32_b512", lambda: IR.InRow(tables, DT, block_dim=512))]
    tr = {}
    R["build"] = {}
    for name, mk in specs:
        t0 = time.time()
        try:
            h = mk()
            h.pack(th0, tn0, to0)
            h.run(1)
            ok = bool(np.all(np.isfinite(h.theta())))
            tr[name] = h
            R["build"][name] = {"ok": ok, "seconds": round(time.time() - t0, 1)}
        except Exception as e:                                       # noqa: BLE001
            err(f"BUILD_{name}", e)
            R["build"][name] = {"ok": False, "seconds": round(time.time() - t0, 1)}
            failures += 1
        print(f"IR_BUILD {name} ok={R['build'][name]['ok']} s={R['build'][name]['seconds']}", flush=True)
    save()

    def run_tr(name, n):
        h = tr[name]
        h.pack(th0, tn0, to0)
        h.run(n)
        return h.theta()

    def rms(x):
        return float(np.sqrt(np.mean(np.square(x))))

    # ---- validity: q_u on the start state, and hi untouched by the out-of-place pass ------------------------------
    V = R["validity"] = {}
    try:
        q_ref = np.fft.irfft2(th0, s=(N, N)) * float(N * N)
        h = tr["inrow32"]
        h.pack(th0, tn0, to0)
        q_ir, hi_pres = h.theta_phys()
        t = tr["tilerows32"]
        t.pack(th0, tn0, to0)
        op = dict(t.ops())
        op["expand"]()
        op["cufft_c2r"]()
        t.stream.synchronize()
        q_tr = t.q.cpu().numpy()
        s_ref = rms(q_ref)
        V["theta"] = {"rms_rel_inrow": rms(q_ir - q_ref) / s_ref, "rms_rel_c2r": rms(q_tr - q_ref) / s_ref,
                      "max_rel_inrow": float(np.max(np.abs(q_ir - q_ref))) / float(np.max(np.abs(q_ref))),
                      "max_rel_c2r": float(np.max(np.abs(q_tr - q_ref))) / float(np.max(np.abs(q_ref))),
                      "rms_rel_inrow_vs_c2r": rms(q_ir - q_tr) / s_ref}
        V["THETA"] = bool(V["theta"]["rms_rel_inrow"] <= THETA_TOL)
        V["HI_PRESERVED"] = hi_pres
        print("IR_THETA " + " ".join(f"{k_}={v:.3e}" for k_, v in V["theta"].items())
              + f" valid={V['THETA']} hi_preserved={hi_pres}", flush=True)
    except Exception as e:                                           # noqa: BLE001
        err("THETA", e)
        failures += 1
    save()

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
            H = {name: run_tr(name, n) for name in GATE_ARMS if name in tr}
            r_ = {"fast_vs_u1": rel(J["lean32ek_fast"], J["lean_u1"]), "if32_vs_if64": rel(J["if32"], J["if64"])}
            for a, ref, key in (("inrow32", "lean_u1", "inrow32_vs_u1"), ("tilerows32", "lean_u1", "tilerows32_vs_u1"),
                                ("inrow32_if", "if64", "inrow32if_vs_if64"),
                                ("tilerows32_if", "if64", "tilerows32if_vs_if64")):
                if a in H:
                    r_[key] = rel(H[a], J[ref])
            if "inrow32" in H and "tilerows32" in H:
                r_["inrow32_vs_tilerows32"] = rel(H["inrow32"], H["tilerows32"])
            r_["finite"] = {k_: bool(np.all(np.isfinite(v))) for k_, v in H.items()}
            res["rel"] = r_
            if n == 40:
                var = res["variants"] = {}
                for name in VARIANTS:
                    if name in tr and "inrow32" in H:
                        x = run_tr(name, n)
                        var[name] = {"rel_vs_inrow32": rel(x, H["inrow32"]),
                                     "bitwise_vs_inrow32": bool(np.array_equal(x, H["inrow32"])),
                                     "finite": bool(np.all(np.isfinite(x)))}
            if n == 4000:
                V["HI_ZERO_ROWS"] = {a: tr[a].hi_outside_max() == 0.0 for a in ("inrow32", "inrow32_if") if a in tr}
                Fr = {k_: frames(v) for k_, v in list(J.items()) + list(H.items())}
                res["bands"] = {f"{a}_vs_{b}": {"sub": band_err(Fr[a][0], Fr[b][0], f_start[0]),
                                                 "af": band_err(Fr[a][1], Fr[b][1], f_start[1])}
                                for a, b in (("inrow32", "lean_u1"), ("tilerows32", "lean_u1"),
                                             ("lean32ek_fast", "lean_u1"), ("inrow32_if", "if64"), ("if32", "if64"))
                                if a in Fr}
        except Exception as e:                                       # noqa: BLE001
            err(f"GATES_n{n}", e)
            failures += 1
        gates[str(n)] = res
        if "rel" in res:
            print(f"IR_REL n={n} " + " ".join(f"{k_}={v:.3e}" for k_, v in res["rel"].items() if k_ != "finite")
                  + f" finite={res['rel']['finite']}", flush=True)
        for k_, v in res.get("variants", {}).items():
            print(f"IR_VARIANT {k_} " + json.dumps(v), flush=True)
        save()

    def gate(kh, kr, n):
        r = gates.get(str(n), {}).get("rel")
        return None if not r or kh not in r else bool(r[kh] <= min(1e-4, 2 * r[kr]))

    ver = R["verdict"] = {"G32": [gate("inrow32_vs_u1", "fast_vs_u1", n) for n in (1, 40)],
                          "GIF": [gate("inrow32if_vs_if64", "if32_vs_if64", n) for n in (1, 40)],
                          "TR_G32": [gate("tilerows32_vs_u1", "fast_vs_u1", n) for n in (1, 40)],
                          "TR_GIF": [gate("tilerows32if_vs_if64", "if32_vs_if64", n) for n in (1, 40)]}
    print(f"IR_GATE32 pass={all(x is True for x in ver['G32'])} ({ver['G32']}) tilerows32 {ver['TR_G32']}", flush=True)
    print(f"IR_GATEIF pass={all(x is True for x in ver['GIF'])} ({ver['GIF']}) tilerows32_if {ver['TR_GIF']}",
          flush=True)
    for k_, v in gates.get("4000", {}).get("bands", {}).items():
        print(f"IR_BANDS {k_:>26s} sub " + " ".join(f"{x:.2e}" for x in v["sub"]) + " | af "
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
        for name in ("tilerows32", "inrow32"):
            if name not in tr:
                continue
            h = tr[name]
            h.pack(th0, tn0, to0)
            try:
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, stream=h.stream):
                    h.run(48)
                with torch.cuda.stream(h.stream):
                    bench[f"{name}_graph"] = ev_time(h.stream, g.replay, 5) / 48
                bench[f"{name}_graph_finite"] = bool(np.all(np.isfinite(h.theta())))
            except Exception as e:                                   # noqa: BLE001
                err(f"GRAPH_{name}", e)
            h.pack(th0, tn0, to0)
            ops = h.ops()
            evs = [[torch.cuda.Event(enable_timing=True) for _ in range(len(ops) + 1)] for _ in range(40)]
            for i in range(40):
                evs[i][0].record(h.stream)
                for j, (_, op) in enumerate(ops):
                    op()
                    evs[i][j + 1].record(h.stream)
            h.stream.synchronize()
            bench[f"{name}_split_ms"] = {nm: statistics.median(evs[i][j].elapsed_time(evs[i][j + 1])
                                                               for i in range(5, 40)) for j, (nm, _) in enumerate(ops)}
        if not late(120):
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
    print("IR_BENCH " + json.dumps({k_: (round(v, 4) if isinstance(v, float) else v) for k_, v in bench.items()}),
          flush=True)

    # ---- verdict (PREDICTIONS_inrow.md) -------------------------------------------------------------------------
    fin = all(all(g.get("rel", {}).get("finite", {}).values()) and bool(g.get("rel", {}).get("finite"))
              for g in gates.values() if "rel" in g)
    hz = V.get("HI_ZERO_ROWS", {})
    valid = bool(V.get("THETA") and V.get("HI_PRESERVED") and hz and all(hz.values()) and fin
                 and all(str(n) in gates and "rel" in gates[str(n)] for n in (1, 40)))
    ti, tt = bench.get("inrow32_eager"), bench.get("tilerows32_eager")
    ratio = (ti / tt) if (ti and tt) else None
    g_ok = all(x is True for x in ver["G32"]) and all(x is True for x in ver["GIF"])
    ver.update({"valid": valid, "speed_ratio": ratio,
                "worth": bool(valid and g_ok and ratio is not None and ratio <= WORTH_RATIO)})
    print(f"IR_VERDICT tag={args.tag} valid={valid} G32={ver['G32']} GIF={ver['GIF']} ratio="
          f"{'nan' if ratio is None else f'{ratio:.4f}'} worth={ver['worth']}", flush=True)
    R["failures"] = failures
    R["wall_s"] = round(time.time() - t_wall, 1)
    save()
    print(f"IR_PROBE_DONE failures={failures}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except BaseException as exc:
        first = str(exc).strip().splitlines()[0][:200] if str(exc).strip() else ""
        print(f"ERROR IR_CRASHED {type(exc).__name__}: {first}", flush=True)
        raise
