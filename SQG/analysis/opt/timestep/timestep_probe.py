"""timestep_probe.py -- research round 2's time-step lever, measured (PREDICTIONS_timestep.md, written first).

  "$PY" timestep_probe.py --out <dir> --worker <w> --mode ladder --ckpt <member>:<npz>
  "$PY" timestep_probe.py --out <dir> --worker <w> --mode aux --ckpt <member>:<npz> [--ckpt ...] --acc <member>:<npz>

ladder (one archived state, one A100): one 0.02 lag (o1.2's pair) from the state, at h = 2e-4 / n, for
  prod  production's AB3 (one decay factor for all history terms), n in NSUBS;
  if    the exact integrating-factor AB3 (if_ab3.py), n in NSUBS;
  ref   if at n = 160 (h = 1.25e-6): the reference, 4x finer than production; CFL sampled after every 40-step call.
  control: prod at n = 40 through if_ab3's run-time-weight call vs lean_u1 (bitwise production): rel <= 1e-12.
  Every arm's end state -> o1.2's two frame kinds, both in float64: sub = irfft2(theta_hat)[::16, ::16] / 8 (then its
  256^2 rfft2) and af = the alias-free coarse spectrum (taxonomy.py's layout). Error per band b (taxonomy BANDS):
      e_b = rms_b(X_arm - X_ref) / rms_b(X_ref - X_start)          (X_start = the frame at lag 0)
  i.e. the arm's 0.02-delta error relative to the band's delta, the quantity Part XII's dt ladder measured.
aux: the CFL of every archived state passed (--ckpt), at production's DT; on the --acc state, float32 IF-AB3
  (if_ab3.make_call_if32) vs float64 IF-AB3 next to lean32ek_fast vs lean_u1 (0.02 lag, n = 40); ms/step of
  lean_u1, prod-rt, if-rt (float64) and lean32ek_fast, if32 (float32); and the batched-RFFT lowering check (one
  rfft2 over a leading dim of 2 vs two calls: HLO fft ops + time).
Writes worker_<w>.json (+ frames_w<w>.npz in ladder mode). Prints TS_WORKER_DONE w=<w> last.
"""
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import argparse                                                      # noqa: E402
import hashlib                                                       # noqa: E402
import json                                                          # noqa: E402
import math                                                          # noqa: E402
import re                                                            # noqa: E402
import statistics                                                    # noqa: E402
import sys                                                           # noqa: E402
import time                                                          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OPT = os.path.dirname(HERE)
ANALYSIS = os.path.dirname(OPT)
for p in (os.path.join(OPT, "seed1"), os.path.join(OPT, "delta"), os.path.join(OPT, "taxonomy"), HERE, ANALYSIS):
    sys.path.insert(0, p)
import profile_solver as PS                                          # noqa: E402
import lean_solver as LS                                             # noqa: E402
import if_ab3 as IF                                                  # noqa: E402

NSUBS = (80, 40, 32, 28, 25, 20)
NREF = 160
THETA_SCALE = 8.0
NC, HALF = 256, 128
BANDS = ((1, 2), (2, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 128), (128, 182))   # taxonomy.py's


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
    ap.add_argument("--mode", choices=("ladder", "aux"), required=True)
    ap.add_argument("--ckpt", action="append", required=True, help="<member>:<path to ckpt_t*.npz>")
    ap.add_argument("--acc", default="", help="aux: <member>:<npz> for the float32 IF-AB3 check")
    ap.add_argument("--bench-calls", type=int, default=30)
    ap.add_argument("--deadline", type=float, default=0.0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    W = args.worker

    GD = PS.import_asqg()
    import jax
    import jax.numpy as jnp
    import numpy as np
    from pyqg_jax import steppers as pst
    block = jax.block_until_ready
    N, NK, DT, STRIDE, M = GD.N, GD.N // 2 + 1, GD.DT, PS.STRIDE, PS.M
    assert N // M == NC and STRIDE == 40, (N, M, STRIDE)
    template = GD.sqg.init(GD.p, jax.random.key(GD.SEED_INITIAL_CONDITION))
    res = {"worker": W, "mode": args.mode, "device_kind": jax.devices()[0].device_kind, "jax": jax.__version__,
           "solver_sha256": GD.CONFIG["solver_sha256"], "bands": BANDS, "nsubs": NSUBS, "nref": NREF}

    def save():
        with open(os.path.join(args.out, f"worker_{W}.json"), "w") as fh:
            json.dump(jsonable(res), fh, indent=1)

    # ---- loading, as o1.2 L192-204 (taxonomy.py) -------------------------------------------------------------------
    def config_hash(member):
        cfg = dict(GD.CONFIG, member=member, seed_initial_condition=member)
        return hashlib.sha256(json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def load(spec):
        m, path = spec.split(":", 1)
        if not os.path.exists(path):
            raise SystemExit(f"ERROR TS_CKPT_MISSING {path}")
        own, GD.CONFIG_HASH = GD.CONFIG_HASH, config_hash(int(m))
        try:
            st, completed = GD.load_checkpoint(template, path)
        finally:
            GD.CONFIG_HASH = own
        st = block(st)
        if int(st._ablevel) != 2:
            raise SystemExit(f"ERROR TS_NOT_LEVEL2 {path}")
        return st, int(m), path, float(GD.block_time(completed))

    diag = jax.jit(GD.diagnostics)

    def cfl_dt(st):
        """CFL at production's DT (umax * KMAX * DT), whatever h the state was stepped with."""
        _, _, umax = jax.device_get(diag(st))
        return float(umax) * GD.KMAX * DT

    _, ncol = LS.preconditions(GD)
    base = LS.build_consts(GD, ncol)

    # ---- frames and band errors -------------------------------------------------------------------------------------
    def af_rows(a):
        return jnp.concatenate([a[:HALF, :HALF + 1], a[N - HALF:, :HALF + 1]], 0)

    af_scale = (NC * NC) / (N * N) / THETA_SCALE

    @jax.jit
    def frames(st):
        qh = st.state.qh[0]
        sub = jnp.fft.irfft2(qh, s=(N, N))[::M, ::M] / THETA_SCALE                   # float64 (o1.2 casts to f32)
        af = (af_rows(qh) * af_scale).at[HALF, :].set(0).at[:, HALF].set(0)
        return jnp.fft.rfft2(sub), af

    ky = np.fft.fftfreq(NC, 1.0 / NC)
    kx = np.arange(NC // 2 + 1)
    kap = np.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)
    masks = [(kap >= a) & (kap < b) for a, b in BANDS]

    def band_err(x, ref, x0):
        out = []
        for m in masks:
            num = float(np.sqrt(np.mean(np.abs(x[m] - ref[m]) ** 2)))
            den = float(np.sqrt(np.mean(np.abs(ref[m] - x0[m]) ** 2)))
            out.append(num / den if den > 0 else float("nan"))
        return out

    def fr(st):
        s, a = jax.device_get(frames(st))
        return np.asarray(s), np.asarray(a)

    def finite(st):
        return bool(jnp.all(jnp.isfinite(st.state.qh)))

    def rel_state(a, b):
        return float(jnp.linalg.norm(a.state.qh - b.state.qh) / jnp.linalg.norm(b.state.qh))

    # ================================================================================================================
    if args.mode == "ladder":
        st0, member, path, t_state = load(args.ckpt[0])
        res["state"] = {"member": member, "path": path, "t": t_state, "cfl": cfl_dt(st0)}
        print(f"TS_STATE w={W} member={member} t={t_state:.2f} CFL={res['state']['cfl']:.3f}", flush=True)
        f0 = fr(st0)
        ladder = IF.Ladder(GD, ncol, base)
        ends, runs = {}, {}
        res["runs"] = runs

        def arm(name, fn):
            t0 = time.time()
            end = block(fn())
            ok = finite(end)
            runs[name] = {"seconds": round(time.time() - t0, 1), "finite": ok}
            if ok:
                ends[name] = fr(end)
            print(f"TS_ARM w={W} {name} {runs[name]['seconds']:.0f} s finite={ok}", flush=True)
            save()
            return end

        # control: run-time-weight prod at n = 40 vs lean_u1 (bitwise production)
        u1 = LS.make_advance(GD, STRIDE, ncol=ncol, unroll1=True)

        def lean_u1():
            s = st0
            for _ in range(100):
                s = u1(s, base)
            return s
        e_u1 = arm("lean_u1", lean_u1)
        e_p40 = arm("prod40", lambda: ladder.run(st0, "prod", 40))
        res["control"] = {"prod40_vs_lean_u1_rel": rel_state(e_p40, e_u1),
                          "bitwise": bool(jnp.array_equal(e_p40.state.qh, e_u1.state.qh))}
        print(f"TS_CONTROL w={W} rel={res['control']['prod40_vs_lean_u1_rel']:.2e} "
              f"bitwise={res['control']['bitwise']}", flush=True)
        cfl_ref = []
        arm("ref", lambda: ladder.run(st0, "if", NREF, on_chunk=lambda s: cfl_ref.append(cfl_dt(s))))
        res["cfl_ref"] = {"max": max(cfl_ref) if cfl_ref else None, "median": statistics.median(cfl_ref) if cfl_ref else None,
                          "n": len(cfl_ref)}
        # the decision's arms first (IF-AB3), then production's order check
        order = [("if", n) for n in (40, 28, 25, 20, 32, 80)] + [("prod", n) for n in (80, 32, 28, 25, 20)]
        assert sorted(order) == sorted([(s, n) for s in ("prod", "if") for n in NSUBS if (s, n) != ("prod", 40)])
        for scheme, n in order:
            name = f"{scheme}{n}"
            if args.deadline and time.time() > args.deadline - 240:
                runs[name] = {"skipped": "deadline"}
                continue
            arm(name, lambda: ladder.run(st0, scheme, n))
        # band errors vs ref
        errs = {}
        if "ref" in ends:
            rs, ra = ends["ref"]
            for name, (s, a) in ends.items():
                if name == "ref":
                    continue
                errs[name] = {"sub": band_err(s, rs, f0[0]), "af": band_err(a, ra, f0[1])}
            if "prod40" in ends and "if40" in ends:                  # the data change of adopting IF at n = 40
                ps, pa = ends["prod40"]
                is_, ia = ends["if40"]
                errs["if40_vs_prod40"] = {"sub": band_err(is_, ps, f0[0]), "af": band_err(ia, pa, f0[1])}
        res["errors"] = errs
        np.savez(os.path.join(args.out, f"frames_w{W}.npz"),
                 **{f"{k}__sub": v[0] for k, v in ends.items()}, **{f"{k}__af": v[1] for k, v in ends.items()},
                 start__sub=f0[0], start__af=f0[1])
        for name in sorted(errs):
            e = errs[name]
            print(f"TS_ERR w={W} {name:>15s} sub " + " ".join(f"{v:.2e}" for v in e["sub"])
                  + " | af " + " ".join(f"{v:.2e}" for v in e["af"]), flush=True)

    # ================================================================================================================
    else:
        cfls = []
        for spec in args.ckpt:
            st, member, path, t_state = load(spec)
            c = cfl_dt(st)
            cfls.append({"member": member, "t": t_state, "cfl": c})
            print(f"TS_CFL member={member} t={t_state:.2f} CFL={c:.4f}", flush=True)
            del st
        res["cfl_states"] = cfls
        save()

        # float32 IF-AB3 vs float64 IF-AB3, next to lean32ek_fast vs lean_u1 (n = 40, 0.02 lag)
        import lean32_precise as LP
        import lean32_fast as LF
        st0, member, path, t_state = load(args.acc)
        res["acc_state"] = {"member": member, "path": path, "t": t_state}
        c32 = LP.tables(GD)
        fast = LF.make_call(pst, N, NK, DT, STRIDE, mode="ek", u1=True, ring=3)
        if32 = IF.make_call_if32(pst, N, NK, DT, STRIDE)
        u1 = LS.make_advance(GD, STRIDE, ncol=ncol, unroll1=True)
        ladder = IF.Ladder(GD, ncol, base)
        c_dt = ladder.consts(DT)
        rt_prod, rt_if = ladder.adv("prod", STRIDE), ladder.adv("if", STRIDE)
        calls = {"lean_u1": lambda s: u1(s, base), "prod_rt": lambda s: rt_prod(s, c_dt),
                 "if_rt": lambda s: rt_if(s, c_dt), "lean32ek_fast": lambda s: fast(s, c32),
                 "if32": lambda s: if32(s, c32)}
        f0 = fr(st0)
        ends = {}
        for k in ("lean_u1", "if_rt", "lean32ek_fast", "if32"):
            s = st0
            t0 = time.time()
            for _ in range(100):
                s = calls[k](s)
            s = block(s)
            ends[k] = fr(s)
            print(f"TS_ACC_ARM {k} {time.time() - t0:.0f} s finite={finite(s)}", flush=True)
        acc = {"if32_vs_if64": {"sub": band_err(ends["if32"][0], ends["if_rt"][0], f0[0]),
                                "af": band_err(ends["if32"][1], ends["if_rt"][1], f0[1])},
               "fast_vs_u1": {"sub": band_err(ends["lean32ek_fast"][0], ends["lean_u1"][0], f0[0]),
                              "af": band_err(ends["lean32ek_fast"][1], ends["lean_u1"][1], f0[1])}}
        res["f32_if_accuracy"] = acc
        for k, v in acc.items():
            print(f"TS_F32 {k:>14s} sub " + " ".join(f"{x:.2e}" for x in v["sub"]) + " | af "
                  + " ".join(f"{x:.2e}" for x in v["af"]), flush=True)
        save()

        # ms per step
        ms = {}
        for k, f in calls.items():
            x = st0
            for _ in range(3):
                x = f(x)
            block(x)
            ts = []
            for _ in range(3):
                t0 = time.perf_counter()
                for _ in range(args.bench_calls):
                    x = f(x)
                block(x)
                ts.append(time.perf_counter() - t0)
            ms[k] = 1e3 * statistics.median(ts) / (args.bench_calls * STRIDE)
        res["ms_per_step"] = ms
        print("TS_BENCH " + " ".join(f"{k}={v:.4f}" for k, v in ms.items()), flush=True)
        save()

        # batched RFFT lowering (research #4): one rfft2 over a leading dim of 2 vs two calls
        rb = {}
        for dt_name, dt in (("f32", jnp.float32), ("f64", jnp.float64)):
            x2 = jax.random.normal(jax.random.key(0), (2, N, N), dt)
            fb = jax.jit(lambda x: jnp.fft.rfft2(x))
            fs = jax.jit(lambda x: (jnp.fft.rfft2(x[0]), jnp.fft.rfft2(x[1])))
            hb = fb.lower(x2).compile().as_text()
            hs = fs.lower(x2).compile().as_text()
            fft_b = re.findall(r"=\s*(\S+)\s+fft\(", hb)
            fft_s = re.findall(r"=\s*(\S+)\s+fft\(", hs)

            def t_(f):
                r = f(x2)
                block(r)
                ts = []
                for _ in range(5):
                    t0 = time.perf_counter()
                    for _ in range(20):
                        r = f(x2)
                    block(r)
                    ts.append((time.perf_counter() - t0) / 20 * 1e6)
                return statistics.median(ts)
            rb[dt_name] = {"batched_fft_ops": fft_b, "separate_fft_ops": fft_s, "batched_us": t_(fb),
                           "separate_us": t_(fs)}
            bs, ss = jax.device_get(fb(x2)), jax.device_get(fs(x2))
            rb[dt_name]["bitwise"] = bool(np.array_equal(np.asarray(bs[0]), np.asarray(ss[0]))
                                          and np.array_equal(np.asarray(bs[1]), np.asarray(ss[1])))
            print(f"TS_RFFT {dt_name} batched {rb[dt_name]['batched_us']:.1f} us ops {fft_b} | separate "
                  f"{rb[dt_name]['separate_us']:.1f} us ops {fft_s} | bitwise {rb[dt_name]['bitwise']}", flush=True)
        res["batched_rfft"] = rb

    res["wall_s"] = round(time.time() - t_wall, 1)
    save()
    print(f"TS_WORKER_DONE w={W} wall={res['wall_s']:.0f}s", flush=True)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except BaseException as exc:
        first = str(exc).strip().splitlines()[0][:200] if str(exc).strip() else ""
        print(f"ERROR TS_CRASHED {type(exc).__name__}: {first}", flush=True)
        raise
