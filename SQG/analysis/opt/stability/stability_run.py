"""stability_run.py -- tile-rows + IF-AB3 at 25 steps per frame for a long run (PREDICTIONS_stability.md, written first;
revised after the code / science review wf_8c6d4a2d-cde, before submission).

  "$PY" stability_run.py --out <dir> --worker <w> --ckpt <member>:<npz> --warp-path <dir> --nvmath-path <dir>
                         [--tu 50] [--nsub 25] [--deadline <epoch s>]
  "$PY" stability_run.py --out <dir> --worker 9 --archive --ckpt <m>:<npz> [--ckpt ...]   (production reference)

Run mode, one archived state on one A100:
  1. load it with o1.2's recipe (float64 AB3 state, history at production's DT);
  2. the two IF-AB3 switching steps to h = 2e-4 / nsub in float64 JAX (if_ab3.make_transition with the elapsed-time
     factors E(h1), E(h1 + h2)), so the history is at spacing h;
  3. GATE AT h (TileRows' first use at any dt but DT): from the switched state, TileRows(if_mode) at h vs the float64
     IF-AB3 reference at h (if_ab3.make_advance_rt 'if' + rt_consts at h), next to the JAX float32 IF-AB3 at h
     (if_ab3.make_call_if32 with the h tables), after 1 and 40 steps: rel(TileRows, if64) <= min(1e-4,
     2 rel(if32, if64)) at both (the tile-rows probe's GIF form); and the damping table: max |(1 - e_h) - E(h)| <= 1e-7.
     Octave-band errors of the 40-step increment are reported. A failed gate stops the worker (status GATE_FAIL).
  4. --tu time units in frames of nsub steps. Every 12 frames (0.0024 TU; production samples every 0.0025 TU) the full
     state goes through A-SQG_data.py's own `diagnostics` (SPE, VIE, eps_nu, eps_mu, P, E(k), umax); CFL_h =
     umax * KMAX * h. Every sample's values + umax are stored (the fine series). The run stops only on a non-finite
     sample (UNSTABLE_NONFINITE) or CFL_h >= 1.0 (UNSTABLE_CFL_HARD); crossings of production's 0.72 abort and 0.60
     warning are counted (production policy, reported separately).
  Every 1 TU (frame count): E(k) and a 256^2 float32 sub frame; run_w<w>.json and run_w<w>.npz are rewritten, so a
  run killed mid-way leaves consistent partial data (status RUNNING / CRASHED / DEADLINE).
Archive mode: the same diagnostics for each passed archived state (production numerics, DT).
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
import traceback                                                     # noqa: E402
import types                                                         # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OPT = os.path.dirname(HERE)
ANALYSIS = os.path.dirname(OPT)
for p in (os.path.join(OPT, "seed1"), os.path.join(OPT, "delta"), os.path.join(OPT, "taxonomy"),
          os.path.join(OPT, "timestep"), os.path.join(OPT, "tilerows"), HERE, ANALYSIS):
    sys.path.insert(0, p)

H_FRAME = 2e-4
SAMPLE_FRAMES = 12                                                   # diagnostics every 0.0024 TU
CFL_ABORT, CFL_WARNING = 0.72, 0.60                                  # A-SQG_data.py's (counted, not a stop)
CFL_HARD = 1.0                                                       # this run's only CFL stop
GATE_TOL, EDEC_TOL = 1e-4, 1e-7
BANDS = ((1, 2), (2, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 128), (128, 256), (256, 512), (512, 1024),
         (1024, 1366))


def jsonable(x):
    """NaN / inf become the strings 'nan' / 'inf' (strict JSON); analyze_stability.py coerces them back."""
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
    ap.add_argument("--archive", action="store_true")
    ap.add_argument("--warp-path", default="")
    ap.add_argument("--nvmath-path", default="")
    ap.add_argument("--tu", type=float, default=50.0)
    ap.add_argument("--nsub", type=int, default=25)
    ap.add_argument("--deadline", type=float, default=0.0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import numpy as np
    import torch
    import profile_solver as PS
    GD = PS.import_asqg()
    import jax
    import jax.numpy as jnp
    from pyqg_jax import steppers as pst
    block = jax.block_until_ready
    N, NK, DT, M = GD.N, GD.N // 2 + 1, GD.DT, PS.M
    template = GD.sqg.init(GD.p, jax.random.key(GD.SEED_INITIAL_CONDITION))
    diag = jax.jit(lambda qh: GD.diagnostics(types.SimpleNamespace(state=types.SimpleNamespace(qh=qh))))
    sub = jax.jit(lambda qh: (jnp.fft.irfft2(qh[0], s=(N, N))[::M, ::M] / 8.0).astype(jnp.float32))

    def config_hash(member):
        cfg = dict(GD.CONFIG, member=member, seed_initial_condition=member)
        return hashlib.sha256(json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def load(spec):
        m, path = spec.split(":", 1)
        own, GD.CONFIG_HASH = GD.CONFIG_HASH, config_hash(int(m))
        try:
            st, completed = GD.load_checkpoint(template, path)
        finally:
            GD.CONFIG_HASH = own
        st = block(st)
        if int(st._ablevel) != 2:
            raise SystemExit(f"ERROR STAB_NOT_LEVEL2 {path}")
        return st, int(m), path, float(GD.block_time(completed))

    def diagnose(qh):
        values, spectrum, umax = jax.device_get(diag(qh))
        return np.asarray(values, dtype=float), np.asarray(spectrum, dtype=float), float(umax)

    # ---- archive mode: production reference ------------------------------------------------------------------------
    if args.archive:
        rows, spectra = [], []
        for spec in args.ckpt:
            st, m, path, t = load(spec)
            v, s, umax = diagnose(st.state.qh)
            rows.append({"member": m, "t": t, "values": v.tolist(), "umax": umax, "cfl_DT": umax * GD.KMAX * DT})
            spectra.append(s)
            print(f"STAB_ARCHIVE member={m} t={t:.0f} SPE={v[0]:.6g} VIE={v[1]:.6g} CFL_DT={umax * GD.KMAX * DT:.4f}",
                  flush=True)
        with open(os.path.join(args.out, "archive.json"), "w") as f:
            json.dump(jsonable({"states": rows, "value_names": ["SPE", "VIE", "eps_nu", "eps_mu", "P"]}), f, indent=1)
        np.savez(os.path.join(args.out, "archive.npz"), spectra=np.asarray(spectra))
        print(f"STAB_ARCHIVE_DONE n={len(rows)}", flush=True)
        return

    # ---- run mode -------------------------------------------------------------------------------------------------
    sys.path.insert(0, args.warp_path)
    site.addsitedir(args.nvmath_path)
    import warp as wp
    wp.config.enable_backward = False
    wp.init()
    import lean_solver as LS
    import lean32_precise as LP
    import if_ab3 as IF
    import tilerows_step as TS
    W = args.worker
    res = {"worker": W, "tu": args.tu, "nsub": args.nsub, "sample_frames": SAMPLE_FRAMES, "cfl_abort": CFL_ABORT,
           "cfl_hard": CFL_HARD, "kmax": float(GD.KMAX)}
    S = {"series": [], "spectra": [], "subs": [], "fine": [], "status": "STARTING"}   # fine: tu_frame, 5 values, umax

    def save(status):
        res["status"] = S["status"] = status
        res["wall_s"] = round(time.time() - t_wall, 1)
        res["series"] = S["series"]
        with open(os.path.join(args.out, f"run_w{W}.json"), "w") as f:
            json.dump(jsonable(res), f, indent=1)
        fine = np.asarray(S["fine"], dtype=float).reshape(-1, 7)
        extra = {}
        if "solver" in S:
            extra = {"final_hi": S["solver"].hi.cpu().numpy(), "final_lo": S["solver"].lo.cpu().numpy()}
        np.savez(os.path.join(args.out, f"run_w{W}.npz"), spectra=np.asarray(S["spectra"]), subs=np.asarray(S["subs"]),
                 fine=fine, **extra)

    try:
        st0, member, path, t0 = load(args.ckpt[0])
        v0, s0, u0 = diagnose(st0.state.qh)
        h = H_FRAME / args.nsub
        res["state"] = {"member": member, "path": path, "t": t0, "h": h, "cfl_DT_start": u0 * GD.KMAX * DT,
                        "cfl_h_start": u0 * GD.KMAX * h, "values_start": v0.tolist()}
        print(f"STAB_STATE w={W} member={member} t={t0:.0f} h={h:.3e} CFL_h(start)={u0 * GD.KMAX * h:.4f}", flush=True)

        # 2. the two switching steps (float64, IF-AB3, exact elapsed-time factors), as if_ab3.Ladder.run
        _, ncol = LS.preconditions(GD)
        base = LS.build_consts(GD, ncol)
        tr = IF.make_transition(GD, ncol=ncol, scheme="if")
        c_h = IF.rt_consts(GD, base, h)
        st = st0
        for h1, h2 in ((DT, DT), (h, DT)):
            b0, b1, b2 = IF.ab3_coeffs(h, h1, h2)
            st = tr(st, c_h, jnp.float64(b0), jnp.float64(b1), jnp.float64(b2), IF.decay_table(GD, h1),
                    IF.decay_table(GD, h1 + h2))
        st = block(st)

        # 3. tables at h, and the gate at h
        c32 = LP.tables(GD)
        rate = GD.NU * GD.kap ** 2 + GD.MU * GD.pos / GD.kapi ** 2
        e_h32 = jnp.where(GD.kap <= GD.KMAX, -jnp.expm1(-rate * h), 1.0).astype(jnp.float32)
        e_h = np.asarray(e_h32)
        edec_err = float(jnp.max(jnp.abs((1.0 - e_h32.astype(jnp.float64)) - IF.decay_table(GD, h))))
        k = np.asarray(c32["kinv"])
        k = k.reshape(k.shape[-2:]) if k.ndim == 3 else k
        tables = {"kinv": TS.to_compact(np.broadcast_to(k, (N, NK))), "edec": TS.to_compact(e_h),
                  "f0patch": np.concatenate([np.asarray(c32["f0bot"]), np.asarray(c32["f0top"])], 0).astype(np.complex64)}
        solver = TS.TileRows(tables, h, if_mode=True)
        S["solver"] = solver
        c32_h = dict(c32, edec=e_h32, tdt=jnp.float32(h))
        th_s = np.asarray(st.state.qh[0])
        tn_s, to_s = np.asarray(st._updates[0].qh[0]), np.asarray(st._updates[1].qh[0])

        ky = np.fft.fftfreq(N, 1.0 / N)
        kap = np.sqrt(ky[:, None] ** 2 + np.arange(NK)[None, :] ** 2)
        bmask = [(kap >= a) & (kap < b) for a, b in BANDS]

        def rel(a, b):
            return float(np.linalg.norm(a - b) / np.linalg.norm(b))

        gate = {"edec_err": edec_err, "edec_ok": bool(edec_err <= EDEC_TOL)}
        for n in (1, 40):
            ref = np.asarray(block(IF.make_advance_rt(GD, n, ncol=ncol, scheme="if")(st, c_h)).state.qh[0])
            f32 = np.asarray(block(IF.make_call_if32(pst, N, NK, h, n)(st, c32_h)).state.qh[0])
            solver.pack(th_s, tn_s, to_s)
            solver.run(n)
            trs = solver.theta()
            r_tr, r_32 = rel(trs, ref), rel(f32, ref)
            ok = bool(np.all(np.isfinite(trs)) and r_tr <= min(GATE_TOL, 2 * r_32))
            g = {"rel_tilerows_vs_if64": r_tr, "rel_if32_vs_if64": r_32, "pass": ok}
            if n == 40:
                d = ref - th_s
                g["band_err_tilerows"] = [float(np.linalg.norm((trs - ref)[mk]) / max(np.linalg.norm(d[mk]), 1e-300))
                                          for mk in bmask]
                g["band_err_if32"] = [float(np.linalg.norm((f32 - ref)[mk]) / max(np.linalg.norm(d[mk]), 1e-300))
                                      for mk in bmask]
            gate[str(n)] = g
            print(f"STAB_GATE w={W} n={n} rel(TileRows,if64)={r_tr:.3e} rel(if32,if64)={r_32:.3e} pass={ok}", flush=True)
        gate["pass"] = bool(gate["edec_ok"] and gate["1"]["pass"] and gate["40"]["pass"])
        res["gate_h"] = gate
        print(f"STAB_GATE w={W} edec_err={edec_err:.2e} pass={gate['pass']}", flush=True)
        if not gate["pass"]:
            print(f"ERROR STAB_GATE_FAIL w={W}", flush=True)
            save("GATE_FAIL")
            raise SystemExit(3)
        solver.pack(th_s, tn_s, to_s)
        steps_done = 2

        rows_t = torch.as_tensor(TS.ROW_IDX, device="cuda")
        keep = {}

        def full_qh():
            """The solver's state as a (1, N, NK) complex128 jax array (hi + lo in float64, expanded), on the GPU via
            DLPack (host copy as the fallback; the path taken is recorded once)."""
            solver.stream.synchronize()
            th = solver.hi.to(torch.complex128) + solver.lo.to(torch.complex128)
            # 2-D, then [None] in JAX: a (1, N, NK) tensor's DLPack strides give JAX a non-default layout, which
            # from_dlpack rejects (smoke 7840661 fell back to a host copy, +0.15 ms/step)
            fullt = torch.zeros((N, NK), dtype=torch.complex128, device="cuda")
            fullt[rows_t, :TS.CC] = th
            torch.cuda.synchronize()
            keep["t"] = fullt
            try:
                out = jax.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(fullt))[None]
                res.setdefault("full_qh_path", "dlpack")
                return out
            except Exception as e:                                   # noqa: BLE001
                res.setdefault("full_qh_path", f"host_copy ({type(e).__name__}: {str(e)[:120]})")
                return jnp.asarray(fullt.cpu().numpy())[None]

        # 4. the long run
        frames_total = int(round(args.tu / H_FRAME))
        frames_per_tu = int(round(1.0 / H_FRAME))
        cfl_all = []
        n60 = n72 = streak72 = longest72 = 0
        cfl_max = 0.0
        status = "COMPLETE"
        save("RUNNING")
        t_run = time.time()
        for fr in range(1, frames_total + 1):
            solver.run(args.nsub)
            steps_done += args.nsub
            per_tu = fr % frames_per_tu == 0
            if fr % SAMPLE_FRAMES == 0 or per_tu or fr == frames_total:
                qh = full_qh()
                v, s, umax = diagnose(qh)
                cfl = umax * GD.KMAX * h
                tu_f = fr * H_FRAME
                S["fine"].append([tu_f] + v.tolist() + [umax])
                finite = bool(np.all(np.isfinite(v)) and math.isfinite(umax))
                if not finite:
                    status = "UNSTABLE_NONFINITE"
                else:
                    cfl_all.append(cfl)
                    cfl_max = max(cfl_max, cfl)
                    n60 += cfl >= CFL_WARNING
                    if cfl >= CFL_ABORT:
                        n72 += 1
                        streak72 += 1
                        longest72 = max(longest72, streak72)
                    else:
                        streak72 = 0
                    if cfl >= CFL_HARD:
                        status = "UNSTABLE_CFL_HARD"
                if per_tu or fr == frames_total or status != "COMPLETE":
                    S["series"].append({"tu_frame": tu_f, "tu": steps_done * h, "t": t0 + steps_done * h,
                                        "values": v.tolist(), "cfl": cfl, "wall_s": round(time.time() - t_run, 1)})
                    S["spectra"].append(s)
                    S["subs"].append(np.asarray(sub(qh)))
                    res.update(cfl_max=cfl_max, n_cfl_samples=len(cfl_all), n_cfl_ge_060=int(n60),
                               n_cfl_ge_072=int(n72), longest_run_ge_072=int(longest72), steps=steps_done)
                    print(f"STAB_TU w={W} tu={tu_f:7.3f} SPE={v[0]:.6g} VIE={v[1]:.6g} P={v[4]:.5g} CFL_h={cfl:.4f} "
                          f"max={cfl_max:.4f} >=0.60:{n60} >=0.72:{n72} wall={time.time() - t_run:.0f}s", flush=True)
                    save("RUNNING" if status == "COMPLETE" else status)
                if status != "COMPLETE":
                    print(f"STAB_UNSTABLE w={W} {status} at tu={tu_f:.4f} CFL_h={cfl:.4f}", flush=True)
                    break
            if args.deadline and time.time() > args.deadline - 300:
                status = "DEADLINE"
                print(f"STAB_DEADLINE w={W} at tu={fr * H_FRAME:.3f}", flush=True)
                break
        run_s = time.time() - t_run
        cs = np.asarray(cfl_all)
        res.update(cfl_max=cfl_max, n_cfl_samples=len(cs), n_cfl_ge_060=int(n60), n_cfl_ge_072=int(n72),
                   longest_run_ge_072=int(longest72), steps=steps_done,
                   cfl_quantiles={q: float(np.quantile(cs, q)) for q in (0.5, 0.9, 0.99)} if len(cs) else {},
                   wall_run_s=round(run_s, 1), ms_per_step_wall=1e3 * run_s / max(steps_done - 2, 1))
        save(status)
        print(f"STAB_RUN_DONE w={W} status={status} tu={steps_done * h:.3f} cfl_max={cfl_max:.4f} "
              f">=0.60:{n60}/{len(cs)} >=0.72:{n72} ms/step(wall)={res['ms_per_step_wall']:.4f} "
              f"full_qh={res.get('full_qh_path')}", flush=True)
    except SystemExit:
        raise
    except BaseException as exc:                                     # noqa: BLE001
        res["crash"] = {"error": f"{type(exc).__name__}: {exc}"[:2000], "tb": traceback.format_exc()[-3000:]}
        try:
            save("CRASHED")
        except Exception:                                            # noqa: BLE001
            pass
        raise


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except BaseException as exc:
        first = str(exc).strip().splitlines()[0][:200] if str(exc).strip() else ""
        print(f"ERROR STAB_CRASHED {type(exc).__name__}: {first}", flush=True)
        raise
