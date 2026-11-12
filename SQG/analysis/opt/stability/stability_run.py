"""stability_run.py -- tile-rows + IF-AB3 at 25 steps per frame for a long run (PREDICTIONS_stability.md, written first).

  "$PY" stability_run.py --out <dir> --worker <w> --ckpt <member>:<npz> --warp-path <dir> --nvmath-path <dir>
                         [--tu 50] [--nsub 25] [--deadline <epoch s>]
  "$PY" stability_run.py --out <dir> --worker 9 --archive --ckpt <m>:<npz> [--ckpt ...]   (production reference)

Run mode, one archived state on one A100:
  1. load it with o1.2's recipe (float64 AB3 state, history at production's DT);
  2. the two IF-AB3 switching steps to h = 2e-4 / nsub in float64 JAX (if_ab3.make_transition, exact elapsed-time
     factors), so the history is at spacing h;
  3. tilerows_step.TileRows(if_mode=True) at dt = h, with the float32 damping table e = -expm1(-rate h) for h;
  4. --tu time units in frames of nsub steps. Every 12 frames (0.0024 TU; production samples every saved frame,
     0.0025 TU) the full state goes through A-SQG_data.py's own `diagnostics` (SPE, VIE, eps_nu, eps_mu, P,
     the shell spectrum E(k), umax), and CFL_h = umax * KMAX * h is checked against production's CFL_ABORT 0.72
     (warning 0.60). Non-finite values or CFL_h >= 0.72 stop the run (UNSTABLE).
  Every 1 TU: the values and E(k) are stored, and a 256^2 float32 frame (o1.2's sub frame) is saved.
Archive mode: the same diagnostics for each passed archived state (production numerics, DT).
Writes run_w<w>.json (+ run_w<w>.npz) or archive.json (+ archive.npz). Prints STAB_RUN_DONE / STAB_ARCHIVE_DONE.
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
import types                                                         # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OPT = os.path.dirname(HERE)
ANALYSIS = os.path.dirname(OPT)
for p in (os.path.join(OPT, "seed1"), os.path.join(OPT, "delta"), os.path.join(OPT, "taxonomy"),
          os.path.join(OPT, "timestep"), os.path.join(OPT, "tilerows"), HERE, ANALYSIS):
    sys.path.insert(0, p)

H_FRAME = 2e-4
SAMPLE_FRAMES = 12                                                   # CFL / diagnostics every 0.0024 TU
CFL_ABORT, CFL_WARNING = 0.72, 0.60                                  # A-SQG_data.py's


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
    res = {"worker": W, "tu": args.tu, "nsub": args.nsub, "sample_frames": SAMPLE_FRAMES, "cfl_abort": CFL_ABORT}

    def save(status):
        res["status"] = status
        res["wall_s"] = round(time.time() - t_wall, 1)
        with open(os.path.join(args.out, f"run_w{W}.json"), "w") as f:
            json.dump(jsonable(res), f, indent=1)

    st0, member, path, t0 = load(args.ckpt[0])
    v0, s0, u0 = diagnose(st0.state.qh)
    h = H_FRAME / args.nsub
    res["state"] = {"member": member, "path": path, "t": t0, "h": h, "cfl_DT_start": u0 * GD.KMAX * DT,
                    "cfl_h_start": u0 * GD.KMAX * h, "values_start": v0.tolist()}
    print(f"STAB_STATE w={W} member={member} t={t0:.0f} h={h:.3e} CFL_h(start)={u0 * GD.KMAX * h:.4f}", flush=True)

    # 2. the two switching steps (float64, IF-AB3 with exact elapsed-time factors), as if_ab3.Ladder.run
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

    # 3. TileRows at dt = h: float32 damping table for h (lean32_precise.tables' edec with DT -> h)
    c32 = LP.tables(GD)
    rate = GD.NU * GD.kap ** 2 + GD.MU * GD.pos / GD.kapi ** 2
    e_h = np.asarray(jnp.where(GD.kap <= GD.KMAX, -jnp.expm1(-rate * h), 1.0).astype(jnp.float32))
    k = np.asarray(c32["kinv"])
    k = k.reshape(k.shape[-2:]) if k.ndim == 3 else k
    tables = {"kinv": TS.to_compact(np.broadcast_to(k, (N, NK))), "edec": TS.to_compact(e_h),
              "f0patch": np.concatenate([np.asarray(c32["f0bot"]), np.asarray(c32["f0top"])], 0).astype(np.complex64)}
    solver = TS.TileRows(tables, h, if_mode=True)
    solver.pack(np.asarray(st.state.qh[0]), np.asarray(st._updates[0].qh[0]), np.asarray(st._updates[1].qh[0]))
    steps_done = 2

    rows_t = torch.as_tensor(TS.ROW_IDX, device="cuda")
    keep = {}

    def full_qh():
        """The solver's state as a (1, N, NK) complex128 jax array (hi + lo summed in float64, expanded), handed to
        JAX on the GPU through DLPack (host copy as the fallback). keep['t'] holds the torch buffer alive."""
        solver.stream.synchronize()
        th = solver.hi.to(torch.complex128) + solver.lo.to(torch.complex128)
        fullt = torch.zeros((1, N, NK), dtype=torch.complex128, device="cuda")
        fullt[0, rows_t, :TS.CC] = th
        torch.cuda.synchronize()
        keep["t"] = fullt
        try:
            return jax.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(fullt))
        except Exception:                                            # noqa: BLE001
            return jnp.asarray(fullt.cpu().numpy())

    # 4. the long run
    frames_total = int(round(args.tu / H_FRAME))
    frames_per_tu = int(round(1.0 / H_FRAME))
    series, spectra, subs, cfl_samples = [], [], [], []
    n_warn, cfl_max, status = 0, 0.0, "COMPLETE"
    ev0 = torch.cuda.Event(enable_timing=True)
    ev0.record(solver.stream)
    t_run = time.time()
    for fr in range(1, frames_total + 1):
        solver.run(args.nsub)
        steps_done += args.nsub
        last = fr == frames_total
        if fr % SAMPLE_FRAMES == 0 or fr % frames_per_tu == 0 or last:
            qh = full_qh()
            v, s, umax = diagnose(qh)
            cfl = umax * GD.KMAX * h
            cfl_samples.append(cfl)
            cfl_max = max(cfl_max, cfl)
            n_warn += cfl >= CFL_WARNING
            if not (np.all(np.isfinite(v)) and math.isfinite(umax)):
                status = "UNSTABLE_NONFINITE"
            elif cfl >= CFL_ABORT:
                status = "UNSTABLE_CFL"
            if fr % frames_per_tu == 0 or last or status != "COMPLETE":
                tnow = t0 + steps_done * h                          # every step since the switch has size h
                series.append({"tu": steps_done * h, "t": tnow, "values": v.tolist(), "cfl": cfl,
                               "wall_s": round(time.time() - t_run, 1)})
                spectra.append(s)
                subs.append(np.asarray(sub(qh)))
                print(f"STAB_TU w={W} tu={steps_done * h:7.3f} SPE={v[0]:.6g} VIE={v[1]:.6g} P={v[4]:.5g} "
                      f"CFL_h={cfl:.4f} max={cfl_max:.4f} warn={n_warn} wall={time.time() - t_run:.0f}s", flush=True)
                res.update(series=series, cfl_max=cfl_max, n_cfl_samples=len(cfl_samples), n_cfl_warn=int(n_warn),
                           steps=steps_done)
                save("RUNNING")
            if status != "COMPLETE":
                print(f"STAB_UNSTABLE w={W} {status} at tu={steps_done * h:.4f} CFL_h={cfl:.4f}", flush=True)
                break
        if args.deadline and time.time() > args.deadline - 300:
            status = "DEADLINE"
            print(f"STAB_DEADLINE w={W} at tu={steps_done * h:.3f}", flush=True)
            break
    ev1 = torch.cuda.Event(enable_timing=True)
    ev1.record(solver.stream)
    ev1.synchronize()
    run_s = time.time() - t_run
    cs = np.asarray(cfl_samples)
    res.update(series=series, cfl_max=cfl_max, n_cfl_samples=len(cs), n_cfl_warn=int(n_warn), steps=steps_done,
               cfl_quantiles={q: float(np.quantile(cs, q)) for q in (0.5, 0.9, 0.99)} if len(cs) else {},
               wall_run_s=round(run_s, 1), ms_per_step_wall=1e3 * run_s / max(steps_done - 2, 1),
               gpu_ms_total=ev0.elapsed_time(ev1))
    np.savez(os.path.join(args.out, f"run_w{W}.npz"), spectra=np.asarray(spectra), subs=np.asarray(subs),
             cfl=cs, final_hi=solver.hi.cpu().numpy(), final_lo=solver.lo.cpu().numpy())
    save(status)
    print(f"STAB_RUN_DONE w={W} status={status} tu={steps_done * h:.3f} cfl_max={cfl_max:.4f} "
          f"warn={n_warn}/{len(cs)} ms/step(wall)={res['ms_per_step_wall']:.4f}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except BaseException as exc:
        first = str(exc).strip().splitlines()[0][:200] if str(exc).strip() else ""
        print(f"ERROR STAB_CRASHED {type(exc).__name__}: {first}", flush=True)
        raise
