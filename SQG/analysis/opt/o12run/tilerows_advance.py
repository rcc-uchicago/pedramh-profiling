"""tilerows_advance.py -- o1.2's per-frame `advance` (one ring frame = STRIDE = 40 steps of DT = 2e-4 TU) by
  if32_<n>         the JAX float32 IF-AB3 (if_ab3.make_call_if32) at n steps per frame (h = 2e-4 / n);
  tilerows40       the tile-row step (tilerows/tilerows_step.TileRows; lean32ek numerics) at DT, 40 steps per frame;
  tilerows_if<n>   the tile-row step with IF-AB3 at n steps per frame.
Option C of the 2026-11-12 plan (tile-rows inside o1.2). Predictions + gates: PREDICTIONS_o12_tilerows.md (first).

  advance = make(GD, stride, scheme, rank=..., log=..., gate=...)        # then: st = advance(st), once per frame

Segments. o1.2 calls `self.state = advance(self.state)` once per frame, and replaces self.state by a freshly loaded
checkpoint (level-2 AB3 state with DT history) at each segment start. advance() continues its own state when st is
the object it returned last; any other st starts a segment:
  IF schemes   the 2 float64 IF-AB3 switching steps to h (if_ab3.make_transition with the exact elapsed-time
               factors, as stability_run.py and if_ab3.Ladder), then n - 2 steps: the frame ends on o1.2's 2e-4 grid,
               and every later frame is n steps.
  tilerows40   the loaded DT history as it is (lean32ek_fast's start).
Tile-row state. TileRows keeps hi + lo (compact block) and its 3 tendency slots on its own stream between calls; at a
segment start they are set on the device from the JAX state. After each frame the compact hi, lo go to JAX by
DLPack and are expanded to the (1, N, N/2+1) complex128 qh that o1.2's `coarse` reads. The returned AB3State has
that qh, t + stride DT, tc + stride, and the INPUT's _updates: they are not the true history (TileRows holds it), and
nothing in o1.2 reads them (o1.2 calls coarse(st) and advance(st) only).
Ordering. A tile-row advance() starts with torch.cuda.synchronize(), which is device-wide (JAX and torch share the
primary context): the previous frame's coarse() has consumed qh / hi / lo before TileRows overwrites them.
Gate (rank 0, first segment, gate=True): from the segment's start state (after the switch), the scheme's frame-1 and
frame-4 states vs the float64 reference of the same scheme (IF: make_advance_rt 'if' at h; tilerows40: lean_u1)
next to its float32 JAX twin (IF: make_call_if32 at h; tilerows40: lean32ek_fast). Tile-row arms pass if
rel(arm, ref64) <= min(1e-4, 2 rel(twin32, ref64)) at both frames (the tile-rows probe's G32 / GIF form). if32 arms
are the twin itself and report rel(if32, if64) (info).
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OPT = os.path.dirname(HERE)
for _p in ("seed1", "delta", "taxonomy", "timestep", "tilerows"):
    if os.path.join(OPT, _p) not in sys.path:
        sys.path.insert(0, os.path.join(OPT, _p))

GATE_TOL = 1e-4
GATE_FRAMES = (1, 4)


def parse(scheme, stride):
    """scheme -> (kind 'if32' | 'tr', if_mode, steps per frame)."""
    if scheme.startswith("if32_"):
        n = int(scheme[len("if32_"):])
        kind, if_mode = "if32", True
    elif scheme.startswith("tilerows_if"):
        n = int(scheme[len("tilerows_if"):])
        kind, if_mode = "tr", True
    elif scheme == f"tilerows{stride}":
        n, kind, if_mode = stride, "tr", False
    else:
        raise ValueError(f"unknown scheme {scheme!r} (stride {stride})")
    if if_mode and not 3 <= n <= stride:
        raise ValueError(f"{scheme}: steps per frame {n} outside [3, {stride}]")
    return kind, if_mode, n


def frame_steps(if_mode, n, first):
    """Steps of the solver's own call in a frame: a segment's first IF frame spends 2 of its n on the switch."""
    return n - 2 if (if_mode and first) else n


def full_from_compact(xp, c, n, nk, km):
    """The (2 km + 1, km + 1) compact block (rows k_y = -km..-1 then 0..km, tilerows_step.ROW_IDX order) as the
    (n, nk) rfft half-plane, zero elsewhere: two contiguous row blocks around a zero block, then zero columns. No
    scatter (job 7840805: the complex128 .at[rows].set expand cost ~26 ms per frame -- suspected, timed in the next
    job). xp = numpy or jax.numpy."""
    cc = km + 1
    mid = xp.zeros((n - 2 * km - 1, cc), dtype=c.dtype)
    return xp.pad(xp.concatenate([c[km:], mid, c[:km]], axis=0), ((0, 0), (0, nk - cc)))


def make(GD, stride, scheme, *, rank=0, log=print, gate=False):
    import jax
    import jax.numpy as jnp
    from pyqg_jax import steppers as pst
    import lean_solver as LS
    import lean32_precise as LP
    import lean32_fast as LF
    import if_ab3 as IF
    kind, if_mode, n = parse(scheme, stride)
    N, NK, DT = GD.N, GD.N // 2 + 1, GD.DT
    h = stride * DT / n
    _, ncol = LS.preconditions(GD)
    base = LS.build_consts(GD, ncol)
    c32 = LP.tables(GD)
    if if_mode:
        rate = GD.NU * GD.kap ** 2 + GD.MU * GD.pos / GD.kapi ** 2
        e_h32 = jnp.where(GD.kap <= GD.KMAX, -jnp.expm1(-rate * h), 1.0).astype(jnp.float32)
        edec_err = float(jnp.max(jnp.abs((1.0 - e_h32.astype(jnp.float64)) - IF.decay_table(GD, h))))
        c32h = dict(c32, edec=e_h32, tdt=jnp.float32(h))
        c_h = IF.rt_consts(GD, base, h)
        tr = IF.make_transition(GD, ncol=ncol, scheme="if")
        sw = []
        for h1, h2 in ((DT, DT), (h, DT)):
            b0, b1, b2 = IF.ab3_coeffs(h, h1, h2)
            sw.append((jnp.float64(b0), jnp.float64(b1), jnp.float64(b2), IF.decay_table(GD, h1),
                       IF.decay_table(GD, h1 + h2)))

        def switch(st):
            for b0, b1, b2, f1, f2 in sw:
                st = tr(st, c_h, b0, b1, b2, f1, f2)
            return st
        log(f"O12_TR_SCHEME {scheme} kind={kind} if=1 steps_per_frame={n} h={h:.6e} edec_err={edec_err:.2e}")
        if edec_err > 1e-7:
            raise SystemExit(f"ERROR O12_TR_EDEC {scheme} max|(1 - e_h) - E(h)| = {edec_err:.2e} > 1e-7")
    else:
        switch = None
        log(f"O12_TR_SCHEME {scheme} kind={kind} if=0 steps_per_frame={n} h={h:.6e}")

    # ---- the gate's references (built lazily, rank 0, first segment) --------------------------------------------
    G = {"on": bool(gate and rank == 0), "start": None, "frame": 0, "rows": [], "done": False}

    def rel(a, b):
        a, b = jnp.asarray(a), jnp.asarray(b)
        return float(jnp.linalg.norm((a - b).ravel()) / jnp.linalg.norm(b.ravel()))

    def gate_frame(qh_arm):
        """Called after each frame of the first segment on rank 0: compare at GATE_FRAMES."""
        G["frame"] += 1
        f = G["frame"]
        if f not in GATE_FRAMES:
            return
        steps = frame_steps(if_mode, n, True) + (f - 1) * n
        st0 = G["start"]
        t = time.time()
        if if_mode:
            ref = IF.make_advance_rt(GD, steps, ncol=ncol, scheme="if")(st0, c_h).state.qh[0]
            twin = IF.make_call_if32(pst, N, NK, h, steps)(st0, c32h).state.qh[0]
        else:
            ref = LS.make_advance(GD, steps, ncol=ncol, unroll1=True)(st0, base).state.qh[0]
            twin = LF.make_call(pst, N, NK, DT, steps, mode="ek", u1=True, ring=3)(st0, c32).state.qh[0]
        r_arm, r_twin = rel(qh_arm[0], ref), rel(twin, ref)
        ok = bool(np.isfinite(r_arm) and (kind == "if32" or r_arm <= min(GATE_TOL, 2 * r_twin)))
        G["rows"].append(ok)
        log(f"O12_TR_GATE {scheme} frame={f} steps={steps} rel_arm_ref64={r_arm:.3e} rel_twin32_ref64={r_twin:.3e} "
            f"pass={ok}{' (if32 is the twin: info)' if kind == 'if32' else ''} ({time.time() - t:.1f} s)")
        if f == GATE_FRAMES[-1]:
            G["done"], G["start"] = True, None
            log(f"O12_TR_GATE_VERDICT {scheme} pass={all(G['rows'])}")

    S = {"last": None, "segments": 0}

    # ---- if32: pure JAX ------------------------------------------------------------------------------------------
    if kind == "if32":
        first = IF.make_call_if32(pst, N, NK, h, frame_steps(True, n, True))
        rest = IF.make_call_if32(pst, N, NK, h, n)

        def advance(st):
            if st is not S["last"]:
                S["segments"] += 1
                sw_st = switch(st)
                if G["on"] and not G["done"] and G["start"] is None and S["segments"] == 1:
                    G["start"] = sw_st
                out = first(sw_st, c32h)
            else:
                out = rest(st, c32h)
            if G["on"] and not G["done"] and S["segments"] == 1:
                gate_frame(out.state.qh)
            S["last"] = out
            return out
        return advance

    # ---- tile rows: Warp + cuFFT on torch buffers ---------------------------------------------------------------
    import torch
    import tilerows_step as TS
    e = e_h32 if if_mode else c32["edec"]
    k = np.asarray(c32["kinv"])
    k = k.reshape(k.shape[-2:]) if k.ndim == 3 else k
    tables = {"kinv": TS.to_compact(np.broadcast_to(k, (N, NK))), "edec": TS.to_compact(np.asarray(e)),
              "f0patch": np.concatenate([np.asarray(c32["f0bot"]), np.asarray(c32["f0top"])], 0).astype(np.complex64)}
    solver = TS.TileRows(tables, h, if_mode=if_mode)
    rows = jnp.asarray(TS.ROW_IDX)
    CC = TS.CC
    c64, c128 = jnp.complex64, jnp.complex128
    path = {}

    @jax.jit
    def split(st):
        """The state's compact hi, lo (float32 pair) and history, and max |theta_hat| outside the kept block."""
        th = st.state.qh[0]
        c = th[rows, :CC]
        hi = c.astype(c64)
        lo = (c - hi.astype(c128)).astype(c64)
        tn = st._updates[0].qh[0][rows, :CC].astype(c64)
        to = st._updates[1].qh[0][rows, :CC].astype(c64)
        outside = jnp.max(jnp.abs(th.at[rows, :CC].set(0)))
        return hi, lo, tn, to, outside

    @jax.jit
    def expand_scatter(hi, lo):                                      # job 7840805's expand
        c = hi.astype(c128) + lo.astype(c128)
        return jnp.zeros((1, N, NK), c128).at[0, rows, :CC].set(c)

    @jax.jit
    def expand_concat(hi, lo):
        c = hi.astype(c128) + lo.astype(c128)
        return full_from_compact(jnp, c, N, NK, TS.KM)[None]

    xmode = os.environ.get("O12_TR_EXPAND", "concat")
    if xmode not in ("concat", "scatter"):
        raise SystemExit(f"ERROR O12_TR_EXPAND={xmode!r} not in (concat, scatter)")
    expand = expand_concat if xmode == "concat" else expand_scatter
    timing = os.environ.get("O12_TR_TIMING") == "1"
    T = {"rows": [], "prev_end": None}
    log(f"O12_TR_EXPAND {scheme} mode={xmode} timing={int(timing)}")

    def timing_row(t0, t1, t2, t3, t4):
        """Per-frame wall times (ms): entry sync, segment start, solver run (+ its stream sync), hand-off + expand
        (blocked), and the time spent outside advance() since the previous call (o1.2's coarse dispatch + push)."""
        out = (t0 - T["prev_end"]) * 1e3 if T["prev_end"] is not None else float("nan")
        T["rows"].append(((t1 - t0) * 1e3, (t2 - t1) * 1e3, (t3 - t2) * 1e3, (t4 - t3) * 1e3, out))
        if len(T["rows"]) in (200, 600, 1200):
            cols = list(zip(*T["rows"][50:]))
            med = [float(np.nanmedian(c)) for c in cols]
            log(f"O12_TR_TIMING {scheme} expand={xmode} frames={len(T['rows'])} entry_sync_ms={med[0]:.2f} "
                f"start_ms={med[1]:.2f} run_ms={med[2]:.2f} handoff_expand_ms={med[3]:.2f} outside_ms={med[4]:.2f}")

    def to_torch(x):
        try:
            t = torch.utils.dlpack.from_dlpack(jax.dlpack.to_dlpack(x))
            path.setdefault("jax_to_torch", "dlpack")
            return t
        except Exception as exc:                                     # noqa: BLE001
            path.setdefault("jax_to_torch", f"host ({type(exc).__name__}: {str(exc)[:100]})")
            return torch.as_tensor(np.asarray(x), device="cuda")

    def to_jax(t):
        try:
            x = jax.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(t))
            path.setdefault("torch_to_jax", "dlpack")
            return x
        except Exception as exc:                                     # noqa: BLE001
            path.setdefault("torch_to_jax", f"host ({type(exc).__name__}: {str(exc)[:100]})")
            return jnp.asarray(t.cpu().numpy())

    def start(st):
        S["segments"] += 1
        st_run = switch(st) if if_mode else st
        hi, lo, tn, to, outside = jax.block_until_ready(split(st_run))
        if float(outside) != 0.0:
            raise ValueError(f"O12_TR {scheme}: state not zero outside the kept block (max {float(outside):.3e})")
        torch.cuda.synchronize()
        for dst, src in ((solver.hi, hi), (solver.lo, lo), (solver.T[0], tn), (solver.T[1], to)):
            dst.copy_(to_torch(src))
        solver.slot = [0, 1, 2]
        torch.cuda.synchronize()
        if G["on"] and not G["done"] and G["start"] is None and S["segments"] == 1:
            G["start"] = st_run
        if S["segments"] == 1:
            log(f"O12_TR_PATH {scheme} jax_to_torch={path.get('jax_to_torch')}")
        return frame_steps(if_mode, n, True)

    def advance(st):
        t0 = time.time()
        torch.cuda.synchronize()                                     # the previous frame's coarse() is done
        t1 = time.time()
        steps = start(st) if st is not S["last"] else n
        t2 = time.time()
        solver.run(steps)
        solver.stream.synchronize()
        t3 = time.time()
        a, b = to_jax(solver.hi), to_jax(solver.lo)
        qh = expand(a, b)
        if "logged" not in path:
            path["logged"] = True
            log(f"O12_TR_PATH {scheme} torch_to_jax={path.get('torch_to_jax')}")
            if G["on"]:                                              # rank 0: the two expands agree bit for bit
                eq = bool(jnp.array_equal(expand_concat(a, b), expand_scatter(a, b)))
                log(f"O12_TR_EXPAND_CHECK {scheme} concat_equals_scatter={eq}")
        if timing:
            qh = jax.block_until_ready(qh)
            timing_row(t0, t1, t2, t3, time.time())
        out = pst.AB3State(state=st.state.update(qh=qh), t=st.t + jnp.float32(stride * DT), tc=st.tc + stride,
                           _ablevel=st._ablevel, _updates=st._updates)
        if G["on"] and not G["done"] and S["segments"] == 1:
            gate_frame(qh)
        S["last"] = out
        T["prev_end"] = time.time()
        return out

    return advance
