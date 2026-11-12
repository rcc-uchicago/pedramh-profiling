"""warp_fft_probe.py -- can NVIDIA Warp carry the SQG solver's custom FFT? (PREDICTIONS_warp.md, written first)

One A100, everything 4096 x 4096 (the solver's N). Phases:
  W0 env      torch 2.3.1, then jax 0.4.28, then warp 1.17.0 (from --warp-path), one process; libmathdx flag.
  W1 copy     attainable bandwidth: torch y.copy_(x), complex64 and complex128.
  W2 cuFFT    torch fft(dim=-1) / fft(dim=0) / fft2 in complex64 and complex128; jax fft2 / irfft2 / rfft2 anchors.
  W3 rows     Warp row FFT (tile (R, 4096)): vec2f R in {1, 2} x block_dim in {128, 256, 512}; inverse and the
              scalar fallback (enable_mathdx_fft False) at (1, 256); vec2d block_dim in {256, 512}.
  W4 columns  Warp column FFT, K columns per block (warp_kernels.col: strided / tstore / tview), and NVIDIA's
              shared-memory transpose kernel (16 x 16, 32 x 32) for its row -> transpose -> row recipe.
  W5 fused    the step's row-kernel proxy (warp_kernels.fused3): 2 loads, 2 inverse FFTs, conj(a)*b, 1 forward.
  W6 accuracy rel L2 vs numpy complex128 of the same input (32 rows / 32 columns / the full 2-D transform).
  W7 bridge   warp_xla.py under jax.jit and lax.scan; host overhead per custom call; 2 kernels in one jit; torch
              still works afterwards.
  W8 verdict  byte-model projection and the decision rule of PREDICTIONS_warp.md.
Every config runs in its own try block: a compile or launch failure (e.g. shared memory) is recorded as data.
Writes --out (JSON) after every phase. Tokens: WARP_ENV_OK, WARP_BRIDGE_OK / WARP_BRIDGE_FAIL, WARP_VERDICT,
WARP_PROBE_DONE failures=N. Bandwidth = unique bytes read + written / time; MB = 1e6 B; MB / (GB/s) = ms.
"""
import argparse
import json
import os
import statistics
import sys
import time
import traceback

N = 4096
MB = 1e6
HERE = os.path.dirname(os.path.abspath(__file__))
KEPT = 2 * 1365 + 1                                                  # |ky| <= 1365: rows of the kept block
KEPT_X = 1366                                                        # 0 <= kx <= 1365


def byte_model(csize, tsize, compensated):
    """MB per fused kernel (A, B, C), PREDICTIONS_warp.md: complex element size csize, real table size tsize."""
    blk = KEPT * KEPT_X * csize
    tab = KEPT * KEPT_X * tsize
    half = N * KEPT_X * csize
    w = N * KEPT * csize
    n_state = 2 if compensated else 1                                # hi + lo, or the float64 state
    a = blk + tab + w + half                                         # read theta (hi) + kinv; write W, Theta
    b = (w + half) + 2 * half                                        # read W, Theta rows; write 2 product half rows
    c = 2 * half + (n_state + 2) * blk + tab + (n_state + 1) * blk   # read products, state, t_n, t_o, decay; write
    return a / MB, b / MB, c / MB


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--warp-path", required=True)
    ap.add_argument("--deadline", type=float, default=0.0, help="epoch seconds; late sweeps are skipped")
    args = ap.parse_args()
    R = {"phases": {}, "errors": [], "n": N}
    failures = 0
    t_start = time.time()

    def save():
        with open(args.out, "w") as f:
            json.dump(R, f, indent=1, default=str)

    def late(margin=420):
        return bool(args.deadline) and time.time() > args.deadline - margin

    def err(where, e):
        R["errors"].append({"where": where, "error": f"{type(e).__name__}: {e}"[:2000]})
        print(f"ERROR WARP_{where}: {type(e).__name__}: {str(e)[:300]}", flush=True)

    # ---- W0 env (torch first, as o1.2 imports it) ------------------------------------------------------------------
    import numpy as np
    import torch
    import jax
    import jax.numpy as jnp
    from jax import lax
    sys.path.insert(0, args.warp_path)
    sys.path.insert(0, HERE)
    import warp as wp
    wp.config.enable_backward = False
    wp.init()
    import warp_kernels as WK
    import warp_xla as WX
    core = wp._src.context.runtime.core
    rt = wp._src.context.runtime
    env = {"torch": torch.__version__, "jax": jax.__version__, "warp": wp.__version__, "warp_file": wp.__file__,
           "mathdx": int(core.wp_is_mathdx_enabled()), "toolkit": getattr(rt, "toolkit_version", None),
           "driver": getattr(rt, "driver_version", None), "gpu": torch.cuda.get_device_name(0),
           "jax_devices": [str(d) for d in jax.devices()], "xla_flags": os.environ.get("XLA_FLAGS", "")}
    try:
        import jaxlib
        env["jaxlib"] = jaxlib.__version__
    except Exception:                                                # noqa: BLE001
        pass
    R["phases"]["env"] = env
    dev = wp.get_cuda_device(0)
    tstream = torch.cuda.current_stream()
    ws = wp.stream_from_torch(tstream)                               # one stream: torch events time Warp too
    xs = torch.arange(8, dtype=torch.float32, device="cuda")
    ys = torch.zeros_like(xs)
    wp.launch(WK.double_k, dim=8, inputs=[wp.from_torch(xs)], outputs=[wp.from_torch(ys)], stream=ws)
    torch.cuda.synchronize()
    ok_env = bool(torch.equal(ys, 2 * xs)) and float(jnp.sum(jnp.arange(4.0))) == 6.0
    print(f"env {json.dumps(env, default=str)}", flush=True)
    if ok_env:
        print("WARP_ENV_OK", flush=True)
    else:
        failures += 1
        print("ERROR WARP_ENV: trivial kernel or jax check failed", flush=True)
    save()

    # ---- data -----------------------------------------------------------------------------------------------------
    rng = np.random.default_rng(0)
    x_np = (rng.standard_normal((N, N), dtype=np.float32)
            + 1j * rng.standard_normal((N, N), dtype=np.float32)).astype(np.complex64)
    x64 = torch.from_numpy(x_np).cuda()
    x128 = x64.to(torch.complex128)
    y64, z64 = torch.empty_like(x64), torch.empty_like(x64)
    y128, z128 = torch.empty_like(x128), torch.empty_like(x128)

    def wv(t):
        """Warp vec2 view of a torch complex tensor (zero copy)."""
        return wp.from_torch(torch.view_as_real(t), dtype=wp.vec2f if t.dtype == torch.complex64 else wp.vec2d)

    W = {k: wv(t) for k, t in (("x64", x64), ("y64", y64), ("z64", z64), ("x128", x128), ("y128", y128),
                                 ("z128", z128))}

    def tstats(ts):
        return {"med_us": statistics.median(ts), "min_us": min(ts), "max_us": max(ts)}

    def time_gpu(fn, reps=20, batches=5, warm=3):
        for _ in range(warm):
            fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(batches):
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record(tstream)
            for _ in range(reps):
                fn()
            e1.record(tstream)
            e1.synchronize()
            ts.append(e0.elapsed_time(e1) / reps * 1e3)
        return tstats(ts)

    def time_jax(f, *a, reps=20, batches=5, warm=3):
        for _ in range(warm):
            f(*a).block_until_ready()
        ts = []
        for _ in range(batches):
            t0 = time.perf_counter()
            for _ in range(reps):
                r = f(*a)
            r.block_until_ready()
            ts.append((time.perf_counter() - t0) / reps * 1e6)
        return tstats(ts)

    def gbs(nbytes, st):
        return nbytes / (st["med_us"] * 1e-6) / 1e9

    B64, B128 = 2 * N * N * 8, 2 * N * N * 16                        # one full read + write of a 4096^2 array

    # ---- W1 copy --------------------------------------------------------------------------------------------------
    s64 = time_gpu(lambda: y64.copy_(x64))
    s128 = time_gpu(lambda: y128.copy_(x128))
    copy = {"c64": dict(s64, GBs=gbs(B64, s64)), "c128": dict(s128, GBs=gbs(B128, s128))}
    R["phases"]["copy"] = copy
    print(f"copy c64 {s64['med_us']:.1f} us = {copy['c64']['GBs']:.0f} GB/s; c128 {s128['med_us']:.1f} us = "
          f"{copy['c128']['GBs']:.0f} GB/s", flush=True)
    save()

    # ---- W2 cuFFT baselines ---------------------------------------------------------------------------------------
    cu = R["phases"]["cufft"] = {}
    for name, fn, nb in (("torch_row_c64", lambda: torch.fft.fft(x64, dim=-1), B64),
                         ("torch_col_c64", lambda: torch.fft.fft(x64, dim=0), B64),
                         ("torch_fft2_c64", lambda: torch.fft.fft2(x64), 2 * B64),
                         ("torch_row_c128", lambda: torch.fft.fft(x128, dim=-1), B128),
                         ("torch_col_c128", lambda: torch.fft.fft(x128, dim=0), B128),
                         ("torch_fft2_c128", lambda: torch.fft.fft2(x128), 2 * B128)):
        try:
            s = time_gpu(fn)
            cu[name] = dict(s, GBs=gbs(nb, s))                       # fft2: both passes' bytes
        except Exception as e:                                       # noqa: BLE001
            err(f"CUFFT_{name}", e)
    xj = jnp.asarray(x_np)
    hj = jnp.asarray(np.fft.rfft2(x_np.real).astype(np.complex64))
    rj = jnp.asarray(x_np.real)
    for name, f, a in (("jax_fft2_c64", jax.jit(jnp.fft.fft2), xj),
                       ("jax_irfft2_c64", jax.jit(lambda h: jnp.fft.irfft2(h, s=(N, N))), hj),
                       ("jax_rfft2_c64", jax.jit(jnp.fft.rfft2), rj)):
        try:
            cu[name] = time_jax(f, a)
        except Exception as e:                                       # noqa: BLE001
            err(f"CUFFT_{name}", e)
    for k, v in cu.items():
        print(f"cufft {k}: {v['med_us']:.1f} us" + (f" ({v['GBs']:.0f} GB/s, all passes)" if "GBs" in v else ""),
              flush=True)
    save()

    # ---- the Warp sweep helper ------------------------------------------------------------------------------------
    def smem_of(kernel, B):
        try:
            return int(kernel.module.load(dev, B).get_kernel_hooks(kernel).forward_smem_bytes)
        except Exception:                                            # noqa: BLE001
            return None

    def run_cfg(name, kernel, dims, B, inputs, outputs, nbytes, ref_copy, table):
        """First launch (= compile, timed), then time; recorded in table[name]. Returns the stats or None."""
        rec = {"block_dim": B, "dims": list(dims)}
        table[name] = rec
        if late():
            rec["skipped"] = "deadline"
            save()
            return None
        try:
            t0 = time.time()
            wp.launch_tiled(kernel, dim=list(dims), inputs=inputs, outputs=outputs, block_dim=B, stream=ws)
            torch.cuda.synchronize()
            rec["compile_s"] = round(time.time() - t0, 2)
            rec["smem_bytes"] = smem_of(kernel, B)
            s = time_gpu(lambda: wp.launch_tiled(kernel, dim=list(dims), inputs=inputs, outputs=outputs,
                                                 block_dim=B, stream=ws))
            rec.update(s)
            rec["GBs"] = gbs(nbytes, s)
            rec["frac_copy"] = rec["GBs"] / copy[ref_copy]["GBs"]
            print(f"{name}: {s['med_us']:.1f} us, {rec['GBs']:.0f} GB/s ({100 * rec['frac_copy']:.0f}% of copy), "
                  f"smem {rec['smem_bytes']}, compile {rec['compile_s']} s", flush=True)
        except Exception as e:                                       # noqa: BLE001
            rec["error"] = f"{type(e).__name__}: {e}"[:1500]
            print(f"{name}: FAILED {rec['error'][:300]}", flush=True)
            s = None
        save()
        return s

    def ok(table, name):
        return name in table and "med_us" in table[name]

    kernels = {}

    # ---- W3 rows --------------------------------------------------------------------------------------------------
    rows = R["phases"]["rows"] = {}
    for RR, B in ((1, 256), (1, 128), (1, 512), (2, 256)):
        k = WK.row(wp.vec2f, RR)
        kernels[f"row_f32_R{RR}_B{B}"] = (k, (N // RR,), B, "f")
        run_cfg(f"row_f32_R{RR}_B{B}", k, (N // RR,), B, [W["x64"]], [W["y64"]], B64, "c64", rows)
    k = WK.row(wp.vec2f, 1, inverse=True)
    kernels["irow_f32_R1_B256"] = (k, (N,), 256, "f")
    run_cfg("irow_f32_R1_B256", k, (N,), 256, [W["x64"]], [W["y64"]], B64, "c64", rows)
    for B in (256, 512):
        k = WK.row(wp.vec2d, 1)
        kernels[f"row_f64_R1_B{B}"] = (k, (N,), B, "d")
        run_cfg(f"row_f64_R1_B{B}", k, (N,), B, [W["x128"]], [W["y128"]], B128, "c128", rows)
    kfb = None
    try:                                                             # module options are read from wp.config
        wp.config.enable_mathdx_fft = False
        kfb = WK.row_fallback(wp.vec2f, 1)
        wp.launch_tiled(kfb, dim=[N], inputs=[W["x64"]], outputs=[W["z64"]], block_dim=256, stream=ws)
        torch.cuda.synchronize()
    except Exception as e:                                           # noqa: BLE001
        kfb = None
        err("FALLBACK_COMPILE", e)
    finally:
        wp.config.enable_mathdx_fft = True
    if kfb is not None:
        run_cfg("fallback_row_f32_R1_B256", kfb, (N,), 256, [W["x64"]], [W["z64"]], B64, "c64", rows)
        ref = torch.fft.fft(x64, dim=-1)
        rows["fallback_rel_l2_vs_torch"] = float(torch.linalg.vector_norm(z64 - ref) / torch.linalg.vector_norm(ref))

    # ---- W4 columns -----------------------------------------------------------------------------------------------
    cols = R["phases"]["cols"] = {}
    views = {"f": (W["x64"], W["y64"], "c64", B64), "d": (W["x128"], W["y128"], "c128", B128)}
    col_cfgs = [("f", "strided", 1, 256), ("f", "strided", 2, 256), ("f", "strided", 4, 512),
                ("f", "tstore", 1, 256), ("f", "tstore", 2, 256), ("f", "tstore", 4, 512),
                ("f", "tview", 4, 512), ("f", "tview", 8, 1024),
                ("d", "strided", 1, 256), ("d", "tstore", 1, 256), ("d", "tview", 4, 512)]
    for p, mode, K, B in col_cfgs:
        name = f"col_{'f32' if p == 'f' else 'f64'}_{mode}_K{K}_B{B}"
        xv, yv, rc, nb = views[p]
        if mode == "tview":
            xv, yv = xv.transpose(), yv.transpose()
        try:
            k = WK.col(wp.vec2f if p == "f" else wp.vec2d, K, mode)
        except Exception as e:                                       # noqa: BLE001
            cols[name] = {"error": f"{type(e).__name__}: {e}"[:1500]}
            continue
        kernels[name] = (k, (N // K,), B, p)
        run_cfg(name, k, (N // K,), B, [xv], [yv], nb, rc, cols)
    for p, TT in (("f", 16), ("f", 32), ("d", 16)):
        name = f"transpose_{'f32' if p == 'f' else 'f64'}_T{TT}"
        xv, yv, rc, nb = views[p]
        k = WK.transpose(wp.vec2f if p == "f" else wp.vec2d, TT)
        kernels[name] = (k, (N // TT, N // TT), 256, p)
        run_cfg(name, k, (N // TT, N // TT), 256, [xv], [yv], nb, rc, cols)

    # ---- W5 fused row kernel ---------------------------------------------------------------------------------------
    fused = R["phases"]["fused"] = {}
    for p, B in (("f", 256), ("f", 512), ("d", 256)):
        name = f"fused3_{'f32' if p == 'f' else 'f64'}_B{B}"
        a, b, z = (W["x64"], W["y64"], W["z64"]) if p == "f" else (W["x128"], W["y128"], W["z128"])
        nb = 3 * N * N * (8 if p == "f" else 16)
        k = WK.fused3(wp.vec2f if p == "f" else wp.vec2d)
        kernels[name] = (k, (N,), B, p)
        run_cfg(name, k, (N,), B, [a, b], [z], nb, "c64" if p == "f" else "c128", fused)

    # ---- W6 accuracy (rel L2 vs numpy complex128) -----------------------------------------------------------------
    acc = R["phases"]["accuracy"] = {}

    def rel(a, ref):
        a = np.asarray(a, dtype=np.complex128)
        return float(np.linalg.norm(a - ref) / np.linalg.norm(ref))

    def launch(name, ins, outs):
        k, dims, B, _p = kernels[name]
        wp.launch_tiled(k, dim=list(dims), inputs=ins, outputs=outs, block_dim=B, stream=ws)

    try:
        xr = x_np.astype(np.complex128)
        ref_rows = np.fft.fft(xr[:32], axis=-1)
        ref_irows = np.fft.ifft(xr[:32], axis=-1) * N
        ref_cols = np.fft.fft(xr[:, :32], axis=0)
        acc["torch_row_c64"] = rel(torch.fft.fft(x64, dim=-1)[:32].cpu().numpy(), ref_rows)
        acc["torch_col_c64"] = rel(torch.fft.fft(x64, dim=0)[:, :32].cpu().numpy(), ref_cols)
        acc["torch_row_c128"] = rel(torch.fft.fft(x128, dim=-1)[:32].cpu().numpy(), ref_rows)
        acc["torch_col_c128"] = rel(torch.fft.fft(x128, dim=0)[:, :32].cpu().numpy(), ref_cols)
        for name, (k, dims, B, p) in kernels.items():
            table = rows if "row" in name and not name.startswith("fused") else cols
            if name.startswith(("transpose", "fused")) or not ok(table, name):
                continue
            xin, yout = (W["x64"], y64) if p == "f" else (W["x128"], y128)
            try:
                yout.zero_()
                if "row" in name:
                    launch(name, [xin], [wv(yout)])
                    torch.cuda.synchronize()
                    acc[name] = rel(yout[:32].cpu().numpy(), ref_irows if name.startswith("irow") else ref_rows)
                else:
                    mode = name.split("_")[2]
                    if mode == "tview":
                        launch(name, [xin.transpose()], [wv(yout).transpose()])
                    else:
                        launch(name, [xin], [wv(yout)])
                    torch.cuda.synchronize()
                    got = yout[:32, :].cpu().numpy().T if mode == "tstore" else yout[:, :32].cpu().numpy()
                    acc[name] = rel(got, ref_cols)
            except Exception as e:                                   # noqa: BLE001
                err(f"ACC_{name}", e)
        ref2 = np.fft.fft2(xr)
        acc["torch_fft2_c64"] = rel(torch.fft.fft2(x64).cpu().numpy(), ref2)
        acc["jax_fft2_c64"] = rel(np.asarray(jnp.fft.fft2(xj)), ref2)
        if ok(rows, "row_f32_R1_B256") and ok(cols, "col_f32_strided_K1_B256"):
            launch("row_f32_R1_B256", [W["x64"]], [W["z64"]])
            launch("col_f32_strided_K1_B256", [W["z64"]], [W["y64"]])
            torch.cuda.synchronize()
            acc["warp_2d_row_colstrided_c64"] = rel(y64.cpu().numpy(), ref2)
        if ok(rows, "row_f32_R1_B256") and ok(cols, "transpose_f32_T16"):
            launch("row_f32_R1_B256", [W["x64"]], [W["y64"]])
            launch("transpose_f32_T16", [W["y64"]], [W["z64"]])
            launch("row_f32_R1_B256", [W["z64"]], [W["y64"]])
            torch.cuda.synchronize()
            acc["warp_2d_nvidia_3pass_c64"] = rel(y64.cpu().numpy().T, ref2)
        if ok(fused, "fused3_f32_B256"):
            b_np = (rng.standard_normal((N, N), dtype=np.float32)
                    + 1j * rng.standard_normal((N, N), dtype=np.float32)).astype(np.complex64)
            y64.copy_(torch.from_numpy(b_np).cuda())
            launch("fused3_f32_B256", [W["x64"], W["y64"]], [W["z64"]])
            torch.cuda.synchronize()
            ia = np.fft.ifft(xr[:16], axis=-1) * N
            ib = np.fft.ifft(b_np[:16].astype(np.complex128), axis=-1) * N
            acc["fused3_f32_B256"] = rel(z64[:16].cpu().numpy(), np.fft.fft(np.conj(ia) * ib, axis=-1))
    except Exception as e:                                           # noqa: BLE001
        err("ACCURACY", e)
        failures += 1
    print("accuracy " + json.dumps({k: f"{v:.2e}" for k, v in acc.items()}), flush=True)
    save()

    # ---- W7 bridge (warp_xla.py under jax.jit / lax.scan) ---------------------------------------------------------
    br = R["phases"]["bridge"] = {}
    bridge_ok = False
    try:
        if not ok(rows, "row_f32_R1_B256"):
            raise RuntimeError("row_f32_R1_B256 did not run; nothing to bridge")
        rowcall = WX.warp_call(kernels["row_f32_R1_B256"][0], out_specs=[((N, N, 2), "float32")],
                               launch_dims=(N, 256), block_dim=256)
        xr2 = jnp.asarray(np.stack([x_np.real, x_np.imag], -1).astype(np.float32))

        def cplx(a):
            return lax.complex(a[..., 0], a[..., 1])

        f1 = jax.jit(lambda a: rowcall(a)[0])
        y1 = f1(xr2).block_until_ready()
        ref1 = jnp.fft.fft(cplx(xr2), axis=-1)
        br["jit_rel_l2"] = float(jnp.linalg.norm(cplx(y1) - ref1) / jnp.linalg.norm(ref1))
        br["stats_after_jit"] = dict(WX.STATS)

        def body_w(c, _):
            return rowcall(c)[0] * jnp.float32(1 / 64), None

        def body_j(c, _):
            z = jnp.fft.fft(cplx(c), axis=-1) * jnp.float32(1 / 64)
            return jnp.stack([jnp.real(z), jnp.imag(z)], -1), None

        S = 50
        fw = jax.jit(lambda c: lax.scan(body_w, c, None, length=S)[0])
        fj = jax.jit(lambda c: lax.scan(body_j, c, None, length=S)[0])
        cb0 = WX.STATS["callbacks"]
        yw = fw(xr2).block_until_ready()
        br["scan_callbacks_first_call"] = WX.STATS["callbacks"] - cb0
        yj = fj(xr2).block_until_ready()
        br["scan_rel_l2"] = float(jnp.linalg.norm(yw - yj) / jnp.linalg.norm(yj))
        tw, tj = time_jax(fw, xr2, reps=3, batches=3, warm=1), time_jax(fj, xr2, reps=3, batches=3, warm=1)
        br["scan_us_per_iter_warp"] = tw["med_us"] / S
        br["scan_us_per_iter_jnp"] = tj["med_us"] / S

        M, L = 1024, 2000                                            # host overhead: tiny kernel, 2,000 iterations
        tiny = WX.warp_call(WK.half_k, out_specs=[((M, 2), "float32")], launch_dims=(M,), block_dim=256)
        c0 = jnp.ones((M, 2), jnp.float32)
        gw = jax.jit(lambda c: lax.scan(lambda c, _: (tiny(c)[0] * jnp.float32(2), None), c, None, length=L)[0])
        gj = jax.jit(lambda c: lax.scan(lambda c, _: (c * jnp.float32(0.5) * jnp.float32(2), None), c, None,
                                        length=L)[0])
        ow, oj = gw(c0).block_until_ready(), gj(c0).block_until_ready()
        br["tiny_exact"] = bool(jnp.array_equal(ow, oj))
        cb1 = WX.STATS["callbacks"]
        sw, sj = time_jax(gw, c0, reps=3, batches=3, warm=1), time_jax(gj, c0, reps=3, batches=3, warm=1)
        br["tiny_callbacks_per_iter"] = (WX.STATS["callbacks"] - cb1) / (10 * L)    # 1 warm-up + 9 timed calls
        br["tiny_us_per_iter_warp"] = sw["med_us"] / L
        br["tiny_us_per_iter_jnp"] = sj["med_us"] / L
        br["overhead_us_per_call"] = (sw["med_us"] - sj["med_us"]) / L

        if ok(cols, "col_f32_strided_K1_B256"):                      # two Warp kernels in one jit = a 2-D FFT
            colcall = WX.warp_call(kernels["col_f32_strided_K1_B256"][0], out_specs=[((N, N, 2), "float32")],
                                   launch_dims=(N, 256), block_dim=256)
            f2 = jax.jit(lambda a: colcall(rowcall(a)[0])[0])
            y2 = f2(xr2).block_until_ready()
            ref2j = jnp.fft.fft2(cplx(xr2))
            br["two_kernel_rel_l2"] = float(jnp.linalg.norm(cplx(y2) - ref2j) / jnp.linalg.norm(ref2j))
            br["two_kernel_us"] = time_jax(f2, xr2)["med_us"]
        a = torch.randn(512, 512, device="cuda", dtype=torch.float64)
        br["torch_after_ok"] = bool(torch.allclose((a @ a.T).cpu(), a.cpu() @ a.cpu().T, rtol=1e-10, atol=1e-10))
        br["stats"] = dict(WX.STATS)
        bridge_ok = bool(br["jit_rel_l2"] <= 1e-6 and br["scan_rel_l2"] <= 1e-5 and br["overhead_us_per_call"] <= 50
                         and br["torch_after_ok"] and WX.STATS["errors"] == 0)
    except Exception as e:                                           # noqa: BLE001
        err("BRIDGE", e)
        br["traceback"] = traceback.format_exc()[-3000:]
    br["BRIDGE_OK"] = bridge_ok
    print("bridge " + json.dumps({k: v for k, v in br.items() if k != "traceback"}, default=str), flush=True)
    if bridge_ok:
        print("WARP_BRIDGE_OK", flush=True)
    else:
        why = WX.STATS.get("last_error") or (R["errors"][-1]["error"] if R["errors"] else "a threshold")
        print(f"WARP_BRIDGE_FAIL {why[:300]}", flush=True)
    save()

    # ---- W8 projection + verdict (PREDICTIONS_warp.md's decision rule) -------------------------------------------
    def best(table, prefix):
        c = [(v["GBs"], k) for k, v in table.items() if k.startswith(prefix) and isinstance(v, dict) and "GBs" in v]
        return max(c) if c else (None, None)

    proj = R["phases"]["projection"] = {}
    a32, b32, c32 = byte_model(8, 4, True)
    a64, b64, c64 = byte_model(16, 8, False)
    proj["bytes_MB_f32"] = [a32, b32, c32, a32 + b32 + c32]
    proj["bytes_MB_f64"] = [a64, b64, c64, a64 + b64 + c64]
    bwc32, kc32 = best(cols, "col_f32")
    bwf32, kf32 = best(fused, "fused3_f32")
    bwc64, kc64 = best(cols, "col_f64")
    bwr64, kr64 = best(rows, "row_f64")
    bwr32, kr32 = best(rows, "row_f32_R")                            # excludes irow_ and fallback_
    proj.update({"BW_col_f32": [bwc32, kc32], "BW_fused_row_f32": [bwf32, kf32], "BW_col_f64": [bwc64, kc64],
                 "BW_row_f64": [bwr64, kr64], "BW_row_f32": [bwr32, kr32]})
    t32 = (a32 + c32) / bwc32 + b32 / bwf32 if bwc32 and bwf32 else None          # MB / (GB/s) = ms
    t64 = (a64 + c64) / bwc64 + b64 / bwr64 if bwc64 and bwr64 else None
    proj["t_step_ms_f32"], proj["t_step_ms_f64"] = t32, t64
    e_col, e_row = acc.get(kc32), acc.get(kr32)
    acc_ok32 = bool(e_col is not None and e_row is not None and "torch_col_c64" in acc
                    and e_col <= 2 * acc["torch_col_c64"] and e_row <= 2 * acc["torch_row_c64"])
    proj["acc_check_f32"] = {"col": [kc32, e_col, acc.get("torch_col_c64")],
                             "row": [kr32, e_row, acc.get("torch_row_c64")], "ok": acc_ok32}
    go32 = bool(bridge_ok and acc_ok32 and t32 is not None and t32 <= 1.6)
    go64 = bool(bridge_ok and t64 is not None and t64 <= 2.70)
    proj["verdict"] = {"f32": "GO_F32" if go32 else "NO_GO", "f64": "GO_F64" if go64 else "NO_GO",
                       "inputs": {"bridge_ok": bridge_ok, "acc_ok32": acc_ok32, "t32": t32, "t64": t64}}

    def fmt(v):
        return "n/a" if v is None else f"{v:.3f}"

    print(f"projection f32 {fmt(t32)} ms ({a32 + b32 + c32:.1f} MB; col {kc32} {fmt(bwc32)} GB/s, fused {kf32} "
          f"{fmt(bwf32)} GB/s); f64 {fmt(t64)} ms ({a64 + b64 + c64:.1f} MB; col {kc64} {fmt(bwc64)}, row {kr64} "
          f"{fmt(bwr64)})", flush=True)
    print(f"WARP_VERDICT f32={proj['verdict']['f32']} f64={proj['verdict']['f64']}", flush=True)
    R["elapsed_s"] = round(time.time() - t_start, 1)
    R["failures"] = failures
    save()
    print(f"WARP_PROBE_DONE failures={failures}", flush=True)


if __name__ == "__main__":
    main()
