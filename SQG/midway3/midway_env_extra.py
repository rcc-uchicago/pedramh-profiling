"""midway_env_extra.py -- the Midway3-specific half of Job 1 (MIDWAY3_INROW_PROMPT.md §4); the cluster-agnostic half
is SQG/polaris/sqg_env_probe.py. Run by sqg_env_probe_midway.sh on a 4-GPU compute node, never on a login node.

  1. torch: CUDA available, every visible device's name; jax: jax.devices() (expect 4 x H100).
  2. Warp 1.17.0 from --warp-path: wp.init(), the libmathdx flag ("mathdx": 1 is required -- without cuFFTDx the
     tile FFT falls back to a scalar path ~4.6x slower and every speed number would be invalid), then a 4096-point
     wp.tile_fft row kernel (warp_kernels.row, the kernel warp_fft_probe.py timed on A100) on a 4096^2 complex64
     array at block_dim 256, checked against torch.fft.fft (relative L2 <= 1e-5) and timed (information).
  3. nvmath from --nvmath-path: `from nvmath.bindings import cufft`, cufft.get_version() (11002 on Polaris: the
     wheel's cuFFT 11.0.2), and the libcufft / libcudart / libmathdx files actually mapped into this process.
Token: SQG_MIDWAY_ENV_OK (all of the above) or ERROR MIDWAY_ENV_<what>. Writes <out>/midway_env_extra.json.
"""
import argparse
import json
import os
import site
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
WARP_DIR = os.path.join(os.path.dirname(HERE), "analysis", "opt", "warp")
N = 4096


def mapped_libs(names=("libcufft", "libcudart", "libmathdx", "libnvJitLink", "libcublas", "libcudnn")):
    """Paths of the shared libraries matching `names` that this process has mapped (from /proc/self/maps)."""
    out = set()
    with open("/proc/self/maps") as fh:
        for line in fh:
            path = line.rstrip("\n").split(" ")[-1]
            if any(os.path.basename(path).startswith(n) for n in names):
                out.add(path)
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--warp-path", required=True)
    ap.add_argument("--nvmath-path", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    R = {"errors": [], "env": {"LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", ""),
                               "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "")}}
    fails = []

    def err(what, e):
        fails.append(what)
        R["errors"].append({"where": what, "error": f"{type(e).__name__}: {e}"[:1000]})
        print(f"ERROR MIDWAY_ENV_{what}: {type(e).__name__}: {str(e)[:300]}", flush=True)

    # ---- 1. torch + jax (torch first, as o1.2 imports it) -------------------------------------------------------------
    import numpy as np
    import torch
    import jax
    R["env"].update(numpy=np.__version__, torch=torch.__version__, jax=jax.__version__,
                    torch_cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version())
    gpus = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    jdev = [f"{d.platform}:{d.device_kind}" for d in jax.devices()]
    R["torch_gpus"], R["jax_devices"] = gpus, jdev
    print(f"torch {torch.__version__} gpus={gpus}", flush=True)
    print(f"jax {jax.__version__} devices={jdev}", flush=True)
    if len(gpus) != 4 or not all("H100" in g for g in gpus):
        err("TORCH_GPUS", RuntimeError(f"expected 4 x H100, torch sees {gpus}"))
    if len(jdev) != 4 or not all("H100" in d for d in jdev):
        err("JAX_DEVICES", RuntimeError(f"expected 4 x H100, jax sees {jdev}"))

    # ---- 2. Warp + cuFFTDx tile FFT -------------------------------------------------------------------------------
    try:
        sys.path.insert(0, a.warp_path)
        sys.path.insert(0, WARP_DIR)
        import warp as wp
        wp.config.enable_backward = False
        wp.init()
        core = wp._src.context.runtime.core
        rt = wp._src.context.runtime
        R["warp"] = {"version": wp.__version__, "file": wp.__file__, "mathdx": int(core.wp_is_mathdx_enabled()),
                     "toolkit": getattr(rt, "toolkit_version", None), "driver": getattr(rt, "driver_version", None)}
        print(f"warp {json.dumps(R['warp'], default=str)}", flush=True)
        if wp.__version__ != "1.17.0":
            err("WARP_VERSION", RuntimeError(wp.__version__))
        if R["warp"]["mathdx"] != 1:
            err("NO_MATHDX", RuntimeError("wp_is_mathdx_enabled() != 1: tile FFT would be the scalar fallback"))
        import warp_kernels as WK
        rng = np.random.default_rng(0)
        x = torch.from_numpy((rng.standard_normal((N, N), dtype=np.float32)
                              + 1j * rng.standard_normal((N, N), dtype=np.float32)).astype(np.complex64)).cuda()
        y = torch.empty_like(x)
        wx = wp.from_torch(torch.view_as_real(x), dtype=wp.vec2f)
        wy = wp.from_torch(torch.view_as_real(y), dtype=wp.vec2f)
        ws = wp.stream_from_torch(torch.cuda.current_stream())
        k = WK.row(wp.vec2f, 1)
        t0 = time.time()
        wp.launch_tiled(k, dim=[N], inputs=[wx], outputs=[wy], block_dim=256, stream=ws)
        torch.cuda.synchronize()
        R["tile_fft"] = {"compile_and_first_launch_s": round(time.time() - t0, 2)}
        ref = torch.fft.fft(x, dim=-1)
        rel = float(torch.linalg.vector_norm(y - ref) / torch.linalg.vector_norm(ref))
        ts = []
        for _ in range(5):
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            for _ in range(20):
                wp.launch_tiled(k, dim=[N], inputs=[wx], outputs=[wy], block_dim=256, stream=ws)
            e1.record()
            torch.cuda.synchronize()
            ts.append(e0.elapsed_time(e1) / 20 * 1e3)
        R["tile_fft"].update(rel_l2_vs_torch=rel, med_us=statistics.median(ts), batches_us=ts)
        print(f"tile_fft 4096-pt rows f32 B256: rel_l2_vs_torch={rel:.2e} med={statistics.median(ts):.1f} us", flush=True)
        if not rel <= 1e-5:
            err("TILE_FFT_WRONG", RuntimeError(f"rel_l2={rel:.3e} > 1e-5"))
    except Exception as e:                                            # noqa: BLE001
        err("WARP", e)

    # ---- 3. nvmath cuFFT bindings + what is mapped --------------------------------------------------------------------
    try:
        site.addsitedir(a.nvmath_path)
        from nvmath.bindings import cufft as cb
        import nvmath
        R["nvmath"] = {"version": getattr(nvmath, "__version__", None), "file": nvmath.__file__,
                       "cufft_version": int(cb.get_version())}
        print(f"nvmath {json.dumps(R['nvmath'])}", flush=True)
    except Exception as e:                                            # noqa: BLE001
        err("NVMATH", e)
    R["mapped_libs"] = mapped_libs()
    for p in R["mapped_libs"]:
        print(f"mapped {p}", flush=True)
    if not any("libcufft" in os.path.basename(p) for p in R["mapped_libs"]):
        err("NO_CUFFT_MAPPED", RuntimeError("no libcufft in /proc/self/maps"))

    R["failures"] = fails
    with open(os.path.join(a.out, "midway_env_extra.json"), "w") as fh:
        json.dump(R, fh, indent=1, default=str)
    if fails:
        print(f"ERROR MIDWAY_ENV_FAILED {','.join(fails)}", flush=True)
        sys.exit(1)
    print(f"SQG_MIDWAY_ENV_OK gpus=4xH100 mathdx=1 tile_fft_rel={R['tile_fft']['rel_l2_vs_torch']:.1e} "
          f"cufft={R['nvmath']['cufft_version']}", flush=True)


if __name__ == "__main__":
    main()
