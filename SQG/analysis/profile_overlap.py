"""profile_overlap.py -- Job B: do the JAX solver and the PyTorch UViT step overlap on one GPU?

o1.2-TrainUViT_m16_tendency.py claims the S = 480 solver steps of each optimizer step "are dispatched asynchronously
by JAX before each ML step and run alongside it on the same GPU", yet the measured 2.6 s/step = 2.2 s solver + 0.4 s
network, i.e. a sum. Hypothesis (two independent code reviews): pyqg_jax's AB3 `lax.switch` on the runtime AB level
makes every solver step sync with the host, so each jitted 40-step call blocks the Python thread until done.

This harness reproduces o1.2's per-step loop on ONE GPU (no DDP) and times, per optimizer step (wall clock):
  ml_only            : sample batch (pageable host buffer gather + H2D) + UViT fwd/bwd, clip, AdamW, 2x loss.item()
  solver_only_<v>    : OnlinePairs.advance equivalent: flush previous 12 frames (dlpack -> ring -> 2 pageable D2H
                       copies per pair), dispatch 12 x (40 steps + coarse)
  serial_<v>         : o1.2 order: advance() then the ML step (what production does)
  thread_<v>         : solver chunk dispatched AND waited on in a background thread while the main thread trains;
                       the main thread joins it before the next step (same 480 solver steps per optimizer step)
for v in {ref (pyqg_jax AB3, as production), fixed (AB level fixed at 2; bitwise-identical, see profile_solver.py)}.
Network time is also measured with CUDA events on torch's stream (never torch.cuda.synchronize: it would wait for
JAX's streams too). Overlap efficiency = (solver_only + ml_only - combined) / min(solver_only, ml_only).

Same solver/model/optimizer as production: A-SQG_data.py imported as in o1.2 (--member 1), real AB3 checkpoint
(level 2); ConvNextUViT(n_feat=32, depths=[2,2,4,4,8], k=7, drop_path=0.1, n_attn=4, heads=8, residual), batch 32,
AdamW(5e-4, wd 0.05, betas 0.9/0.95), clip 1.0, strict fp32 (torch 2.3: allow_tf32 = False for matmul and cuDNN).
Differences: host buffer 8192 pairs (not 262144; gather cost from a 137 GB array may differ), no DDP all-reduce,
no wandb, no OneCycle (LR schedule does not affect cost).

The data step inside the ML step (sample(): host gather + H2D) is also timed on its own, on the host and with a CUDA
event between it and the network (NVTX range "sample"), so the network time can be reported without it.

Writes <out>/profile_overlap.json; prints OVERLAP_OK.

Polaris overrides (profile_overlap_polaris.pbs; each one unset = the Midway path above, unchanged):
  SQG_ASQG_PY, SQG_TRAINING_ROOT, SQG_INIT=random   as in profile_solver.py
  SQG_BATCH (default 32), SQG_BUF (default 8192)    batch and host-buffer pairs (an A100 has 40 GB)

Per-phase sync audit (profile_phases_polaris.pbs -> summarize_nsys_phases.py; each one unset = the path above):
  SQG_NVTX=1     NVTX ranges around every phase of the optimizer step, on the thread that runs it:
                   step > advance > flush | dispatch > solver, coarse   (serial; thread: step > join, flush)
                   step > sample, forward, backward (zero_grad + backward), clip, optimizer, loss_item, harness_sync
                   worker thread (thread arms): dispatch > solver, coarse; wait.   solver_only: step > advance, wait
                 `sample` is pushed unconditionally, as before. harness_sync (the CUDA-event wait that times the step)
                 exists only in the harness, not in o1.2.
  --arms a,b     run only these configurations (default: all seven, in the order below)
  --nsys         exactly one arm: drain the pipeline after the warm-up, then cudaProfilerStart/Stop around the timed
                 steps, for `nsys profile --capture-range=cudaProfilerApi`. Writes profile_overlap_nsys_<arm>.json
                 (its timings include profiler overhead; take timings from the plain run).
"""
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import argparse
import contextlib
import importlib.util
import io
import json
import signal
import statistics
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ASQG_PY = os.environ.get("SQG_ASQG_PY", "/project/pedramh/weidong/SQG/fixed_ring/A-SQG_data.py")
CKPT_NAME = "ckpt_t0100.00.npz"
STRIDE, CALLS, M = 40, 12, 16            # o1.2: STRIDE, S // STRIDE, subsampling
LAG = 100                                # ring frames between x and y (o1.2 LAG at DT_SAVE = 0.02)
THETA_SCALE = 8.0
BATCH = int(os.environ.get("SQG_BATCH", 32))
BUF = int(os.environ.get("SQG_BUF", 8192))
NVTX = os.environ.get("SQG_NVTX", "0") == "1"
ARMS = ["ml_only", "solver_only_ref", "solver_only_fixed", "serial_ref", "serial_fixed", "thread_ref", "thread_fixed"]


def patch_training_root(src, root):
    """`src` with its one `TRAINING_ROOT = ...` line replaced by `TRAINING_ROOT = Path(r"<root>")`.

    Same as profile_solver.patch_training_root (the two harnesses are standalone copies): A-SQG_data.py mkdirs under
    the hard-coded Midway TRAINING_ROOT at import and hashes its own bytes, so the file itself must not be edited.
    """
    lines = src.split("\n")
    hits = [i for i, line in enumerate(lines) if line.startswith("TRAINING_ROOT = ")]
    if len(hits) != 1:
        raise RuntimeError(f"expected one TRAINING_ROOT line in {ASQG_PY}, found {len(hits)}")
    lines[hits[0]] = f'TRAINING_ROOT = Path(r"{root}")'
    return "\n".join(lines)


def import_asqg():
    """A-SQG_data.py as a module, as o1.2 (--member 1, silenced, signal handlers restored); see profile_solver.py."""
    argv, handlers = sys.argv, {s: signal.getsignal(s) for s in (signal.SIGUSR1, signal.SIGTERM)}
    sys.argv = [ASQG_PY, "--member", "1"]
    try:
        spec = importlib.util.spec_from_file_location("asqg_data", ASQG_PY)
        mod = importlib.util.module_from_spec(spec)
        root = os.environ.get("SQG_TRAINING_ROOT")
        with contextlib.redirect_stdout(io.StringIO()):
            if root:
                with open(ASQG_PY, "rb") as fh:
                    src = patch_training_root(importlib.util.decode_source(fh.read()), root)
                exec(compile(src, ASQG_PY, "exec", dont_inherit=True), mod.__dict__)
            else:
                spec.loader.exec_module(mod)
    finally:
        sys.argv = argv
        for s, h in handlers.items():
            signal.signal(s, h)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20, help="timed optimizer steps per configuration")
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", default=",".join(ARMS), help="comma-separated subset of " + ",".join(ARMS))
    ap.add_argument("--nsys", action="store_true", help="cudaProfilerStart/Stop around the timed steps (one arm)")
    args = ap.parse_args()
    arms = args.arms.split(",")
    if set(arms) - set(ARMS):
        ap.error(f"unknown arms {sorted(set(arms) - set(ARMS))}")
    if args.nsys and len(arms) != 1:
        ap.error("--nsys captures exactly one arm (one nsys process per arm)")
    os.makedirs(args.out, exist_ok=True)

    GD = import_asqg()
    import jax
    import jax.numpy as jnp
    from jax import lax
    from pyqg_jax import steppers as pst
    import torch
    import torch.nn.functional as F
    sys.path.insert(0, os.path.dirname(HERE))
    from model_UViT import ConvNextUViT

    torch.backends.cuda.matmul.allow_tf32 = False          # strict fp32 (o1.2 uses the torch>=2.9 fp32_precision API)
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(0)
    dev = torch.device("cuda:0")

    def rng(name):
        """NVTX range of the per-phase audit (SQG_NVTX=1); a no-op otherwise. NVTX stacks are per thread."""
        return torch.cuda.nvtx.range(name) if NVTX else contextlib.nullcontext()
    N = GD.N
    n = N // M

    # ---- solver (as profile_solver.py) -----------------------------------------------------------------------
    template = GD.sqg.init(GD.p, jax.random.key(GD.SEED_INITIAL_CONDITION))
    init = os.environ.get("SQG_INIT", "checkpoint")
    if init == "random":                                 # as profile_solver.py: same code per step, AB3 level 2
        st0 = GD.advance(template, 2)
    elif init == "checkpoint":
        st0, _ = GD.load_checkpoint(template, GD.CHECKPOINT_DIR / CKPT_NAME)
    else:
        raise ValueError(f"SQG_INIT={init!r}: expected random or checkpoint")
    assert int(st0._ablevel) == 2
    dt = GD.sm.stepper.dt
    c1, c2, c3 = (23 / 12) * dt, (-16 / 12) * dt, (5 / 12) * dt

    def inner_fixed(state, _):
        updates = GD.sm.model.get_updates(state.state)
        updates = updates.update(qh=updates.qh + GD.F0_hat)
        up, upp = state._updates
        new = pst._nostep_tree_map(lambda v, u, u_p, u_pp: v + (c1 * u) + (c2 * u_p) + (c3 * u_pp),
                                   state.state, updates, up, upp)
        state = pst.AB3State(state=new, t=state.t + jnp.float32(dt), tc=state.tc + 1, _ablevel=jnp.uint8(2),
                             _updates=(pst._map_state_remove_nostep(updates), up))
        qh = GD.sm.model.postprocess_state(state.state).qh * GD.decay
        return state.update(state=state.state.update(qh=qh)), None

    advs = {"ref": jax.jit(lambda st: lax.scan(GD.inner, st, None, length=STRIDE)[0]),
            "fixed": jax.jit(lambda st: lax.scan(inner_fixed, st, None, length=STRIDE)[0])}
    coarse = jax.jit(lambda st: jnp.fft.irfft2(st.state.qh[0], s=(N, N))[::M, ::M].astype(jnp.float32))

    class Online:
        """o1.2 OnlinePairs (ring on GPU, pageable host FIFO, flush-then-dispatch), without segment loads."""

        def __init__(self, adv):
            self.adv, self.state = adv, jax.tree_util.tree_map(lambda x: jnp.array(x, copy=True), st0)
            self.ring = torch.empty(LAG + 1, 1, n, n, device=dev)
            self.bx = torch.empty(BUF, 1, n, n)
            self.by = torch.empty_like(self.bx)
            self.k, self.n_pairs, self.pending = 0, 0, []

        def _push(self, frame):
            f = torch.from_dlpack(frame)[None] / THETA_SCALE
            self.ring[self.k % (LAG + 1)] = f
            if self.k >= LAG:
                j = self.n_pairs % BUF
                self.bx[j].copy_(self.ring[(self.k - LAG) % (LAG + 1)])
                self.by[j].copy_(f)
                self.n_pairs += 1
            self.k += 1

        def flush(self):
            with rng("flush"):
                for fr in self.pending:
                    self._push(fr)
                self.pending = []

        def dispatch(self):
            with rng("dispatch"):
                for _ in range(CALLS):
                    with rng("solver"):
                        self.state = self.adv(self.state)
                    with rng("coarse"):
                        self.pending.append(coarse(self.state))

        def advance(self):
            with rng("advance"):
                self.flush()
                self.dispatch()

        def wait(self):
            with rng("wait"):
                jax.block_until_ready((self.state, self.pending))

        def sample(self, b):
            idx = torch.randint(min(self.n_pairs, BUF), (b,))
            return self.bx[idx].to(dev, non_blocking=True), self.by[idx].to(dev, non_blocking=True)

    # ---- model ------------------------------------------------------------------------------------------------
    model = ConvNextUViT(in_ch=1, out_ch=1, n_feat=32, depths=[2, 2, 4, 4, 8], kernel_size=7, drop_path=0.1,
                         n_attn=4, n_heads=8, residual=True).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.05, betas=(0.9, 0.95))
    model.train()

    def ml_step(online, ev, sample_s):
        """o1.2's ML step. ev = (start, end, after_sample): ev[0]..ev[1] is the step as before (ml_gpu_ms), ev[2]
        splits it into sample() (gather + H2D, kept out of the compute numbers) and the network."""
        ev[0].record()
        t0 = time.perf_counter()
        torch.cuda.nvtx.range_push("sample")
        x, y = online.sample(BATCH)
        torch.cuda.nvtx.range_pop()
        sample_s.append(time.perf_counter() - t0)
        ev[2].record()
        with rng("forward"):
            loss = F.mse_loss(model(x), y)
        with rng("backward"):
            opt.zero_grad(set_to_none=True)
            loss.backward()
        with rng("clip"):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        with rng("optimizer"):
            opt.step()
        ev[1].record()
        with rng("loss_item"):
            _ = loss.item(), loss.item()                 # o1.2: loss_sum += loss.item(); postfix loss.item()

    def run(name, mode, variant):
        online = Online(advs[variant or "ref"])
        while online.n_pairs < 2 * BATCH:                # fill a little so sampling works (pairs need LAG frames)
            online.advance()
            online.wait()
        online.flush()
        wall, ml_ms, sample_ms, net_ms, sample_s = [], [], [], [], []
        worker = None
        for i in range(args.warmup + args.steps):
            if args.nsys and i == args.warmup:           # capture the timed steps only, from a drained pipeline, so
                if worker is not None:                   # every captured step owns exactly one solver chunk
                    worker.join()
                    worker = None
                online.wait()
                torch.cuda.profiler.start()
            ev = tuple(torch.cuda.Event(enable_timing=True) for _ in range(3))
            t0 = time.perf_counter()
            with rng("step"):
                if mode == "ml_only":
                    ml_step(online, ev, sample_s)
                elif mode == "solver_only":
                    online.advance()
                    online.wait()
                elif mode == "serial":                   # production order
                    online.advance()
                    ml_step(online, ev, sample_s)
                elif mode == "thread":
                    if worker is not None:
                        with rng("join"):
                            worker.join()
                    online.flush()                       # main thread: frames of the previous chunk -> buffer

                    def chunk():
                        online.dispatch()
                        online.wait()
                    worker = threading.Thread(target=chunk)
                    worker.start()
                    ml_step(online, ev, sample_s)
                if mode != "solver_only":
                    with rng("harness_sync"):
                        ev[1].synchronize()              # torch stream only
            t1 = time.perf_counter()
            if i >= args.warmup:
                wall.append(t1 - t0)
                if mode != "solver_only":
                    ml_ms.append(ev[0].elapsed_time(ev[1]))
                    sample_ms.append(ev[0].elapsed_time(ev[2]))
                    net_ms.append(ev[2].elapsed_time(ev[1]))
        if worker is not None:
            worker.join()
        if mode in ("serial", "thread"):                 # the last chunk's GPU work is part of the cost
            online.wait()
        if args.nsys:
            torch.cuda.profiler.stop()
        r = {"wall_s_median": round(statistics.median(wall), 4), "wall_s_mean": round(statistics.mean(wall), 4)}
        if ml_ms:
            r["ml_gpu_ms_median"] = round(statistics.median(ml_ms), 2)
            r["sample_host_ms_median"] = round(1e3 * statistics.median(sample_s[args.warmup:]), 2)
            r["sample_gpu_ms_median"] = round(statistics.median(sample_ms), 2)
            r["net_gpu_ms_median"] = round(statistics.median(net_ms), 2)
        print(f"  {name:18s} {r}", flush=True)
        return r

    configs = [("ml_only", "ml_only", None), ("solver_only_ref", "solver_only", "ref"),
               ("solver_only_fixed", "solver_only", "fixed"), ("serial_ref", "serial", "ref"),
               ("serial_fixed", "serial", "fixed"), ("thread_ref", "thread", "ref"), ("thread_fixed", "thread", "fixed")]
    res = {"device": torch.cuda.get_device_name(0), "torch": torch.__version__, "jax": jax.__version__,
           "init": init, "steps": args.steps, "warmup": args.warmup, "buffer_pairs": BUF, "batch": BATCH,
           "results": {}}
    if args.nsys or NVTX or arms != ARMS:                # recorded only off the default path, which stays unchanged
        res.update(arms=arms, nvtx=NVTX, nsys=args.nsys)
    print(f"overlap harness on {res['device']} (torch {torch.__version__}, jax {jax.__version__})", flush=True)
    for name, mode, v in configs:
        if name in arms:
            res["results"][name] = run(name, mode, v)

    R = res["results"]
    res["overlap_efficiency"] = {}
    for v in ("ref", "fixed"):
        for mode in ("serial", "thread"):
            if {"ml_only", f"solver_only_{v}", f"{mode}_{v}"} <= set(R):
                ml, so = R["ml_only"]["wall_s_median"], R[f"solver_only_{v}"]["wall_s_median"]
                comb = R[f"{mode}_{v}"]["wall_s_median"]
                res["overlap_efficiency"][f"{mode}_{v}"] = round((so + ml - comb) / min(so, ml), 3)
    if "ml_only" in R:
        res["sample_fraction_of_ml_only"] = round(R["ml_only"]["sample_gpu_ms_median"]
                                                  / R["ml_only"]["ml_gpu_ms_median"], 3)
    res["torch_peak_GiB"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    ms = jax.devices()[0].memory_stats() or {}
    res["jax_peak_GiB"] = round(ms.get("peak_bytes_in_use", 0) / 2**30, 2)
    out_json = os.path.join(args.out, f"profile_overlap_nsys_{arms[0]}.json" if args.nsys else "profile_overlap.json")
    with open(out_json, "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"overlap efficiency (1 = solver fully hidden behind the ML step, 0 = serial): {res['overlap_efficiency']}")
    if "ml_only" in R:
        print(f"sample() gather+H2D, ml_only: {R['ml_only']['sample_gpu_ms_median']} ms of "
              f"{R['ml_only']['ml_gpu_ms_median']} ms ({res['sample_fraction_of_ml_only']}), batch {BATCH}")
    print(f"peak memory: torch {res['torch_peak_GiB']} GiB, jax {res['jax_peak_GiB']} GiB")
    print(f"OVERLAP_OK -> {out_json}")


if __name__ == "__main__":
    main()
