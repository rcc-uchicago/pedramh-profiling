"""sqg_multisim.py -- 2-node smoke (8 A100s over Slingshot): independent SQG simulations, and simulators streaming
training pairs to a trainer GPU ("option B": simulation and training on separate GPUs). One rank per GPU, launched by
mpiexec through rank_env.sh (RANK, WORLD_SIZE, LOCAL_RANK, MASTER_ADDR, MASTER_PORT; one NUMA-local GPU visible).

  --mode independent  every rank integrates its own trajectory (the solver's random IC, seed = rank + 1, stepped to AB3
                      level 2) for --calls jitted 40-step calls, all ranks at once; ms/step per rank and per node, and
                      the aggregate vs --ref-ms (the 1-GPU number), i.e. how well independent simulations scale.
  --mode stream       rank 0 trains o1.2's ConvNextUViT (batch --batch, AdamW 5e-4 / 0.05 / (0.9, 0.95), clip 1.0, strict
                      fp32) from a host replay buffer of --buf pairs; ranks 1.. integrate their own trajectories in o1.2's
                      pattern (12 x 40 steps, a 256^2 frame theta[::16, ::16] / 8 after each 40), pair frame k with frame
                      k - LAG (4,000 steps = 0.02 apart, as o1.2) and send every 12 pairs to rank 0 (torch.distributed
                      gloo, TCP over the HSN). Reports trainer s/step, fresh pairs per trainer step (o1.2 gets 12),
                      receive rate and MB/s, per-simulator ms/step and send time, intra- vs inter-node senders.
  --mode ncclprobe    a 64 MB NCCL all_reduce over all ranks (values checked at both ends, then 10 timed rounds -> busbw).
                      The NCCL group has --nccl-timeout (a wedged collective aborts); a wedged communicator init, as in
                      7815096, is bounded only by the job script's `timeout`.
  --mode replicate    "option A" (round 1c, 2026-11-05): o1.2 on every rank. Each rank runs its own solver in o1.2's
                      order (store the previous call's 12 frames, dispatch 12 x 40 steps + coarse, GPU ring of LAG + 1
                      frames, host FIFO of pairs), and trains a DDP rank of the UViT over NCCL (all ranks, batch --batch
                      per rank). --steps synced steps, then --nosync-steps under DDP.no_sync() (no all_reduce), so the
                      DDP cost is the difference. The global batch grows with the ranks (o1.2's recipe changes).
  --mode decouple     "option B" with DDP (round 1c): on every node the rank with local rank --trainer-local-rank trains
                      (DDP among the trainers, over NCCL across nodes) from the pairs its node's other ranks stream to it
                      (gloo, as --mode stream). --free-steps free-running steps, then --matched-steps where step i waits
                      until --fresh-per-step x (i + 1) pairs have arrived since the phase began (o1.2's 12 fresh pairs
                      per rank-step, i.e. its recipe).
  --solver ref|lean32 ref = o1.2's step (GD.inner under the 40-step lax.scan, float64); lean32 = seed 2's float32 step
                      (round 1b: gate passed, 3.19 ms; the user's choice to carry toward o1.2, 2026-11-05).

This is a throughput smoke, not training: the IC is the tiny random field (every step costs the same whatever the field),
and the trainer's loss is not looked at. Rank 0 prints MULTISIM_OK <mode> <solver> [<tag>] and writes
<out>/multisim_<mode>_<solver>[_<tag>].json.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OPT = os.path.dirname(HERE)                                           # SQG/analysis/opt
ANALYSIS = os.path.dirname(OPT)                                       # SQG/analysis
SQG_DIR = os.path.dirname(ANALYSIS)                                   # SQG
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")      # as production
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
os.environ.setdefault("SQG_ASQG_PY", os.path.join(SQG_DIR, "fixed_ring", "A-SQG_data.py"))

import argparse                                                      # noqa: E402
import datetime                                                      # noqa: E402
import json                                                          # noqa: E402
import socket                                                        # noqa: E402
import statistics                                                    # noqa: E402
import threading                                                     # noqa: E402
import time                                                          # noqa: E402
import traceback                                                     # noqa: E402

import numpy as np                                                   # noqa: E402
import torch                                                         # noqa: E402
import torch.distributed as dist                                     # noqa: E402

sys.path.insert(0, ANALYSIS)
sys.path.insert(0, os.path.join(OPT, "seed2"))
sys.path.insert(0, SQG_DIR)
import profile_solver as PS                                          # noqa: E402  (import_asqg, STRIDE, CALLS, M)

N, M = 4096, PS.M                                                    # A-SQG grid and o1.2's subsampling
NC = N // M                                                          # 256: the training view
STRIDE, CALLS = PS.STRIDE, PS.CALLS                                  # 40 steps per call, 12 calls per optimizer step
PAIR_GAP_T = 0.02                                                    # o1.2: K = 1 frame of the 0.02 archive
THETA_SCALE = 8.0                                                    # o1.2: x = theta / 8
CHUNK = 12                                                           # pairs per message (o1.2's fresh pairs per step)
MSG = CHUNK * 2 * NC * NC + 2                                        # float32: the pairs, then [flag, sequence]


def log(*a):
    print(*a, flush=True)


# ---- the solver on this rank's GPU -----------------------------------------------------------------------------------
def build_solver(solver):
    """(advance(st) -> st after 40 steps, coarse(st) -> 256^2 float32 frame, start state, LAG) on this rank's GPU."""
    GD = PS.import_asqg()
    import jax
    import jax.numpy as jnp
    from jax import lax
    from pyqg_jax import steppers as pst

    dt = GD.DT
    lag = int(round(PAIR_GAP_T / dt)) // STRIDE                       # 4,000 steps / 40 = 100 frames
    rank = int(os.environ.get("RANK", "0"))
    template = GD.sqg.init(GD.p, jax.random.key(rank + 1))
    st0 = GD.advance(template, 2)                                     # level 2, as SQG_INIT=random elsewhere
    coarse = jax.jit(lambda st: jnp.fft.irfft2(st.state.qh[0], s=(N, N))[::M, ::M].astype(jnp.float32))
    if solver == "ref":
        f = jax.jit(lambda st: lax.scan(GD.inner, st, None, length=STRIDE)[0])
        return f, coarse, st0, lag
    import seed2_opt as S2                                            # seed 2's lean32 (round 1)
    mdl = GD.sm.model
    F0 = np.asarray(GD.F0_hat[0])
    iy, ix = np.nonzero(F0)
    P = S2.PATCH
    if not (ix.max() < P and np.all((iy < P) | (iy >= N - P))):
        raise SystemExit("ERROR MULTISIM_FORCING_OUTSIDE_PATCH")
    tables = {"kinv": (mdl.f_0 / mdl.Nb) * jnp.sqrt(mdl.wv2i), "kx": mdl.kk, "ky": mdl.ll, "decay": GD.decay,
              "f0top": jnp.asarray(F0[:P, :P]), "f0bot": jnp.asarray(F0[N - P:, :P]), "tdt": jnp.float32(dt)}
    down = {np.dtype("complex128"): jnp.complex64, np.dtype("float64"): jnp.float32}
    C = {k: v.astype(down.get(np.dtype(v.dtype), v.dtype)) for k, v in tables.items()}
    call, _, _ = S2.make_variant("lean32", pst, N, N // 2 + 1, dt, STRIDE)
    return (lambda st: call(st, C)), coarse, st0, lag


# ---- shared: the NCCL group, the trainer -----------------------------------------------------------------------------
def nccl_group(args, ranks=None):
    """A NCCL group beside the default gloo group (every rank must call this, members or not). Its timeout lets the
    watchdog abort a wedged *collective*. It does not bound torch 2.3.1's blocking communicator init, where 7815096's
    probe wedged (4 ranks failed in connect, the other 4 sat in init); the job script's `timeout` bounds that."""
    torch.cuda.set_device(0)
    return dist.new_group(ranks=ranks, backend="nccl", timeout=datetime.timedelta(seconds=args.nccl_timeout))


def build_model():
    """o1.2's ConvNextUViT and AdamW, strict fp32 (no TF32), on this rank's GPU."""
    from model_UViT import ConvNextUViT
    torch.backends.cuda.matmul.allow_tf32 = False                     # strict fp32, as o1.2
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(0)                                              # every rank starts from the same weights
    model = ConvNextUViT(in_ch=1, out_ch=1, n_feat=32, depths=[2, 2, 4, 4, 8], kernel_size=7, drop_path=0.1,
                         n_attn=4, n_heads=8, residual=True).to(torch.device("cuda:0"))
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.05, betas=(0.9, 0.95))
    model.train()
    return model, opt


def train_step(model, opt, x, y):
    """o1.2's ML step (MSE, clip 1.0, AdamW); loss.item() every step, as o1.2 (a host sync)."""
    loss = torch.nn.functional.mse_loss(model(x), y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    return loss.item()


def med(xs, nd=4):
    return round(statistics.median(xs), nd) if xs else None


# ---- mode ncclprobe: record the NCCL transport before any DDP phase ---------------------------------------------------
def run_ncclprobe(args, rank, world):
    """A GPU all_reduce over a separate NCCL group (the default group is gloo). The job script greps NCCL's INFO lines
    for 'Selected Provider is cxi' (Slingshot) and P2P vs SHM. Checks the value before anything overwrites it, at both
    ends of the buffer (combine_handoffs: a probe once read 8/8 OK while its collective was timing out)."""
    pg = nccl_group(args)
    x = torch.full((args.probe_mb * 2 ** 20 // 4,), float(rank + 1), device="cuda:0")
    expect = world * (world + 1) / 2
    dist.all_reduce(x, group=pg)
    torch.cuda.synchronize()
    ok = bool(x[0].item() == expect and x[-1].item() == expect)
    times = []
    for _ in range(10):
        x.fill_(1.0)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        dist.all_reduce(x, group=pg)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    ok = ok and bool(x[0].item() == world and x[-1].item() == world)
    t = statistics.median(times)
    nbytes = x.numel() * 4
    return {"probe_ok": ok, "allreduce_MB": nbytes / 1e6, "allreduce_ms_median": round(1e3 * t, 3),
            "busbw_GBps": round(2 * (world - 1) / world * nbytes / t / 1e9, 3),
            "nccl_version": nccl_version()}


def nccl_version():
    """Information only: never raises."""
    try:
        import torch.cuda.nccl as tnccl
        v = tnccl.version()                                           # a tuple in torch 2.x, an int in older builds
        return ".".join(str(x) for x in v) if isinstance(v, (tuple, list)) else str(v)
    except Exception as exc:                                          # noqa: BLE001
        return f"unknown ({type(exc).__name__})"


# ---- mode independent ------------------------------------------------------------------------------------------------
def run_independent(args, rank):
    import jax
    adv, coarse, st, _ = build_solver(args.solver)
    for _ in range(3):
        st = adv(st)
    jax.block_until_ready(st)
    dist.barrier()                                                    # every rank starts the clock together
    times = []
    for _ in range(args.repeats):
        t0 = time.perf_counter()
        for _ in range(args.calls):
            st = adv(st)
        jax.block_until_ready(st)
        times.append(time.perf_counter() - t0)
    ms = 1e3 * statistics.median(times) / (args.calls * STRIDE)
    return {"ms_per_step": round(ms, 4), "ms_per_step_all": [round(1e3 * t / (args.calls * STRIDE), 4) for t in times]}


# ---- mode stream: simulators -----------------------------------------------------------------------------------------
def run_simulator(args, rank, dst=0):
    """Integrate this rank's trajectory in o1.2's pattern and send every 12 pairs to the trainer rank `dst` (gloo)."""
    import jax
    adv, coarse, st, lag = build_solver(args.solver)
    st = adv(st)                                                      # compile outside the clock
    jax.block_until_ready(coarse(st))
    ring = np.empty((lag + 1, NC, NC), dtype=np.float32)
    msg = torch.zeros(MSG, dtype=torch.float32)
    pairs, k, seq = [], 0, 0
    solver_s, send_s, chunk_s = 0.0, [], []
    dist.barrier()
    ring[0] = np.asarray(coarse(st)) / THETA_SCALE
    k = 1
    t_start = time.perf_counter()
    first_send = None
    while time.perf_counter() - t_start < args.seconds:
        t0 = time.perf_counter()
        for _ in range(CALLS):                                        # o1.2's pattern: 12 x (40 steps, coarse frame)
            st = adv(st)
            fr = np.asarray(coarse(st)) / THETA_SCALE
            ring[k % (lag + 1)] = fr
            if k >= lag:
                pairs.append((ring[(k - lag) % (lag + 1)].copy(), fr))
            k += 1
        t1 = time.perf_counter()
        solver_s += t1 - t0
        chunk_s.append(t1 - t0)
        while len(pairs) >= CHUNK:
            body = np.stack([np.stack(p) for p in pairs[:CHUNK]])     # (12, 2, 256, 256)
            msg[:-2] = torch.from_numpy(body.reshape(-1))
            msg[-2], msg[-1] = 0.0, float(seq)
            t2 = time.perf_counter()
            dist.send(msg, dst=dst)
            send_s.append(time.perf_counter() - t2)
            first_send = first_send or (time.perf_counter() - t_start)
            pairs, seq = pairs[CHUNK:], seq + 1
    msg.zero_()
    msg[-2] = 1.0                                                     # done
    dist.send(msg, dst=dst)
    steps = len(chunk_s) * CALLS * STRIDE
    return {"ms_per_step": round(1e3 * solver_s / steps, 4) if steps else None, "steps": steps, "chunks_sent": seq,
            "pairs_sent": seq * CHUNK, "first_send_s": round(first_send, 2) if first_send else None,
            "send_ms_median": round(1e3 * statistics.median(send_s), 3) if send_s else None,
            "send_ms_max": round(1e3 * max(send_s), 3) if send_s else None,
            "chunk_s_median": round(statistics.median(chunk_s), 4) if chunk_s else None}


# ---- mode stream: the trainer ----------------------------------------------------------------------------------------
class Receiver:
    """The trainer's host replay buffer (pageable, as o1.2's) filled by a thread that receives 12-pair messages (gloo,
    from any simulator) until `n_senders` simulators have sent their done flag. Start it after the last collective on
    the default group that the main thread makes before join()."""

    def __init__(self, buf, n_senders):
        self.bx = torch.empty(buf, 1, NC, NC)
        self.by = torch.empty_like(self.bx)
        self.buf, self.n_senders, self.lock = buf, n_senders, threading.Lock()
        self.n, self.done, self.arrivals, self.bytes, self.per_src = 0, 0, [], 0, {}
        self.th = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        msg = torch.empty(MSG, dtype=torch.float32)
        while self.done < self.n_senders:
            src = dist.recv(msg)                                      # from any simulator sending to this rank
            t = time.perf_counter()
            if msg[-2].item() == 1.0:
                self.done += 1
                continue
            body = msg[:-2].view(CHUNK, 2, 1, NC, NC)
            with self.lock:
                for i in range(CHUNK):
                    j = self.n % self.buf
                    self.bx[j].copy_(body[i, 0])
                    self.by[j].copy_(body[i, 1])
                    self.n += 1
            self.arrivals.append((t, src))
            self.bytes += MSG * 4
            self.per_src[src] = self.per_src.get(src, 0) + 1

    def start(self):
        self.th.start()

    def alive(self):
        return self.th.is_alive()

    def sample(self, b):
        with self.lock:
            idx = torch.randint(min(self.n, self.buf), (b,))
            x, y = self.bx[idx], self.by[idx]
        return x.to("cuda:0", non_blocking=True), y.to("cuda:0", non_blocking=True)

    def join(self, args, t_start):
        """Wait for every sender's done flag, but not forever: a simulator that died after the barrier never sends
        one. Raises (-> this rank's ERROR line and a FAILED JSON) instead of hanging the arm to the job timeout."""
        self.th.join(timeout=max(30.0, args.seconds + 120.0 - (time.perf_counter() - t_start)))
        if self.alive():
            raise RuntimeError(f"receiver_timeout: {self.done}/{self.n_senders} simulators sent done")

    def wait_for(self, n, poll=0.002):
        """Block until n pairs have arrived (or every sender is done); seconds waited."""
        t0 = time.perf_counter()
        while self.n < n and self.alive():
            time.sleep(poll)
        return time.perf_counter() - t0

    def summary(self):
        arr = sorted(t for t, _ in self.arrivals)
        window = (arr[-1] - arr[0]) if len(arr) > 1 else float("nan")
        ok = len(arr) > 1 and window > 0
        return {"pairs_received": self.n, "chunks_received": len(arr), "receive_window_s": round(window, 2),
                "pairs_per_s": round(CHUNK * (len(arr) - 1) / window, 3) if ok else None,   # (n-1)/window: biased high
                "MB_per_s": round(self.bytes / 1e6 / window, 3) if ok else None,          # for few chunks (Part VIII)
                "chunks_per_src": {str(k): v for k, v in sorted(self.per_src.items())}}


def run_trainer(args, rank, world):
    model, opt = build_model()
    rx = Receiver(args.buf, world - 1)
    dist.barrier()
    t_start = time.perf_counter()
    rx.start()
    rx.wait_for(args.batch, poll=0.05)                                # first pairs need LAG frames per simulator
    t_fill = time.perf_counter() - t_start
    step_s, fresh, steps = [], [], 0
    n_prev = rx.n
    while rx.alive():
        t0 = time.perf_counter()
        x, y = rx.sample(args.batch)
        train_step(model, opt, x, y)                                  # o1.2 reads the loss every step (a sync)
        steps += 1
        if steps > 3:                                                 # skip warm-up steps
            step_s.append(time.perf_counter() - t0)
            fresh.append(rx.n - n_prev)
        n_prev = rx.n
    rx.join(args, t_start)
    return {"fill_wait_s": round(t_fill, 2), "train_steps": steps,
            "train_s_per_step_median": med(step_s),
            "fresh_pairs_per_step_mean": round(statistics.mean(fresh), 3) if fresh else None, **rx.summary()}


# ---- mode replicate: o1.2 on every rank (solver + DDP rank) -------------------------------------------------------------
class Pairs:
    """o1.2's OnlinePairs without segments or checkpoints: a GPU ring of LAG + 1 frames, a host FIFO of pairs, and
    o1.2's order (advance() first stores the previous call's frames, then dispatches the next 12 x 40 steps)."""

    def __init__(self, adv, coarse, st, lag, buf):
        self.adv, self.coarse, self.st, self.lag, self.buf = adv, coarse, st, lag, buf
        self.ring = torch.empty(lag + 1, 1, NC, NC, device="cuda:0")
        self.bx = torch.empty(buf, 1, NC, NC)
        self.by = torch.empty_like(self.bx)
        self.n, self.k, self.pending = 0, 0, []
        self._push(coarse(st))

    def _push(self, frame):
        f = torch.from_dlpack(frame)[None] / THETA_SCALE              # o1.2: JAX -> torch on the GPU, x = theta / 8
        self.ring[self.k % (self.lag + 1)] = f
        if self.k >= self.lag:
            j = self.n % self.buf
            self.bx[j].copy_(self.ring[(self.k - self.lag) % (self.lag + 1)])
            self.by[j].copy_(f)
            self.n += 1
        self.k += 1

    def advance(self):
        for fr in self.pending:
            self._push(fr)
        self.pending = []
        for _ in range(CALLS):
            self.st = self.adv(self.st)
            self.pending.append(self.coarse(self.st))

    def sample(self, b):
        idx = torch.randint(min(self.n, self.buf), (b,))
        return self.bx[idx].to("cuda:0", non_blocking=True), self.by[idx].to("cuda:0", non_blocking=True)


def run_replicate(args, rank, world):
    from torch.nn.parallel import DistributedDataParallel as DDP
    pg = nccl_group(args)
    adv, coarse, st, lag = build_solver(args.solver)
    data = Pairs(adv, coarse, st, lag, args.buf)
    model, opt = build_model()
    t0 = time.perf_counter()
    ddp = DDP(model, device_ids=[0], process_group=pg)                # o1.2: DDP(model, device_ids=[0]); NCCL init
    t_ddp = time.perf_counter() - t0
    t0 = time.perf_counter()
    while data.n < args.batch:                                        # o1.2 starts with a filled buffer
        data.advance()
    t_fill = time.perf_counter() - t0

    def step():
        ta = time.perf_counter()
        data.advance()
        tb = time.perf_counter()
        x, y = data.sample(args.batch)
        train_step(ddp, opt, x, y)
        return time.perf_counter() - ta, tb - ta

    for _ in range(3):
        step()
    dist.barrier()                                                    # gloo: start the clock together
    synced, adv_s = [], []
    for _ in range(args.steps):
        s, a = step()
        synced.append(s)
        adv_s.append(a)
    nosync = []
    with ddp.no_sync():                                               # the same steps without the gradient all_reduce
        for _ in range(args.nosync_steps):
            nosync.append(step()[0])
    return {"ddp_init_s": round(t_ddp, 2), "fill_s": round(t_fill, 2), "steps": args.steps,
            "s_per_step_median": med(synced), "s_per_step_all": [round(x, 4) for x in synced],
            "s_per_step_nosync_median": med(nosync), "advance_host_s_median": med(adv_s),
            "pairs_in_buffer": data.n, "peak_torch_GiB": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)}


# ---- mode decouple: per node one DDP trainer fed by its node's simulators ------------------------------------------------
def run_ddp_trainer(args, rank, pg, n_senders):
    from torch.nn.parallel import DistributedDataParallel as DDP
    model, opt = build_model()
    t0 = time.perf_counter()
    ddp = DDP(model, device_ids=[0], process_group=pg)                # NCCL among the trainers only
    t_ddp = time.perf_counter() - t0
    rx = Receiver(args.buf, n_senders)
    dist.barrier()                                                    # gloo, all ranks: simulators start their clock
    t_start = time.perf_counter()
    rx.start()
    rx.wait_for(args.batch, poll=0.05)
    t_fill = time.perf_counter() - t_start

    def step():
        x, y = rx.sample(args.batch)
        train_step(ddp, opt, x, y)

    for _ in range(3):
        step()
    free = []
    t_free0 = time.perf_counter()
    for _ in range(args.free_steps):                                  # free-running: the trainer's own speed with DDP
        t0 = time.perf_counter()
        step()
        free.append(time.perf_counter() - t0)
    t_free = time.perf_counter() - t_free0
    matched, waits, fresh, starved = [], [], [], 0
    n0 = n_prev = rx.n
    t_m0 = time.perf_counter()
    for i in range(args.matched_steps):                               # o1.2's recipe: fresh_per_step new pairs/step
        t0 = time.perf_counter()
        waits.append(rx.wait_for(n0 + args.fresh_per_step * (i + 1)))
        if rx.n < n0 + args.fresh_per_step * (i + 1):
            starved += 1                                              # every simulator already done
        step()
        matched.append(time.perf_counter() - t0)
        fresh.append(rx.n - n_prev)
        n_prev = rx.n
    t_m = time.perf_counter() - t_m0
    rx.join(args, t_start)
    # The paced phase must be read by its MEAN (phase wall / steps). Its simulators start together, so their 12-pair
    # messages arrive in bursts: most steps do not wait and one in ~3 waits for the next burst. 7815637 reported the
    # median (0.424 s, wait median 0.0) while its 720 fresh pairs at ~23 pairs/s took ~0.52 s per step (Part IX).
    nf, nm = max(len(free), 1), max(len(matched), 1)
    return {"ddp_init_s": round(t_ddp, 2), "fill_wait_s": round(t_fill, 2), "senders": n_senders,
            "free_steps": args.free_steps, "s_per_step_free_median": med(free),
            "free_phase_s": round(t_free, 3), "s_per_step_free_mean": round(t_free / nf, 4),
            "matched_steps": args.matched_steps, "s_per_step_matched_median": med(matched),
            "matched_phase_s": round(t_m, 3), "s_per_step_matched_mean": round(t_m / nm, 4),
            "wait_s_matched_median": med(waits), "wait_s_matched_mean": round(statistics.mean(waits), 4) if waits else None,
            "fresh_pairs_per_step_matched_mean": round(statistics.mean(fresh), 3) if fresh else None,
            "starved_steps": starved, **rx.summary()}


def decouple_roles(args, rank, world):
    """(trainer ranks, this rank's trainer, simulators of this rank's node) from the actual placement (gloo gather)."""
    me = (socket.gethostname(), int(os.environ.get("LOCAL_RANK", "0")))
    allp = [None] * world
    dist.all_gather_object(allp, me)
    hosts = sorted({h for h, _ in allp})
    trainers = []
    for h in hosts:
        tr = [r for r, (hh, lr) in enumerate(allp) if hh == h and lr == args.trainer_local_rank]
        if len(tr) != 1:
            raise RuntimeError(f"host {h}: {len(tr)} ranks with local rank {args.trainer_local_rank}")
        trainers.append(tr[0])
    mine = [r for r in trainers if allp[r][0] == me[0]][0]
    node_sims = [r for r, (hh, _) in enumerate(allp) if hh == me[0] and r != mine]
    return trainers, mine, node_sims


def summarize(args, allm, world, ref_ms):
    """Rank 0: the per-mode summary, the JSON and the MULTISIM / MULTISIM_OK lines."""
    hosts = sorted({m["host"] for m in allm})
    res = {"mode": args.mode, "solver": args.solver, "world_size": world, "hosts": hosts, "ref_ms_1gpu": ref_ms,
           "ranks": allm, "errors": [m["rank"] for m in allm if "error" in m]}
    sims = [m for m in allm if m["role"] == "simulator" and m.get("ms_per_step")]
    if sims:
        agg = sum(1e3 / m["ms_per_step"] for m in sims)          # solver steps per second, all simulators
        res["aggregate_steps_per_s"] = round(agg, 2)
        res["scaling_efficiency_vs_1gpu"] = round(agg / (len(sims) * 1e3 / ref_ms), 4)
        res["ms_per_step_by_host"] = {h: round(statistics.mean(m["ms_per_step"] for m in sims if m["host"] == h), 4)
                                      for h in hosts if any(m["host"] == h for m in sims)}
        res["ms_per_step_min_max"] = [min(m["ms_per_step"] for m in sims), max(m["ms_per_step"] for m in sims)]
        res["simulated_time_units_per_hour"] = round(agg * 3600 * 5e-6, 3)
    if args.mode == "stream":
        tr = allm[0]
        host0 = allm[0]["host"]
        res["senders_same_node"] = [m["rank"] for m in sims if m["host"] == host0]
        res["senders_other_node"] = [m["rank"] for m in sims if m["host"] != host0]
        for side in ("same_node", "other_node"):
            ms_ = [m["send_ms_median"] for m in sims if m.get("send_ms_median") is not None
                   and (m["host"] == host0) == (side == "same_node")]
            res[f"send_ms_median_{side}"] = round(statistics.median(ms_), 3) if ms_ else None
        fresh = tr.get("fresh_pairs_per_step_mean")
        res["o12_fresh_pairs_per_step"] = CHUNK
        res["verdict"] = (None if fresh is None else
                          "trainer-bound: >= 12 fresh pairs per step" if fresh >= CHUNK else
                          "solver-bound: < 12 fresh pairs per step at full trainer speed")
    if args.mode in ("stream", "decouple") and sims:                # 12 pairs per (chunk period + send) per sim:
        res["pairs_per_s_from_period"] = round(sum(                 # unbiased, unlike (n - 1) / window
            CHUNK / (m["chunk_s_median"] + (m.get("send_ms_median") or 0.0) / 1e3)
            for m in sims if m.get("chunk_s_median")), 3)
    if args.mode == "ncclprobe":
        res["probe_ok_all"] = all(m.get("probe_ok") for m in allm)
        res["busbw_GBps_min"] = min((m["busbw_GBps"] for m in allm if "busbw_GBps" in m), default=None)
        res["nccl_version"] = allm[0].get("nccl_version")
        if not res["probe_ok_all"] or res["busbw_GBps_min"] is None:
            res["errors"] = res["errors"] or [m["rank"] for m in allm if not m.get("probe_ok")] or ["no_busbw"]
    if args.mode == "replicate":
        reps = [m for m in allm if m.get("s_per_step_median")]
        if len(reps) == world:
            s = max(m["s_per_step_median"] for m in reps)            # DDP steps in lockstep: the slowest rank
            ns = [m["s_per_step_nosync_median"] for m in reps if m.get("s_per_step_nosync_median")]
            res.update({"s_per_step": s, "s_per_step_nosync_max": max(ns) if ns else None,
                        "ddp_cost_s": round(s - max(ns), 4) if ns else None,
                        "global_batch": world * args.batch, "fresh_pairs_per_rank_step": CHUNK,
                        "rank_steps_per_s": round(world / s, 4), "rank_steps_per_s_per_gpu": round(1 / s, 4)})
        else:
            res["errors"] = res["errors"] or ["missing_replica_timing"]
    if args.mode == "decouple":
        trs = [m for m in allm if m["role"] == "trainer"]
        res["trainers"] = [m["rank"] for m in trs]
        if trs and all(m.get("s_per_step_matched_mean") for m in trs):
            s_m = max(m["s_per_step_matched_mean"] for m in trs)        # mean, not median (bursty paced phase)
            s_f = max(m["s_per_step_free_mean"] for m in trs)
            res.update({"n_trainers": len(trs), "n_simulators": len(sims), "global_batch": len(trs) * args.batch,
                        "s_per_step_free": s_f, "s_per_step_matched": s_m,
                        "fresh_pairs_per_step_matched": min(m["fresh_pairs_per_step_matched_mean"] for m in trs),
                        "starved_steps": sum(m["starved_steps"] for m in trs),
                        "rank_steps_per_s_matched": round(len(trs) / s_m, 4),
                        "rank_steps_per_s_per_gpu_matched": round(len(trs) / s_m / world, 4)})
        else:
            res["errors"] = res["errors"] or ["missing_trainer_timing"]
    os.makedirs(args.out, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    path = os.path.join(args.out, f"multisim_{args.mode}_{args.solver}{tag}.json")
    with open(path, "w") as fh:
        json.dump(res, fh, indent=1)
    line = (f"MULTISIM {args.mode} {args.solver}: {len(sims)} simulators on {len(hosts)} nodes, "
            f"ms/step {res.get('ms_per_step_min_max')}, efficiency {res.get('scaling_efficiency_vs_1gpu')}")
    if args.mode == "ncclprobe":
        line = (f"MULTISIM ncclprobe{' ' + args.tag if args.tag else ''}: {world} ranks on {len(hosts)} nodes, "
                f"all_reduce values {'OK' if res['probe_ok_all'] else 'WRONG'}, busbw min {res['busbw_GBps_min']} "
                f"GB/s ({allm[0].get('allreduce_MB')} MB, NCCL {res['nccl_version']})")
    if args.mode == "replicate":
        line = (f"MULTISIM replicate {args.solver}: {world} ranks (solver + DDP rank each) on {len(hosts)} nodes, "
                f"s/step {res.get('s_per_step')} (no_sync {res.get('s_per_step_nosync_max')}, DDP cost "
                f"{res.get('ddp_cost_s')} s), global batch {res.get('global_batch')}, "
                f"{res.get('rank_steps_per_s_per_gpu')} rank-steps/s per GPU")
    if args.mode == "decouple":
        line = (f"MULTISIM decouple {args.solver}: {res.get('n_trainers')} DDP trainers + {res.get('n_simulators')} "
                f"simulators on {len(hosts)} nodes, s/step free {res.get('s_per_step_free')} / matched "
                f"{res.get('s_per_step_matched')} ({res.get('fresh_pairs_per_step_matched')} fresh pairs/step, "
                f"starved {res.get('starved_steps')}), {res.get('rank_steps_per_s_per_gpu_matched')} rank-steps/s "
                f"per GPU; simulators ms/step {res.get('ms_per_step_min_max')}, "
                f"{res.get('pairs_per_s_from_period')} pairs/s")
    if args.mode == "stream":
        line += (f"; trainer {tr.get('train_s_per_step_median')} s/step, fresh pairs/step "
                 f"{tr.get('fresh_pairs_per_step_mean')} (o1.2: 12), {tr.get('pairs_per_s')} pairs/s, "
                 f"{tr.get('MB_per_s')} MB/s, send ms same/other node {res.get('send_ms_median_same_node')}/"
                 f"{res.get('send_ms_median_other_node')}")
    log(line)
    who = f"{args.mode} {args.solver}{' ' + args.tag if args.tag else ''}"
    log(f"MULTISIM_OK {who} -> {path}" if not res["errors"] else
        f"ERROR MULTISIM_FAILED {who} ranks {res['errors']} -> {path}")




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["ncclprobe", "independent", "stream", "replicate", "decouple"], required=True)
    ap.add_argument("--probe-mb", type=int, default=64, help="ncclprobe: all_reduce size")
    ap.add_argument("--solver", choices=["ref", "lean32"], default="ref")
    ap.add_argument("--out", required=True)
    ap.add_argument("--calls", type=int, default=30, help="independent: 40-step calls per timing repeat")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--ref-ms", type=float, default=None, help="1-GPU ms/step for the efficiency (default per solver)")
    ap.add_argument("--seconds", type=float, default=150.0, help="stream: simulator run time after its first frame")
    ap.add_argument("--batch", type=int, default=16, help="trainer batch per rank (o1.2: 32 on an 80+ GB GPU)")
    ap.add_argument("--buf", type=int, default=16384, help="trainer replay-buffer pairs (per trainer / replica)")
    ap.add_argument("--tag", default="", help="suffix for the JSON name and the MULTISIM_OK line (probe arms)")
    ap.add_argument("--nccl-timeout", type=float, default=120.0,
                    help="NCCL group timeout, s (watchdog abort of a collective; not of communicator init)")
    ap.add_argument("--steps", type=int, default=20, help="replicate: timed DDP steps")
    ap.add_argument("--nosync-steps", type=int, default=6, help="replicate: steps under DDP.no_sync()")
    ap.add_argument("--free-steps", type=int, default=40, help="decouple: free-running DDP trainer steps")
    ap.add_argument("--matched-steps", type=int, default=60, help="decouple: steps paced to --fresh-per-step")
    ap.add_argument("--fresh-per-step", type=int, default=CHUNK, help="decouple: new pairs per trainer step (o1.2: 12)")
    ap.add_argument("--trainer-local-rank", type=int, default=1,
                    help="decouple: the trainer's local rank on every node (1 = GPU 46000, GDR-capable under PHB)")
    args = ap.parse_args()
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    ref_ms = args.ref_ms or {"ref": 9.2911, "lean32": 3.2048}[args.solver]   # job 7814300, one A100 alone
    dist.init_process_group("gloo", init_method="env://", rank=rank, world_size=world,   # short: a wedge dies in
                            timeout=datetime.timedelta(minutes=6))                     # minutes, not at walltime
    me = {"rank": rank, "host": socket.gethostname(), "local_rank": int(os.environ.get("LOCAL_RANK", "0")),
          "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "role": None}
    try:
        if args.mode == "ncclprobe":
            me["role"] = "probe"
            me.update(run_ncclprobe(args, rank, world))
        elif args.mode == "independent":
            me["role"] = "simulator"
            me.update(run_independent(args, rank))
        elif args.mode == "replicate":
            me["role"] = "replica"
            me.update(run_replicate(args, rank, world))
        elif args.mode == "decouple":
            trainers, mine, node_sims = decouple_roles(args, rank, world)
            pg = nccl_group(args, trainers)                           # every rank calls new_group
            me["trainers"] = trainers
            if rank == mine:
                me["role"] = "trainer"
                me.update(run_ddp_trainer(args, rank, pg, len(node_sims)))
            else:
                me["role"] = "simulator"
                me["sends_to"] = mine
                me.update(run_simulator(args, rank, dst=mine))
        elif rank == 0:
            me["role"] = "trainer"
            me.update(run_trainer(args, rank, world))
        else:
            me["role"] = "simulator"
            me.update(run_simulator(args, rank))
    except Exception as exc:                                          # one greppable line per rank, then gather
        me["error"] = f"{type(exc).__name__}: {exc}"[:500]
        me["traceback"] = traceback.format_exc()[-3000:]
        log(f"ERROR MULTISIM_RANK_FAILED rank={rank} {type(exc).__name__}: {str(exc)[:200]}")
    log(f"rank {rank} {me['host']} gpu {me['cuda_visible_devices']} {me['role']}: "
        + json.dumps({k: v for k, v in me.items() if k in ("ms_per_step", "train_s_per_step_median", "pairs_sent",
                                                           "send_ms_median", "fresh_pairs_per_step_mean", "error",
                                                           "s_per_step_median", "s_per_step_nosync_median",
                                                           "s_per_step_free_median", "s_per_step_matched_median")}))
    allm = [None] * world
    dist.all_gather_object(allm, me)
    if rank == 0:
        try:                                                          # an aggregation slip must not park the
            summarize(args, allm, world, ref_ms)                      # other ranks in the final barrier
        except Exception as exc:                                      # noqa: BLE001
            log(f"ERROR MULTISIM_FAILED {args.mode} {args.solver} rank0_aggregation "
                f"{type(exc).__name__}: {str(exc)[:200]}")
            try:
                os.makedirs(args.out, exist_ok=True)
                tag = f"_{args.tag}" if args.tag else ""
                with open(os.path.join(args.out, f"multisim_{args.mode}_{args.solver}{tag}.json"), "w") as fh:
                    json.dump({"mode": args.mode, "solver": args.solver, "ranks": allm,
                               "errors": [f"rank0_aggregation: {type(exc).__name__}: {exc}"[:300]]}, fh,
                              indent=1, default=str)
            except Exception:                                         # noqa: BLE001
                pass
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
