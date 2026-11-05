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
  --solver ref|lean32 ref = o1.2's step (GD.inner under the 40-step lax.scan, float64); lean32 = seed 2's float32 step
                      (round 1's fastest variant; timing only, its gate is round 1b's).

This is a throughput smoke, not training: the IC is the tiny random field (every step costs the same whatever the field),
there is no DDP, and the trainer's loss is not looked at. Rank 0 prints MULTISIM_OK <mode> <solver> and writes
<out>/multisim_<mode>_<solver>.json.
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


# ---- mode ncclprobe: record the NCCL transport before any DDP phase ---------------------------------------------------
def run_ncclprobe(args, rank, world):
    """A GPU all_reduce over a separate NCCL group (the default group is gloo). The job script greps NCCL's INFO lines
    for 'Selected Provider is cxi' (Slingshot) and P2P vs SHM. Checks the value before anything overwrites it, at both
    ends of the buffer (combine_handoffs: a probe once read 8/8 OK while its collective was timing out)."""
    pg = dist.new_group(backend="nccl")
    torch.cuda.set_device(0)
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
            "busbw_GBps": round(2 * (world - 1) / world * nbytes / t / 1e9, 3)}


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
def run_simulator(args, rank):
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
            dist.send(msg, dst=0)
            send_s.append(time.perf_counter() - t2)
            first_send = first_send or (time.perf_counter() - t_start)
            pairs, seq = pairs[CHUNK:], seq + 1
    msg.zero_()
    msg[-2] = 1.0                                                     # done
    dist.send(msg, dst=0)
    steps = len(chunk_s) * CALLS * STRIDE
    return {"ms_per_step": round(1e3 * solver_s / steps, 4) if steps else None, "steps": steps, "chunks_sent": seq,
            "pairs_sent": seq * CHUNK, "first_send_s": round(first_send, 2) if first_send else None,
            "send_ms_median": round(1e3 * statistics.median(send_s), 3) if send_s else None,
            "send_ms_max": round(1e3 * max(send_s), 3) if send_s else None,
            "chunk_s_median": round(statistics.median(chunk_s), 4) if chunk_s else None}


# ---- mode stream: the trainer ----------------------------------------------------------------------------------------
def run_trainer(args, rank, world):
    from model_UViT import ConvNextUViT
    torch.backends.cuda.matmul.allow_tf32 = False                     # strict fp32, as o1.2
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(0)
    dev = torch.device("cuda:0")
    model = ConvNextUViT(in_ch=1, out_ch=1, n_feat=32, depths=[2, 2, 4, 4, 8], kernel_size=7, drop_path=0.1,
                         n_attn=4, n_heads=8, residual=True).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.05, betas=(0.9, 0.95))
    model.train()
    bx = torch.empty(args.buf, 1, NC, NC)                             # host replay buffer, pageable as o1.2's
    by = torch.empty_like(bx)
    lock = threading.Lock()
    state = {"n": 0, "done": 0, "arrivals": [], "bytes": 0, "per_src": {}}

    def receiver():
        msg = torch.empty(MSG, dtype=torch.float32)
        while state["done"] < world - 1:
            src = dist.recv(msg)                                      # from any simulator
            t = time.perf_counter()
            if msg[-2].item() == 1.0:
                state["done"] += 1
                continue
            body = msg[:-2].view(CHUNK, 2, 1, NC, NC)
            with lock:
                for i in range(CHUNK):
                    j = state["n"] % args.buf
                    bx[j].copy_(body[i, 0])
                    by[j].copy_(body[i, 1])
                    state["n"] += 1
            state["arrivals"].append((t, src))
            state["bytes"] += MSG * 4
            state["per_src"][src] = state["per_src"].get(src, 0) + 1

    dist.barrier()
    t_start = time.perf_counter()
    th = threading.Thread(target=receiver, daemon=True)
    th.start()
    while state["n"] < args.batch and th.is_alive():                  # first pairs need LAG frames per simulator
        time.sleep(0.05)
    t_fill = time.perf_counter() - t_start
    step_s, fresh, steps = [], [], 0
    n_prev = state["n"]
    while th.is_alive():
        t0 = time.perf_counter()
        with lock:
            hi = min(state["n"], args.buf)
            idx = torch.randint(hi, (args.batch,))
            x, y = bx[idx], by[idx]
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        loss = torch.nn.functional.mse_loss(model(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        _ = loss.item()                                               # o1.2 reads the loss every step (a sync)
        steps += 1
        if steps > 3:                                                 # skip warm-up steps
            step_s.append(time.perf_counter() - t0)
            fresh.append(state["n"] - n_prev)
        n_prev = state["n"]
    th.join()
    arr = sorted(t for t, _ in state["arrivals"])
    window = (arr[-1] - arr[0]) if len(arr) > 1 else float("nan")
    pairs_rx = state["n"]
    return {"fill_wait_s": round(t_fill, 2), "train_steps": steps,
            "train_s_per_step_median": round(statistics.median(step_s), 4) if step_s else None,
            "fresh_pairs_per_step_mean": round(statistics.mean(fresh), 3) if fresh else None,
            "pairs_received": pairs_rx, "chunks_received": len(arr),
            "receive_window_s": round(window, 2),
            "pairs_per_s": round(CHUNK * (len(arr) - 1) / window, 3) if len(arr) > 1 and window > 0 else None,
            "MB_per_s": round(state["bytes"] / 1e6 / window, 3) if len(arr) > 1 and window > 0 else None,
            "chunks_per_src": {str(k): v for k, v in sorted(state["per_src"].items())}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["ncclprobe", "independent", "stream"], required=True)
    ap.add_argument("--probe-mb", type=int, default=64, help="ncclprobe: all_reduce size")
    ap.add_argument("--solver", choices=["ref", "lean32"], default="ref")
    ap.add_argument("--out", required=True)
    ap.add_argument("--calls", type=int, default=30, help="independent: 40-step calls per timing repeat")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--ref-ms", type=float, default=None, help="1-GPU ms/step for the efficiency (default per solver)")
    ap.add_argument("--seconds", type=float, default=150.0, help="stream: simulator run time after its first frame")
    ap.add_argument("--batch", type=int, default=16, help="stream: trainer batch (o1.2: 32 on an 80+ GB GPU)")
    ap.add_argument("--buf", type=int, default=16384, help="stream: trainer replay-buffer pairs")
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
                                                           "send_ms_median", "fresh_pairs_per_step_mean", "error")}))
    allm = [None] * world
    dist.all_gather_object(allm, me)
    if rank == 0:
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
        if args.mode == "ncclprobe":
            res["probe_ok_all"] = all(m.get("probe_ok") for m in allm)
            res["busbw_GBps_min"] = min((m["busbw_GBps"] for m in allm if "busbw_GBps" in m), default=None)
            if not res["probe_ok_all"]:
                res["errors"] = res["errors"] or [m["rank"] for m in allm if not m.get("probe_ok")]
        os.makedirs(args.out, exist_ok=True)
        path = os.path.join(args.out, f"multisim_{args.mode}_{args.solver}.json")
        with open(path, "w") as fh:
            json.dump(res, fh, indent=1)
        line = (f"MULTISIM {args.mode} {args.solver}: {len(sims)} simulators on {len(hosts)} nodes, "
                f"ms/step {res.get('ms_per_step_min_max')}, efficiency {res.get('scaling_efficiency_vs_1gpu')}")
        if args.mode == "ncclprobe":
            line = (f"MULTISIM ncclprobe: {world} ranks on {len(hosts)} nodes, all_reduce values "
                    f"{'OK' if res['probe_ok_all'] else 'WRONG'}, busbw min {res['busbw_GBps_min']} GB/s "
                    f"({allm[0].get('allreduce_MB')} MB)")
        if args.mode == "stream":
            line += (f"; trainer {tr.get('train_s_per_step_median')} s/step, fresh pairs/step "
                     f"{tr.get('fresh_pairs_per_step_mean')} (o1.2: 12), {tr.get('pairs_per_s')} pairs/s, "
                     f"{tr.get('MB_per_s')} MB/s, send ms same/other node {res.get('send_ms_median_same_node')}/"
                     f"{res.get('send_ms_median_other_node')}")
        log(line)
        log(f"MULTISIM_OK {args.mode} {args.solver} -> {path}" if not res["errors"] else
            f"ERROR MULTISIM_FAILED {args.mode} {args.solver} ranks {res['errors']} -> {path}")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
