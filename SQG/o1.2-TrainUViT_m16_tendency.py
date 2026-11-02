"""o1.2-TrainUViT_m16_tendency.py -- ONLINE twin of 1-TrainUViT_m16_tendency.py, third design (o1.1, 3500 x 0.5:
val flat at ~0.0043 over epochs 50-69 while train kept falling, val/train ~1.3, level with offline at the same epoch).
ConvNextUViT on m=16 subsampled fields (256^2), tendency theta(t) -> theta(t + 0.02), the model returns x + net(x).

Same model, hyper-parameters, loss, OneCycle shape, validation pairs and plots as 1-TrainUViT_m16_tendency.py; the
TRAINING DATA come from the solver, with these differences to o1.1 (more variance: the whole archive, a 2x buffer,
the SAME number of optimizer steps as offline):
  solver   : fixed_ring/A-SQG_data.py itself (imported, left untouched), the script that wrote the 7 training members.
  segments : ALL 3500 archived checkpoints of the 7 members (ckpt_t020 .. ckpt_t519), one seeded permutation, rank r
             takes every world_size-th one; from each the solver integrates SEG_T = 1.0 (200000 steps, up to the next
             checkpoint) -> 3500 time units in total = the whole offline archive, each checkpoint used once.
  ring     : every STRIDE = 40 solver steps (2e-4) theta[::16, ::16] goes into a ring of LAG + 1 = 101 frames; frame k
             and frame k - LAG are exactly 4000 steps = 0.02 apart -> one pair (k >= LAG), 4901 pairs per segment.
  steps    : N_EPOCH x steps_per_epoch = 250 x 781 = 195k optimizer steps on 7 GPUs, exactly as the offline run (same
             OneCycle schedule), so the two differ ONLY in the data.
  S        : S = 480 solver steps (12 new pairs) per ML step: 480 x 195250 = 9.4e7 steps = 469 segments per rank, so
             the 195k steps consume 3281 of the 3500 checkpoints (94% of the archive); S is the largest multiple of
             STRIDE that fits (520 would need 508 segments per rank > 500).
  buffer   : FIFO of BUF_SIZE = 262144 pairs in HOST memory (~137 GB per rank, ~960 GB on 7 ranks = 53 segments =
             53 time units per rank, 2x o1.1); each batch is drawn uniformly from it, so a pair is sampled 32 / 12 =
             ~2.7 times on average (o1.1: ~10, offline: 250). Training starts once START_SEGS = 4 segments are in the
             buffer (~1 h, as o1.1).
  check    : the first segment loaded by every rank is compared bitwise with train_m16.zarr at t_c + 0.02 j (the
             restart reproduces the archived trajectory); the run aborts on any mismatch.
  resume   : a restart ALWAYS resumes from <RUN_TAG>_best.pt (as ML/0-TrainUViT_dt.py; from scratch if it does not
             exist): weights, optimizer, scheduler, step, histories, segment position and wandb run were written
             together, so the epochs after the best one are re-run (OneCycle LR goes back to that point too);
             only the buffer is rebuilt (START_SEGS segments, ~1 h).
The solver steps are dispatched asynchronously by JAX before each ML step and run alongside it on the same GPU.
"""
import os
# One GPU per process for BOTH JAX and PyTorch: remap CUDA_VISIBLE_DEVICES to this rank's GPU before either is
# imported (otherwise JAX would put every rank's solver on the first visible GPU).
if "LOCAL_RANK" in os.environ and "," in os.environ.get("CUDA_VISIBLE_DEVICES", ""):
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"].split(",")[int(os.environ["LOCAL_RANK"])]
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")   # JAX shares the GPU with PyTorch
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
import contextlib
import glob
import hashlib
import importlib.util
import io
import json
import re
import signal
import sys
import threading
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import wandb
from tqdm import tqdm
import zarr
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Dataset, Subset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_UViT import ConvNextUViT

# ============================================================================
# Config
# ============================================================================
RESIDUAL = True                    # tendency (big skip x + net(x)), as 1-TrainUViT_m16_tendency.py
RUN_TAG = "UViT_m16_dt0.02_tendency_online3"

DATA_ROOT = "/data/weidong/SQG_data/GenData_deterForcing/coarse_grained"
TRAIN_STORE = f"{DATA_ROOT}/train_m16.zarr"     # only for the bitwise check of the first segment
VAL_STORE = f"{DATA_ROOT}/val_m16.zarr"
CKPT_DIR = "/data/weidong/SQG_data/GenData_deterForcing/model_ckpt"
ASQG_PY = "/home/nvidia/weidong/SQG/SQG_deterForcing/fixed_ring/A-SQG_data.py"

K = 1                              # lead in saved frames: dt_pred = K * 0.02
THETA_SCALE = 8.0                  # x = theta / THETA_SCALE (mean 0)
M = 16                             # subsampling theta[::M, ::M], as MLE/0-coarse_grain.py

# online data
STRIDE = 40                        # solver steps between ring frames (40 x 5e-6 = 2e-4)
S = 480                            # solver steps per ML step (= STRIDE x new pairs per ML step): the whole archive in 195k steps
SEG_T = 1.0                        # time integrated from each checkpoint (200000 steps, ~1667 ML steps)
BUF_SIZE = 262144                  # replay buffer (pairs per rank, host memory, ~137 GB, ~53 segments)
START_SEGS = 4                     # segments in the buffer before the first optimizer step (~1 h)

# model (as 1-TrainUViT_m16_tendency.py)
N_FEAT, DEPTHS, KERNEL_SIZE, DROP_PATH = 32, [2, 2, 4, 4, 8], 7, 0.1
N_ATTN, N_HEADS = 4, 8

# optimisation (as 1-TrainUViT_m16_tendency.py, including the number of steps)
SEED = 42
BATCH_SIZE = 32                    # per GPU; effective batch = 32 x n GPUs
N_EPOCH = 250
N_OFFLINE_PAIRS = 174993           # train pairs of 1-TrainUViT_m16_tendency.py -> same steps per epoch (781 on 7 GPUs)
N_VAL = 200                        # val pairs, drawn at random once (seed SEED) from all val trajectories
LR = 5e-4                          # OneCycle peak
PCT_START, DIV_FACTOR, FINAL_DIV_FACTOR = 0.1, 25, 1e4
WEIGHT_DECAY, BETAS, GRAD_CLIP = 0.05, (0.9, 0.95), 1.0

# wandb
WANDB_PROJECT = "MZ-SQG"
WANDB_ENTITY = "beauwdli-the-university-of-chicago"
WANDB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")


# ============================================================================
# Distributed (torchrun sets RANK / WORLD_SIZE; plain python = 1 GPU). Each process sees only its own GPU.
# ============================================================================
if "RANK" in os.environ:
    rank, world_size = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(0)
    dist.init_process_group("nccl", rank=rank, world_size=world_size, device_id=torch.device("cuda:0"))
else:
    rank, world_size = 0, 1
device = torch.device("cuda:0")
is_main = rank == 0
def log(*a, **k):                  # rank 0 only; tqdm.write keeps the progress bars intact
    if is_main:
        tqdm.write(" ".join(str(v) for v in a))
torch.manual_seed(SEED + rank)
np.random.seed(SEED + rank)

# ============================================================================
# Precision: strict IEEE FP32 everywhere (no TF32, no autocast); checked against float64
# ============================================================================
torch.backends.fp32_precision = "ieee"
torch.backends.cuda.matmul.fp32_precision = "ieee"
torch.backends.cudnn.fp32_precision = "ieee"
torch.backends.cudnn.conv.fp32_precision = "ieee"
torch.backends.cudnn.rnn.fp32_precision = "ieee"


def check_fp32():
    from torch.nn.attention import SDPBackend, sdpa_kernel
    g = torch.Generator(device=device).manual_seed(0)
    r = lambda *s: torch.randn(*s, device=device, generator=g)
    rel = lambda a, b: ((a.double() - b).abs().max() / b.abs().max()).item()
    x, w, a, b = r(1, 64, 128, 128), r(64, 64, 7, 7) / 50, r(1024, 1024), r(1024, 1024)
    q, k, v = r(1, 8, 1024, 64), r(1, 8, 1024, 64), r(1, 8, 1024, 64)
    with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
        o = F.scaled_dot_product_attention(q, k, v)
    errs = {"conv": rel(F.conv2d(x, w, padding=3), F.conv2d(x.double(), w.double(), padding=3)),
            "matmul": rel(a @ b, a.double() @ b.double()),
            "attention": rel(o, torch.softmax(q.double() @ k.double().transpose(-1, -2) / 8, -1) @ v.double())}
    assert all(e < 2e-5 for e in errs.values()), f"not IEEE fp32: {errs}"
    log("strict fp32 check passed: " + ", ".join(f"{n} {e:.1e}" for n, e in errs.items()))


check_fp32()
log(f"{RUN_TAG}: {world_size} GPU(s), {torch.cuda.get_device_name(0)}")

# ============================================================================
# Solver: import A-SQG_data.py as a module (its argparse gets --member 1, its prints are silenced, the
# SIGUSR1 / SIGTERM handlers it installs are restored). Imported after torch so both share this GPU.
# ============================================================================
import jax
import jax.numpy as jnp
from jax import lax


def import_asqg():
    argv, handlers = sys.argv, {s: signal.getsignal(s) for s in (signal.SIGUSR1, signal.SIGTERM)}
    sys.argv = [ASQG_PY, "--member", "1"]
    try:
        spec = importlib.util.spec_from_file_location("asqg_data", ASQG_PY)
        mod = importlib.util.module_from_spec(spec)
        with contextlib.redirect_stdout(io.StringIO()):
            spec.loader.exec_module(mod)
    finally:
        sys.argv = argv
        for s, h in handlers.items():
            signal.signal(s, h)
    return mod


GD = import_asqg()
N, DT = GD.N, GD.DT
LAG = int(round(K * GD.DT_SAVE / DT)) // STRIDE     # ring frames between x and y (4000 / 40 = 100)
SEG_STEPS = int(round(SEG_T / DT))                  # 200000 solver steps per segment
FRAMES_PER_STEP = S // STRIDE                       # new ring frames (= new pairs) per ML step: 12
FRAMES_PER_SEG = SEG_STEPS // STRIDE                # ring frames after frame 0 of a segment: 5000
PAIRS_PER_SEG = FRAMES_PER_SEG - LAG + 1            # 4901
assert LAG * STRIDE * DT == K * GD.DT_SAVE and S % STRIDE == 0 and SEG_STEPS % STRIDE == 0 and FRAMES_PER_SEG > LAG
T_END = GD.T_START + GD.T_SPIN + GD.T_PROD          # 520: last archived time
template = GD.sqg.init(GD.p, jax.random.key(GD.SEED_INITIAL_CONDITION))   # stepper-state template, as GD.main


def config_hash(member):
    """CONFIG_HASH of A-SQG_data.py --member member (only member and seed_initial_condition depend on it)."""
    cfg = dict(GD.CONFIG, member=member, seed_initial_condition=member)
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load(path, member):
    """GD.load_checkpoint for a checkpoint of member -> AB3 state on this GPU."""
    own, GD.CONFIG_HASH = GD.CONFIG_HASH, config_hash(member)
    try:
        return GD.load_checkpoint(template, path)[0]
    finally:
        GD.CONFIG_HASH = own


def warm(path):
    """Read a checkpoint file once so the next load() comes from the page cache (prefetch thread)."""
    with open(path, "rb") as f:
        while f.read(1 << 26):
            pass


advance = jax.jit(lambda st: lax.scan(GD.inner, st, None, length=STRIDE)[0])     # STRIDE numerical steps
coarse = jax.jit(lambda st: jnp.fft.irfft2(st.state.qh[0], s=(N, N))[::M, ::M].astype(jnp.float32))

# checkpoints of the 7 members whose whole segment stays inside the archived trajectory (t_c + SEG_T <= 520)
ckpts = []
for member in range(1, 8):
    d = GD.RUN_CHECKPOINT_DIR.parent / GD.RUN_LABEL.replace("_m01_", f"_m{member:02d}_")
    files = sorted(glob.glob(f"{d}/ckpt_t*.npz"))
    with np.load(files[0]) as z:
        assert str(z["config_hash"].item()) == config_hash(member), f"config hash rebuild failed for member {member}"
    for f in files:
        t_c = float(re.search(r"ckpt_t([0-9.]+)\.npz", f).group(1))
        if t_c + SEG_T <= T_END + 1e-9:
            ckpts.append((member, t_c, f))
order = np.random.default_rng(SEED).permutation(len(ckpts))
my_segs = [ckpts[i] for i in order[rank::world_size]]
steps_per_epoch = (N_OFFLINE_PAIRS // world_size) // BATCH_SIZE      # as the offline DistributedSampler (drop_last)
total_steps = N_EPOCH * steps_per_epoch
n_seg_needed = START_SEGS + int(np.ceil(total_steps * FRAMES_PER_STEP / FRAMES_PER_SEG)) + 1   # fill + training
assert len(my_segs) >= n_seg_needed, f"{len(my_segs)} checkpoints per rank < {n_seg_needed} segments needed"
log(f"solver A-SQG_data.py: {len(ckpts)} checkpoints of 7 members (t_c + {SEG_T:g} <= {T_END:g}); "
    f"{len(my_segs)} per rank, ~{n_seg_needed} needed -> each used at most once")
log(f"online: segment {SEG_T:g} = {SEG_STEPS} steps = {FRAMES_PER_SEG} frames (stride {STRIDE} steps), "
    f"S = {S} steps = {FRAMES_PER_STEP} pairs per ML step, ring {LAG + 1} frames (pair lag {LAG * STRIDE} steps = "
    f"{LAG * STRIDE * DT:g}), {PAIRS_PER_SEG} pairs per segment, host buffer {BUF_SIZE} pairs "
    f"({BUF_SIZE / PAIRS_PER_SEG:.0f} segments), training starts at {START_SEGS} segments")
if is_main:
    os.makedirs(CKPT_DIR, exist_ok=True)
    with open(f"{CKPT_DIR}/{RUN_TAG}_segments.json", "w") as f:     # rank r uses segments[r][0 .. n_segments - 1]
        json.dump({"seg_t": SEG_T, "world_size": world_size,
                   "segments": [[[m, t] for m, t, _ in (ckpts[i] for i in order[r::world_size])]
                                for r in range(world_size)]}, f)

# train_m16.zarr rows for the bitwise check of the first segment
train_group = zarr.open_group(TRAIN_STORE, mode="r")
train_src = {int(re.search(r"_m(\d\d)_", s["name"]).group(1)): s for s in train_group.attrs["sources"]}


# ============================================================================
# Online data: ring of subsampled frames (GPU) -> FIFO replay buffer of pairs (host memory)
# ============================================================================
class OnlinePairs:
    def __init__(self, segs, i_seg=0):
        self.segs, self.i_seg, self.n_loaded, self.thread, self.pending = segs, i_seg, 0, None, []
        self.ring = torch.empty(LAG + 1, 1, N // M, N // M, device=device)
        self.bx = torch.empty(BUF_SIZE, 1, N // M, N // M)             # host memory, filled lazily
        self.by = torch.empty_like(self.bx)
        self.n_pairs, self.load_time, self.check = 0, 0.0, []
        self._next_segment()

    def _next_segment(self):
        if self.thread is not None:
            self.thread.join()
            self.thread = None
        if self.i_seg >= len(self.segs):                 # safety net (resumes refill the buffer): wrap around
            log(f"WARNING: all {len(self.segs)} segments of this rank used, reusing them from the start")
            self.i_seg = 0
        t0 = time.time()
        self.member, self.t_c, path = self.segs[self.i_seg]
        self.state = load(path, self.member)
        self.load_time += time.time() - t0
        self.i_seg, self.n_loaded, self.k, self.steps_left = self.i_seg + 1, self.n_loaded + 1, 0, SEG_STEPS
        if self.i_seg < len(self.segs):
            self.thread = threading.Thread(target=warm, args=(self.segs[self.i_seg][2],), daemon=True)
            self.thread.start()
        self._push(coarse(self.state))                 # frame 0 at t_c

    def _push(self, frame):
        f = torch.from_dlpack(frame)[None] / THETA_SCALE
        if self.n_loaded == 1 and self.k % LAG == 0 and self.k > 0:   # first segment: bitwise vs train_m16.zarr
            s = train_src[self.member]
            row = s["offset"] + int(round((self.t_c + self.k * STRIDE * DT - s["t_first"]) / GD.DT_SAVE))
            ref = torch.from_numpy(train_group["theta"][row]).to(device)
            self.check.append((f[0] * THETA_SCALE - ref).abs().max().item())
        self.ring[self.k % (LAG + 1)] = f
        if self.k >= LAG:
            j = self.n_pairs % BUF_SIZE
            self.bx[j].copy_(self.ring[(self.k - LAG) % (LAG + 1)])
            self.by[j].copy_(f)
            self.n_pairs += 1
        self.k += 1

    def _flush(self):
        for fr in self.pending:
            self._push(fr)
        self.pending = []

    def advance(self):
        """Store the frames of the previous call, then dispatch S more solver steps (asynchronous). A segment may end
        between two of the FRAMES_PER_STEP chunks: its last frames are stored before the next one is loaded."""
        self._flush()
        for _ in range(FRAMES_PER_STEP):
            if self.steps_left == 0:
                self._flush()
                self._next_segment()
            self.state = advance(self.state)
            self.steps_left -= STRIDE
            self.pending.append(coarse(self.state))

    def sample(self, b):
        idx = torch.randint(min(self.n_pairs, BUF_SIZE), (b,))
        return self.bx[idx].to(device, non_blocking=True), self.by[idx].to(device, non_blocking=True)


# ============================================================================
# Resume: a restart always resumes from <RUN_TAG>_best.pt (from scratch if it does not exist)
# ============================================================================
best_path, last_path = f"{CKPT_DIR}/{RUN_TAG}_best.pt", f"{CKPT_DIR}/{RUN_TAG}_last.pt"
resume_path = best_path
resume = torch.load(resume_path, map_location="cpu", weights_only=False) if os.path.exists(resume_path) else None
start_epoch = resume["epoch"] + 1 if resume else 0
i_seg0 = resume["i_seg"] if resume else 0
if resume:
    log(f"resuming from {os.path.basename(resume_path)}: epoch {resume['epoch']} done, step {resume['global_step']}, "
        f"best val {resume['best_val_loss']:.4f}, {i_seg0} segments used per rank")

t0 = time.time()
online = OnlinePairs(my_segs, i_seg0)
start_pairs = START_SEGS * PAIRS_PER_SEG
while online.n_pairs < start_pairs:                 # fill the buffer before the first optimizer step
    online.advance()
online.advance()                                    # store the pending frames of the fill
check = torch.tensor(max(online.check), device=device)
if world_size > 1:
    dist.all_reduce(check, op=dist.ReduceOp.MAX)
assert check.item() == 0.0, f"online frames differ from train_m16.zarr: max |diff| {check.item():.3e}"
log(f"buffer filled: {online.n_pairs} pairs from {online.n_loaded} segments in {time.time() - t0:.0f} s; "
    f"first segment bitwise equal to train_m16.zarr on all ranks ({len(online.check)} frames each)")
x0, y0 = online.sample(1)
log(f"sample: x {tuple(x0.shape[1:])} {x0.dtype}, x std {x0.std():.3f}, (y - x) std {(y0 - x0).std():.4f}")


# ============================================================================
# Validation data: as 1-TrainUViT_m16_tendency.py (the same N_VAL pairs of val_m16.zarr)
# ============================================================================
def make_pairs(store):
    """rows[n] = store row of x of pair n (y is row + K, same trajectory), trajs[n] = its trajectory id."""
    rows, trajs = [], []
    for s in zarr.open_group(store, mode="r").attrs["sources"]:     # trajectories in storage order
        first, last = s["offset"], s["offset"] + s["n_records"] - 1   # first and last row of this trajectory
        r = np.arange(first, last - K + 1)
        rows.append(r)
        trajs.append(np.full(len(r), s["id"]))
    return np.concatenate(rows), np.concatenate(trajs)


class PairDataset(Dataset):
    """sample n -> (x, y, traj) = (theta[row] / 8, theta[row + K] / 8, trajs[n]), row = rows[n]; x, y (1, 256, 256)."""

    def __init__(self, store, rows, trajs):
        self.store, self.rows, self.trajs, self.theta = store, rows, trajs, None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, n):
        if self.theta is None:                       # opened lazily in each DataLoader worker
            self.theta = zarr.open_group(self.store, mode="r")["theta"]
        row = int(self.rows[n])
        x = torch.from_numpy(self.theta[row] / THETA_SCALE)[None]
        y = torch.from_numpy(self.theta[row + K] / THETA_SCALE)[None]
        return x, y, int(self.trajs[n])


val_rows, val_traj = make_pairs(VAL_STORE)
n_val_traj = int(val_traj.max()) + 1
val_ds = PairDataset(VAL_STORE, val_rows, val_traj)
val_sel = np.sort(np.random.default_rng(SEED).choice(len(val_ds), N_VAL, replace=False))
val_loader = DataLoader(Subset(val_ds, val_sel[rank::world_size].tolist()), batch_size=BATCH_SIZE, num_workers=4,
                        pin_memory=True)
log(f"val: {N_VAL} random pairs of {len(val_ds)} (per trajectory: "
    f"{np.bincount(val_traj[val_sel], minlength=n_val_traj).tolist()}); "
    f"per epoch: {steps_per_epoch} steps x {BATCH_SIZE} x {world_size} GPUs (as offline)")

# ============================================================================
# Model
# ============================================================================
model = ConvNextUViT(in_ch=1, out_ch=1, n_feat=N_FEAT, depths=DEPTHS, kernel_size=KERNEL_SIZE, drop_path=DROP_PATH,
                     n_attn=N_ATTN, n_heads=N_HEADS, residual=RESIDUAL).to(device)
assert all(p.dtype == torch.float32 for p in model.parameters())
raw_model = model
if world_size > 1:
    model = DDP(model, device_ids=[0])
log(f"ConvNextUViT {sum(p.numel() for p in raw_model.parameters()) / 1e6:.2f}M params, "
    f"{'tendency (big skip)' if RESIDUAL else 'full field'}, circular padding")


def criterion(pred, y):
    return F.mse_loss(pred, y)


# ============================================================================
# Optimizer & LR schedule (OneCycle, stepped every optimizer step)
# ============================================================================
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=BETAS)
scheduler = OneCycleLR(optimizer, max_lr=LR, total_steps=total_steps, pct_start=PCT_START,
                       div_factor=DIV_FACTOR, final_div_factor=FINAL_DIV_FACTOR)

global_step, best_val, train_hist, val_hist, wandb_run_id = 0, float("inf"), [], [], None
if resume:
    raw_model.load_state_dict(resume["model_state_dict"])
    optimizer.load_state_dict(resume["optimizer_state_dict"])
    scheduler.load_state_dict(resume["scheduler_state_dict"])
    global_step, best_val, wandb_run_id = resume["global_step"], resume["best_val_loss"], resume.get("wandb_run_id")
    train_hist, val_hist = resume["train_loss_history"], resume["val_loss_history"]
    del resume

# ============================================================================
# wandb (rank 0 only; a resumed run continues its wandb run)
# ============================================================================
config = dict(run_tag=RUN_TAG, residual=RESIDUAL, k=K, dt_pred=0.02 * K, theta_scale=THETA_SCALE, m=M,
              online=True, solver=ASQG_PY, stride=STRIDE, s=S, seg_t=SEG_T, seg_steps=SEG_STEPS, lag_frames=LAG,
              buf_size=BUF_SIZE, start_segs=START_SEGS, n_ckpts=len(ckpts), val_store=VAL_STORE,
              n_val_total=len(val_rows), n_val=N_VAL, n_feat=N_FEAT, depths=DEPTHS, kernel_size=KERNEL_SIZE,
              drop_path=DROP_PATH, n_attn=N_ATTN, n_heads=N_HEADS, batch_size=BATCH_SIZE, world_size=world_size,
              n_epoch=N_EPOCH, steps_per_epoch=steps_per_epoch, lr=LR, pct_start=PCT_START, div_factor=DIV_FACTOR,
              final_div_factor=FINAL_DIV_FACTOR, weight_decay=WEIGHT_DECAY, betas=BETAS, grad_clip=GRAD_CLIP,
              precision="strict fp32")
if is_main:
    os.makedirs(WANDB_DIR, exist_ok=True)
    run = wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, name=RUN_TAG, dir=WANDB_DIR,
                     config=config, id=wandb_run_id, resume="allow")
    wandb_run_id = run.id
    log(f"wandb: {run.url}")


def plot_sample(x, y, pred, title):
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    for a, d, t in zip(ax, [x[0], (y - x)[0], (pred - x)[0]], ["x(t)", "true increment", "predicted increment"]):
        v = float(d.abs().max())
        im = a.imshow(d, cmap="bwr", vmin=-v, vmax=v, origin="lower")
        a.set_title(t)
        plt.colorbar(im, ax=a)
    fig.suptitle(title)
    plt.tight_layout()
    return fig


x_viz, y_viz, _ = val_ds[0]                  # first pair of val trajectory 0, plotted every epoch

# ============================================================================
# Training
# ============================================================================
def mean_over_ranks(total, count):
    t = torch.tensor([total, count], device=device, dtype=torch.float64)
    if world_size > 1:
        dist.all_reduce(t)
    return (t[0] / t[1]).item()


def save(path, epoch, val_loss):
    ckpt = {"epoch": epoch, "global_step": global_step, "val_loss": val_loss, "best_val_loss": best_val,
            "model_state_dict": {k: v.cpu() for k, v in raw_model.state_dict().items()},
            "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(),
            "train_loss_history": train_hist, "val_loss_history": val_hist, "config": config,
            "wandb_run_id": wandb_run_id, "i_seg": i_seg_all[0], "n_pairs": online.n_pairs}
    torch.save(ckpt, path + ".tmp")
    os.replace(path + ".tmp", path)


log(f"{'resuming' if start_epoch else 'training from scratch'}: epochs {start_epoch}..{N_EPOCH - 1} x "
    f"{steps_per_epoch} steps -> {best_path}")
for epoch in tqdm(range(start_epoch, N_EPOCH), desc="epochs", disable=not is_main, position=0, dynamic_ncols=True):
    # ---- train ----
    model.train()
    t0, loss_sum, n_steps, seg0, load0 = time.time(), 0.0, 0, online.n_loaded, online.load_time
    pbar = tqdm(range(steps_per_epoch), desc=f"epoch {epoch}", disable=not is_main, position=1, leave=False,
                dynamic_ncols=True)
    for _ in pbar:
        online.advance()                               # next S solver steps run alongside this ML step
        x, y = online.sample(BATCH_SIZE)
        loss = criterion(model(x), y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        if global_step < total_steps - 1:
            scheduler.step()
        loss_sum, n_steps, global_step = loss_sum + loss.item(), n_steps + 1, global_step + 1
        pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")
        if n_steps == 2 and epoch == start_epoch:
            log(f"peak GPU memory (PyTorch) {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
        if is_main:
            wandb.log({"train/loss": loss.item(), "train/lr": optimizer.param_groups[0]["lr"], "epoch": epoch},
                      step=global_step)
    train_loss = mean_over_ranks(loss_sum, n_steps)
    t_train = time.time() - t0

    # ---- validate ----
    model.eval()
    sums = torch.zeros(3, n_val_traj, device=device, dtype=torch.float64)   # per trajectory: model, persistence, count
    with torch.no_grad():
        for x, y, traj in val_loader:
            x, y, traj = x.to(device, non_blocking=True), y.to(device, non_blocking=True), traj.to(device)
            per_sample = lambda a: ((a - y) ** 2).mean((1, 2, 3)).double()
            sums[0].index_add_(0, traj, per_sample(model(x)))
            sums[1].index_add_(0, traj, per_sample(x))           # persistence: predict x(t + dt) = x(t)
            sums[2].index_add_(0, traj, torch.ones_like(traj, dtype=torch.float64))
    if world_size > 1:
        dist.all_reduce(sums)
    val_traj = (sums[0] / sums[2]).tolist()                  # val loss of each val trajectory
    val_loss = (sums[0].sum() / sums[2].sum()).item()        # all val pairs (equal number per trajectory)
    persistence = (sums[1].sum() / sums[2].sum()).item()
    train_hist.append(train_loss)
    val_hist.append(val_loss)
    # segments used so far: the minimum over the ranks is what a resume can safely continue from
    i_seg_all = torch.tensor([online.i_seg], device=device)
    if world_size > 1:
        dist.all_reduce(i_seg_all, op=dist.ReduceOp.MIN)
    i_seg_all = i_seg_all.tolist()

    # ---- log & checkpoint ----
    log(f"epoch {epoch}: train {train_loss:.4f}  val {val_loss:.4f} (persistence {persistence:.4f})  lr {optimizer.param_groups[0]['lr']:.2e}  "
        f"{t_train / n_steps:.3f} s/step  ({time.time() - t0:.0f} s)", flush=True)
    log("  val per trajectory: " + "  ".join(f"traj {j} {v:.4f}" for j, v in enumerate(val_traj)))
    log(f"  online: {online.n_loaded - seg0} segments this epoch ({online.i_seg} used per rank, "
        f"{online.i_seg * SEG_T * world_size:.1f} time units on all ranks), buffer {min(online.n_pairs, BUF_SIZE)} pairs, "
        f"checkpoint loads {online.load_time - load0:.0f} s")
    improved = val_loss < best_val
    best_val = min(best_val, val_loss)
    if is_main:
        with torch.no_grad():
            pred_viz = raw_model(x_viz[None].to(device))[0].cpu()
        fig = plot_sample(x_viz, y_viz, pred_viz, f"epoch {epoch}  val {val_loss:.4f}")
        wandb.log({f"val/loss_traj{j}": v for j, v in enumerate(val_traj)}, step=global_step)
        wandb.log({"val/loss": val_loss, "val/best": best_val, "val/persistence": persistence,
                   "train/epoch_loss": train_loss, "samples/increment": wandb.Image(fig),
                   "online/segments": online.i_seg, "online/time_units": online.i_seg * SEG_T * world_size,
                   "online/buffer_pairs": min(online.n_pairs, BUF_SIZE)}, step=global_step)
        plt.close(fig)
        save(last_path, epoch, val_loss)
        if improved:
            save(best_path, epoch, val_loss)
            log(f"  best val {best_val:.4f} -> {os.path.basename(best_path)}")
    if world_size > 1:
        dist.barrier()

log(f"done: best val {best_val:.4f}")
if is_main:
    wandb.finish()
if world_size > 1:
    dist.destroy_process_group()
