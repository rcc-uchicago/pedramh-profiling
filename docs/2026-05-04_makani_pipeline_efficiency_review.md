# PlaSim → SFNO training pipeline: efficiency review vs. NVIDIA Makani

Date: 2026-05-04
Status: review only — no code changes proposed for immediate implementation.
Scope: read-only audit of `src/sfno_training/`, `makani-src/makani/`, and the
SLURM launchers, comparing against NVIDIA Makani's reference implementation.

This document answers: *what are the likely bottlenecks in our current
pipeline, what concrete changes would help, and which should we prioritize?*
It is structured as: (1) characterization of what we run today, (2) issue
catalog by area, (3) recommendation matrix with risk/impact, (4) suggested
priority ordering, (5) open questions for the user.

---

## 1. What we currently run (one-paragraph baseline)

The "full" production run (`submit_zgplev_full.slurm`) launches a **single
process** via plain `python -m sfno_training.train_plasim` on a single H100
node with `--disable_ddp`, using `--amp_mode bf16` and
`--checkpointing_level 2`. It reads HDF5 multifiles
(`/fields_state`, `/fields_diagnostic`, `/forcing` per file) through a
PyTorch `DataLoader` with `num_workers=4`, `pin_memory=True`, **no
`persistent_workers`**, **default `prefetch_factor=2`**. Per-sample
normalization (`(x − bias)/scale`) and the `GridConverter` (equiangular →
Legendre-Gauss) run on **CPU inside the worker**. Optimizer is `AdamW`
(`foreach=True` inherited from Makani), TF32 is on
(`train_plasim.py:148-149`), `torch.compile` and `channels_last` are off.
Training is single-step (`multistep_count=1`); validation is multi-step
(`valid_autoreg_steps=3`) and runs *twice* per epoch (raw + EMA second
pass). Checkpointing is synchronous `.tar` via `torch.save`. The
multi-GPU "baseline" (`submit_zgplev_baseline.slurm`) is the only place we
use `torchrun --nproc_per_node=4` (on `gh` GH200 nodes); it sees 4× DDP and
slightly larger batch (8 vs 4).

NVIDIA Makani, by comparison, supports DDP + spatial (h/w) + tensor
(fin/fout) parallelism through a `DistributedManager` abstraction, has an
optional DALI-based dataloader, an explicit `torch.compile(inductor)` mode,
custom DDP gradient-reduction comm hooks (with futures for
comm/compute overlap), `gradient_accumulation_steps` with `model.no_sync()`,
and either rank-sharded or rank-0-gather checkpoint formats — none of which
we currently exercise from the AI-RES side.

---

## 2. Issue catalog

### 2.1 Data loading / HDF5 I/O

| # | Finding | Where | Note |
|---|---------|-------|------|
| D1 | `persistent_workers` not set → workers torn down each epoch, h5py handles re-opened, dataset state re-instantiated | `plasim_trainer.py:166-174` | Default is `False`. With our 4-worker, multi-file dataset, every epoch pays handle-open + index-build cost. |
| D2 | `prefetch_factor` not set (default `2`) | same | On H100 with bf16 + `embed_dim=256, num_layers=12`, an SFNO step is fast enough that 2-deep prefetch is plausibly the bottleneck. |
| D3 | Normalization runs on CPU, in numpy, per sample (state + diagnostic + forcing) | `plasim_forcing_dataset.py:310-313, 331, 340` | Stats are precomputed `(1, C, 1, 1)` arrays, so this is a CPU bandwidth tax that scales with channels × H × W per worker. |
| D4 | `GridConverter` runs on CPU per sample | `plasim_forcing_dataset.py:346` | Equiangular → Legendre-Gauss interpolation is non-trivial; doing it on CPU per sample wastes worker time. |
| D5 | Three separate h5py reads per sample (`state`, `diagnostic`, `forcing`) | `plasim_forcing_dataset.py:221-290` | Three HDF5 reads → three POSIX `pread`s into separate buffers per sample. With small chunks this can be the dominant I/O cost. |
| D6 | No in-memory cache | n/a | PlaSim sim52 is small enough (~tens of GB at 64×128, single-precision) that the entire training set might fit in RAM on a Stampede3 node — see open question Q3. |
| D7 | Smoke config has `num_data_workers=0` | `plasim_sim52_zgplev_smoke.yaml:112` | Acceptable for a 1-epoch gate; flagging only because if we ever flip "smoke" to a real-shaped run, this would silently kill throughput. |
| D8 | Sparse channel selection via `np.argsort` on every read | `plasim_forcing_dataset.py:221-256` | Almost certainly cheap, but the index map could be precomputed once at `__init__` if profiling shows it matters. |

### 2.2 GPU utilization / distributed training

| # | Finding | Where | Note |
|---|---------|-------|------|
| G1 | "Full" production run is **single-GPU** with `--disable_ddp` | `submit_zgplev_full.slurm:101-109` | The biggest unknown is whether Stampede3's `h100` partition has 1 GPU/node or multiple — see Q1. If multiple, we're leaving GPUs idle. |
| G2 | `torch.compile` is off (`jit_mode == "none"` implicitly) | not set | Makani supports `--jit_mode inductor`. SFNO has FFT and SHT ops that may not compile cleanly, but a partial-graph compile of MLPs / norm layers could be tried. Risk: known torch.compile pitfalls with complex tensors. |
| G3 | `channels_last` not used | not set | SFNO is mostly NCHW-friendly conv/FFT; H100 cuDNN benefits from `channels_last` for 2D ops. Unclear whether SFNO's spectral path supports it. Low priority unless profiling shows conv kernels are the bottleneck. |
| G4 | No gradient accumulation | `multistep_count=1` and no `gradient_accumulation_steps` key | At BS=4 on H100, we may be undersized for activation utilization. Macro-batch via accumulation costs nothing if we are not memory-bound. |
| G5 | `bf16` AMP is on, `checkpointing_level=2` is on | `submit_zgplev_full.slurm:107-108` | Good. No change. |
| G6 | TF32 is enabled | `train_plasim.py:148-149` | Good. No change. |
| G7 | EMA shadow weights live on GPU | `ema.py:112` | Doubles parameter memory. For the current 256-dim/12-layer SFNO this is fine; flag if we scale up. |
| G8 | No NCCL/UCX env vars set | SLURM scripts | Default NCCL settings are usually fine on a single node, but for the multi-GPU baseline (4× GH200) it is worth setting `NCCL_DEBUG=WARN` and confirming the IB/NVLink topology is being used. |

### 2.3 Checkpointing / validation

| # | Finding | Where | Note |
|---|---------|-------|------|
| C1 | Validation runs twice per epoch (raw + EMA) | `plasim_trainer.py:513-560` | This is a deliberate design choice (we want to track both losses), but it doubles validation cost. EMA-only-every-N-epochs is a free 1.5× cut for ~95% of the signal. |
| C2 | Synchronous checkpoint write blocks training | `plasim_trainer.py:449-511` | `torch.save` writes a `.tar` via pickle, then re-opens the file to append EMA keys. On a 47.5h job with per-epoch saves this matters less, but for shorter runs it adds up. |
| C3 | Two-pass save: write then re-open to append EMA | `plasim_trainer.py:502-511` | Should be a single write. Low-risk cleanup. |
| C4 | `valid_autoreg_steps=3` even when training is single-step | configs | Reasonable for tracking 24h skill, but no knob to make this cheaper during early epochs. Could subsample validation set during the first N epochs. |

### 2.4 Preprocessing / caching

| # | Finding | Where | Note |
|---|---------|-------|------|
| P1 | Per-sample CPU normalization (D3) is the headline preprocessing cost | `plasim_forcing_dataset.py:310-313` | Move to GPU as a fused op after `pin_memory` transfer. |
| P2 | Per-sample CPU `GridConverter` (D4) | `plasim_forcing_dataset.py:346` | Either move to GPU, or **bake the conversion into a one-time preprocessed dataset** stored in the model's native grid. The latter is the bigger win and we already have the `plasim_makani_packager` to do it. |
| P3 | Forcings (sst, rsdt, sic, …) are stored alongside state in same HDF5 file | dataset structure | This is actually fine — Makani also reads forcings from the same `fields` dataset. The split into `forcing` vs `fields_state` is in our dataset, not a separate file. No change. |
| P4 | Stats files `(1, C, 1, 1)` are loaded once at init | `plasim_trainer.py:106-107` | Already cached. No change. |

### 2.5 Launcher / backend

| # | Finding | Where | Note |
|---|---------|-------|------|
| L1 | "Full" run uses bare `python` not `torchrun` | `submit_zgplev_full.slurm:101` | If `h100` partition is 1 GPU/node, this is correct. If multi-GPU, we should `torchrun --nproc_per_node=N` or `srun --ntasks-per-node=N python ...`. See Q1. |
| L2 | Distributed init has a single-rank fast path | `train_plasim.py:61-78` | Good. Single-GPU jobs don't pay process-group setup. |
| L3 | We do **not** use `mpi4py` anywhere | grep clean | NCCL-only is the right default for a single node. MPI bootstrap is only useful when (a) crossing nodes and (b) the cluster's PMI is more reliable than `c10d::TCPStore`. |
| L4 | `MASTER_ADDR/PORT` not set explicitly when using `torchrun` (baseline) | `submit_zgplev_baseline.slurm:55-60` | Fine — `torchrun` handles this on a single node. Would matter only for multi-node. |

---

## 3. Recommendations: risk × impact matrix

Estimated impact is per-run wallclock unless noted. Risk is the chance of
silent correctness regressions (NaN, divergence, off-by-one in
normalization, etc.), not the engineering effort.

### 3.1 Low risk, low effort — do first

| # | Change | Expected gain | Why low-risk |
|---|--------|--------------|--------------|
| R1 | Set `persistent_workers=True` and `prefetch_factor=4` (or 8) in the train DataLoader | Modest steady-state throughput; large reduction in per-epoch warmup | Pure DataLoader knob. Worker recreation is the only thing that changes; if it breaks, we'll see it on epoch boundaries immediately. |
| R2 | Collapse the EMA-key second-write into a single `torch.save` | Cuts a per-epoch I/O round-trip | Local refactor in `plasim_trainer.py:502-511`. |
| R3 | Add `NCCL_DEBUG=WARN` and `NCCL_ASYNC_ERROR_HANDLING=1` to the multi-GPU baseline SLURM | Diagnostic, not perf | Pure env vars. |
| R4 | Drop EMA validation to every K epochs (e.g. K=5) and keep raw validation per epoch | ~30–40% cut in validation cost | A config flag with a deterministic switch. EMA loss curve becomes coarser but we keep best-EMA tracking. |
| R5 | Add `gradient_accumulation_steps` knob (use Makani's already-existing path) | Effective larger batch without OOM; sometimes improves convergence | Already wired in Makani's trainer (we inherit it). Just a config key. |

### 3.2 Medium risk, medium effort — do second

| # | Change | Expected gain | Why medium-risk |
|---|--------|--------------|--------------|
| R6 | Move per-sample normalization from CPU/numpy to GPU/torch (after `H2D` copy in trainer's `step` method) | Frees worker CPU time; usually the difference between data-bound and compute-bound | Have to thread the `(bias, scale)` tensors onto the right device and confirm that diagnostic and forcing channels are still scaled with their separate stats. Need careful tests. |
| R7 | Move `GridConverter` to GPU (or precompute a Legendre-Gauss-grid dataset via the existing `plasim_makani_packager`) | Eliminates the most expensive per-sample CPU op | The "precompute once" variant is the safer of the two — it's just another packaging variant, validated by existing tests. The "move to GPU" variant requires a torch port of `GridConverter`. |
| R8 | If `h100` partition has ≥2 GPUs/node: switch full SLURM to `srun --ntasks-per-node=N python -m sfno_training.train_plasim` or `torchrun --nproc_per_node=N` (drop `--disable_ddp`) | Near-linear speedup with N | Standard DDP. Just need to confirm GPU count (Q1) and that the global batch size scales sensibly. Makani's `comm.init` and `DistributedSampler` codepath is already exercised by the GH200 baseline. |
| R9 | Try `torch.compile(model, mode="reduce-overhead")` behind a `--jit_mode inductor` flag | 5–20% step-time reduction if it works | SFNO's complex/spectral ops are a known pain point for `torch.compile`. Gate behind a flag, A/B against eager. |
| R10 | Stage data to `$SCRATCH` (or per-node `/tmp` if it has enough space) at job start | Removes Lustre metadata + IOPS variability | Adds a copy step to the SLURM script; needs disk-usage check. Risk: cleanup if job dies mid-stage. |

### 3.3 Higher risk, higher effort — only if profiling justifies

| # | Change | Expected gain | Why higher-risk |
|---|--------|--------------|--------------|
| R11 | In-memory dataset cache (load the entire HDF5 into `torch.from_numpy(...).pin_memory()` once at start) | Removes I/O entirely after warmup | Memory-cap dependent. Fine for sim52 64×128 (~tens of GB), but breaks the "scales to ERA5-class" assumption. Implement as opt-in. |
| R12 | Switch from `python torch.save .tar` to PyTorch async/dist-checkpoint format | ~Free overlap of checkpoint write and next epoch | New format means resume compatibility risk; need migration plan for existing run history. |
| R13 | Switch to NVIDIA DALI for the dataloader (Makani has it as an option) | Higher peak throughput; relieves CPU | Significant porting work; DALI HDF5 path is less battle-tested than numpy/h5py. Only justified if profiling shows we are CPU-bound after R6+R7. |
| R14 | Enable spatial / tensor model parallelism | Lets us run a much bigger model | Big architectural change. Not warranted at current model size. Park it. |
| R15 | Switch launcher to `srun --mpi=pmi2 python ...` MPI-bootstrapped distributed | Multi-node bootstrap reliability | **Not recommended right now.** Only matters when crossing nodes; we don't yet, and `torchrun` is the well-trodden path. NCCL-only is what Makani upstream does. |

---

## 4. Suggested priority ordering

Phase 1 — **a single afternoon's work, near-zero correctness risk**
(picks up the obvious idle CPU/GPU cycles):

1. R1: `persistent_workers=True`, `prefetch_factor=4`.
2. R2: collapse the EMA second-write.
3. R4: EMA validation every K epochs.
4. R3: NCCL diagnostics on the baseline.

Phase 2 — **the actual throughput levers** (once Phase 1 is merged and we
have a fresh baseline timing):

5. Profile *first* with `torch.profiler` to confirm we're CPU-bound on the
   dataloader. Only then move on.
6. R7 (precompute Legendre-Gauss variant via `plasim_makani_packager`) —
   this is almost certainly the single biggest win and is the safest to
   validate (existing packager tests cover it).
7. R6 (GPU-side normalization).
8. **Resolve Q1 (GPU count per H100 node)**, then R8 (multi-GPU full
   training).
9. R5 (gradient accumulation knob, only if R8 makes batch tuning relevant).
10. R10 (data staging) only if Lustre variability is showing up in the
    profiler.

Phase 3 — **only if Phase 2 leaves a measurable gap**:

11. R9 (torch.compile, behind a flag).
12. R11 (in-memory cache for sim52-class runs).
13. R12 (async checkpointing).
14. R13 (DALI), R14, R15 — likely not worth doing.

---

## 5. Open questions for the user

These would change the prioritization and are worth resolving before
Phase 2.

- **Q1.** How many H100 GPUs are on a Stampede3 `h100` partition node? The
  current full SLURM hard-codes `-N 1 -n 1 --disable_ddp`. If the answer
  is "1", we are correct; if "≥2", **R8 is the single biggest win** and
  changes the rest of the plan. (The script comment says the partition
  "does not advertise GPU GRES", which doesn't tell us the count.)
- **Q2.** What is the size of the sim52 zgplev training set on disk?
  Determines whether R11 (in-memory cache) is feasible.
- **Q3.** Is the `gh` (GH200) baseline still the canonical multi-GPU run,
  or is it being phased out in favor of multi-GPU `h100`? If the latter,
  the SLURM scripts converge.
- **Q4.** What is the actual measured per-step time and GPU utilization
  on the current full run? Without this, the recommendations above are
  based on structural inspection only and the priority ordering is a
  guess. A 5-minute `nvidia-smi dmon` + `torch.profiler` snapshot would
  reorder this list with much higher confidence.

---

## Appendix: file:line index of cited code

- `src/sfno_training/train_plasim.py:34-78` — `_world_size_from_env`,
  `_should_skip_distributed_init` (single-rank fast path).
- `src/sfno_training/train_plasim.py:113-122` — `comm.init` call.
- `src/sfno_training/train_plasim.py:148-149` — TF32 enable.
- `src/sfno_training/trainer/plasim_trainer.py:105-107` — stats load.
- `src/sfno_training/trainer/plasim_trainer.py:166-174` — DataLoader
  construction (no `persistent_workers`/`prefetch_factor`).
- `src/sfno_training/trainer/plasim_trainer.py:268` — EMA post-step hook.
- `src/sfno_training/trainer/plasim_trainer.py:449-511` — checkpoint save
  (with two-write pattern at 502-511).
- `src/sfno_training/trainer/plasim_trainer.py:513-560` — two-pass
  validation (raw + EMA).
- `src/sfno_training/data/plasim_forcing_dataset.py:192-210` — lazy h5py
  handle cache.
- `src/sfno_training/data/plasim_forcing_dataset.py:221-290` — three
  separate `_read_*` paths.
- `src/sfno_training/data/plasim_forcing_dataset.py:310-346` — per-sample
  normalization + GridConverter on CPU.
- `src/sfno_training/submit_zgplev_full.slurm:101-109` — bare `python`
  launch, `--disable_ddp`.
- `src/sfno_training/submit_zgplev_baseline.slurm:55-60` — `torchrun
  --nproc_per_node=4`.
- `makani-src/makani/utils/driver.py:657-665` — `foreach=True` optimizers.
- `makani-src/makani/utils/training/deterministic_trainer.py:484-522` —
  gradient accumulation + `model.no_sync()` path.
- `makani-src/makani/utils/training/deterministic_trainer.py:310-311` —
  `torch.compile` gate behind `jit_mode == "inductor"`.
- `makani-src/makani/utils/comm.py:114-196` — `DistributedManager`-based
  process-group init (DDP + spatial + tensor parallel).
