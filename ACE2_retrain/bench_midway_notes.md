# ACE2 (ai2cm `fme`) on Midway — bring-up and first profile

Living notes for ACE2 on Midway, in the style of `si/bench_midway_notes.md`:
narrative first, then a dated decisions log. Cross-cutting summary lives in
`CHANGELOG.md`; this file carries the measured detail.

## What exists

| Piece | Path |
|---|---|
| Vendored model | `ACE2_retrain/ace_exp/` (ai2cm/ace @ `1c3ebad80`, `fme` 2026.5.1) |
| Midway config | `ACE2_retrain/config_midway.yaml` — port of the Delta `config_nsight.yaml` |
| Smoke | `ACE2_retrain/midway_smoke_train.sh` → `ACE2_SMOKE_OK` |
| nsys profile | `ACE2_retrain/midway_bench_nsys.sh` → `ACE2_NSYS_OK` |
| Env | `/project/rcc/mehta5/envs/fme` — torch 2.7.1+cu126, fme 2026.5.1 |

The original `train.sh` is the **Delta/NCSA** launcher and is left untouched
(rule #7). Its env (`/scratch/midway3/krucker01/envs/fme`) is not readable by
us, which is why a Midway env had to be built from
`ace_exp/Makefile::create_environment` (minus `[docs,graphcast]` and the
healpix/analysis extras — not needed for ERA5 lat-lon).

## Cluster facts confirmed on-node 2026-08-18

Run on `--account=rcc-staff -p test`, as requested — **not** the project's usual
`pi-pedramh`/`pedramh-gpu`.

| Item | Value |
|---|---|
| Partition | `test` — `Hidden=YES`, `AllowAccounts=rcc-staff`, **`AllowQos=test`** (so `--qos=test` is mandatory) |
| Default walltime | **`00:05:00`** — omit `--time` and you silently get 5 minutes |
| Hardware | **mixed**: `beagle3-*` = A100, `midway3-02xx` = V100, `midway3-0320` = A30 ⇒ **`--constraint=a100` is load-bearing** |
| GPU actually allocated | **A100-PCIE-40GB** ×4 on `beagle3-0012` — PCIe, *not* SXM/NVLink. This dominates the profile below. |
| nsys | `module load cuda/12.6` (matches torch's cu126 build) |

## Measured — jobs 53478978 (smoke) and 53478979 (nsys)

Both at `batch_size=4` global (1/rank), `stepper_training.n_forward_steps=2`,
AMP on, eager. **Not production shape**: the Delta config uses `batch_size 16`,
sized for 96 GB GH200s; 40 GB A100 headroom is still unmeasured.

- **455,831,040 trainable parameters** (455.8 M).
- **`tf32=True`** is logged at startup — i.e. the vendored `67242e348` perf
  commit is *active*. It has never been equivalence-checked (DESIGN §4).
- **Step time 0.54 s** (smoke, cold) / **0.56 s** (under nsys, warm) at
  1.84 training samples/s/rank ⇒ nsys overhead ≈ 4%.
- **Page cache dominates wall-clock**, exactly as recorded for PanguWeather:

  | | job 53478978 (cold) | job 53478979 (warm, same node) |
  |---|---|---|
  | launch → "Starting Training Loop" | 80 s | 50 s |
  | epoch total | 231 s (64 samples) | 117 s (**512** samples) |

  The second run trained **8× more samples in half the wall-clock**. Any
  timing compared across a cold/warm boundary is meaningless.
- **The epoch is dominated by validation, not training.** In the cold smoke,
  16 training steps took ~48 s and the remaining **~184 s (80%)** went to
  validation + train-evaluation aggregators.

## First profile — job 53478979, 275 MB report, whole-run capture

Bucketed from `CUPTI_ACTIVITY_KIND_KERNEL` over all 4 ranks
(352.3 s of kernel time, 2,380,116 launches, 148.6 s wall):

| bucket | % GPU kernel time | seconds | launches |
|---|---|---|---|
| NCCL (comm **+ wait**) | **45.8%** | 161.2 | 8,648 |
| elementwise / copy | **32.2%** | 113.5 | 1,702,640 |
| GEMM | 10.4% | 36.6 | 178,840 |
| norm / cudnn | 4.4% | 15.4 | 80,072 |
| optimizer | 3.8% | 13.3 | 57,856 |
| FFT / SHT | 2.2% | 7.7 | 38,304 |
| reduction | 0.9% | 3.2 | 110,336 |
| other | 0.4% | 1.4 | 203,420 |

Memory traffic: HtoD **29.9 GiB / 2.08 s**, DtoH 0.31 GiB / 0.03 s,
DtoD **5,665 GiB / 10.14 s**.

Occupancy: **do not read one off this capture** — it spans startup, training and
validation, and a summed-kernel-time average across those phases (~59%) is not a
quantity that means anything. See the per-phase occupancy under the windowed
capture below: 91% during training, 3.3% during validation.

### How to read this — three caveats that change the conclusion

1. **NCCL kernel time is not comm cost.** Ring kernels spin while waiting for
   peers, so that 45.8% conflates real transfer with load imbalance and
   straggler wait. The single largest AllReduce instance is **4.16 s** against a
   median of 11.2 ms — that is waiting, not bandwidth. Treat 45.8% as an upper
   bound on "time not spent computing", not as "time spent on the wire".
2. **This capture includes startup and validation**, not just the training hot
   path — and validation is ~80% of an epoch (above). Job 53479120 re-runs it
   windowed (`ACE2_NSYS_DELAY=45 ACE2_NSYS_DURATION=110`) to isolate training.
3. **`batch_size=4`, not 16.** Smaller batches make the per-step gradient
   all-reduce a larger share of the step. Expect the NCCL fraction to fall at
   production batch size.

### Windowed re-capture — job 53479120 (`ACE2_NSYS_DELAY=45 ACE2_NSYS_DURATION=110`)

205 MB report, 100.5 s trace span. The window covers ~71 s of training plus the
trailing validation, so it is the training-dominated view the unbounded capture
could not give:

| bucket | whole window | **first 71 s (training)** | unbounded (53478979) |
|---|---|---|---|
| NCCL (comm + wait) | 40.9% | **40.6%** | 45.8% |
| elementwise / copy | 35.2% | **35.4%** | 32.2% |
| GEMM | 11.2% | **11.3%** | 10.4% |
| norm / cudnn | 4.7% | 4.7% | 4.4% |
| optimizer | 4.1% | 4.2% | 3.8% |
| FFT / SHT | 2.4% | 2.4% | 2.2% |

The shape is stable across all three views, so it is not an artifact of where
the capture window fell.

**Occupancy — do NOT quote a whole-window average.** Summing kernel time over
the whole window gives "70% busy", which is meaningless: it averages a busy
training phase with an idle validation tail. Measured properly (union of kernel
intervals per device, so multi-stream overlap is not double-counted, in 5 s
bins):

| phase | GPU occupancy |
|---|---|
| training (steady, ~0–58 s) | **91%** |
| validation tail | **3.3%** |

The 9% idle during training is **launch latency, not a stall**: 326,176 idle
gaps totalling 4.74 s over 55 s on device 0, largest single gap **9.76 ms**, no
sync bubble. Device 0 issues **397,207 kernels in 55 s = 7,222 launches/s**, one
every ~138 µs. 38% of the idle sits in 0.1–1 ms gaps and 32% in 1–10 ms gaps.
Same root cause as the 35% elementwise share — ~2,900 tiny elementwise kernels
per step per rank cannot keep the launch queue ahead of the GPU — so fusion
(`torch.compile`, CUDA graphs) would attack both at once.

**This is NOT comparable to PanguWeather's "0.7% loader idle"**
(`polaris_bench_report.md`). That is `loader_wait_frac` — the fraction of
*training-loop* wall time blocked on the data loader — not kernel occupancy over
a capture window. ACE2 has no instrumentation, so its loader-wait equivalent is
unmeasured. `polaris_bench_report.md` records this exact trap already
("`cpu_prep_frac` is not loader idle; built `loader_wait_*`").

**New finding — validation is CPU-bound, not GPU-bound.** 280.2 s of the
window's 282.5 s of kernel time falls in the first 71 s. Validation therefore
contributes **~1% of GPU kernel time while consuming ~40% of the window's
wall-clock** (and ~80% of a cold epoch, above). The aggregators, not the GPU,
are what make an ACE2 epoch long. Any "speed up ACE2" work that only touches
the training step is optimizing the smaller half of the epoch.

Even discounted, two things look real: this is an **elementwise-bound** model
(32% of kernel time, 1.7 M launches, versus 10% GEMM) — the same shape the
PanguWeather profile found — and **fp32 gradient all-reduce over PCIe** is
expensive for a 455.8 M-parameter model on non-NVLink A100s.

## No instrumentation exists in fme

There is **no** `cudaProfilerApi`, `torch.profiler`, or NVTX anywhere in the
SFNO lat-lon training path — the only NVTX in the tree is in the HEALPix layers
and the downscaling module, neither of which this config touches. Consequences:

- The house `--capture-range=cudaProfilerApi --capture-range-end=stop` flags
  would capture **nothing** here, so `midway_bench_nsys.sh` uses a time window
  instead. This is the one place it deliberately departs from the s2s/SI/port
  scripts.
- `parse_nsys.py` produces no useful NVTX summary for ACE2 — it keys on
  `data_prep`/`forward_loss`/`backward`/`optimizer`, which this model never
  emits. The tables above came from querying the sqlite directly.
- `GlobalTimer`'s category breakdown only reaches wandb, which the house rule
  disables. **Set `logging.metrics_log_dir`** to get those scalars on disk.

Adding `ACE2_*` bench knobs + NVTX that emits the **shared** range names is the
follow-up that makes ACE2 comparable to the other models. Per CLAUDE.md #10 the
names must match the existing contract, not invent new ones.

## 8 GPUs (2 nodes x 4 H200) — **GREEN**, job 53483666

`ACE2_SMOKE_2NODE_OK train_loss=60.243 valid_loss=57.124 world=8 batch=16`, 2:26
wall, on `midway3-[0603,0604]` — both `gold-6542Y`, so homogeneous by luck
despite the smoke's loose `--constraint=H200`.

| | 4x A100-PCIE (batch 4) | 8x H200, 2 nodes (batch 16) |
|---|---|---|
| samples/s/rank | 1.84 | **6.82** (3.7x) |
| aggregate samples/s | 7.4 | **54.2** (7.4x) |
| per-rank batch | 1 | 2 |

The 3.7x is **not** a pure per-GPU comparison: H200 ran 2 samples/rank against
A100's 1, so part of it is better utilisation at larger per-rank batch.
`ACE2_BATCH_SIZE=8` gives the like-for-like 1/rank number.

Three things this settles:

- **Multi-node NCCL works on Midway.** `NCCL_SOCKET_IFNAME=^lo,docker0`, flagged
  as inherited-and-unverified, is now confirmed — no hang, no fallback.
- **The production `batch_size=16` fits**, exercised for the first time in this
  project. The A100s had to drop to 4 for 40 GB.
- **Validation gets *worse* on faster hardware, as predicted.** Training ended
  21:46:21 and the epoch ended 21:46:52: **30 s of a 49 s epoch (61%) is
  validation**, up from ~40% on A100. Speeding up the training step raises
  validation's share of the epoch — it does not shrink it.

`nproc` reads "2 cores" in that job's banner. That is an artifact of the
node-info probe (an `srun` overriding `--ntasks` without `--cpus-per-task`,
which binds the step to one core), NOT what training ran with: `sacct` shows
step `.2` with AllocCPUS=96 over 2 nodes = 48/node, and the throughput confirms
it. Fixed in both scripts.

## Still queued


`midway_smoke_train_2node.sh` (→ `ACE2_SMOKE_2NODE_OK`) and
`midway_bench_nsys_2node.sh` (→ `ACE2_NSYS_2NODE_OK`), jobs **53483263** and
**53483265** (the latter chained `afterok`). Both **queued as of 2026-08-18**;
no result yet. They are siblings — the single-node scripts are untouched.

**The launcher had to change, which is why these are separate scripts.**
`torchrun --standalone` binds rendezvous to localhost and *cannot* span nodes.
Multi-node needs one launcher per node sharing a c10d rendezvous — the shape the
Delta `train.sh` already proved for this codebase:

```
--ntasks-per-node=1   # ONE launcher per node; torch.distributed.run forks the 4 local ranks
srun python -m torch.distributed.run --nnodes 2 --nproc_per_node 4 \
     --rdzv_id $SLURM_JOB_ID --rdzv_backend c10d --rdzv_endpoint <head_ip>:29500
```

`--ntasks-per-node=4` here would start 4 launchers per node = 16 ranks, not 8.

**H100, not H200** — measured with `sbatch --test-only` on 2026-08-18:

| constraint | est. start | nodes |
|---|---|---|
| **H100** | **08-18 09:44** | `midway3-[0372,0423]` |
| H200 | 08-18 20:45 | mixed flavours |
| `H200&gold-6542Y` | 08-19 08:31 | homogeneous |
| `H200&epyc-9335` | 08-20 03:34 | homogeneous |

H100 was both soonest *and* automatically homogeneous: `--constraint=H100` with
`--gres=gpu:4` can only match `gold-6346,512g` 32-core nodes, because the other
H100 box (`midway3-0432`, Gold-6448Y/1TB) has `gpu:2` and is excluded by the
4-GPU request. Homogeneity is not cosmetic here — the single-node profile showed
NCCL ring kernels spin while waiting for peers, so a slower partner node gets
recorded as communication cost that does not exist. Retarget without editing:
`sbatch --constraint=H200 ...` (then prefer `"H200&gold-6542Y"` to measure).

**Batch size 16 — the production value — runs for the first time here.** fme
requires `batch_size % world_size == 0`; world size is 8, so 16 gives 2/rank.
The A100 runs had to drop to 4 for 40 GB. `ACE2_BATCH_SIZE=8` gives 1/rank, the
like-for-like weak-scaling comparison against the 4-GPU A100 runs.

**What this is meant to answer.** The single-node profile put NCCL at 40.6% of
GPU kernel time on **A100-PCIE, which has no NVLink**. Two nodes of H100 change
both variables at once — NVLink within a node, and an inter-node hop across
InfiniBand (`ib0`). So expect the split to move; the useful question is whether
the elementwise 35.4% share holds, since that is the part no interconnect can
fix. nsys writes **one report per node** (`_node0`/`_node1`); read them together
or inter-node imbalance is invisible.

Open risk: `NCCL_SOCKET_IFNAME=^lo,docker0` is inherited from the repo's legacy
`midway_training.sh`, not confirmed against a working Midway multi-node NCCL
run. If job 53483263 hangs at startup, that is the first thing to suspect —
re-run with `ACE2_NCCL_DEBUG=INFO`.

## Decisions / changes log

- **2026-08-18** — First ACE2 bring-up and profile on Midway. Env built at
  `/project/rcc/mehta5/envs/fme`; `config_midway.yaml` ported from the Delta
  config (paths + wandb only, model/loss/optimizer/variables byte-identical);
  `midway_smoke_train.sh` and `midway_bench_nsys.sh` added beside the untouched
  `train.sh`. Jobs **53478978** `ACE2_SMOKE_OK` (train 35.783 / valid 36.213,
  5:55) and **53478979** `ACE2_NSYS_OK` (275 MB report, 4:24). Numbers above.
  Job **53479120** windowed capture `ACE2_NSYS_OK` (205 MB, 4:00) — confirms the
  bucket shape is stable and shows **validation is CPU-bound** (~1% of GPU kernel
  time for ~40% of the window's wall-clock).
  - Smoke/bench shorten the production config with `--override` rather than
    forking a second config, so every deviation is visible in the script.
  - A pre-flight `python -m fme.ace.validate_config` runs before the GPU work:
    fme parses strict (dacite `strict=True`), so one stale key aborts the run —
    catching that in seconds beats catching it after a 4-GPU allocation opens a
    2.39 TB file.
  - **Open**: production `batch_size=16` is unvalidated on 40 GB A100; the
    `tf32=True` hot-path change is unvalidated; no equivalence baseline exists
    for ACE2 at all.
