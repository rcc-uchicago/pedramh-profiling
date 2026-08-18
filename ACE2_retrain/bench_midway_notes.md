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

Occupancy: 148.6 s wall × 4 ranks = 594.6 GPU-seconds available; 352.3 s of
kernel time ⇒ **~59% GPU-busy**, and only **~32%** once NCCL is removed.

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

## Decisions / changes log

- **2026-08-18** — First ACE2 bring-up and profile on Midway. Env built at
  `/project/rcc/mehta5/envs/fme`; `config_midway.yaml` ported from the Delta
  config (paths + wandb only, model/loss/optimizer/variables byte-identical);
  `midway_smoke_train.sh` and `midway_bench_nsys.sh` added beside the untouched
  `train.sh`. Jobs **53478978** `ACE2_SMOKE_OK` (train 35.783 / valid 36.213,
  5:55) and **53478979** `ACE2_NSYS_OK` (275 MB report, 4:24). Numbers above.
  Job **53479120** (windowed capture) submitted.
  - Smoke/bench shorten the production config with `--override` rather than
    forking a second config, so every deviation is visible in the script.
  - A pre-flight `python -m fme.ace.validate_config` runs before the GPU work:
    fme parses strict (dacite `strict=True`), so one stale key aborts the run —
    catching that in seconds beats catching it after a 4-GPU allocation opens a
    2.39 TB file.
  - **Open**: production `batch_size=16` is unvalidated on 40 GB A100; the
    `tf32=True` hot-path change is unvalidated; no equivalence baseline exists
    for ACE2 at all.
