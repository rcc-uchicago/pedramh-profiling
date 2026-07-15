# pedramh-profiling: Design Specification

A benchmarking + GPU-profiling + optimization workbench for the Pedram
Hassanzadeh group's probabilistic **subseasonal-to-seasonal (S2S)** weather
models. The goal is to make these models **faster on HPC GPUs without changing
what they compute** — measured, gated, and reproducible across clusters.

> **Current focus: `PanguWeather/`** (2026-07-15). It is a 95%-identical fork of
> `s2s/v2.0` (§2c) and the one that actually runs on Polaris today — `s2s/v2.0` is blocked on
> an ERA5 stage. Two consequences worth knowing before you read further: PanguWeather carries
> **none** of s2s's NVTX/`S2S_BENCH` instrumentation, so profiling it starts with porting
> that; and the two forks are **copies, not shared imports**, so a fix in one silently does
> not reach the other. Six codebases now live here (§2, §2b), not three.

Development guide and conventions are in **CLAUDE.md** (read that for how to
work here — it is also the single source of truth for cluster facts). This
document covers *what* we are building and *why*.

---

## Table of Contents

1. [Goals and Non-Goals](#1-goals-and-non-goals)
2. [The models and how they relate](#2-the-models-and-how-they-relate)
3. [Architecture: the shared model pipeline](#3-architecture-the-shared-model-pipeline)
4. [The correctness oracle: numerical-equivalence-vs-baseline](#4-the-correctness-oracle-numerical-equivalence-vs-baseline)
5. [Optimization thesis and ROI ladder](#5-optimization-thesis-and-roi-ladder)
6. [Clusters and hardware](#6-clusters-and-hardware)
7. [Validation and testing strategy](#7-validation-and-testing-strategy)
8. [Roadmap](#8-roadmap)
9. [Repository layout](#9-repository-layout)
10. [Open questions and risks](#10-open-questions-and-risks)

---

## 1. Goals and Non-Goals

### Goals

- **Measure** training and inference throughput of the models on HPC GPUs,
  with the existing NVTX / `*_BENCH` / CSV instrumentation, reproducibly.
- **Optimize** the hot path (torch.compile, FlexAttention, DDP comm hooks, fused
  optimizers, vectorized loss) — each optimization **gated on numerical
  equivalence** against the pre-optimization baseline.
- **Port** the models to run on multiple clusters (Midway/SLURM today; Polaris/PBS
  next) so results are comparable across A100 / H100-class hardware.
- **Keep a durable record** — every benchmark, every decision, every dead-end — in
  a living document so a fresh session (or a teammate) can pick up mid-stream.

### Non-Goals (things we deliberately do NOT do)

- **We do NOT change the science.** The CRPS+KL loss, latitude weighting, the VAE
  reparameterization, the ensemble construction, and normalize↔inverse behavior
  are frozen. An "optimization" that changes model outputs beyond tolerance is a
  **bug**, not a win. (See §4.)
- **We do NOT chase cross-hardware parity numbers.** A 40 GB A100 is not an H100
  NVL; a slower A100 step is expected, not a regression to "fix" by altering the
  model.
- **We do NOT re-train, re-tune, or reproduce forecasts.** This is a
  performance/correctness workbench, not a modeling effort. Accuracy of the
  *science* is out of scope except as the equivalence baseline.
- **We do NOT hand-write custom CUDA/Triton kernels as a first move.** Per the ROI
  analysis (§5), compiler- and framework-level wins come first; bespoke kernels
  are last-resort.
- **We do NOT diverge S2S and the port.** The model/loss/loader code under
  `s2s/v2.0/` is shared and imported by the Lightning port — edits there must
  serve all consumers, never one harness.

---

## 2. The models and how they relate

| Dir | Model | Harness | 4-GPU launch | Instrumentation env |
|---|---|---|---|---|
| `s2s/v2.0/` | **S2S** — Pangu/Plasim 3D-Swin + VAE ensembles, lat-weighted CRPS. The canonical, benchmark-instrumented codebase. | plain PyTorch DDP via `torchrun` | `torchrun --standalone --nproc_per_node=4` (single launcher, spawns 4 local ranks) | `S2S_BENCH_*`, `S2S_NVTX`, `S2S_AMP_DTYPE`, `TORCH_COMPILE_MODE` |
| `s2s-lightning/` | **S2S-Lightning** — a PyTorch Lightning restructuring of S2S. **Imports** `s2s/v2.0` (no copy); only the harness differs. | Lightning `Trainer` + `DDPStrategy` | **Midway:** `srun` with `--ntasks-per-node=4` == devices (Lightning SLURM launcher, one process/GPU). **Polaris:** one `python` (subprocess launcher, no `srun`). | `S2S_BENCH_*`, `S2S_NVTX`, `S2S_PRECISION`, `S2S_TORCH_COMPILE`, `S2S_DDP_BUCKET_CAP_MB` |
| `si/` | **SI (stochastic interpolants)** — the sibling generative-forecasting project (DiT/SiT interpolant models, plus SFNO/UNet/AE variants); the Lightning-layout template S2S-Lightning mirrors. | Lightning `Trainer` + `DDPStrategy` | **Midway:** `srun` with `--ntasks-per-node=4` == devices (identical to the port). **Polaris:** one `python` (no `srun`). | `SI_BENCH_*`, `SI_NVTX`, `SI_PRECISION`, `SI_DDP_*` |

### 2b. The three SFNO codebases (added 2026-07-14 as git subtrees)

Three more trees joined the repo during the Polaris bring-up. They are **not** part of the
S2S/port/SI trio above, and two of them are not ours at all:

| Dir | What it is | Ours? | Model | Stochastic? | Harness | Data |
|---|---|---|---|---|---|---|
| `PanguWeather/` | The group's PanguWeather work — **the current focus** (§2c) | ✅ group | `pangu_plasim` (3D-Swin + VAE) **or** `sfno_plasim` (SFNO) | depends on nettype | plain torch DDP + `torchrun` | E3SM / PLASIM / ERA5 |
| `makani_sfno/` | NVIDIA **makani** + the group's `sfno_training` wrapper | ⚠️ vendor + our wrapper | SFNO (spherical harmonics) | no (as configured) | makani's own trainer (`train_plasim`) | E3SM / PLASIM |
| `physicsnemo_sfno/` | NVIDIA **PhysicsNeMo** `unified_recipe` | ❌ vendor | AFNO / **SFNO** / GraphCast | no (as configured) | hydra recipe | E3SM / ARCO-ERA5 |

Why this matters when you touch them: the vendor trees behave nothing like ours, and their
traps are of a different kind — makani **auto-resumes** and will exit 0 having trained zero
steps; PhysicsNeMo hardcodes `load_checkpoint("./checkpoints")` *relative to CWD* and its
hydra defaults use the PATH form, so `model=sfno` on the CLI is impossible. Both are recorded
in `polaris_pbs_notes.md`.

**Three ways of being stochastic — the split is not "SI vs the rest":**

| family | how | loss |
|---|---|---|
| S2S, the port, PanguWeather `pangu_plasim` | **VAE reparameterization → N ensemble members** | lat-weighted CRPS + KL |
| SI | **stochastic interpolants** (DiT/SiT, diffusion-family) | interpolant objective |
| PanguWeather `sfno_plasim`, makani, physicsnemo | **deterministic** single-shot | `raw_l2` / L2 |

This is why §4.0's VAE noise-fixing hook matters for S2S/Pangu and not only for SI.

### 2c. PanguWeather is a FORK of `s2s/v2.0` — and it is now the focus

`s2s/v2.0` and `PanguWeather/v2.0` are **the same codebase, diverged by purpose.** They both
define `PanguModel_Plasim`, and `networks/pangu.py` is **95% identical** (1093 of ~1148 lines
in common). `s2s/v2.0/utils/losses.py` is a strict **subset** of PanguWeather's (171/171 lines
shared; PanguWeather adds `Raw_MSELoss` and an unweighted `CRPSLoss`).

They diverged along one axis — instrumentation vs science:

| | `s2s/v2.0` | `PanguWeather/v2.0` |
|---|---|---|
| NVTX ranges | **39** | **0** |
| `S2S_BENCH` | 6 | **0** |
| `TORCH_COMPILE_MODE` | 2 | **0** |
| DDP `static_graph` | 4 | **0** |
| SFNO nettype | 0 | **8** |
| bias correction | 0 | **59** |
| finetuning | 1 | **32** |
| `train.py` size | 1,975 lines | **3,985 lines** |

**Both keep the VAE** (`reparameterize` at s2s:463, PanguWeather:448) — the fork did not
remove it. What changes is the **nettype**: `train.py:594` takes the VAE path for
`pangu_plasim`, and `:613` the SFNO path for `sfno_plasim`. The SFNO net
(`networks/modulus_sfno/`) has **no VAE at all**, and the E3SM config uses `loss: "raw_l2"`.
So "s2s vs the Pangu SFNO run" differs by the VAE; "s2s vs PanguWeather `pangu_plasim`" does
not.

> ⚠️ **The forks do not share code.** Unlike S2S↔the port (which share by *import*),
> `PanguWeather/v2.0` is a **copy**. A fix to `s2s/v2.0/networks/pangu.py` does **not** reach
> it, and vice versa. Rule #5 in CLAUDE.md ("shared code serves both") does not apply here —
> this is worse: nothing tells you the other copy drifted.

**FOCUS: the work is now on PanguWeather.** Practical consequences:
1. **PanguWeather has zero instrumentation.** Profiling it means porting the NVTX ranges and
   the `S2S_BENCH` harness from `s2s/v2.0` — that is a prerequisite, not a detail. Keep the
   range names identical to S2S's or `parse_nsys.py` and every prior comparison break
   (CLAUDE.md #10).
2. **It is the only one of the two that runs on Polaris today** — its E3SM SFNO path is GREEN
   (job 7252271, re-verified as a second user by 7253591 at an identical loss of 0.3411),
   while `s2s/v2.0` is blocked on the ERA5 stage.
3. Its full training needs **no data prep** — it reads the E3SM archive directly.
4. Fixes made in `s2s/v2.0` during the S2S phase (e.g. the `--seed` knob in
   `utils/seeding.py`) are **not** in PanguWeather. Port them deliberately; do not assume.

**Key relationship:** S2S and S2S-Lightning are the *same model* with two harnesses
(the port shares `s2s/v2.0` by import — a change to `s2s/v2.0/networks/pangu.py` is
live for both). SI is a *different* model that happens to share the SI Lightning
layout the port was modeled on. So a fix in the shared code affects two of the
three; SI is independent.

> **Naming trap (do not invert):** In `s2s/v2.0/`, `train.py`/`inference.py` are the
> **actively-maintained, bench-instrumented** files (`find_unused_parameters=False,
> static_graph=True` in `train.py`'s DDP wrap, the `S2S_BENCH` framework, live NVTX).
> `train_optimized.py`/`inference_optimized.py` are **older** despite the name
> (`train_optimized.py` uses `find_unused_parameters=True`). Never swap this.

---

## 3. Architecture: the shared model pipeline

The scientific pipeline (identical across S2S and the port; SI is analogous with a
DiT/SiT interpolant core):

```
ERA5 HDF5  ──►  GetDataset / get_data_loader     (normalize; group vars:
(per-cluster    (utils/data_loader_multifiles.py)  upper-air / surface / diagnostic /
 data_dir)                                          land / ocean / const+varying boundary)
                        │
                        ▼
            PanguModel_Plasim  (networks/pangu.py)
            Earth-Specific 3D Swin Transformer
            + VAE reparameterization ──► N ensemble members
                        │
                        ▼
            Loss = latitude-weighted CRPS  (utils/losses.py: Latitude_weighted_CRPSLoss)
                 + KL term                 (Kl_divergence_gaussians)
                        │
                        ▼
            DDP (static_graph=True, find_unused_parameters=False) · AMP · optimizer step
```

**Instrumentation is load-bearing** and must survive every change: `S2S_BENCH`
(warmup/steps/CSV env knobs) times steps GPU-accurately (`cuda.synchronize`
bracketing each step in the training loop); `S2S_NVTX` emits the ~11 NVTX ranges —
for S2S these are `to_ensemble_batch`, `data_prep`, `forward_loss`, `backward`,
`optimizer`, `step_N`, and the `val_*` ranges (across `train.py`'s train/val loops).
**SI's range names differ**
(`preprocess`, `forward_loss`, … per `si/CLAUDE.md`) — do not assume S2S names in an
SI trace or vice-versa. A benchmark whose instrumentation drifted (a dropped range,
a renamed range, a missing CSV column) is not comparable — **treat instrumentation
as part of the contract** and never rename a range casually (it breaks
`parse_nsys.py` and historical comparability).

---

## 4. The correctness oracle: numerical-equivalence-vs-baseline

This is the single most important idea in the project — the analog of "CLASS is the
oracle" for a Boltzmann solver. **Every optimization must reproduce the
pre-optimization model output within a stated tolerance.**

### 4.0 Prerequisites (NONE of these exist yet — build them before optimizing)

The gate is not executable today. Three pieces must be built first (Roadmap item):

- **A seed mechanism.** Canonical `s2s/v2.0/train.py` has **no** `--seed` (it
  hardcodes `torch.manual_seed(world_rank)` in its setup); `si/bench.py` has
  `--seed`; the port defaults to 42. Add a `--seed`/env knob to `train.py` so a
  baseline is reproducible.
- **A tiny deterministic baseline config.** No small config exists — `test.yaml` is
  the full ~79M-param model (it OOMed a 93 GiB H100 at its defaults; the smokes
  only fit it via a `batch_size=1` override). Add a real `tiny_baseline.yaml` (few
  layers/channels or batch 1, `num_data_workers=0`, no wandb/checkpoint).
- **A noise-fixing hook for the VAE.** The reparameterization draw is stochastic,
  and `torch.compile`/FlexAttention can change RNG kernel selection/consumption
  order — so ensemble outputs can differ *on a correct optimization*. The
  comparison must fix the noise (seed a dedicated `torch.Generator` for the reparam
  draw, or inject a fixed epsilon) or compare a deterministic pre-sample quantity.
  **Never** compare a bitwise hash of the stochastic output.

### 4.1 The procedure (once the above exist)

1. **Capture a baseline** before touching the hot path: fixed seed, world size 1
   (add a separate 4-GPU baseline when the change touches DDP), with
   `torch.use_deterministic_algorithms(True)`, `cudnn.benchmark=False`,
   `CUBLAS_WORKSPACE_CONFIG=:4096:8`, and seeded dataloader workers. Over K=20
   steps, record the **per-step loss trajectory** and — with the VAE noise fixed —
   **summary stats** (mean/std/min/max) of the forward output plus the loss scalar.
2. **Make one change** (e.g. enable `TORCH_COMPILE_MODE`).
3. **Re-run the identical config+seed** and compare. **Metric:** max elementwise
   `|a − b| / (|b| + 1e-8)`, reporting the location of the max (never dump raw
   tensors). **Tolerance:** eager-vs-eager fp32 ≤ 1e-5; bf16 or compiled paths
   ≤ 1e-2 (state the exact number used for each change). If it doesn't match, the
   change is wrong — find the cause; **do NOT loosen the tolerance to pass**
   (CLAUDE.md Things-NOT-to-do #1/#11).
4. Record the measured speedup **and** the equivalence result in the living doc.

### 4.2 Storage policy

Baselines are **not** committed as tensors — `.gitignore` blocks `*.pt` and
CLAUDE.md #8 forbids them. Commit only the **text summary** (JSON/CSV of the
per-step losses + output stats + the tolerances used) under `baselines/<model>/`;
keep any raw reference tensor on per-cluster shared storage (e.g.
`/project/pedramh/…/baselines/`), with the path recorded in the living doc.

### 4.3 Invariants the gate protects

- **CRPS sign & normalization** (skill − spread, divided by `num_ensemble_members`) and the **cos-latitude weighting**.
- **VAE / KL** term and the reparameterization draw.
- **normalize ↔ inverse-normalize symmetry** and the **predict-delta add-back**.
- **No train/val leakage**; the `os.path.isfile` guard before `restore_checkpoint`.
- Under Lightning: **no hand-rolled AMP/backward** inside automatic optimization; precision via `Trainer(precision=…)`, not manual autocast/GradScaler; DDP `static_graph` + the dead-module freeze preserved.

---

## 5. Optimization thesis and ROI ladder

Hand-written kernels are usually the *wrong* first lever for this model. The
expected-ROI order (highest leverage first). **Several knobs already exist — enable
them, don't re-implement:**

1. **`torch.compile`** — canonical S2S already plumbs `TORCH_COMPILE_MODE=reduce-overhead|max-autotune`
   in `train.py` (currently unset); the port has `S2S_TORCH_COMPILE`. Turning
   it on is the biggest single lever. Gate on equivalence; expect longer warmup.
   Do **not** write new compile wiring into the shared code.
2. **FlexAttention** for the bias-disabled `EarthAttention3D` path (reproduce the
   SDPA additive-mask output within tolerance; confirm gradients flow through the
   learned bias).
3. **bf16 DDP communication hook** — compress all-reduce. (Precision itself is
   already selectable via the `S2S_AMP_DTYPE=bf16|fp16` env knob in `train.py`.)
4. **Fused AdamW.**
5. **Vectorize the CRPS pairwise/ensemble loop** — last, and only if it profiles hot.

Each rung is a separate small commit with its own equivalence check (§4) and a bench
delta recorded in the living doc. Custom Triton/CUDA is below rung 5 and only if a
profile proves a specific kernel dominates.

---

## 6. Clusters and hardware

**Single source of truth for cluster facts is CLAUDE.md §Cluster facts** — the table
below is a summary; when they disagree, CLAUDE.md wins, and confirmed Polaris values
ultimately live in `polaris_pbs_notes.md`.

| | **Midway** (RCC/UChicago) | **Polaris** (ALCF) — bring-up next |
|---|---|---|
| Scheduler | SLURM (`sbatch`) | PBS Pro (`qsub`) |
| GPU | H100 NVL, ~94 GB, Intel Ice Lake host, PCIe Gen4, NVLink within socket-pairs | **4× A100 40 GB SXM4**/node, AMD "Milan" 32-core |
| Data | ERA5 HDF5 at `/project/pedramh/h5data/h5data` | must be Globus-staged to `/eagle/<project>/…` |
| Env | conda/mamba module (see CLAUDE.md for the exact incantation + the SI conda variant) | `module use /soft/modulefiles && module load conda` |
| Launch | `torchrun` (S2S) / Lightning `srun`, `ntasks-per-node=4` (port, SI) | `torchrun` (S2S) / Lightning **without `srun`** (port, SI) |

**The A100's 40 GB is the binding constraint on Polaris** — much tighter than
Midway's ~94 GB. Midway bench settings (e.g. `exp2` batch 8 → 2/GPU, bf16) may OOM;
Polaris smokes start at per-GPU batch 1. The full Polaris bring-up procedure is
`polaris_handoff_prompt.md` (on `main`), which will produce `polaris_pbs_notes.md`.

---

## 7. Validation and testing strategy

There is **no pytest suite yet** (the `s2s/v2.0/test/` files are ad-hoc scripts, and
no `conftest.py`/`--fast` mode exists). **Building the harness is itself a Roadmap
item** — until it lands, "run the tests" means "run the relevant smoke". Three test
tiers, cheapest first, matched to "small commits, tests pass":

1. **Unit / equivalence tests** *(to be built — run before every commit once they exist)*:
   - CRPS/KL numerical checks (sign, normalization, lat-weighting) vs a reference.
   - normalize↔inverse round-trip identity.
   - A tiny-model forward+backward that runs a few steps and asserts finite loss.
   - The §4 baseline-equivalence diff for any hot-path change.
2. **Smoke run** (1-GPU then 4-GPU, per cluster; **available today**): the model
   completes a handful of steps and writes its bench CSV / prints its success token
   (`SMOKE_OK` for the port smokes). This is the "does it run on this hardware" gate
   for cluster bring-up, and the commit gate until tier-1 exists.
3. **Bench parity** (informational): the `*_BENCH` CSV + `nsys` trace, compared
   within a cluster (never across hardware) to measure a change.

Test-output hygiene (borrowed from clax): tests print ≤10 lines on success, ~20 on
failure; report *max relative error and where it occurs*, not raw tensors; log
verbose diagnostics to files, keep `ERROR <reason>` greppable on one line.

---

## 8. Roadmap

- [ ] **Polaris bring-up** — probe → 1-GPU → 4-GPU smoke for each model via
  PBS; produce `polaris_pbs_notes.md`. (See `polaris_handoff_prompt.md`.)
- [ ] **§4 prerequisites** — add a `--seed` knob to `train.py`, a `tiny_baseline.yaml`,
  and the VAE noise-fixing hook. *(Blocks baseline capture and all optimization.)*
- [ ] **Baseline capture** — the §4 baselines for each model on each cluster.
- [ ] **Test harness** — the tier-1 equivalence/unit tests + a `conftest`-registered
  `--fast` option.
- [ ] **Optimization passes** — the §5 ladder, one gated commit per rung.
- [ ] **Cross-cluster bench report** — A100 vs H100 NVL, per model, honest about
  hardware differences.

Track live status in **CHANGELOG.md** (the living doc), not here.

## 9. Repository layout

```
pedramh-profiling/
├── README.md                 # repo overview + contribution flow
├── DESIGN.md                 # this file — what & why
├── CLAUDE.md                 # how to work here (conventions, don'ts, model policy); SSOT for cluster facts
├── CHANGELOG.md              # the living doc: dated progress + decisions log
├── polaris_handoff_prompt.md # Polaris (PBS) bring-up brief
├── s2s/            v2.0/     # canonical S2S (model, losses, loaders, HPC scripts)
├── s2s-lightning/            # the Lightning port (imports s2s/v2.0)
└── si/                       # the SI model (own CLAUDE.md for SI-specific bench)
```

## 10. Open questions and risks

- **§4 is not yet executable** — the seed knob, tiny config, and VAE noise-fixing
  hook (§4.0) do not exist. Until they do, "equivalence" has nothing reproducible to
  compare to. This is the first real risk; build §4.0 before optimizing.
- **Lightning launcher on PBS** — the port/SI rely on Lightning's SLURM launcher
  (`srun`, `ntasks==devices`) on Midway; the PBS path must not let Lightning think
  it's under SLURM (single `python`, no `srun`; see the handoff). Single-node only
  for now; multi-node needs a `ClusterEnvironment`.
- **A100 memory** — some Midway configs will not fit; document the smallest config
  that does, don't silently shrink the model.
- **Instrumentation drift** — an edit that drops an NVTX range or a CSV column
  silently invalidates comparisons, and S2S vs SI range names differ. The
  equivalence gate does not catch this; review bench plumbing explicitly.
- **Shared-code blast radius** — a change under `s2s/v2.0/` touches S2S *and* the
  port. Run both models' smokes after any shared-code edit.
