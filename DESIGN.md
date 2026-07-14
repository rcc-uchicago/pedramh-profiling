# pedramh-profiling: Design Specification

A benchmarking + GPU-profiling + optimization workbench for the Pedram
Hassanzadeh group's probabilistic **subseasonal-to-seasonal (S2S)** weather
models. The goal is to make these models **faster on HPC GPUs without changing
what they compute** — measured, gated, and reproducible across clusters.

Development guide and conventions are in **CLAUDE.md** (read that for how to
work here). This document covers *what* we are building and *why*.

---

## Table of Contents

1. [Goals and Non-Goals](#1-goals-and-non-goals)
2. [The three models and how they relate](#2-the-three-models-and-how-they-relate)
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

- **Measure** training and inference throughput of all three models on HPC GPUs,
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
- **We do NOT diverge the three models.** The model/loss/loader code under
  `s2s/v2.0/` is shared and imported by the Lightning port — edits there must
  serve all consumers, never one harness.

---

## 2. The three models and how they relate

| Dir | Model | Harness | 4-GPU launch | Instrumentation env prefix |
|---|---|---|---|---|
| `s2s/v2.0/` | **S2S** — Pangu/Plasim 3D-Swin + VAE ensembles, lat-weighted CRPS. The canonical, benchmark-instrumented codebase. | plain PyTorch DDP via `torchrun` | `torchrun --standalone --nproc_per_node=4` | `S2S_BENCH_*`, `S2S_NVTX` |
| `s2s-lightning/` | **S2S-Lightning** — a PyTorch Lightning restructuring of S2S. **Imports** `s2s/v2.0` (no copy); only the harness differs. | Lightning `Trainer` + `DDPStrategy` | one `python` w/ `devices=4` (SLURM launcher on Midway) | `S2S_BENCH_*`, `S2S_PRECISION` |
| `si/` | **SI** — the sibling DiT/SiT-style generative weather model; the Lightning-layout template S2S-Lightning mirrors. | Lightning `Trainer` + `DDPStrategy` | one `python` w/ `devices=4` | `SI_BENCH_*`, `SI_NVTX`, `SI_PRECISION` |

**Key relationship:** S2S and S2S-Lightning are the *same model* with two harnesses
(the port shares `s2s/v2.0` by import — a change to `s2s/v2.0/networks/pangu.py` is
live for both). SI is a *different* model that happens to share the SNFO→SI
Lightning layout the port was modeled on. So a fix in the shared code affects two
of the three; SI is independent.

> **Naming trap (do not invert):** In `s2s/v2.0/`, `train.py`/`inference.py` are the
> **actively-maintained, bench-instrumented** files (`find_unused_parameters=False,
> static_graph=True`, the `S2S_BENCH` framework, live NVTX). `train_optimized.py`/
> `inference_optimized.py` are **older** despite the name. Never swap this attribution.

---

## 3. Architecture: the shared model pipeline

The scientific pipeline (identical across S2S and the port; SI is analogous with a
DiT/SiT core):

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
(warmup/steps/CSV env knobs) times steps GPU-accurately (`cuda.synchronize` around
each step); `S2S_NVTX` emits the ~12 NVTX ranges (`preprocess`, `forward_loss`,
`backward`, `optimizer`, …) that the `nsys` traces key on. SI mirrors this with
`SI_BENCH_*` / `SI_NVTX`. **A benchmark whose instrumentation drifted is not
comparable — treat instrumentation as part of the contract.**

---

## 4. The correctness oracle: numerical-equivalence-vs-baseline

This is the single most important idea in the project — the analog of "CLASS is the
oracle" for a Boltzmann solver, or "GCC is the oracle" for a C compiler.

**Every optimization must reproduce the pre-optimization model output within a
stated tolerance.** The workflow:

1. **Capture a baseline** *before* touching the hot path: with a fixed seed and a
   tiny deterministic config, record (a) the loss trajectory over K steps, (b) a
   forward-pass output tensor (ensemble members) hash/stats, and (c) the bench CSV.
   Store it under `baselines/<model>/<name>.{pt,csv}`.
2. **Make one change** (e.g. enable torch.compile).
3. **Re-run the same fixed config** and diff against the baseline: loss curve and
   forward output must match to tolerance (bf16 paths: relative error ≤ ~1e-2;
   fp32/eager-vs-eager: much tighter). If it doesn't match, the change is wrong —
   find the real cause, do **not** loosen the tolerance to pass.
4. Only then keep the change, and record the measured speedup + the equivalence
   result in the living doc.

Correctness invariants the equivalence gate protects (these are where silent
science bugs hide):
- **CRPS sign & normalization** (skill − spread, divided by `num_ensemble_members`) and the **cos-latitude weighting**.
- **VAE / KL** term and the reparameterization draw.
- **normalize ↔ inverse-normalize symmetry** and the **predict-delta add-back**.
- **No train/val leakage**; the `os.path.isfile` guard before `restore_checkpoint`.
- Under Lightning: **no hand-rolled AMP/backward** inside automatic optimization; precision via `Trainer(precision=…)`, not manual autocast/GradScaler; DDP `static_graph` + the dead-module freeze preserved.

---

## 5. Optimization thesis and ROI ladder

Hand-written kernels are usually the *wrong* first lever for this model. The
expected-ROI order (highest leverage first):

1. **`torch.compile`** — currently OFF; turning it on (with the right mode) is the
   biggest single lever. Gate on equivalence; expect longer warmup.
2. **FlexAttention** for the bias-disabled `EarthAttention3D` path (reproduce the
   SDPA additive-mask output within tolerance; confirm gradients flow through the
   learned bias).
3. **bf16 DDP communication hook** — compress all-reduce.
4. **Fused AdamW.**
5. **Vectorize the CRPS pairwise/ensemble loop** — last, and only if it profiles hot.

Each rung is a separate small commit with its own equivalence check and a bench
delta recorded in the living doc. Custom Triton/CUDA is below rung 5 and only if a
profile proves a specific kernel dominates.

---

## 6. Clusters and hardware

| | **Midway** (RCC/UChicago) | **Polaris** (ALCF) — bring-up next |
|---|---|---|
| Scheduler | SLURM (`sbatch`) | PBS Pro (`qsub`) |
| GPU | H100 NVL, ~94 GB, Intel Ice Lake host, PCIe Gen4, NVLink within socket-pairs | **4× A100 40 GB SXM4**/node, AMD "Milan" 32-core |
| Data | ERA5 HDF5 at `/project/pedramh/h5data/h5data` | must be Globus-staged to `/eagle/<project>/…` |
| Env | `module load python/miniforge-25.3.0 && mamba activate …` | `module use /soft/modulefiles && module load conda` |
| Launch | `torchrun` (S2S) / Lightning `srun` (port, SI) | `torchrun` (S2S) / Lightning **without `srun`** (port, SI) |

**The A100's 40 GB is the binding constraint on Polaris** — much tighter than
Midway's ~94 GB. Midway bench settings (e.g. `exp2` batch 8 → 2/GPU, bf16) may OOM;
Polaris smokes start at per-GPU batch 1. Full Polaris bring-up procedure:
`polaris_handoff_prompt.md` → will produce `polaris_pbs_notes.md`.

---

## 7. Validation and testing strategy

There is **no inherited pytest suite** (the `s2s/v2.0/test/` files are ad-hoc
scripts). Building the harness is itself a roadmap item. Three test tiers, cheapest
first, matched to "small commits, tests pass":

1. **Unit / equivalence tests** (fast, CPU-or-1-GPU, run before every commit):
   - CRPS/KL numerical checks (sign, normalization, lat-weighting) vs a reference.
   - normalize↔inverse round-trip identity.
   - A tiny-model forward+backward that runs a few steps and asserts finite loss.
   - The §4 baseline-equivalence diff for any hot-path change.
2. **Smoke run** (1-GPU then 4-GPU, per cluster): the model completes a handful of
   steps and writes its bench CSV / prints its success token. This is the "does it
   run on this hardware" gate for cluster bring-up.
3. **Bench parity** (informational): the `*_BENCH` CSV + `nsys` trace, compared
   within a cluster (never across hardware) to measure a change.

Test-output hygiene (borrowed from clax): tests print ≤10 lines on success, ~20 on
failure; report *max relative error and where it occurs*, not raw tensors; log
verbose diagnostics to files, keep `ERROR <reason>` greppable on one line.

---

## 8. Roadmap

- [ ] **Polaris bring-up** — probe → 1-GPU → 4-GPU smoke for all three models via
  PBS; produce `polaris_pbs_notes.md`. (See `polaris_handoff_prompt.md`.)
- [ ] **Baseline capture** — the §4 baselines for each model on each cluster.
- [ ] **Test harness** — the tier-1 equivalence/unit tests + a `--fast` smoke.
- [ ] **Optimization passes** — the §5 ladder, one gated commit per rung.
- [ ] **Cross-cluster bench report** — A100 vs H100 NVL, per model, honest about
  hardware differences.

Track live status in **CHANGELOG.md** (the living doc), not here.

## 9. Repository layout

```
pedramh-profiling/
├── DESIGN.md                 # this file — what & why
├── CLAUDE.md                 # how to work here (conventions, don'ts, Fable 5)
├── CHANGELOG.md              # the living doc: dated progress + decisions log
├── polaris_handoff_prompt.md # Polaris (PBS) bring-up brief
├── s2s/            v2.0/     # canonical S2S (model, losses, loaders, HPC scripts)
├── s2s-lightning/            # the Lightning port (imports s2s/v2.0)
└── si/                       # the SI model (own CLAUDE.md for SI-specific bench)
```

## 10. Open questions and risks

- **No baseline captured yet** — until §4 baselines exist, "equivalence" has nothing
  to compare to. This is the first real risk; capture baselines before optimizing.
- **Lightning launcher on PBS** — the port/SI rely on Lightning's SLURM launcher on
  Midway; the PBS path must not let Lightning think it's under SLURM (see the
  handoff). Single-node only for now; multi-node needs a `ClusterEnvironment`.
- **A100 memory** — some Midway configs will not fit; document the smallest config
  that does, don't silently shrink the model.
- **Instrumentation drift** — an edit that drops an NVTX range or a CSV column
  silently invalidates comparisons. The equivalence gate does not catch this;
  review bench plumbing explicitly.
- **Shared-code blast radius** — a change under `s2s/v2.0/` touches S2S *and* the
  port. Run both models' smokes after any shared-code edit.
