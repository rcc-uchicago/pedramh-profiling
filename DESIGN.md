# pedramh-profiling: Design Specification

A bring-up, training, and GPU-performance workbench for the Pedram Hassanzadeh
group's probabilistic **subseasonal-to-seasonal (S2S)** weather models. The work
runs in **two phases, per model, per cluster**:

- **Phase 1 — bring-up:** get the model **training and running inference on real
  data on the target cluster** — environment, data prep, scheduler scripts, then
  real training runs that produce evaluatable models. "Green" means reproducible
  by a second user, not just the installer.
- **Phase 2 — performance:** profile it, then optimize the hot path one gated
  step at a time, **without changing what it computes** (§4).

Phases overlap across models **and clusters**: **`PanguWeather/` is the current
focus**, already in Phase 2 on Polaris (profiled — `polaris_bench_report.md`).
On Polaris, `s2s`/the port sit in Phase 1, blocked on an ERA5 stage — but both
already have Phase-2 evidence **on Midway** (`s2s/v2.0/bench_report.md`,
`s2s-lightning/LIGHTNING_PORT.md`). PanguWeather is a ~95%-identical **copy** of
`s2s/v2.0` (§2c) — no shared code, so a fix in one silently misses the other.

**CLAUDE.md** is how to work here and the **single source of truth for cluster
facts** — this document deliberately does not repeat that table. **CHANGELOG.md**
is live status. This document is *what* we are building and *why*.

## 1. Goals and Non-Goals

*(Scope widened 2026-07-16, owner's decision: training is now in scope. Older
docs — including CLAUDE.md's scope line — may still say "NOT retraining".)*

### Goals

- **Phase 1 — bring each model up**: environment, staged/prepared data,
  scheduler scripts, then **training runs that produce evaluatable models** and
  **inference** on real data (data prep is a Phase-1 prerequisite — §8).
- **Phase 2 — measure, then optimize**: profile with the NVTX / `*_BENCH` / CSV
  instrumentation, then climb the §5 ladder — **every hot-path change gated on
  numerical equivalence** against the pre-change baseline (§4).
- **Comparable across clusters** (Midway H100 NVL, Polaris 4×A100-40GB), honest
  about hardware differences; **a durable record** of every benchmark, decision,
  and dead-end in CHANGELOG.md + the per-cluster notes.

### Non-Goals (still deliberate)

- **We do NOT change the science.** The division of labor: **bring-up and
  training are ours; the science is jesswan's** — variable sets, fill values,
  channel roles, loss definitions, the physics. Training runs that produce
  evaluatable models are Phase-1 work; *changing what a model computes* is out
  of bounds without jesswan's sign-off. In the hot path this stays mechanical:
  an "optimization" that moves outputs beyond tolerance is a **bug** (§4).
- **We do NOT chase forecast reproduction for its own sake** — we train to get
  evaluatable models and exercise the real pipeline; matching published skill
  scores is not the deliverable.
- **We do NOT chase cross-hardware parity numbers** — a slower A100 step is
  expected, not a regression to "fix" by altering the model.
- **We do NOT hand-write custom CUDA/Triton kernels as a first move** —
  compiler- and framework-level wins first (§5); bespoke kernels last-resort.
- **We do NOT diverge S2S and the port** — `s2s/v2.0/` is shared by **import**
  with the Lightning port; edits there must serve both. (The PanguWeather fork
  is the opposite trap: a **copy**, so fixes do NOT propagate — §2c.)

## 2. The models — all six codebases

| Dir | Ours? | Model (loss) | Stochastic? | Harness · 4-GPU launch | Bench env prefix | Data |
|---|---|---|---|---|---|---|
| `s2s/v2.0/` | ✅ canonical | **S2S** — Pangu/Plasim 3D-Swin + VAE ensembles (lat-weighted CRPS + KL). Bench-instrumented reference. | ✅ VAE reparameterization → N ensemble members | plain torch DDP · `torchrun --standalone --nproc_per_node=4` (both clusters) | `S2S_BENCH_*`, `S2S_NVTX`, `S2S_AMP_DTYPE`, `TORCH_COMPILE_MODE`; `--seed`/`--deterministic` | ERA5 HDF5 (not staged on Polaris → blocked there) |
| `s2s-lightning/` | ✅ | **S2S-Lightning** — the *same model*: **imports** `s2s/v2.0` (no copy); only the harness differs. | ✅ same VAE | Lightning `Trainer`+`DDPStrategy` · Midway: `srun`, `--ntasks-per-node=4` == devices; Polaris: one `python`, **never `srun`** | `S2S_BENCH_*`, `S2S_NVTX`, `S2S_PRECISION`, `S2S_TORCH_COMPILE`, `S2S_DDP_BUCKET_CAP_MB` | ERA5 HDF5 (same block) |
| `si/` | ✅ | **SI** — stochastic interpolants (DiT/SiT + SFNO/UNet/AE variants); interpolant objective. A *different* model that shares only the Lightning layout the port mirrors. | ✅ interpolants (diffusion-family) | Lightning `Trainer`+`DDPStrategy` · same launch shape as the port | `SI_BENCH_*`, `SI_NVTX`, `SI_PRECISION`, `SI_DDP_*` | AMIP (E3SM staged on Polaris; GREEN there) |
| `PanguWeather/` | ✅ group | **Fork of `s2s/v2.0`** (§2c). nettype `pangu_plasim` = 3D-Swin + VAE (CRPS + KL); nettype `sfno_plasim` = SFNO (`raw_l2`) — the green Polaris path, **1.18 B params** in the E3SM config (not ~79M; `polaris_bench_report.md` §1). | `pangu_plasim` ✅ (VAE) · `sfno_plasim` ❌ | plain torch DDP · `torchrun --standalone --nproc_per_node=$NPROC` | `PANGU_BENCH_*`, `PANGU_NVTX`, `TORCH_COMPILE_MODE` (harness ported 2026-07-15; knobs renamed from `S2S_*` 2026-07-16 — PanguWeather owns its own, nothing shared read them; **range names + CSV columns stay identical to s2s**. A stale `S2S_BENCH*` now errors: `LEGACY_BENCH_ENV`); seed via `--global_seed`; precision from the YAML (`amp_dtype`), **not** an env knob | E3SM h5 directly (staged) / PLASIM (`pangu_plasim` blocked on it) / ERA5 |
| `makani_sfno/` | ⚠️ vendor + our wrapper | NVIDIA **makani** + the group's `sfno_training` wrapper — SFNO (L2). | ❌ (as configured) | makani's own trainer (`train_plasim`) · Polaris: `python -m torch.distributed.run` from the isolated SFNO venv; `--batch_size` is **global** (the rank count must divide it) | none | E3SM (packed) / PLASIM |
| `physicsnemo_sfno/` | ❌ vendor | NVIDIA **PhysicsNeMo** `unified_recipe` — AFNO / **SFNO** / GraphCast (L2). | ❌ (as configured) | hydra recipe · Polaris: `python -m torch.distributed.run` from the SFNO venv | none | E3SM → zarr (conversion NOT yet cleared for the full run — §8) / ARCO-ERA5 |

- **The stochastic column is why §4.0's VAE noise hook matters for S2S / the
  port / Pangu `pangu_plasim`, not only for SI** — three mechanisms (VAE draw,
  interpolants, none); the split is not "SI vs the rest".
- **Vendor traps** (full list: `polaris_pbs_notes.md`): makani **auto-resumes**
  and can exit 0 having trained zero steps; PhysicsNeMo hardcodes CWD-relative
  checkpoints and its hydra PATH-form defaults break `model=sfno` on the CLI.
- **Naming trap** (CLAUDE.md #4): in `s2s/v2.0/`, `train.py`/`inference.py` are
  the maintained, instrumented files; the `_optimized` ones are **older**.

### 2c. PanguWeather is a FORK of `s2s/v2.0` — divergence evidence

*(Numbered §2c so external citations — CHANGELOG, `polaris_bench_report.md`, the
handoff prompts — stay valid; the former §2/§2b tables are merged above.)*

The two are **the same codebase, diverged by purpose**, sharing code by **copy,
not import** — nothing tells you the other fork drifted. At the 2026-07-15
audit: `networks/pangu.py` ~95% identical (1093 of ~1148 lines);
`s2s/v2.0/utils/losses.py` a strict subset of PanguWeather's (171/171 shared;
PanguWeather adds `Raw_MSELoss` + an unweighted `CRPSLoss`). One axis of
divergence — instrumentation vs science (counts **as audited 2026-07-15, before
the harness port**):

| | `s2s/v2.0` | `PanguWeather/v2.0` |
|---|---|---|
| NVTX ranges | **39** | 0 → **ported 2026-07-15** |
| bench harness | 6 (`S2S_BENCH`) | 0 → **ported**, renamed `PANGU_BENCH` 2026-07-16 |
| `TORCH_COMPILE_MODE` | 2 | 0 → **wired 2026-07-15** |
| DDP `static_graph` | 4 | **0** — do NOT copy blindly; needs s2s's dead-module freeze (`polaris_bench_report.md` §6b) |
| SFNO nettype | 0 | **8** |
| bias correction | 0 | **59** |
| finetuning | 1 | **32** |

The ported harness reproduces the legacy path **bit-identically** (job 7255505
== the green 7253591, loss 0.3411). The drift is **bidirectional** — each fork
has things the other lacks (e.g. `gradient_as_bucket_view` only in
PanguWeather); full table: `polaris_bench_report.md` §6b. **Audit before
assuming any fix reached the other fork.**

**Both forks keep the VAE** (`reparameterize`: s2s `pangu.py:463`, PanguWeather
`pangu.py:449`). What changes is the **nettype**: PanguWeather `train.py:621`
takes the VAE path for `pangu_plasim`, `:640` the SFNO path for `sfno_plasim`;
the SFNO net (`networks/modulus_sfno/`) has **no VAE at all**, and the E3SM
config uses `loss: "raw_l2"`. So "s2s vs the Pangu SFNO run" differs by the
VAE; "s2s vs PanguWeather `pangu_plasim`" does not.

## 3. Architecture: the shared model pipeline

Identical across S2S and the port (SI is analogous with a DiT/SiT interpolant
core; PanguWeather's `sfno_plasim` branch bypasses this model entirely):

```
ERA5/E3SM HDF5 (per-cluster data_dir)
  ─► GetDataset / get_data_loader   utils/data_loader_multifiles.py — normalize; group vars
                                    (upper-air/surface/diagnostic/land/ocean/const+varying boundary)
  ─► PanguModel_Plasim              networks/pangu.py — Earth-Specific 3D Swin Transformer
                                    + VAE reparameterization ─► N ensemble members
  ─► lat-weighted CRPS + KL         utils/losses.py — Latitude_weighted_CRPSLoss,
                                    Kl_divergence_gaussians
  ─► DDP (static_graph=True, find_unused_parameters=False) · AMP · optimizer step
```

**Instrumentation is part of the contract** (CLAUDE.md #10): the bench harness
(`S2S_BENCH` in s2s/the port, `PANGU_BENCH` in PanguWeather since 2026-07-16 —
the env knobs are per-project, the **range names and CSV columns are not**) times
steps GPU-accurately (`cuda.synchronize`-bracketed); `S2S_NVTX`/`PANGU_NVTX` emits the named
ranges (`to_ensemble_batch`, `data_prep`, `forward_loss`, `backward`,
`optimizer`, `step_N`, `val_*`; PanguWeather adds `ema` and has no `val_*` ranges). **SI's range names
differ** (`preprocess`, `forward_loss`, … — `si/CLAUDE.md`). A dropped/renamed
range or CSV column silently invalidates every comparison and breaks
`parse_nsys.py`; the §4 gate does NOT catch this — review bench plumbing explicitly.

## 4. The correctness oracle: numerical-equivalence-vs-baseline

The single most important idea in the project. **Every optimization must
reproduce the pre-optimization model output within a stated tolerance.**

### 4.0 Prerequisites — status (all three MET on PanguWeather; partial on s2s)

- **A seed mechanism** — ✅ both forks. s2s: `--seed`/`$S2S_SEED`/YAML +
  `--deterministic` (`s2s/v2.0/utils/seeding.py`; GPU-verified `SEEDING_OK`).
  PanguWeather **already had** `--global_seed` → `seed_torch()`, stronger than
  s2s's legacy path — do **not** port `seeding.py` across (two mechanisms
  racing the same global RNGs is a regression).
- **A tiny deterministic baseline config** — ✅ PanguWeather
  `config/tiny_baseline.yaml`: 7.17M params (165× under the real 1.18 B),
  0.023 s/step, 1.00 GB, run green (job 7255583). ❌ s2s: none — its `test.yaml`
  is the full ~79M-param Swin model (CLAUDE.md #12).
- **A VAE noise-fixing hook** — the reparameterization draw is stochastic, and
  `torch.compile`/FlexAttention can change RNG kernel selection/consumption
  order, so ensembles can differ *on a correct optimization*. Fix the noise
  (dedicated seeded `torch.Generator` or a fixed epsilon) or compare a
  deterministic pre-sample quantity; **never** hash the stochastic output.
  ✅ PanguWeather `utils/vae_noise.py` (16 tests, `VAE_NOISE_OK`; inert on
  `sfno_plasim` — no VAE). ❌ s2s: none.

⇒ **Baseline capture on PanguWeather is unblocked**; an s2s baseline still
needs its own tiny config + noise hook.

### 4.1 The procedure

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
   (CLAUDE.md rules #1/#11).
4. Record the measured speedup **and** the equivalence result in the living doc.

### 4.2 Storage policy

Baselines are **not** committed as tensors — `.gitignore` blocks `*.pt` and
CLAUDE.md #8 forbids them. Commit only the **text summary** (JSON/CSV of the
per-step losses + output stats + the tolerances used) under `baselines/<model>/`;
keep any raw reference tensor on per-cluster shared storage, with the path
recorded in the living doc.

### 4.3 Invariants the gate protects

- **CRPS sign & normalization** (skill − spread, divided by `num_ensemble_members`) and the **cos-latitude weighting**.
- **VAE / KL** term and the reparameterization draw.
- **normalize ↔ inverse-normalize symmetry** and the **predict-delta add-back**.
- **No train/val leakage**; the `os.path.isfile` guard before `restore_checkpoint`.
- Under Lightning: **no hand-rolled AMP/backward** inside automatic optimization; precision via `Trainer(precision=…)`, not manual autocast/GradScaler; DDP `static_graph` + the dead-module freeze preserved.

## 5. Phase 2: optimization thesis and ROI ladder

Phase 2 starts only after the model is up (Phase 1) and **profiled**. Two
independent profiles agree on the thesis: the 2026-05 **Midway H100 campaign**
(`s2s/v2.0/bench_report.md` — elementwise ops the single largest GPU-time
consumer, matmul 6th) and the first **Polaris** profile (PanguWeather SFNO,
4×A100: **GPU-bound**, loader idle 0.7%; **elementwise-bound**, 61% of GPU time
pointwise vs 15% GEMM ⇒ fusion-starved — `polaris_bench_report.md`). Expected-ROI
order, highest leverage first. **Several knobs already exist — enable, don't re-implement:**

1. **`torch.compile`** — plumbed as `TORCH_COMPILE_MODE=reduce-overhead|max-autotune`
   in both `s2s/v2.0/train.py` and (since 2026-07-15) `PanguWeather/v2.0`; the
   port has `S2S_TORCH_COMPILE`. Deliberately left unset until the §4 baseline
   exists. Gate on equivalence; expect longer warmup (raise `PANGU_BENCH_WARMUP`, or `S2S_BENCH_WARMUP` on s2s).
   Do **not** write new compile wiring into the shared code.
2. **FlexAttention** for the bias-disabled `EarthAttention3D` path (reproduce
   the SDPA additive-mask output within tolerance; confirm gradients flow
   through the learned bias).
3. **bf16 DDP communication hook** — compress all-reduce. (Precision itself is
   already selectable: `S2S_AMP_DTYPE` in s2s; the YAML `amp_dtype` in PanguWeather.)
4. **Fused AdamW.**
5. **Vectorize the CRPS pairwise/ensemble loop** — last, and only if it profiles hot.

Each rung is a separate small commit with its own equivalence check (§4) and a
bench delta recorded in the living doc. Custom Triton/CUDA is below rung 5 and
only if a profile proves a specific kernel dominates.

## 6. Clusters and hardware

**All cluster facts live in CLAUDE.md §Cluster facts**; confirmed Polaris
detail in `polaris_pbs_notes.md`. The one hardware fact that shapes design:
**the A100's 40 GB is the binding constraint on Polaris** — Midway configs
sized for ~94 GB may OOM; Polaris smokes start at per-GPU batch 1. Document the
smallest config that fits; don't silently shrink the model.

## 7. Validation and testing strategy

There is **no pytest suite yet** — three self-running test files exist
(`SEEDING_OK`, `BENCH_INSTR_OK`, `VAE_NOISE_OK`; the rest of the `test/` dirs
are ad-hoc scripts) but no `conftest.py`/`--fast`;
building the harness is a Roadmap item. Until then, "run the tests" = "run the
relevant smoke". Three tiers, cheapest first:

1. **Unit / equivalence** *(partially built)*: CRPS/KL checks vs a reference;
   normalize↔inverse round-trip; tiny-model forward+backward with finite loss;
   the §4 baseline diff for any hot-path change.
2. **Smoke run** (1-GPU then 4-GPU, per cluster; **available today**): a few
   steps ending in the success token / bench-CSV row. The bring-up gate, and
   the commit gate until tier 1 exists. Key on the token, never exit code
   alone — makani exited 0 having trained zero steps.
3. **Bench parity** (informational): `*_BENCH` CSV + `nsys` trace, compared
   within a cluster only.

Output hygiene: ≤10 lines on success, ~20 on failure; report *max relative
error and where it occurs*, not raw tensors; keep `ERROR <reason>` greppable
on one line.

## 8. Roadmap

Live status is **CHANGELOG.md** (status table + smoke matrix), not here.

### Phase 1 — bring-up → training → inference (per model, per cluster)

- [x] **Polaris bring-up** — all 4 runnable models GREEN on 4×A100
  (`polaris_pbs_notes.md`). Pangu + SI additionally pass the second-user
  *simulation* (`PYTHONNOUSERSITE=1`; the first **real** second-user run is
  still jesswan's); makani/physicsnemo have no second-user check yet. s2s +
  port scripts delivered (import chain verified; the data smoke has never
  executed), blocked on the ERA5 stage.
- [ ] **Clear the E3SM data prep for the full conversion** — 4 open converter
  defects + 5 open decisions (jesswan/us — one is ours:
  `polaris_data_prep_decisions.md`): `polaris_data_prep_handoff_prompt.md`;
  measured variable reference + risks R1–R12: `polaris_e3sm_variable_reference.md`;
  training impact: `data_for_training.md`. makani's converter is unaudited.
- [ ] **Full training runs → evaluatable models** — PanguWeather-SFNO first
  (reads the staged E3SM h5 directly; no zarr prep in its path).
- [ ] **Inference on the trained models**, on-cluster.
- [ ] **ERA5 Globus stage** → unblocks s2s + the port on Polaris.

### Phase 2 — profile → optimize (per model; PanguWeather is here now)

- [x] **Instrumentation port + first Polaris profile** (PanguWeather SFNO,
  4×A100) — `polaris_bench_report.md`; SI / makani / physicsnemo not yet
  profiled on Polaris (SI's Midway bench: `si/bench_midway_notes.md`).
- [x] **§4.0 prerequisites on PanguWeather** — all three met.
- [ ] **Capture the §4.1 baseline** — unblocked; nothing left to build.
- [ ] **The §5 ladder**, one gated commit per rung, starting `torch.compile`.
- [ ] **Profile SI / makani / physicsnemo** (SI cheapest: harness + green bench exist).
- [ ] **Test harness** — tier-1 tests + a `conftest`-registered `--fast`.
- [ ] **Cross-cluster bench report** — A100 vs H100 NVL, honest about hardware.

## 9. Repository layout

```
pedramh-profiling/
├── README.md                    # repo overview + contribution flow
├── DESIGN.md                    # this file — what & why
├── CLAUDE.md                    # how to work here; SSOT for cluster facts
├── CHANGELOG.md                 # the living doc: dated progress + decisions log
├── polaris_pbs_notes.md         # confirmed Polaris facts, traps, smoke evidence
├── polaris_bench_report.md      # PanguWeather SFNO profile + fork-drift table (§6b)
├── polaris_data_prep_*.md       # converter defects + open science decisions
├── polaris_e3sm_variable_reference.md   # measured variable reference, risks R1–R12
├── data_for_training.md         # which data risks actually affect training
├── s2s/            v2.0/        # canonical S2S (model, losses, loaders, HPC scripts)
├── s2s-lightning/               # the Lightning port (imports s2s/v2.0)
├── si/                          # the SI model (own CLAUDE.md for SI bench detail)
├── PanguWeather/                # group fork of s2s/v2.0 (§2c) — current focus
├── makani_sfno/                 # NVIDIA makani + group wrapper (git subtree)
└── physicsnemo_sfno/            # NVIDIA PhysicsNeMo unified_recipe (git subtree)
```

## 10. Open questions and risks

- **The sharpest open science question is R2, frozen ocean forcing** — binary,
  and only jesswan can settle it; if it is a defect, everything trained on the
  archive used wrong forcing (`data_for_training.md`). Until the data-prep
  decisions land, the full ~1 TB conversion stays uncleared (§8).
- **An s2s baseline is still uncapturable** — its tiny config and VAE noise
  hook don't exist (§4.0); PanguWeather's do.
- **Lightning on PBS is single-node only** — the no-`srun` launch is proven
  green; multi-node needs a `ClusterEnvironment`.
- **Fork drift and instrumentation drift are silent** — the §4 gate catches
  neither; audit deliberately (§2c, §3).
