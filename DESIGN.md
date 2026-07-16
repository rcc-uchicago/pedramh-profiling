# pedramh-profiling: Design Specification

Bring-up, training, and GPU-performance workbench for the Pedram Hassanzadeh
group's S2S weather models. Two phases, **per model, per cluster** (they overlap):

- **Phase 1 — bring-up:** env → staged/prepared data → scheduler scripts →
  **training runs that produce evaluatable models** + inference on real data.
  "Green" = reproducible by a second user, not just the installer.
- **Phase 2 — performance:** profile, then optimize the hot path one gated step
  at a time, without changing what it computes (§4).

Position: **`PanguWeather/` is the focus** — profiled on Polaris (Phase-2
evidence: `polaris_bench_report.md`) while its Phase-1 full training run is still
open (§8). s2s + the port: Phase 1 on Polaris blocked on an ERA5 stage; Phase-2
evidence exists on Midway (`s2s/v2.0/bench_report.md`,
`s2s-lightning/LIGHTNING_PORT.md`). CLAUDE.md owns house rules + cluster facts —
not repeated here; CHANGELOG.md is live status.

## 1. Goals and Non-Goals

*(Scope widened 2026-07-16, owner's decision: training is in scope. Docs
predating that may still say "NOT retraining".)*

**Goals:** Phase 1 per model per cluster (data prep is a prerequisite — §8);
Phase 2 = profile with the NVTX / `*_BENCH` / CSV instrumentation, then the §5
ladder, **every hot-path change gated on numerical equivalence** (§4); results
comparable across clusters; every benchmark, decision, and dead-end recorded in
CHANGELOG.md + the per-cluster notes.

**Non-goals (deliberate):**

- **No science changes.** Bring-up and training are ours; **the science is
  jesswan's** — variable sets, fill values, channel roles, loss definitions,
  physics. Changing what a model computes needs her sign-off; an "optimization"
  that moves outputs beyond tolerance is a **bug** (§4).
- **No forecast-repro chasing** — matching published skill scores is not the
  deliverable.
- **No cross-hardware parity chasing** — a slower A100 step is expected, not a
  regression to "fix" by altering the model.
- **No hand-written CUDA/Triton as a first move** — compiler/framework wins
  first (§5).
- **No diverging S2S and the port** — `s2s/v2.0/` is shared **by import**;
  edits must serve both. (PanguWeather is the opposite trap: a **copy**, so
  fixes do NOT propagate — §2c.)

## 2. The models — all six codebases

| Dir | Ours? | Model (loss) | Stochastic? | Harness · 4-GPU launch | Bench env prefix | Data |
|---|---|---|---|---|---|---|
| `s2s/v2.0/` | ✅ canonical | **S2S** — Pangu/Plasim 3D-Swin + VAE ensembles (lat-weighted CRPS + KL). Bench-instrumented reference. | ✅ VAE reparameterization → N ensemble members | plain torch DDP · `torchrun --standalone --nproc_per_node=4` (both clusters) | `S2S_BENCH_*`, `S2S_NVTX`, `S2S_AMP_DTYPE`, `TORCH_COMPILE_MODE`; `--seed`/`--deterministic` | ERA5 HDF5 (not staged on Polaris → blocked there) |
| `s2s-lightning/` | ✅ | **S2S-Lightning** — the *same model*: **imports** `s2s/v2.0` (no copy); only the harness differs. | ✅ same VAE | Lightning `Trainer`+`DDPStrategy` · Midway: `srun`, `--ntasks-per-node=4` == devices; Polaris: one `python`, **never `srun`** | `S2S_BENCH_*`, `S2S_NVTX`, `S2S_PRECISION`, `S2S_TORCH_COMPILE`, `S2S_DDP_BUCKET_CAP_MB` | ERA5 HDF5 (same block) |
| `si/` | ✅ | **SI** — stochastic interpolants (DiT/SiT + SFNO/UNet/AE variants); interpolant objective. A *different* model that shares only the Lightning layout the port mirrors. | ✅ interpolants (diffusion-family) | Lightning `Trainer`+`DDPStrategy` · same launch shape as the port | `SI_BENCH_*`, `SI_NVTX`, `SI_PRECISION`, `SI_DDP_*` | AMIP (E3SM staged on Polaris; GREEN there) |
| `PanguWeather/` | ✅ group | **Fork of `s2s/v2.0`** (§2c). nettype `pangu_plasim` = 3D-Swin + VAE (CRPS + KL); nettype `sfno_plasim` = SFNO (`raw_l2`) — the green Polaris path, **1.18 B params** in the E3SM config (not ~79M; `polaris_bench_report.md` §1). | `pangu_plasim` ✅ (VAE) · `sfno_plasim` ❌ | plain torch DDP · `torchrun --standalone --nproc_per_node=$NPROC` | `PANGU_BENCH_*`, `PANGU_NVTX`, `TORCH_COMPILE_MODE`; a stale `S2S_BENCH*`/`S2S_NVTX` errors `LEGACY_BENCH_ENV`; **range names + CSV columns identical to s2s**; seed via `--global_seed`; precision from the YAML `amp_dtype`, **not** an env knob | E3SM h5 directly (staged) / PLASIM (`pangu_plasim` blocked on it) / ERA5 |
| `makani_sfno/` | ⚠️ vendor + our wrapper | NVIDIA **makani** + the group's `sfno_training` wrapper — SFNO (L2). | ❌ (as configured) | makani's own trainer (`train_plasim`) · Polaris: `python -m torch.distributed.run` from the isolated SFNO venv; `--batch_size` is **global** (the rank count must divide it) | none | E3SM (packed) / PLASIM |
| `physicsnemo_sfno/` | ❌ vendor | NVIDIA **PhysicsNeMo** `unified_recipe` — AFNO / **SFNO** / GraphCast (L2). | ❌ (as configured) | hydra recipe · Polaris: `python -m torch.distributed.run` from the SFNO venv | none | E3SM → zarr (full conversion not cleared — §8) / ARCO-ERA5 |

**Six projects; TWO pairs are live-coupled, the rest borrow by copy.** Know which
you are in before editing — "separate projects" is true of some pairs and
dangerously false of others (audited 2026-07-16; an earlier draft of this
paragraph claimed *one* shared codebase and was refuted):

| pair | coupling | consequence |
|---|---|---|
| `s2s/v2.0/` ↔ `s2s-lightning/` | live, **PYTHONPATH import** | port has no copy — one edit changes both (CLAUDE.md #5) |
| `physicsnemo_sfno/` → `makani_sfno/` | live, **editable install** ⚠ | `polaris_setup_sfno_venv.sh:78-79` pip-installs makani from a GitHub pin and `physicsnemo_sfno` as `-e` into **one shared venv**; makani imports it (`makani/utils/comm.py:19`). **Editing `physicsnemo_sfno/physicsnemo/` changes what makani runs.** Neither directory says so |
| `PanguWeather/` ← `s2s/v2.0/` | **copy** (fork) | fixes do NOT propagate — §2c |
| `si/` + the rest | copy / unrelated | no live coupling |

**makani is not vendored here** — 0 files in-repo; it is a pip pin. Only
`physicsnemo_sfno/` is an in-repo tree. And PhysicsNeMo's SFNO *is* makani's
model (entry point `[physicsnemo.models] SFNO = makani.models.networks.sfnonet:SFNO`).

**One rule spans the copy boundary by design:** #10. `PanguWeather/v2.0/train.py:206`
pins its NVTX names to `s2s/v2.0/train.py:188`, and its
`test/bench_instrumentation_test.py` asserts s2s's CSV columns — so the forks stay
benchmark-comparable. Knobs are per-project (`PANGU_*` vs `S2S_*`); range names and
CSV columns are shared contract. Do not "fix" that asymmetry.

> **⚠ The copy boundary is convention, not structure.** `s2s/v2.0/` and
> `PanguWeather/v2.0/` export the **same top-level module names** (`utils`,
> `networks`, `config`) and both import **unqualified**, so `networks.pangu`
> resolves to whichever tree is first on `PYTHONPATH` — and the two files differ
> by **106 lines** (measured). Set `PYTHONPATH` to exactly one tree; never both.
> Nothing tells you which one you actually loaded.

- **The stochastic column is why §4.0's VAE noise hook matters for S2S / the
  port / Pangu `pangu_plasim`, not only for SI** — three mechanisms (VAE draw,
  interpolants, none); the split is not "SI vs the rest".
- **Vendor traps** (full list: `polaris_pbs_notes.md`): makani **auto-resumes**
  and can exit 0 having trained zero steps; PhysicsNeMo hardcodes CWD-relative
  checkpoints and its hydra PATH-form defaults break `model=sfno` on the CLI.

### 2c. PanguWeather is a FORK of `s2s/v2.0` — divergence evidence

*(Stays numbered §2c — CHANGELOG, `polaris_bench_report.md`, and the handoff
prompts cite it by number.)*

Same codebase, diverged by purpose, shared by **copy, not import** — nothing
tells you the other fork drifted; **audit before assuming any fix reached the
other fork.** At the 2026-07-15 audit: `networks/pangu.py` ~95% identical
(1093 of ~1148 lines); `s2s/v2.0/utils/losses.py` a strict subset of
PanguWeather's (171/171 shared; PanguWeather adds `Raw_MSELoss` + an unweighted
`CRPSLoss`). One axis of divergence — instrumentation vs science (counts **as
audited 2026-07-15, before the harness port**):

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
PanguWeather); full table: `polaris_bench_report.md` §6b.

**Both forks keep the VAE** (`reparameterize`: s2s `pangu.py:463`, PanguWeather
`pangu.py:449`). What changes is the **nettype**: PanguWeather
`train.py::get_model` takes the VAE path for `pangu_plasim` (`:651`), the SFNO
path for `sfno_plasim` (`:670`); the SFNO net (`networks/modulus_sfno/`) has
**no VAE at all**, and the E3SM config uses `loss: "raw_l2"`. So "s2s vs the
Pangu SFNO run" differs by the VAE; "s2s vs PanguWeather `pangu_plasim`" does not.

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
(`S2S_BENCH` in s2s/the port, `PANGU_BENCH` in PanguWeather — env knobs are
per-project, **range names and CSV columns are not**) times steps
`cuda.synchronize`-bracketed; `S2S_NVTX`/`PANGU_NVTX` emits `to_ensemble_batch`,
`data_prep`, `forward_loss`, `backward`, `optimizer`, `step_N`, `val_*`
(PanguWeather adds `ema` and has no `val_*` ranges). **SI's range names differ**
(`preprocess`, `forward_loss`, … — `si/CLAUDE.md`). The §4 gate does NOT catch
a dropped/renamed range or CSV column (it silently invalidates every comparison
and breaks `parse_nsys.py`) — review bench plumbing explicitly.

## 4. The correctness oracle: numerical-equivalence-vs-baseline

**Every optimization must reproduce the pre-optimization model output within a
stated tolerance.**

### 4.0 Prerequisites — status (all three MET on PanguWeather; partial on s2s)

- **Seed mechanism** — ✅ both forks. s2s: `--seed`/`$S2S_SEED`/YAML +
  `--deterministic` (`s2s/v2.0/utils/seeding.py`; GPU-verified `SEEDING_OK`).
  PanguWeather **already had** `--global_seed` → `seed_torch()`, stronger than
  s2s's legacy path — do **not** port `seeding.py` across (two mechanisms
  racing the same global RNGs is a regression).
- **Tiny deterministic baseline config** — ✅ PanguWeather
  `config/tiny_baseline.yaml`: 7.17M params (165× under the real 1.18 B),
  0.023 s/step, 1.00 GB, run green (job 7255583). ❌ s2s: none — its `test.yaml`
  is the full ~79M-param Swin model (CLAUDE.md #12).
- **VAE noise-fixing hook** — the reparameterization draw is stochastic, and
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

No tensors in git (`.gitignore` blocks `*.pt`; CLAUDE.md #8). Commit only the
**text summary** (JSON/CSV of per-step losses + output stats + the tolerances
used) under `baselines/<model>/`; keep any raw reference tensor on per-cluster
shared storage, path recorded in the living doc.

### 4.3 Invariants the gate protects

- **CRPS sign & normalization** (skill − spread, divided by `num_ensemble_members`) and the **cos-latitude weighting**.
- **VAE / KL** term and the reparameterization draw.
- **normalize ↔ inverse-normalize symmetry** and the **predict-delta add-back**.
- **No train/val leakage**; the `os.path.isfile` guard before `restore_checkpoint`.
- Under Lightning: **no hand-rolled AMP/backward** inside automatic optimization; precision via `Trainer(precision=…)`, not manual autocast/GradScaler; DDP `static_graph` + the dead-module freeze preserved.

## 5. Phase 2: optimization ROI ladder

Phase 2 starts only after the model is up (Phase 1) and **profiled**. Two
independent profiles agree on the thesis: Midway H100 (`s2s/v2.0/bench_report.md`
— elementwise ops the largest GPU-time consumer, matmul 6th) and Polaris 4×A100
(PanguWeather SFNO: **GPU-bound**, loader idle 0.7%; 61% of GPU time pointwise
vs 15% GEMM ⇒ fusion-starved — `polaris_bench_report.md`). Highest leverage
first. **Several knobs already exist — enable, don't re-implement:**

1. **`torch.compile`** — plumbed as `TORCH_COMPILE_MODE=reduce-overhead|max-autotune`
   in both `s2s/v2.0/train.py` and `PanguWeather/v2.0`; the port has
   `S2S_TORCH_COMPILE`. Deliberately left unset until the §4 baseline exists.
   Gate on equivalence; expect longer warmup (raise `PANGU_BENCH_WARMUP`, or
   `S2S_BENCH_WARMUP` on s2s). Do **not** write new compile wiring into shared code.
2. **FlexAttention** for the bias-disabled `EarthAttention3D` path (reproduce
   the SDPA additive-mask output within tolerance; confirm gradients flow
   through the learned bias).
3. **bf16 DDP communication hook** — compress all-reduce. (Precision itself is
   already selectable: `S2S_AMP_DTYPE` in s2s; the YAML `amp_dtype` in PanguWeather.)
4. **Fused AdamW.**
5. **Vectorize the CRPS pairwise/ensemble loop** — last, and only if it profiles hot.

Each rung = one small commit with its own §4 check + a bench delta in the living
doc. Custom Triton/CUDA is below rung 5 and only if a profile proves a specific
kernel dominates.

## 6. Clusters and hardware

Facts live in CLAUDE.md §Cluster facts (+ `polaris_pbs_notes.md`). The one fact
that shapes design: **the A100's 40 GB is the binding constraint on Polaris** —
Midway configs sized for ~94 GB may OOM; Polaris smokes start at per-GPU batch 1.
Document the smallest config that fits; don't silently shrink the model.

## 7. Validation and testing strategy

**No pytest suite yet** — three self-running test files exist (`SEEDING_OK`,
`BENCH_INSTR_OK`, `VAE_NOISE_OK`; the rest of the `test/` dirs are ad-hoc
scripts) but no `conftest.py`/`--fast` (a §8 item). Until then, "run the tests"
= "run the relevant smoke". Three tiers, cheapest first:

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
  Channel contract (2026-07-16): all three E3SM pipelines read the same
  **108 of 162** channels — clouds `CLDICE`/`CLDLIQ`/`CLOUD` excluded
  (`EXCLUDED_VARS` in `physicsnemo_sfno/polaris/e3sm_h5_to_seqzarr.py`;
  asserts in `makani_sfno/polaris/convert_e3sm_to_makani_alldata.py`);
  PhysicsNeMo store = **103 predicted + 5 unpredicted**, **~1.43 TB**.
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

## 9. Open questions and risks

- **The sharpest open science question is R2, frozen ocean forcing** — binary,
  and only jesswan can settle it; if it is a defect, everything trained on the
  archive used wrong forcing (`data_for_training.md`). Until the data-prep
  decisions land, the full ~1.43 TB conversion stays uncleared (§8).
- **An s2s baseline is still uncapturable** — its tiny config and VAE noise
  hook don't exist (§4.0); PanguWeather's do.
- **Lightning on PBS is single-node only** — the no-`srun` launch is proven
  green; multi-node needs a `ClusterEnvironment`.
- **Fork drift and instrumentation drift are silent** — the §4 gate catches
  neither; audit deliberately (§2c, §3).
