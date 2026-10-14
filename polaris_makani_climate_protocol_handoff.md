# HANDOFF — makani E3SM emulator: the ACE2-style multi-year climate evaluation

Written 2026-09-24 at the end of a long Q&A session with the user about the lagged
ensemble, jesswan's proposed multi-year protocol, and what "compare to climatology"
should mean. **Decision taken by the user at the end: "Let's do what ACE2 does."**
This document is what the next session needs to build it without re-deriving anything.

Read this, then CHANGELOG `2026-09-24` (three entries), then
`makani_sfno/docs/2026-09-21_lagged_ensemble_result.md` (why the lagged ensemble is
closed) and `makani_sfno/docs/2026-09-10_longroll_blowup_analysis.md` (the stability
question this protocol runs into). `ace2_vs_makani_differences.md` §5–§6 is the
side-by-side that motivated copying ACE2's evaluation rather than inventing one.

**The one thing to read first (§2): the current checkpoint does not blow up within 36
days — measured today, job 7648967 — it decorrelates by day 10 and then sits at the
climatological ceiling with amplitude preserved, except for two near-surface
geopotential levels and the soil channels, which drift. jesswan's earlier observation
puts the eventual blow-up near day 120. A five-year run from Oct 2044 therefore
probably survives her 92-day discard and dies inside the first scored year. Build the
tooling anyway — it is what evaluates the *next* checkpoint too — but pre-register the
run as a stability measurement first and a climate evaluation second.**

---

## 0. Where the discussion stands

jesswan's proposal (verbatim, relayed by the user): *"we run inference from Oct 2044,
but then we only compare 2045-2049 to climatology and don't use Oct-Dec 2044 in that
comparison"*, with **8 members initialised on evenly spaced October days**.

Assessed this session (full critique in CHANGELOG `2026-09-24 (makani, cont.)`):

| | verdict |
|---|---|
| shape of the protocol | ✅ right: spin-up spent inside training years; scored window = the whole held-out record (2045–47 valid + 2048–49 test), five whole calendar years; 8 ICs ~4 d apart decorrelate in 2–3 weeks and become an IC ensemble for climate statistics |
| "2044 is known to the model" | ✅ **not** a contamination: training was single-step teacher-forced; the trajectory forgets its IC within weeks; the 92-day discard is 4–6× that. **Measured (§2): a 6–8 % training-year advantage exists at leads ≤ 7 d and is gone by 14 d** |
| stability | 🔴 the checkpoint diverges near step 500 (~120 d, jesswan's observation) ⇒ late January 2045, the first scored month. Until fixed, the run measures the failure mode |
| "compare to climatology" | ⚠ was unspecified. **Resolved: do what ACE2 does (§3)** — whole-window time-mean vs the same-window truth mean, plus monthly means |
| held-out years | ⚠ the 7 forcing channels are **bitwise identical every year** (measured §2) ⇒ 2045–49 differ from training only in atmospheric truth; this is a forced-stationarity test, not a response test. Checkpoint was selected on 2045–47 valid loss; 2048–49 are the only fully clean years |
| noise floor | ⚠ E3SM truth is one realisation; 8×5 model-years vs 5 truth years needs the 30-year interannual spread as the floor |
| plumbing | 🔴 nothing that exists can run it (§4) |

The user's instruction after this: adopt the ACE2 recipe. Nothing has been built yet
beyond the probe (§2). The lagged ensemble is **closed** (negative result, do not reopen
— `2026-09-21_lagged_ensemble_result.md`).

---

## 1. Facts established this session — do not re-derive

Data and indexing (all confirmed in code or on disk):

- **E3SM pack**: `$MEMBER_ROOT/data/e3sm_makani_alldata_production/{train,valid,test}/YYYY.h5`,
  **1460 frames per year** (noleap 365 d × 4; `polaris_pack_e3sm_alldata_full.pbs` asserts
  `T == 1460`), 38.1 GiB each. Split **train 2015–2044 / valid 2045–2047 / test 2048–2049**
  (`polaris_pack_alldata_production.pbs`). ⚠ **1455/1459 is the PlaSim `MOST.*.h5`
  length** (Aug-1 anchor); it does not apply here — a mistake made once this session.
- h5 layout: `fields_state (T,100,180,360)`, `fields_diagnostic (T,1,…)` (PRECT),
  `forcing (T,7,…)`, `time_plasim (T,)`, `channel_state/diagnostic/forcing` name
  arrays, attrs `year`, `lat_order = "descending (row 0 = +89.5)"`. No `plasim_time_units`
  anchor — E3SM outputs are **step-index labelled** (decision E).
- Noleap month boundaries in frames: cumulative days `[0,31,59,90,120,151,181,212,243,
  273,304,334,365] × 4`. **Oct 1 00:00 = frame 1092.**
- Model: SFNO `embed_dim 384, num_layers 8, scale_factor 3`, 107 in / 101 out, bf16
  autocast, trained **single-step** (`n_future 0`, `valid_autoreg_steps 3`). Checkpoint
  `prod1n_b32_sgdr/training_checkpoints/best_ckpt_mp0.tar`, selected on single-step
  valid loss over 2045–2047.
- Lagged-sweep semantics, now documented in `sfno_ensemble/starts.py` (`Parameters`
  block, commit `87f4962a`): everything in frames; `K` = rollout length and oldest-member
  age; `stride` = start spacing = member age spacing; `n_targets` = consecutive target
  frames that must receive all `K//stride` members; `first_start` = pure translation.
- Climatology conventions (§3): ACE2 = whole-window time mean + monthly reference;
  makani = one all-time mean (`time_means.npy`, annual, biases ACC high); our PLaSim
  scorer = per-frame-of-year (366×4 bins). **The 1460-bin per-frame version is only
  needed for ACC at short leads, not for climate statistics.**

Process facts (also in memory `makani-worktree-and-test-python`):

- **origin/main does not contain the makani ensemble code.** Branch worktrees from HEAD:
  `git worktree add .claude/worktrees/<name> -b <branch> HEAD`, then `EnterWorktree(path=…)`.
- Login-node `python3` is 3.6 with no pytest/h5py. Torch-free tests run under
  `$MEMBER_ROOT/conda-envs/sfno-venv/bin/python` (3.12); expect a core dump at
  interpreter teardown after the pass line (CLAUDE.md #3). h5py in that venv needs the
  compute-node modules (`libcudart.so.13`) — do not open h5 on the login node.
- `grep`/`find`/`head` are denied to the assistant on this login node; use python3
  heredocs (`os.walk` inside the repo only).

---

## 2. The probe — job 7648967, `LONGROLL_PROBE_OK`, 2026-09-24

`makani_sfno/polaris/polaris_longroll_probe.pbs` (commit `a648e1ea`): two single
rollouts from **frame 1092 (Oct 1 00:00)** of **2044** (train) and **2048** (test) on the
production checkpoint, scored with `score_rollout_nc.py`, read by
`longroll_probe_summary.py`. Log: `makani_sfno/makani_longroll_probe.o7648967` (in the
worktree — copy it somewhere durable if the worktree is deleted). Outputs:
`$MEMBER_ROOT/runs/makani_eval/prod1n_b32_sgdr_longroll_probe_f1092/{train_2044,test_2048}/`.

**Cost facts** (needed for §4 sizing):

| | measured |
|---|---|
| `rollout_one_ic` at K=200 | 🔴 **OOM** — "35.01 GiB in use, tried to allocate 4.88 GiB" on a 39.49 GiB A100. Confirms ~6 K-frame copies live at once (26.2 MB/frame) |
| K=144 (36 d) | ✅ 191.7 s / 189.1 s ⇒ **~1.3 s per step** including I/O; 5 GB NetCDF each |
| scoring 144 leads × 101 ch | 4.2 min CPU per arm |

**Survival read-out** (median over 101 channels; NRMSE = RMSE / truth anomaly amplitude,
1.0 = collapsed to climatological mean, 1.414 = decorrelated with correct variance):

| lead | train arm NRMSE / VR / ACC | test arm NRMSE / VR / ACC |
|---|---|---|
| 6 h | 0.107 / 0.995 / 0.994 | 0.115 / 0.992 / 0.993 |
| 126 h (5 d) | 0.546 / 0.985 / 0.850 | 0.571 / 0.979 / 0.833 |
| 246 h (10 d) | 0.958 / 0.987 / 0.552 | 1.018 / 0.986 / 0.468 |
| 366 h (15 d) | 1.205 / 1.051 / 0.355 | 1.195 / 1.024 / 0.373 |
| 606 h (25 d) | 1.325 / 1.093 / 0.164 | 1.324 / 1.063 / 0.199 |
| 864 h (36 d) | **1.396 / 1.110 / 0.116** | **1.390 / 1.112 / 0.123** |

- Median crosses 1.0 at **258 h / 246 h** (~10 d), never reaches 1.414 within 36 d; slope
  ratio (last 96 h over 30–126 h) **0.29 / 0.07** ⇒ strongly decelerating, i.e. **plateauing
  at the ceiling, not diverging**. VR 1.0–1.1 ⇒ amplitude preserved, no blurring.
- **Channels past the 3× blow-up clause at day 36: exactly two**, both arms — `Z3_l17`
  (NRMSE **152**, bias/a_truth **−81**: a systematic drift on a near-constant level) and
  `Z3_l16` (3.1). Next: `SOILWATER_10CM` 2.5, `TSOI_10CM` 1.9, `Z3_l15` 1.7 / `V_l04` 1.9.
  ⇒ the failure is **channel-specific drift in fed-back near-surface geopotential and
  soil**, not a global blow-up. Whether `Z3_l17` should be prognostic at all is jesswan's
  call (raised in CHANGELOG `2026-09-20`, still open).
- ⚠ **Correction (same day, CHANGELOG "analysis only"):** in physical units `Z3_l17` is a
  **linear −0.35 m/day** drift on a channel whose truth anomaly amplitude is 0.2 m and whose
  6-h tendency (0.09 m) is ~35× below bf16 resolution of its topography-dominated normalisation
  (σ = 816 m). It is ~0.015 σ at day 36 — the "152" is the denominator. It is probably **not**
  the blow-up precursor; watch the ordinary-channel warm drift (`Z3_l10` +44 m, `T_l17` +0.8 K
  at 36 d) instead. The next bullet's prediction is withdrawn.
- Reconciling with "blows up near 500 steps": not contradicted — 144 < 500. The probe
  says the first 36 days are physical apart from those channels; the blow-up, if it
  comes at ~120 d, will be **preceded by Z3_l17/Z3_l16/soil leaving range first**. The
  streaming driver's stability monitor (§4b) must record per-channel first-bad-step.

**Contamination read-out** (train-year IC vs test-year IC, n = 1 each — indicative):

| lead | train/test median NRMSE | channels where train is better |
|---|---|---|
| 6 h | 0.937 | 81 % |
| 24 h | 0.924 | 84 % |
| 72 h | 0.917 | 91 % |
| 168 h | 0.927 | 75 % |
| 336 h | **1.001** | 50 % |

A real but small memorisation tilt (~7 %) that is **gone by 14 days**. Below the flag
threshold (ratio < 0.90 on > 75 %). It cannot survive a 92-day discard. Do not spend
more on this question unless jesswan asks.

**Forcing repetition**: `FORCING_REPEATS_ANNUAL_CYCLE=True` — all 7 forcing channels
(incl. `sst`, `ice`, `solin`, `natveg`) bitwise identical across 2044, 2045 and 2048 at
frames 0, 1092, 1459 (`max_abs_diff = 0.000e+00`); atmospheric state differs by 6.6e3.
The frozen-2015 boundary finding (CHANGELOG, `netcdf-to-h5_e3sm.py` re-slicing) is in
the packed makani data. **Intent unresolved — jesswan** (`CTL_SST0051` reads as a fixed-
SST control). Consequence for the protocol: the model sees forcing it has seen 30
times; 2045–49 is a stationarity test.

---

## 3. What "do what ACE2 does" means, concretely

Source: `ACE2_retrain/ace_exp/fme/ace/aggregator/inference/` (`main.py`
`InferenceEvaluatorAggregatorConfig`, `time_mean.py`, `annual.py`, `seasonal.py`,
`zonal_mean.py`, `spectrum.py`, `histogram.py`) and the ACE2 configs
(`ACE2_retrain/config_polaris.yaml`: inline inference `n_forward_steps: 7300` = 5 yr,
`forward_steps_in_memory: 40`, `time_mean_reference_data: …/time-mean.nc`; evaluator
configs 14600 = 10 yr; AMIP configs add `monthly_reference_data`).

| ACE2 element | what it is | our equivalent to build |
|---|---|---|
| `time_mean_reference_data` | one (var, lat, lon) map: the reference's mean over the **same window as the run** | E3SM truth mean over 2045–2049 (5 files, one pass); also the 2015–2044 training mean (`stats/time_means.npy` already exists) as the persistence-of-climate baseline |
| `TimeMeanEvaluatorAggregator` | bias and RMSE maps of the run's whole-window time mean vs the reference; area-weighted global numbers | same, equiangular lat weights (`sfno_ensemble.scores.equiangular_weights`), 101 channels |
| `monthly_reference_data` + `PairedGlobalMeanAnnualAggregator` | monthly-mean reference → global-mean annual series, model vs reference (drift, trend) | E3SM monthly means 2045–2049 (60 maps × 101 ch); global-mean monthly and annual series per member |
| `SeasonalAggregator` | seasonal cycle by month | 12-month climatology of the run vs truth 2045–49 and vs 2015–44 |
| zonal mean, spectrum, histogram, video | secondary diagnostics | zonal-mean bias (cheap, add); spectra optional (`torch_harmonics` is a dep); skip video |
| `log_step_means` at step 20 | short-lead sanity | keep: we already have the K=56 curves |
| `log_nino34_index` | ENSO | **skip** — SST is prescribed and identical every year; the index would be truth |
| ensemble for error bars | 3 ICs one day apart | jesswan's 8 ICs ~4 d apart in Oct 2044 |
| checkpoint selection | 8 × 5-yr inference runs | not ours to change |

Scoring window per jesswan: **2045-01-01 00:00 to 2049-12-31 18:00** (frames 0..1459 of
five files; 7300 steps), after a discard of 368 steps for the Oct 1 member (256 for
Oct 29). Members: frames 1092 + 16·i, i = 0..7 (Oct 1, 5, …, 29).

**Noise floor** (ACE2 does not need this because ERA5 is long; we do): the interannual
standard deviation of annual means across 2015–2044 per channel and grid cell, from the
monthly climatology pass. A 5-year model-mean bias is only a finding if it exceeds
that spread appropriately scaled.

---

## 4. What has to be built — nothing here exists

### 4a. Reference builders (torch-free, h5py, chunked reads; one PBS job)

- `time_mean` and `monthly_means` over an arbitrary year range and split directories,
  reading `fields_state ‖ fields_diagnostic` in chunks of ~16 frames (the packer's
  `STATS_CHUNK_T` pattern), float64 accumulation. Outputs NetCDF/npz with channel names.
- Needed: 2045–2049 (190 GB read, ~15–30 min) and the 2015–2044 monthly climatology +
  interannual std (1.2 TB read, 1–2 h; `debug` will not fit — `preemptable` or two jobs).
- Unit test on a synthetic 3-file h5 fixture (noleap month edges, channel order).

### 4b. The streaming cross-file rollout driver — the core

The existing `rollout_one_ic` (`src/sfno_inference/rollout_driver.py`) is a verbatim
copy of `validate_one_epoch`: one `dataset[idx]` fetch of the whole K-frame block, all
of it `.to(device)`, predictions kept in a list, then `cat`, then de-normalised copies
of prediction and truth — six K-frame tensors alive at once (26.2 MB/frame). **Leave it
alone** (the scorecard, the lagged sweep and the Stampede3 path use it) and add a
sibling under `src/sfno_inference/`, e.g. `climate_driver.py`:

1. **Constant memory.** Carry only the current state `(1, 100, H, W)`. Fetch forcing in
   chunks of ~40 frames (fme's `forward_steps_in_memory`) straight from h5 (`forcing[t0:t1]`),
   normalise with the dataset's forcing stats, feed one frame per step. Drop truth unless
   a reduction needs it (the time-mean bias does not — it uses the pre-built reference;
   the per-step stability monitor does not either).
2. **Reuse the model contract, do not re-implement it.** `build_wrapper_from_checkpoint`
   and `load_eval_params` (`checkpoint_loader.py`); `PlasimPreprocessor.cache_unpredicted_
   features / flatten_history / append_history` (`src/sfno_training/models/preprocessor.py`)
   with 1-frame target blocks — this is what keeps 107-in/101-out, PRECT-drop-before-
   feedback, and bf16 autocast identical to training (CLAUDE.md #1: do not change what
   the model computes). `_load_run_norm_stats` for de-normalisation. Verify against
   `rollout_one_ic` on a K=56 run: the two must agree to float tolerance — **that is the
   equivalence check for this commit** (CLAUDE.md #6).
3. **Cross-file.** Build a symlinked `all_years/` directory (2044–2049) or index the
   files directly by year; at frame 1459 → next year's frame 0. E3SM is contiguous
   noleap, so this is a provenance question only; `nwp_ic_offsets`' refusal exists for
   PLaSim's Aug-1-anchored files.
4. **Reduce on the fly** (float64 accumulators on CPU or GPU): running time mean over the
   scored window; per-calendar-month sums and counts (12 for climatology, 60 for the
   series); area-weighted global mean per step per channel `(T, C)` — ~3 MB for 7670
   steps; optional zonal mean `(T, C, H)`; optional monthly-mean full fields `(60, C, H, W)`
   = 1.6 GB per member — write these, they are the ACE2 monthly output.
5. **Stability monitor**, per step: `isfinite`, and each channel's area-weighted std
   relative to `global_stds.npy`; record `first_bad_step[c]` at a threshold (start with 3×,
   the prereg's blow-up clause) and the step at which the median crosses. Write it with the
   outputs; the aggregator must **truncate the scored window at the first global blow-up**
   and say so, rather than averaging garbage.
6. **Output** one NetCDF per member: time mean, monthly means, monthly climatology,
   global-mean series, stability record, provenance (start frame, member id, checkpoint,
   sha). No per-step fields.

Cost: model step is ~0.1–0.3 s; I/O per 40-frame chunk is ~1 GB. Estimate 30–60 min per
5.25-year member on one A100; 8 members on 4 GPUs = 2 rounds ⇒ `preemptable` (or two
`debug` jobs of 4 members each — surface the queue choice first, per memory).

### 4c. Aggregator / metrics (torch-free numpy)

ACE2 table in §3: time-mean bias and RMSE maps (per channel, plus area-weighted global
bias/RMSE), monthly global-mean series vs truth, seasonal-cycle amplitude and phase,
annual means, zonal-mean bias; ensemble mean and spread across the 8 members; the
30-year interannual std as the noise floor; the training-mean baseline. Reuse
`sfno_ensemble.scores.equiangular_weights/area_mean/rmse`. Report as one markdown table
per channel group plus `CLIMATE_EVAL_OK` token. Unit tests on synthetic arrays.

### 4d. Launcher + pre-registration

- `polaris_climate_run.pbs` sibling; env bootstrap **verbatim** from
  `polaris_eval_inference.pbs` (module ordering is on purpose); `python -m torch.
  distributed.run --standalone --nproc_per_node=1` per member (makani's `comm.init`
  needs a process group even at world size 1 — the launcher header documents why).
- **Write the pre-registration before the first scored run**, following
  `2026-09-20_k56_readout_prereg.md`: which statistics, which channels are headline,
  what number means the climate is "right", what the stability record must show for the
  climate numbers to be admissible at all. The K=56 prereg's defect (a max-over-channels
  clause dominated by `Z3_l17`) is the trap to avoid: decide how `Z3_l17` is handled
  before looking.

---

## 5. Questions for jesswan — ask before the scored run, not after

1. Which **climatology reference window**: same-window truth 2045–2049 (ACE2) plus the
   2015–2044 training mean as baseline — is that acceptable, or does she want 2015–2044
   only?
2. Which **statistics and channels are headline**? (Time-mean bias/RMSE, seasonal cycle,
   annual series are ACE2's; anything else is hers to add.)
3. **Frozen forcing**: is the identical annual cycle in `sst/ice/…` intended
   (`CTL_SST0051` control) or the re-slicing defect? It decides whether 2045–49 is a
   response test at all.
4. **`Z3_l17` (and `Z3_l16`) prognostic or not?** They drift 81 truth-amplitudes in 36
   days and will end every long run. Removing them from the fed-back state changes what
   the model computes — her sign-off.
5. **Start Oct 2044 (her proposal) or Oct 2047** (only 2048–49 scored, no validation
   exposure)? Recommendation: build for either; run 2044 first because she asked.
6. If the run dies near day 120 as expected, which training arm does she want next —
   deeper unroll (`n_future` 2–4; the depth-4 × 24-epoch run 7630639 is the arm that
   answers whether it compounds) or channel removal?

---

## 6. Do NOT re-try

- ❌ `--mode climate` on E3SM/Polaris as-is: `rollout_one_ic` OOMs above K≈150 on 40 GB
  (K=200 measured OOM). One E3SM year is ~41 GB of block alone.
- ❌ Lagged/snapshot ensembles to improve the deterministic forecast — closed, negative
  (`2026-09-21_lagged_ensemble_result.md`, job 7643271).
- ❌ Building a 1460-bin per-frame climatology for the climate comparison — monthly
  suffices; per-frame is for ACC at short leads only.
- ❌ Scoring the run against `time_means.npy` alone — annual mean, hides the seasonal
  cycle (why ACC reads high).
- ❌ Treating the K=56 scorer's `DRIFT_FIRST` verdict as a climate result — it is the
  prereg's branch rule, fired by `Z3_l17`.
- ❌ Using `nwp_ic_offsets` / `plan_lagged_sweep` to pick climate-run starts — they refuse
  cross-file and are the wrong tool; the streaming driver takes (year, frame) directly.
- ❌ Running anything with torch on the login node; re-deriving 1455 vs 1460.

---

## 7. Definition of done

1. Reference builders + tests; 2045–49 references and 2015–44 monthly climatology + std
   on disk (`REFERENCE_OK`).
2. Streaming driver + equivalence check vs `rollout_one_ic` at K=56 (`CLIMATE_DRIVER_EQUIV_OK`);
   a smoke crossing the 2044→2045 boundary for ~400 steps on `debug` (`CLIMATE_SMOKE_OK`).
3. Pre-registration doc committed **before** step 4, with jesswan's answers to §5 1–5 or
   explicit assumptions.
4. 8-member run from Oct 2044; per-member stability record; aggregator report
   (`CLIMATE_EVAL_OK`); CHANGELOG entry with the measured numbers, whichever way they go.
5. If the run dies before 2049: the per-channel first-bad-step table is the deliverable,
   handed to jesswan with §5 question 6.

Branch at handoff: `docs/lagged-ensemble-docstrings` (worktree
`.claude/worktrees/lagged-docstrings`, pushed), commits `87f4962a` (docstrings),
`a648e1ea` (probe), `afaa28d1` + this doc's commit (assessment, result). Not merged —
main is protected; a PR against `feat/multinode-ddp-port` shows only these commits.
