# HANDOFF — makani E3SM emulator: the streaming multi-year rollout driver

Written 2026-09-24. **Scope: build and prove the driver only**: §4b of
`polaris_makani_climate_protocol_handoff.md`, narrowed. The references (§4a), the aggregator
(§4c), the pre-registration (§4d) and the 8-member scored run are **out of scope**. They come
after this lands. A monitor session watches this work from `MONITOR_makani_streaming_driver.md`.

Read in this order: this file → CHANGELOG `2026-09-24` entries (four; the newest, "analysis
only", reframes `Z3_l17`) → `polaris_makani_climate_protocol_handoff.md` §1–§2, §4b, §6 →
the four code references in §2 below. Do not re-derive anything in §1.

---

## 0. Why this exists, in one paragraph

`rollout_one_ic` (`makani_sfno/src/sfno_inference/rollout_driver.py:116`) fetches the whole
K-frame block with one `dataset[idx]`, keeps every prediction, then `cat`s and de-normalises
prediction and truth. That is ~6 K-frame copies on the GPU (26.2 MB/frame) and it **OOMs at K=200**
on a 40 GB A100 (probe 7648967). A 5-year run from Oct 2044 is **7,667 steps**. The driver carries
**one state**, streams forcing in chunks, crosses year files, and reduces to climate statistics
on the fly. It writes no per-step fields.

---

## 1. Facts you inherit (verified 2026-09-24; re-open the primary before relying on one)

| fact | value | primary |
|---|---|---|
| pack | `$MEMBER_ROOT/data/e3sm_makani_alldata_production/{train,valid,test}/YYYY.h5`; train 2015–2044, valid 2045–47, test 2048–49 | `polaris_pack_alldata_production.pbs` |
| frames per year | **1460** (noleap × 4). *Not* 1455/1459; that is PlaSim | `polaris_pack_e3sm_alldata_full.pbs` asserts `T == 1460` |
| month edges (frames) | cumulative days `[0,31,59,90,120,151,181,212,243,273,304,334,365] × 4`; **Oct 1 00:00 = frame 1092** | noleap arithmetic |
| step counts from Oct 1 2044 | 2045-01-01 00:00 is **step 368**; 2049-12-31 18:00 is **step 7,667** | 1460−1092 = 368; 368 + 7300 − 1 |
| model contract | 107 in (100 state + 7 forcing) / 101 out (100 state + `PRECT`), bf16 autocast, `n_history 0` | `makani_sfno/polaris/e3sm_alldata_full.yaml` |
| forcing | 7 channels `lsm,topo,glacier,natveg,sst,solin,ice`, **bitwise identical every year** | CHANGELOG 2026-09-24 probe |
| stats | `$PACK/stats/{global_means,global_stds,time_means,time_diff_stds,forcing_global_*}.npy`, 101 ch | on disk |
| checkpoints | **A** `runs/makani_mn_scaling/e3sm_mn_scaling/prod1n_b32_sgdr/training_checkpoints/best_ckpt_mp0.tar` (single-step production); **B** `…/nf4_prod_b16_r1/training_checkpoints/best_ckpt_mp0.tar` (job 7630639, `n_future=4`, warm-started from A; the file's mtime suggests epoch 1, **confirm the epoch field inside the checkpoint**) | `warmstart_provenance.txt` in B's dir |
| 7630639 status | **training completed 24/24 epochs, `rc=0`**; job exit 1 is `SCALING_CSV_SCHEMA_DRIFT` (new `provider` column) in post-processing only. Valid loss best at epoch 1 (0.01341) → 0.01361 at 24; LR never annealed (cosine `T_max=100` vs 24 epochs) | `makani_sfno/makani_mn_scaling.o7630639`, `$B/out.log` |
| probe cost | K=144 rollout ~1.3 s/step **including** block I/O; scoring separate | `makani_sfno/makani_longroll_probe.o7648967` |
| `Z3_l17` | truth anomaly 0.2 m, 6-h tendency 0.09 m, σ 816 m (topography); drift −0.35 m/day **linear**. ~0.015 σ at day 36, so **not** a blow-up signal in σ units | CHANGELOG "analysis only" |

Process facts: branch the worktree from **HEAD of `docs/lagged-ensemble-docstrings`** (origin/main
lacks `sfno_ensemble`): `git worktree add .claude/worktrees/climate-driver -b feat/makani-climate-driver HEAD`,
then `EnterWorktree(path=…)`. Torch-free tests run under `$MEMBER_ROOT/conda-envs/sfno-venv/bin/python`
(expect a core dump at teardown **after** the pass line). **Torch never on the login node**, and
h5py in that venv needs compute-node modules. `grep/find/head` are denied, so search with python3
heredocs inside the repo. **Git commands that walk the working tree (`status`, `diff`) can wedge on
Lustre and eat login-node process slots**, so use object-only commands where possible.

---

## 2. Existing examples — what to copy, what not to

Four streaming or long-rollout implementations already exist. **None can be reused as-is.** Each
is a reference for one part.

| # | where | what it does | take from it | do NOT take |
|---|---|---|---|---|
| 1 | **makani stock `Inferencer._inference_indexlist`**, `$VENV/site-packages/makani/utils/inference/inferencer.py:453-702` | Streams frame-by-frame through a `DataLoader` with `n_future=0` and `SortedIndexSampler` (`:500`); carries `inptlist` forward (`:573`, `:620`); bf16 autocast per step (`:593`); `cache_unpredicted_features` per step (`:589`); running reductions via buffers (`:631-673`) | **The loop shape**: one frame per iteration, state carried, reductions updated in-loop, `torch.inference_mode()`. NVTX range per step (`:527`) | Its buffers. `TemporalAverageBuffer`/`MeanStdBuffer` (`utils/inference/rollout_buffer.py:363-636`) are **per-lead** `(num_rollout_steps, C, H, W)` float32 Welford: 7,667 × 26 MB ≈ 200 GB. It also reads full truth every step. It uses stock `Preprocessor2D`, not our `PlasimPreprocessor` wrapper path |
| 2 | **our `rollout_one_ic`**, `src/sfno_inference/rollout_driver.py:116-280` | The **model contract**: `cache_unpredicted_features(*gdata)` (`:206`) → `flatten_history` → `wrapper(inpt)` under autocast (`:224-229`) → `append_history(inpt, pred, idt)` (`:243`); de-norm via `_load_run_norm_stats` | **Exactly this step body.** It mirrors `validate_one_epoch`, which is what keeps 107→101, the `PRECT` strip before feedback, and bf16 identical to training (CLAUDE.md #1) | Its memory pattern (whole block in, list of preds, `cat`). **Do not edit this file**: the scorecard, the lagged sweep and the Stampede3 path depend on it |
| 3 | **group `long_inference.py`**, `PanguWeather/v2.0/long_inference.py` (`predict_sync` `:800-1060`; year rollover `:841-850`, `:1014-1021`) | Year-by-year long rollout with async NetCDF writes | **The year-rollover bookkeeping idea** only (next-year datetime, reallocate per-year buffers) | Any import: PanguWeather is a **copy/fork**, not coupled (CLAUDE.md §Repo architecture). It writes **full per-year fields**: at 101 ch that is ~38 GB per member-year, ~1.5 TB for 8×5 yr. Its own `Stepper`, not the makani wrapper |
| 4 | **ACE2 / fme**, `ACE2_retrain/ace_exp/fme/ace/inference/loop.py:29-81` (`run_dataset_comparison`), `fme/ace/data_loading/inference.py:246-350`, aggregators `fme/ace/aggregator/inference/{time_mean,annual,main}.py` | Chunked inference: `forward_steps_in_memory` windows of forcing, advance `i_time += forward_steps_in_memory`; aggregators consume each window | **The chunking contract** (forcing window of C+1 frames, state carried across windows) and **which reductions** (whole-window time mean, monthly means, global-mean annual series) | Its data model or code. Different framework |

Also relevant: `scripts/eval_inference.py::run_climate` (`:378-440`) is the existing "climate
mode". It is `rollout_one_ic` with `K = n_samples − 1` inside **one** file. It is the path that
OOMs, and it cannot cross files. **Leave it alone**; the new CLI is a sibling.

### 2a. Two silent traps in the preprocessor (verified in stock `makani/models/preprocessor.py`)

`PlasimPreprocessor` (`src/sfno_training/models/preprocessor.py`) overrides only
`append_history`. Forcing caching is stock `Preprocessor2D`:

1. **Stale forcing, no error.** `append_history(x1, x2, step)` copies target forcing frame
   `step` into the input forcing **only if `step < unpredicted_tar_eval.shape[1]`**
   (`preprocessor.py:219`). With chunked forcing, a chunk-local index that overruns **silently
   keeps the previous step's forcing**. Assert `0 <= local_step < chunk_len` yourself on every step.
2. **Wiped input forcing.** `cache_unpredicted_features(x, y, xz, yz)` with `xz=None` **sets
   the input forcing to `None`** (`:386-389`). At each chunk boundary, re-cache with
   `xz = the current input forcing` (`get_unpredicted_features()[0]`) and `yz = the next chunk`.

Both traps pass any test that only checks shapes. §5 requires a seeded test for each.

---

## 3. What to build

Files (all new; siblings, no edits to existing drivers):

- `makani_sfno/src/sfno_inference/climate_driver.py`: the library.
- `makani_sfno/scripts/climate_rollout.py`: CLI, one member per process.
- `makani_sfno/polaris/polaris_climate_smoke.pbs`: launcher. Env bootstrap **verbatim** from
  `polaris/polaris_eval_inference.pbs`, launched with `python -m torch.distributed.run --standalone
  --nproc_per_node=1` (makani's `comm.init` needs a process group at world size 1; bare
  invocation dies in `build_wrapper_from_checkpoint`, `polaris_eval_inference.pbs:160`, job
  7632577). One process per GPU with `CUDA_VISIBLE_DEVICES`; `wait` on **each PID** and fail if
  any fails.
- `makani_sfno/tests/sfno_inference/test_climate_driver.py`: mirror the synthetic fixture in
  `tests/sfno_inference/test_rollout_driver.py`.

### 3a. Driver behaviour

1. **Inputs:** `run_dir`, `ckpt`, `start=(year, frame)`, `n_steps`, `score_start=(year, frame)`,
   `member_id`, `chunk_len` (default 40), output path, optional `snapshot_every` (default 0).
2. **Years as one axis.** Build a directory of symlinks `YYYY.h5` for the needed years (they live
   in three split dirs) and hand it to `_plasim_get_dataloader(eval_params, dir, device,
   mode="eval")` with `valid_autoreg_steps=0`. `PlasimForcingDataset` treats files in
   filename order as one continuous axis (`plasim_forcing_dataset.py:97-161`; `_get_indices` maps
   `global_idx + dt·off` across files). **Confirm on the h5s that E3SM's per-file timestamps
   either continue or trigger the synthesised-axis warning (`:133`)**, and log which.
   **Refuse** a gap (missing year) loudly.
3. **IC:** `dataset[global_idx]` for the start frame. **Model contract = `rollout_one_ic`'s step
   body (§2 #2), unchanged.** Load via `load_eval_params` + `build_wrapper_from_checkpoint`.
4. **Forcing:** read chunks with the dataset's own `_read_forcing(global_idx, a, b)` and normalise
   with `dataset.forcing_bias/forcing_scale`. The dataset normalises with the same stats, so the
   result is identical by construction. Do not read state truth in the loop. Truth is not needed:
   references are pre-built (§4a of the protocol handoff).
5. **Reductions, float64**, over the scored window only (steps before `score_start` still run and
   feed the stability monitor):
   - whole-window time mean `(C,H,W)`;
   - **current-month** sum `(C,H,W)` + count, flushed to CPU/disk at each noleap month edge, giving
     monthly means `(n_months, C, H, W)` on disk (60 × 26 MB fp32 ≈ 1.6 GB per member at 5 yr; keep
     only one month on GPU);
   - area-weighted global mean per step `(T, C)` (equiangular weights,
     `sfno_ensemble.scores.equiangular_weights`), **for all steps including spin-up**;
   - optional zonal mean `(T, C, H)`.
6. **Stability monitor, every step:** `isfinite` over the whole state; per channel, area-weighted std
   of `pred` and area-weighted RMS of `pred − time_means`, both **in units of `global_stds`**; record
   `first_bad_step[c]` at 3× and 10× (the protocol's clause), and the step where the median crosses.
   **Also log the physical-unit global mean of every channel** so linear drifts like `Z3_l17`'s are
   visible in metres. On the first non-finite value: stop, write everything so far, set
   `truncated_at_step`, exit with `CLIMATE_ROLLOUT_TRUNCATED step=… channel=…`, not an ordinary error.
7. **Snapshots (optional):** a full-field fp32 state every `snapshot_every` steps (e.g. 120 = 30 d),
   for post-mortem of a blow-up.
8. **Output:** one NetCDF per member with every reduction, the stability record, and provenance
   (start year/frame, member id, ckpt path + sha256 prefix, git sha, chunk_len, per-year file
   paths, and the timestamp-axis mode from 3a.2).
9. **Channel-agnostic.** Take every channel count and name from `eval_params`/the checkpoint and the
   run's stats. **Never hard-code 101/100/7.** A soil-free model (99 out, 105 in) is next in line.
   Assert `len(global_means) == N_out` rather than reshaping silently.
10. **Concise output** (CLAUDE.md): ≤10 lines on success, one greppable `ERROR <reason>` line on failure.

---

## 4. Gates, in order. Key on the token, never on `rc` (CLAUDE.md #14)

| # | gate | green = | where |
|---|---|---|---|
| G1 | unit tests (CPU, synthetic, torch-free where possible) | `CLIMATE_DRIVER_TEST_OK` and **every seeded failure in §5 shown red** | compute node or the venv python; not login-node torch |
| G2 | **equivalence** vs `rollout_one_ic` (GPU, `debug`) | ckpt A, test year 2048, start frame 1092, **K=56**: every step's prediction matches `rollout_one_ic`'s. **Pre-register the tolerance in the commit before running**: expected bitwise (same ops, same order, same autocast); if not bitwise, state the max relative error and where, and explain it before proceeding. **Chunk invariance:** `chunk_len ∈ {1, 7, 40}` bitwise identical to each other | `CLIMATE_DRIVER_EQUIV_OK` |
| G3 | **smoke** (`debug`, 1 node) | ckpts **A and B** on 2 GPUs in parallel, start 2044 frame 1092, **600 steps** (150 d; crosses into 2045 at step 368, ends ~2045-02-28). Both write NetCDFs; provenance shows the 2044→2045 hand-off at step 368 with the forcing frame index verified; s/step and peak GPU memory **measured and printed** | `CLIMATE_SMOKE_OK` |
| G4 | record | CHANGELOG entry: measured s/step, memory, per-channel `first_bad_step` for A vs B at 150 d, median-crossing step, `Z3_l17`/`Z3_l10`/`T_l17` global-mean drift in physical units | CHANGELOG.md |

G3 is also the first stability measurement on B (does `n_future=4` push back the ~day-120
failure?). Report it **as measured**, as n=1 per checkpoint. It is not a climate result.

Queue: `debug` 1 node ≤1 h is pre-authorized (memory). 600 steps at ≤1.3 s/step ≈ 13 min per member.
**Surface `capacity`/`preemptable` before any use.** One job in flight.

---

## 5. Tests that must exist, each shown red on a seeded fault first

1. **Streaming = block rollout** on the synthetic fixture (6–12 steps), chunk sizes 1/2/4: identical.
2. **Forcing index**: seed an off-by-one in the chunk-local index, and the test fails (trap §2a.1).
3. **Chunk boundary re-cache**: seed `xz=None` at the boundary, and the test fails (trap §2a.2).
4. **Noleap month binning**: steps crossing Dec 31 18:00 → Jan 1 00:00 and Feb 28 → Mar 1 land
   in the right bins; bin counts `[124,112,124,120,124,120,124,124,120,124,120,124]`.
5. **Accumulators** vs a numpy float64 reference, including the scored-window start.
6. **Stability monitor**: seeded NaN at step k gives `truncated_at_step == k` and outputs written;
   a seeded 5σ ramp on one channel gives the right `first_bad_step`.
7. **Cross-file**: a 2-file fixture where each file's forcing encodes its year and frame; every
   step's forcing is the right (year, frame).
8. **Channel-agnostic**: fixture with a non-101 channel count runs; mismatched stats length raises.

---

## 6. Do NOT

- ❌ Edit `rollout_one_ic`, `run_climate`, `validate_one_epoch`, the preprocessor or the model.
  Anything that changes what the model computes is out of scope (CLAUDE.md #1) and needs jesswan.
- ❌ Loosen the G2 tolerance, or skip or `xfail` a failing test (CLAUDE.md #11).
- ❌ Import from `PanguWeather/` or put `PanguWeather/v2.0` on `PYTHONPATH` alongside makani.
- ❌ Use makani's per-lead `TemporalAverageBuffer` as a time mean, or float32 accumulators.
- ❌ Use `nwp_ic_offsets`/`plan_lagged_sweep` for starts; they refuse cross-file by design.
- ❌ Run torch on the login node; submit to `capacity`/`preemptable` without asking; resubmit a
  job stuck on `queue_tags` (CLAUDE.md #12).
- ❌ Fix `SCALING_CSV_SCHEMA_DRIFT` (7630639's post-processing) here. Note it; separate commit
  or session.
- ❌ Push to `main`, merge, or bypass hooks. Commit small; push the feature branch.

## 7. Open for the operator (do not block on these; state the default)

1. **Soil channels** removed entirely, or kept as prescribed inputs? *Default:* the driver is
   channel-agnostic, so neither blocks this work.
2. **Checkpoint B's epoch**: if `best_ckpt_mp0.tar` is epoch 1, is that the B arm we want, or the
   final epoch (`ckpt_mp0_v3.tar`)? *Default:* `best_ckpt`, labelled with its epoch.

## 8. Definition of done

G1–G4 green on `feat/makani-climate-driver`, pushed; a draft PR against the branch it was cut
from (main is protected); CHANGELOG updated with measured numbers; this file's §7 answers or
defaults recorded. Next session: §4a references + §4c aggregator + §4d prereg, then the 8-member run.
