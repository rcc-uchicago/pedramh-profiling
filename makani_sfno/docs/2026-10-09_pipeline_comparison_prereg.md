# Pipeline comparison protocol and the G1 fp32 run: pre-registration (2026-10-09)

Written before any score under this protocol is computed and before G1 is submitted. Thresholds do not move
after results are seen. Changes go in dated amendments at the end.

**Goal (operator).** The best accuracy on **all channels** over 5-year rollouts: stable *and* a faithful
climate. A new from-scratch training run is allowed.

**Operator rulings (2026-10-09):**
1. **Soil may be dropped.** 99-channel models are eligible. Everything is ranked on the 99 shared channels;
   soil scores of 101-channel models are reported separately and do not enter the ranking.
2. **TF32 is fine.** "fp32" means no autocast (fp32 weights, activations and carried state); TF32 matmuls
   stay allowed in training (`train_plasim.py:302-303`). Inference with `--amp off` keeps PyTorch's
   defaults and records the TF32 flags in each member log.
3. **G1 may run on `capacity`.**

Basis: a 3-debater adversarial review (workflow `wf_e7709935-e37`; full record
`EVAL/pipeline_debate_2026-10-09/debate.json`), judged by the main session. Its consensus points are adopted
below. Two rejections: the 2×2×2 factorial (it detects about 0.05σ; the gaps among survivors are
0.005–0.01σ), and a 90th-percentile-channel primary (chosen after seeing that it favoured FSEMA).

## 1. What a candidate is

A candidate is the tuple (weights, raw or EMA, inference precision bf16/fp32, inference dry-air fix on/off).
Every tuple scored counts toward **k**, which is logged per lineage in CHANGELOG next to every selected
number. No inference variant is assumed to carry over between checkpoints: FSEMA + fix went 0/8, and
F-depth4 + fix moved TREFHT by ±3–4 K.

## 2. Selection and reporting

- **Selection set S:** the existing 8 October-2044 starts (frames 1092 + 16i). The score window is
  **2045-01-01 … 2047-12-31**.
- **Reporting:** the window is **2048-01-01 … 2049-12-31**, on S and on a new start set **R**: 8 April-2044
  starts (frames 364 + 16i), rolled to 2049-12-31. R is run for finalists only.
- **Seal.** No candidate, existing ones included, is scored on 2048–49, and nothing is rolled from R, until
  `finalists_frozen.txt` (checkpoint paths + tuples) is committed. One look.
- Existing 2045–49 numbers are **selection-contaminated** and are quoted only as history.

## 3. Metric

- **e_c:** area-weighted pattern RMSE of the member's window time-mean map against truth's. This is what the
  existing scorer computes, now taken from the per-member `monthly_mean`, weighted by step count.
- **floor_c:** RMS over the 28 member pairs of the pattern RMSE between two members' window time-mean maps.
  That is internal variability under the same prescribed SST. It is pooled over the reference arms B22,
  FSEMA, DRYAIR24 and FD22 (existing bf16 runs) and fixed once for every candidate.
  - Truth 2042–44 vs 2045–47 is reported **only as an upper bracket**. It contains SST-forced interannual
    differences, so it must never be used to cancel work.
  - Fallback: in any group where the floor and the bracket differ by more than 3×, that group uses legacy
    `rmse_sigma` instead of r_c.
- **r_c = e_c / floor_c.**
- **Groups (24):**
  - {T, U, V, Z3, RELHUM} × {top l00–03, mid l04–12, low l13–17} = 15 groups.
  - PS, TREFHT, U10, RHREFHT, PSL, TMQ, FSNT, FSNTOA, PRECT = 9 groups.
- **Primary A:** median over members of the mean over the 24 groups of the group-mean r_c.
- **Worst-group guard W:** max over groups of the median-member group r.
- Always published alongside:
  - the legacy mean `rmse_sigma` (99 ch);
  - an aggregator-sensitivity table: mean, median, p90 and max channel, and the mean excluding l00–l02;
  - 6 h / 24 h RMSE vs B22, as context only.

## 4. Gates (fail any → out, whatever the score)

1. **8/8 finite** to 2049-12-31 on S. Finalists also need 8/8 on R (16/16 overall, reported with its
   Clopper-Pearson bound).
2. **Water stays physical:** global-mean TMQ > 0 at every lead, and last-month negative-TMQ area ≤ 1 %.
3. **Mass:** |dPS| ≤ 1 hPa (budget read-out, last lead − lead 1).
4. **Drift census:**
   - Per channel, drift = (model − truth) annual global mean in 2047 minus the same in 2045.
   - It is expressed in units of truth's detrended interannual SD of the annual global mean, from 2015–2044
     truth.
   - A group drifts if its median channel |drift| > 2.
   - Gate: at most 3 drifting groups.
   - Reporting uses 2049 − 2045, same units.
5. **Variability** (required before a ship decision; reported for every finalist):
   - Model/truth ratio of the SD of the monthly global-mean anomaly, and of the seasonal-cycle amplitude,
     per group.
   - A finalist with more than 3 groups outside [0.8, 1.25] is excluded. This stops a damped model winning on
     time-mean error.

The 1-yr screen and single-step validation loss are **retired as selection gates** and reported as context
only. F-scratch raw e22/e23 passed the 1-yr screen at both starts, then went 2/8 and 0/8 at 5 yr. The screen's
f1092 start is also member 0 of S.

## 5. Decision rule

1. Within each lineage, rank the gate-passers by A on S/2045–47.
2. Finalists are the best per lineage plus the next best overall, up to 6. Freeze them: commit
   `finalists_frozen.txt`.
3. Report every finalist on 2048–49 for S and for R.
4. **Winner:** the lowest reporting A, if all of these hold:
   - (a) it passes every gate on R;
   - (b) W ≤ 1.25 × the lowest W among finalists;
   - (c) its lead over the runner-up exceeds 3.3 × the paired member SE, with the same sign on selection.
5. Otherwise it is a **tie**, broken in this order: fewer drifting groups (2049 − 2045), then lower W, then
   better 24 h RMSE.
6. A genuine group trade-off that the tie-breaks cannot settle goes to jesswan.

## 6. Infrastructure gates (PASS tokens, each run as a debug job, before the result it guards)

| gate | what | status |
|---|---|---|
| `LAUNCHER_DEFAULT_EQUIV_OK` | `submit_f_nosoil.sh` with AMP_MODE/SPARE/DEPEND unset is byte-identical to bac79ea9 | **PASS**, arm/g1-fp32 `9018ec79` |
| `AMP_OFF_DEFAULT_EQUIV_OK` | `climate_rollout.py` with `--amp` at its default reproduces an existing member's `global_mean` bitwise over 120 steps | pending |
| `EMA_EXTRACT_OK` | EMA weights rebuilt from F-scratch `ckpt_mp0_v22.tar` (`ema_state`) equal `best_ckpt_ema_mp0_e23_stable.tar` bitwise; version→epoch map confirmed (same check on F-depth4) | pending |
| `WINDOW_SCORER_OK` | the monthly_mean window scorer reproduces legacy full-window B22 0.2394 and FSEMA 0.2341 within 1e-3 | pending |
| throughput probe | fp32 s/step and peak memory at the production layouts: single-step 2 nodes LB4 (7731735); depth-4 2 nodes LB2 (7731736) | running |

## 7. The existing pool (no training)

- **F-scratch EMA e15–e43:** 29 candidates. Every `ckpt_mp0_vN.tar` holds `ema_state`.
- **F-depth4 EMA:** every 4th epoch, e4…e63 (~16).
- **DRYAIR raw e15–e23:** 9 (101 ch; ranked on the 99 shared).
- **Already tested arms:** B22, B24, B22 ± fix, B24 ± fix, B22posclamp, FSEMA, DRYAIR24, FD14/22/24 ± fix. Their
  existing runs are re-scored on S/2045–47. B cannot be swept: only v0–v3 and e22/e24 are kept.
- **fp32 inference** (`--amp off`): the top 3 per lineage after the bf16 sweep, plus B22 and FSEMA, paired
  against bf16.
- **Inference fix variants:** only for the top 3 per lineage.

Run order: debug and debug-scaling, one queued job each, about 8 checkpoints per job. Preemptable is proposed
to the operator before use.

## 8. G1: fp32 from scratch (approved for `capacity`)

**Stage A** = the F-scratch recipe, unchanged except for precision:
- launcher: `AMP_MODE=none SPARE=0 DEPEND=7725884 bash polaris/submit_f_nosoil.sh 2 23 <wall> scratch`
  from worktree `arm-g1-fp32` (branch `arm/g1-fp32`), which is FROZEN while the job is queued;
- 99 channels, random init, makani's default seed (the same as F-scratch);
- SGDR T₀ 20 with 3 warmup epochs, peak LR 2e-3, β₂ 0.95, global batch 32;
- EMA 0.9995 saved in every epoch's checkpoint, `CKPT_VERSIONS 250`;
- **23 epochs**, which ends cycle 1: the FSEMA analogue;
- 2 nodes, no spare;
- walltime set from the probe.

**Stage B** gets its own addendum before submission. Its intended shape:
- depth-4 (`MULTISTEP=5`), warm-started from Stage A's EMA e23 (extracted under `EMA_EXTRACT_OK`);
- LR 4e-4, cosine annealed to 1e-6 at the final epoch (`SCHED_TMAX = EPOCHS − 2`), global batch 16;
- EMA on, fp32, about 24 epochs.

**Candidates** are every EMA epoch of Stage A e15–e23 and of Stage B, all through the §3–§5 harness.

**What G1 can and cannot show:**
- It is one seed, so it supports checkpoint claims only, not recipe claims.
- Stage A vs F-scratch (same recipe, same default seed, fp32 vs bf16) is a descriptive precision comparison
  (n = 1).
- No outcome is predicted.

## 9. Not in scope here

Recipe and attribution claims (≥ 2 seeds each), the constraint bundle (capped tendency-unit loss, Z3 output-only,
positivity clamp, trained-in dry air) and ACE2 each need their own pre-registration.

## Amendment 1 (2026-10-10, before submission): the stop-early trained-in dry-air test, and how the gates read the window

**Gates vs scores (clarifies §2 and §4).** Gates 1–3 are pass/fail on the whole run to 2049-12-31: finite, TMQ > 0,
and |dPS| measured last lead − lead 1. That is what "survives 5 yr" means. Every *score* that ranks candidates is
read only on 2045–47 until `finalists_frozen.txt` exists: A, W, the drift census, and any channel-level climate
error. Ranking from `readout.log` drift values at leads after 2047-12-31 is forbidden.

**Operator, 2026-10-09/10.** Test training the dry-air constraint in on F-depth4, stop-early, with the first block
on capacity ahead of G1. If G1 starts first because of queue order, the block follows G1.

**Run `fsF_dryair_nf4_b16_rtmax100_scratch`**, from worktree `arm-fd-dryair` (branch `arm/fd-dryair` at
`9018ec79`; frozen while queued):
- **Settings:** 7719538's exact settings, plus `CONSERVE_DRY_AIR=1`.
  - Warm start from F-scratch raw e23 (`best_ckpt_mp0_e23_stable.tar`) with fresh optimizer, scheduler,
    counters and loss.
  - `MULTISTEP=5`, LR 4e-4, `CosineAnnealingLR` `T_max 100`, 1 warmup epoch at `LR_START 0.01`.
  - 2 nodes × LB 2 (global batch 16), 99 channels (`e3sm_alldata_nosoil.yaml`), bf16.
  - EMA 0.9995 (the config default), `CKPT_VERSIONS 250`.
- **Prerequisite:** smoke debug 7732540, 60 steps on the same settings. It must print `MAKANI_MN_SCALING_OK` with
  a finite loss. Otherwise nothing is submitted.
- **Block 1 = epochs 1–4** (about 3 h).

**Continue rule.** After each block, run the 5-yr protocol (§2 S set) on raw and EMA at the block's last epoch and
the epoch two before it. Continue by 4 more epochs, resuming in place with `LOAD_*=1`, only if every condition
holds:
1. At least one of them passes gates 1–3.
2. Its global TREFHT drift (2047 − 2045 annual) is within 0.5 K.
3. Its drift census has at most 3 groups.
4. It is not worse on A (2045–47) than F-depth4 EMA at the same epoch from the sweep. "Worse" means a gap larger
   than 3.3 × the paired member SE.

Otherwise stop. The arm's total cost is then the blocks already run (about 6 node-h per block). Block 1 cannot
show that later epochs will be good; it can only show cheaply that the approach fails.

**What it decides.** Whether G1 Stage B gets a trained-in dry-air variant. If the arm is stopped, Stage B runs
plain only. No prediction is made.
