# SQG online training: what else to test (three-agent team, 2026-11-06)

**Ask (user, 2026-11-06):**
- Show the math behind the delta-fidelity table. → `../delta/MATH_delta.md`
- Then explore, "thinking like AI for science", what else could make online training more efficient within the
  known limits, using two Opus agents and one Fable principal scientist that communicate.
- Follow-up: find analogues in other fields. → `ANALOGUES.md`

**How it ran.**
- **Workflow:** `wf_0391b43a-95e`, 8 agent runs, 0 errors, ~74 min, all read-only on polaris-login-02.
- **Agents:**
  - NUMERICS (Opus): solver, precision, resolution, statistics.
  - LEARNING (Opus): learner, data pipeline, schedule.
  - PS (Fable `claude-fable-5`): principal scientist.
- **The exchange:**

  | round | who | what |
  |---|---|---|
  | R1 | both explorers, independently | proposals |
  | R2 | both explorers | each reads the other's and replies: endorse / challenge / combine / ≤ 3 new |
  | R3 | PS | verified 15 claims, sent challenges, gave a provisional ranking and rejects |
  | R4 | both explorers | answered every challenge, revised or withdrew proposals |
  | R5 | PS | final ranked plan with pre-registered predictions and gates |

- **Full record:** every round's structured output is in `team_results.json`.

**Main-session verification.** I checked these claims against the files before carrying them:

| claim | verdict | evidence |
|---|---|---|
| channels_last is a gain (+38.5%), as stated in my R1 brief | **WRONG, caught by LEARNING**: it is 38.5% *slower* (576 vs 416 ms, all of it in backward) | notes Part III-C L287–304; memory corrected |
| subsample aliasing puts ≈ 0.002 of the 0.0043 val MSE out of reach | **confirmed independently** | m16 − m1 increment band excess 0.041 (k < 128), ≈ 0.052 with the corners, × Var(Δ) 0.0764 = 0.0040; half of it unpredictable (θ at k ≥ 128 has C(0.02) = 1e-4) ≈ 0.0020. Team's adopted lo = 0.00222 |
| the denominator: 168 h (handoff) vs ~142 h (CHANGELOG L449) | **explained, not resolved**: 142 h is the 11-02 estimate; the user gave 7 days on 11-05 | only the stdout parse (item 1) pins it |
| o1.2 resumes only from `best.pt` and rebuilds a 4-segment buffer | **confirmed** | o1.2 L321–335 |
| o1.2 is not loss-deterministic (so replay is data-bitwise only) | **confirmed** | L128–135 set only strict IEEE fp32; no `use_deterministic_algorithms` |
| GPU 0 of the H200 box is not used by o1.2 | **confirmed** | `o1.2-TrainUViT_m16_tendency.sh`: `GPUS=1,...,7` |
| o1.2 logs s/step and epoch seconds every epoch | **confirmed** | L526–527 |
| 47% of UViT GPU time is fusible glue | **confirmed** (elementwise 19.8 + LayerNorm 15.3 + copies 5.2 + memcpy 3.8 + reduce 3.1) | notes L329–353 |
| the solver itself runs at ≈ 4.7 ms/step on Hopper | **confirmed**: "2.2e8 steps (~287 h on an H100)" → 4.70 ms | `A-SQG_data.py` L141 |

**What the H100 comment implies** (main session, an inference): 480 × 4.7 ms = 2.26 s of solver plus 0.39 s of
trainer = 2.65 s, against the user's ≈ 3.1 s/step. So ≈ 0.45 s/step (≈ 24 h of 168 h) is not steady-state solver.
This supports the PS's "reading (a)" (≈ 26 h/run of restart / fill / validation overhead) before any log is
parsed.

**What the literature search adds** (`ANALOGUES.md`; the PS did not see it):

1. **`lean32ek` is the published fix for this failure.** Klöwer et al. 2022 (JAMES) used compensated time
   integration for Float16 shallow water. It was faster than mixed precision there too (3.6× vs ≈ 2.75×). Cite it
   in the item-4 memo.
2. **Item 3:** add a stochastic-rounding `lean32` arm as a *noise-only* calibration control (Paxton 2022; Kimpson
   2023).
   - Predicted: k1–2 e_Δ ≈ 3e-5–1e-4 with zero signed mean; overall ≈ `lean32ek`.
   - Together with the decay-quantization (pure-bias) control, this brackets the taxonomy from both sides.
3. **Before any "where to simulate" idea:** run a deep-ensemble disagreement scan over checkpoint segments on
   stored frames (the FLARE / ADEPT / AL4PDE pattern). It tells whether active selection has any epistemic signal
   above the aliasing floor.
4. **The online loop that adds information is on-policy:** model-visited states (Rasp 2020) or an unrolled rollout
   loss (Frezat 2022; Yan 2024 on pyqg-JAX). "More of the archive, denser" does not add information. Raise it with
   the owners as a research direction, after item 2.

---


## The principal scientist's final report (R5, verbatim)


**Objective:** trained-model quality per GPU-hour. Production: o1.2 online = 7 days on 7 H200 (handoff: 168 h, 147 h solver; CHANGELOG: ~142 h, ~120 h solver — unresolved, item 1 settles it) vs 21 h offline at identical 195,250 steps.

### 1. The math behind the delta-fidelity table (user's question, verified vs `polaris_delta_7819310/delta_fidelity.log` + `MATH_delta.md`)

Metric: `e_Δ = ‖Δ_v − Δ_ref‖/‖Δ_ref‖` over the 0.02-TU pair lag (4,000 steps); frame error is magnified ×1.83 into Δ (×~25 inside k1–2, where Δ holds only 6.8e-5 of its energy).

| variant | mechanism | arithmetic | measured |
|---|---|---|---|
| lean32 (all f32) | f32 **state** rounds every increment: per-step error ≤ u·|θ|/|i|, u=2⁻²⁴; at k1–2 the increment is 9.8e-6 of θ → 6.1e-3/step, matching the measured 1-step 5.3e-3. Slow modes keep the sub-ulp remainder's sign → **coherent bias**, flat in L. f32 decay table (rate error −15.7% at k≈8.4) adds ~20% | bias, not √L | 5.46e-4 overall, 3.6e-3 at k1–2; **red** per-mode spectrum (260× spread) |
| lean32ek (compensated f32 state) | TwoSum/Fast2Sum ≈ 48-bit state → only the f32 **tendency** errs: 3.29e-5 × (incr/θ = 8.44e-4) = 2.8e-8 of θ/step; correlated ~40 steps (e_frame(40)=3.8e-7), then random-walks: 3.8e-7·√100·1.28 = 4.8e-6; ×1.83 → 8.8e-6 | noise, √L | 8.84e-6, 2.2e-5 at k1–2; **near-white** (5× spread) |
| mixed (f64 state, f32 FFTs) | same tendency error, same walk | — | 8.67e-6 (= lean32ek to 2 digits) |
| f64ic32 floor | one f32 rounding of the state (2.5e-8 rms of θ) → e_frame 4.0e-8, ×3.1 chaotic growth over 0.02 → 2.3e-7 | reference floor | 2.32e-7 |

ms/step (A100, memory-bound): lean32 3.20; lean32ek 3.35 (+0.149 ms = the extra complex64 `lo` array, 134 MB at 0.90 TB/s); mixed 4.73 (+1.38 ms = complex128 state+tendencies). Vs production 9.29 → 2.90× / 2.77× / 1.96×. All failed the pre-set 1e-6 gate → owners' call (item 4), never a post-hoc loosening.

**Learner's view:** lean32ek's per-mode error variance (~1e-15) is ~1e-7 of the per-mode alias floor (1.8–4.4e-8) — invisible as noise; only a conditional mean could matter, and item 3 measures exactly that with calibrated positive controls.

### 2. The one floor table (adopted; verified from `learning_14_archive.out`, stable under fit variants, out-of-sample: m1-only fit predicts m16 alias excess to 1.2% at k64–128, 2.2% at the corner)

| coarse band | signal S | alias A | A/(S+A) | Wiener W |
|---|---|---|---|---|
| k<32 | — | 5.8e-5 | ≤0.002 | 5.8e-5 |
| k32–64 | 1.49e-2 | 1.97e-4 | 0.013 | 1.93e-4 |
| k64–128 | 6.99e-3 | 1.34e-3 | 0.161 | 1.00e-3 |
| k128–183 corner | 5.8e-4 | 6.3e-4 | 0.520 | 2.95e-4 |
| **total** | | **lo = 0.00222** | | **central = 0.00377**, hi = 0.00406 |

val 0.0043 → 1.93× lo, **1.14× central**, 1.06× hi; 86.6% of the floor sits at coarse |k| ≥ 64. So ≤12–17% of val is reducible by ANY data/step/precision change against central. Caveat: 0.0043 is o1.1's **mid-schedule** value; only annealed finals compare (item 1 retrieves them). Pre-step: exact F1 from stored E(k), acceptance band [0.0021, 0.0023], replaces the model table only BEFORE the readout.

### 3. Ranked plan (~26 node-h total; cheapest decisive first)

| # | item | cost (node-h) | where | decides | P(pass) | owner |
|---|---|---|---|---|---|---|
| 1 | STEP0 log/checkpoint parse (N2+L1a+L10a) | 0.05 | CPU, science machine | denominator 142 vs 168 h; restarts; o1.1/o1.2/offline finals; L6 gate | 0.6 reading-(a) | user |
| 2 | STEP1 per-band floor readout (+L5, L9-0, L13 riders) | 2 | sci-machine GPU0 / Midway | pivot: replay+fewer steps vs online value; eps_model,b | 0.65 at-floor; 0.6 parity | us run / owners act |
| 3 | STEP2 generator taxonomy (N1+L3+dt ladder+fast form) | 1 | Midway pref. (Hopper timing free) | lean32ek admissibility; dt bias; mechanism B | 0.8 lean32ek clean; 0.9 controls | us / owners via #4 |
| 4 | N11 acceptance protocol, signed pre-unblinding | 0.1 | none | tolerances g_b fixed before data | 0.7 admits ek, rejects lean32 | jesswan/Weidong |
| 5 | L2a tee/replay (4.30 TB, data-bitwise) | 1 | Polaris debug | solver → one-time cost; exact restart buffers | 0.7 | user(/data) + Weidong |
| 6 | L4 compile + activation ckpt (gated 1e-2 / bitwise) | 0.5 | Polaris debug | trainer 1.2–1.4×; batch 32 on A100 | 0.55 / 0.7 | us, gated |
| 7 | L10 resume-from-last (guarded) | 0.3 | Polaris debug | kills re-run epochs + LR rewind | 0.9 harness | Weidong |
| 8 | L6 compressed OneCycle + **first 2-seed replication** | 21 | sci-machine 7×H200 | step count ÷2; seed-spread calibration for ALL falsifiers | 0.6 spread≤2%; 0.5 1/2-arm | jesswan/Weidong |

Order of operations: 1 → (2 ∥ 5-gate storage question) → 4 signed → 3 → 6/7 anytime → 8 after its gate. Item 8's 1/2-schedule arm is swapped for L11's 1,750-TU annealed arm if the o1.1 parse shows a mid-schedule stop.

### 4. Projection (T = 21 + D/s; D = 120 | 147 solver-h, item 1 decides)

| rung | needs | T (h) | status |
|---|---|---|---|
| today | — | 142 | 168 | user-reported |
| lean_u1 (bitwise) | H200 carry-over only | 104 | 122 | speed **measured** (A100) |
| + lean32ek | items 3+4 pass | 64 | 74 | speed+error measured; acceptance assumed |
| + fast form | bitwise check | 56 | 64 | 2.7 ms **unverified estimate** |
| + restart recovery | items 1,5,7 | −up to ~26 | assumed pending parse |
| pivot: replay (repeat runs) | item 2 parity + tee | ~21 | data-bitwise, measured mechanics |
| + compile 1.2–1.4× | item 6 gates | 15–17.5 | assumed, gated |
| + 1/2 schedule | item 8 | ~8–10.5 | assumed, P=0.5 |

Repeat-run endpoint ≈ 60–75 GPU-h vs 1,176 today (~15×), with one generating run at 56–122 h paid once per stream.

### 5. Not pursued
N6 (2048² restarts — regenerates stored information), N7 (densification — density isn't the information axis, C(0.02)=−0.065), N8 (custom f64 FFT — withdrawn by proposer), N10 (offset views — EV<1), L7 (conditional on the outcome we bet against), L8/L11-B standalone (read o1.1 first). Standing dead ends: plain f32 state, STRIDE thinning, channels_last (+38.5%, measured), fp16/bf16 FFTs, Pallas, cuFFT callbacks, editing A-SQG_data.py, moving any tolerance after the number is seen.

### 6. Open questions for jesswan/Weidong
(1) Sign gates A–D before unblinding; (2) /data space + will runs reuse the stream; (3) lean32ek acceptable in principle; (4) last.pt resume; (5) logs + did o1.2 finish; (6) GPU 0 idle / bigger val for selection; (7) dt=2e-4/30 and y_LP as pre-registered tests; (8) dense members past t=520.

### 7. Dissent (summary)
dt lever demoted vs NUMERICS (kept as free arms; counter: only f64 lever >1.45× if f32 refused). Pivot keyed on paired differences per LEARNING (counter: val/F=1.14 is absolute and survives an unfinished o1.2). L11 deferred to the o1.1 natural experiment (counter: o1.1 is confounded; swap rule in item 8 if the parse disappoints). My round-3 corner-slope value (0.0024) was wrong; the adopted table is LEARNING's, reproduced by NUMERICS, re-verified by me.


## Ranked plan in full (R5, verbatim): test, prediction and gate fixed before any run


### 1. STEP0 (N2+L1a+L10a): Zero-GPU parse of the o1.2/o1.1/offline stdout logs and best/last.pt histories: pin the denominator, count restarts, retrieve final vals

- **Why:** Every projection on the table divides by an unpinned denominator: the handoff says 168 h total / 147 h solver (sqg_opt_handoff.md L70-72), CHANGELOG.md L449 says ~142 h / ~120 h. Both are rate x steps, not a measured wall clock. The same parse retrieves o1.1's and o1.2's final vals (o1.1 '3500 x 0.5, level with offline' is a free production-scale SEG_T/trajectory-time data point, o1.2 docstring L1-2), the offline v125/v200/v249 that gates item 8, and the restart count that prices item 7. o1.2 logs everything needed: per-epoch s/step and epoch seconds (L526-527), buffer-fill time (L340-341), resume lines (L327-328), per-trajectory val (L528).
- **Test:** Stdlib parser over logs/uvit_m16_tendency_online3_*.log (one file per launch, --max_restarts=0), o1.1's logs, and the *_best.pt/*_last.pt val/train histories (not wandb: step rewinds after a best.pt resume, L538-542). Outputs: median steady-state s/step; solver hours = sum((s/step - 0.39) x steps); overhead = wall - sum(epoch seconds); number of restarts and per-restart epoch lag; o1.1/o1.2/offline final vals; offline v125/v200/v249 and sigma_late; val SE from per-trajectory sums.
- **Where:** Science machine / wherever the production logs live (CPU only; needs the user to point at the logs)
- **Cost (node-h):** 0.05
- **Prediction:** P=0.6 that reading (a) holds: median steady-state 2.4-2.9 s/step with >= 10 h of non-step overhead (restarts/fills/validation), supported by the solver's own '~287 h on an H100' comment implying ~4.7 ms/step on Hopper. P=0.5 that >= 1 restart occurred.
- **Gate:** Parse must cover >= 90% of epochs or the denominator stays dual-quoted. Reading (a) is falsified if median steady-state >= 3.0 s/step or overhead < 5 h. No T-projection goes to jesswan/Weidong before this lands.
- **Expected gain:** Pins D at ~120 vs ~147 solver-hours, an 18-25 h spread in every downstream projection; if reading (a), ~26 h/run of non-solver overhead becomes recoverable by items 5+7 with zero numerics change (~182 H200 GPU-h/run); o1.1's annealed endpoint settles most of the trajectory-time question for free.
- **Changes what is computed:** no (read-only analysis)
- **Decides:** user (runs it; must supply log paths)

### 2. STEP1 (L1+N3+N11-A, riders L5/L9-0/L13): Per-band val readout of offline vs o1.2 final checkpoints against the ONE pre-registered floor table, with residual projections on y_alias/x_alias; riders: clip-active fraction + gradient-noise scale, y_LP stage-0, zero-training physics baselines

- **Why:** The verified floor table (learning_14_archive.out, re-checked this round: lo = Var(y_alias) = 0.00222, central +Wiener = 0.00377, hi = 0.00406, 86.6% of it at coarse |k| >= 64; out-of-sample: m1-only fit predicts the m16 alias excess to 1.2% at k64-128 and 2.2% at the corner) says at most ~12-17% of a 0.0043 val is reducible by ANY data/step/precision change against central, 48-51% against lo. This one job decides whether the 120-147 h/run of solver can buy anything, prices every generator decision, and fixes the per-band eps_model that item 3's bias gate needs. Riders cost ~0.3 GPU-h extra and settle batch-scaling direction (L5), y_LP GO/NO-GO (L9), and model-limited vs data-limited bands (L13/N12 coarse-SQG + semi-Lagrangian baselines).
- **Test:** Pre-step (Midway CPU, ~15 min): exact F1 from the archive's stored isotropic E(k) (A-SQG_data.py L408-410) through the lattice-fold script; acceptance band pre-registered. Then 1 GPU, 1-2 h: >= 2,000 val pairs (4096^2 fields where available), per-band (256^2 rfft octaves) errors of offline final, o1.2 final (or o1.2@E* with the paired equal-epoch fallback over E*-9..E*), o1.1 final if it exists; residual projections on y_alias and x_alias; rollout card (RMSE at 0.1/0.5/1 TU; 50-TU free-run spectrum, k<=4 energy, theta PDF); riders.
- **Where:** Science machine GPU 0 (if idle) or Midway H100 (archive is local); Polaris blocked (no checkpoints staged)
- **Cost (node-h):** 2
- **Prediction:** P=0.65: offline final val / 0.00377 in [0.95, 1.3] AND >= 80% of val MSE at coarse |k| >= 64 AND per-band error >= 0.9 x A_b in k32-64/k64-128/corner. P=0.6: o1.2-offline parity (every band within 3 paired SE and within 5% of (error_b - A_b)).
- **Gate:** Fixed now: exact-E(k) pre-step must land F1 in [0.0021, 0.0023], else the exact table replaces the model table BEFORE the readout (never after). Pivot rule: parity + floor-consistency => the program pivots to replay + fewer steps and online generation is only ever paid once; any band with (offline - o1.2) > 3 paired SE AND > 10% of (error_b - A_b), or a rollout difference beyond the across-trajectory spread => online retains value and L7-style designs re-enter. P4: final val >= 0.95 x lo, else the floor model is refuted and all floor-based reasoning is suspended.
- **Expected gain:** Decides the fate of ~840-1,029 H200 GPU-h per run. If parity: every later run is a 21-h trainer job (-121 to -147 h/run), and precision sign-off matters only for generating runs. Fixes eps_model,b (needed by item 3), B_simple direction, and the data-vs-model split per band.
- **Changes what is computed:** no (analysis); acting on parity (train offline/replay) is a data-distribution decision for the owners
- **Decides:** us (run); jesswan/Weidong (act on the pivot rule they pre-signed)

### 3. STEP2 (N1+L3+N4 arms): Generator-error taxonomy on 16 archived states: noise-vs-bias per band with two calibrated positive controls, alias-free split, dt ladder (2e-4/30, /32, 2.5e-6), lean32ek fast-form bitwise+timing

- **Why:** This unblocks the largest surviving solver lever (lean32ek 2.77x measured, fast form ~3.44x estimated, vs lean_u1's bitwise 1.45x). The verified mechanism work says lean32ek's error is near-white FFT noise (per-mode variance ~1e-15, i.e. ~1e-7 of the 1.8-4.4e-8 per-mode alias floor) while lean32's is a red slow-mode bias - but that is from one synthetic state (rms(Delta) 0.825 vs archive ~0.27); the archived-state rerun is owed (notes L1453-54). The decay-quantization control (known beta per shell, 7e-4 at k1-2 vs MDE 6e-6-8e-6) plus the injected-2xMDE control make the job self-calibrating: without them a lean32ek pass proves nothing (LEARNING's power argument, accepted).
- **Test:** dt_ladder.py written first with CPU tests (variable-step AB3 coefficients verified in exact arithmetic; uniform branch at h=DT must reproduce lean_u1 bitwise at lags 40/400/4000 - the bitwise control). Arms per GPU (4 states x >= 4 members, 4 pairs/state): lean_u1 ref + lean32ek on all 16 pairs; decay-quant control, lean32, f64ic32, fast form on first pairs; dt arms on one pair (the 2.5e-6 Richardson arm separates first-order splitting from third-order AB3: predicted ratio -1.50 vs -0.64). Analysis: per-band e_Delta on subsampled + alias-free frames; per-mode error spectra; regression e_q = alpha_q x_q + beta_q Delta_q clustered by state; MDE recomputed from lean32ek's own scatter BEFORE unblinding z-scores. ~25 min/GPU, fits one debug hour.
- **Where:** Midway pedramh-gpu preferred (archive local, yields the Hopper timing calibration free); fallback Polaris debug after staging ~8 GB (lfs quota check first - Eagle was 49.88/50 TB on 11-02)
- **Cost (node-h):** 1
- **Prediction:** P=0.9 both positive controls recover (decay-quant beta within 20%/shell; injected bias z >= 3, beta within 30%) - else the job is void. P=0.8 lean32ek shows no band bias above MDE (theory bound ~1e-7 of Delta, two orders below MDE). P=0.85 lean32 fails the bias gate at k <= 4. P=0.6 dt order ratio in [-1.6, -0.55]. P=0.5 fast form bitwise-to-lean32ek at <= 2.9 ms.
- **Gate:** Fixed now: any positive-control failure voids the run (no post-hoc rescue). lean32ek is admitted only against item 4's owner-signed g_b = 0.1/sqrt(n_b) per-band bias gates with eps_model,b taken from item 2 (so item 2 precedes this job's pre-registration). h=DT bitwise control must be exact or the dt arms are discarded.
- **Expected gain:** If lean32ek admitted: T = 21 + D/2.77 = 64-74 h per generating run (fast form 56-64 h), i.e. -48 to -58 h and -336 to -406 H200 GPU-h per run without a tee, once per stream with one. A100 ratios; H200 carry-over checked by the Midway Hopper timing (pass band: lean_u1/lean32ek ratio in [1.5, 3]).
- **Changes what is computed:** the test: no. Adopting lean32ek for generation: yes (targets differ by ~9e-6 of Delta) - owners' call. Adopting dt=2e-4/30: yes (data distribution) - owners' call, and it kills the first-segment bitwise assert.
- **Decides:** us (run); user (site/staging); jesswan/Weidong (adoption, via item 4)

### 4. N11 (absorbs N5-revised): Floor-referenced acceptance protocol for non-bitwise generators, tolerances signed by the owners BEFORE item 3 is unblinded

- **Why:** Part XI's strict pre-set gate (e_Delta <= 1e-6 vs a rounding floor) was failed by every float32 variant and measures nothing the learner can see (solver errors enter the loss in quadrature: lean32's full 5.5e-4 changes the MSE by 5e-5 relative, MATH_delta s9). The replacement must still be fixed before the data are seen - the project's 'bitwise is not a hard gate / never loosen after the run' rule is the whole point. Gate A: paired per-band error variance < 1% of F1_b. Gate B: |bias| < g_b x eps_model,b with g_b = 0.1/sqrt(n_b) (bias <= 10% of the model's own error accumulated over the band's rollout memory). Gate C (1-TU segment statistics, G2 dropped as powerless) only if repeated non-bitwise generation is planned. Gate D (dose-response fine-tune) only if B fails.
- **Test:** Zero GPU: hand jesswan/Weidong the verified floor table + gates A-D + the MDE table now; they fix g_b and any additional rollout/climate criteria in writing. Items 2 and 3 then supply the inputs. Predicted outcome: lean32ek passes A+B; lean32 fails B at k <= 4 (its mechanism-A bias 7e-4 is >100x the MDE).
- **Where:** No compute - a signed one-page protocol
- **Cost (node-h):** 0.1
- **Prediction:** P=0.7 that lean32ek passes A+B and plain lean32 fails B, i.e. the protocol both admits the useful generator and demonstrates it can reject.
- **Gate:** The protocol is valid only if signed before item 3's regression outputs are unblinded. If the owners add criteria, those are appended before, never after.
- **Expected gain:** Converts item 3's measurement into a decision: -48 to -58 h per generating run. The same yardstick then prices dt (N4) and TF32/compile precision questions without new negotiation.
- **Changes what is computed:** no (decision rule); anything it admits is a data-distribution change the owners have then explicitly accepted
- **Decides:** jesswan/Weidong

### 5. L2a (tee/replay): Deterministic tee of the 256^2 online frame stream to disk (4.30 TB); replay test proves data-bitwise replay, exact post-restart buffer reload, <0.5% writer overhead

- **Why:** The sleeper: o1.2's stream is a deterministic function of (checkpoint, solver build) - 5,001 frames/segment x 3,281 segments x 256 KB = 4.30 TB (verified arithmetic). With a tee, the solver becomes a ONE-TIME cost per stream: every later run (seeds, ablations, o1.3) and every restart is a ~21-h trainer job. It also fixes a fidelity defect found this round: after a restart the FIFO holds 19.6k pairs for ~26 epochs and refill pairs are drawn ~9.6x vs 2.7x steady-state - only an exact buffer reload removes that. Claim restated per my round-3 challenge: DATA-bitwise in production (o1.2 seeds RNG at L124 but never calls use_deterministic_algorithms, L128-135); loss-bitwise only inside the instrumented harness. Free integrity gain: today only the FIRST segment per rank is checked against train_m16.zarr (L283-287/L336-339); with a tee every 0.02 frame of all 3,281 segments is checkable offline.
- **Test:** Polaris debug, 1 node, <= 1 h, from the saved delta_7819310 state: 200 ML steps with writer on (lean_u1 + UViT), then replay solver-off. PASS: per-chunk sha256 of frames equal; sampled index sequence equal; loss bitwise under harness-only torch.use_deterministic_algorithms(True) + CUBLAS_WORKSPACE_CONFIG; writer overhead < 0.5%; kill-mid-epoch-2 + resume reloads a sha256-equal buffer. L2b (bx/by -> single frame ring, 137 -> ~69 GB/rank) split out for separate Weidong sign-off.
- **Where:** Polaris debug (1 node, 1 h); production writer later lands in Weidong's o1.2 on the H200 box
- **Cost (node-h):** 1
- **Prediction:** P=0.7 full PASS (the stream is deterministic by construction; residual risk is writer I/O interference and DataLoader RNG consumption under replay, both unverified).
- **Gate:** Fixed now: any frame/index mismatch or > 1% production-loop slowdown = FAIL, no partial credit. The whole item is conditional on the user confirming >= ~5 TB free on /data; fallbacks (compression, every-10th-frame) are separate proposals - the 10th-frame one is a science call.
- **Expected gain:** Every reuse of the stream: 168 -> ~21 h (-147 h, -1,029 H200 GPU-h per later run; 104-122 -> 21 h after lean_u1). Per restart (de-duplicated against item 7): 2.4 / 8.4 / 28.5 h at 1x2 / 2x5 / 4x10 restart-x-lag, plus the buffer-transient fidelity fix. Pays only if more runs on these trajectories are planned - question 2 to the owners.
- **Changes what is computed:** no - data-bitwise; training noise is the same GPU nondeterminism every production rerun already has
- **Decides:** user (storage + whether future runs reuse the archive); Weidong (writer in his script)

### 6. L4 (compile + act-ckpt): Trainer throughput: torch.compile the ConvNeXt glue (gated at 1e-2 loss-trajectory), activation checkpointing for batch 32 on A100 (bitwise); channels_last stays dead

- **Why:** 47% of the UViT step is fusible glue over ~3,562 launches (notes L329-353); channels_last was measured 38.5% SLOWER (L287-304, L411) and stays rejected. Today this is only 2-4% of 168 h, but it multiplies into the entire trainer term the moment any solver-off path lands (replay, offline pivot, decoupling): 21 h -> 15-17.5 h at 1.2-1.4x. Activation checkpointing makes Polaris recipe-faithful (batch 32 OOMs a 40 GB A100 today, L282-285).
- **Test:** Polaris debug, 30 min, profile_uvit.py arms: eager vs compile (default, max-autotune-no-cudagraphs) at batch 16; batch 32 with per-stage checkpointing. Strict fp32, TF32 off.
- **Where:** Polaris debug; recheck on the production env's newer torch before shipping
- **Cost (node-h):** 0.5
- **Prediction:** P=0.55 compile >= 1.2x within the gate; P=0.7 checkpointing fits batch 32 at <= 1e-5 vs eager.
- **Gate:** Pre-set: compiled K=20 loss-trajectory relative difference <= 1e-2; checkpointing <= 1e-5 vs eager; ship nothing below 1.1x. TF32/bf16 are NOT in scope (break the strict-IEEE check; owners' call via item 4's yardstick).
- **Expected gain:** -3.5 to -6 h (-25 to -42 H200 GPU-h) per trainer-dominated run; on the replay+1/2-schedule endpoint it is the difference between ~10.5 and ~8 h.
- **Changes what is computed:** compile: rounding only (gated); activation checkpointing: no (bitwise)
- **Decides:** us (gated); user ships into o1.2/offline

### 7. L10 (resume fix): Resume from last.pt with a finite-and-<=1.5x-best guard (best.pt as fallback): kills the re-run epochs and the OneCycle rewind after a restart

- **Why:** Verified one-line reality: o1.2 ALWAYS resumes from best.pt (L322, docstring L25-28) while last.pt with the identical field set is saved every epoch (L464-471, L544). With a flat val whose 200-pair argmin is noise-driven (SE ~2%), each restart re-runs the post-best epochs at 3.1 s/step and rewinds the LR. Standalone gain is re-run epochs ONLY (R x lag x 0.67 h) - the refill hour and transient belong to item 5; double-counting removed per my round-3 challenge.
- **Test:** (1) Free: item 1's parse counts restarts and lags; propose only if sum(lag) x 0.67 h >= 5 h/run. (2) Polaris debug ~20 min: toy harness (3 epochs x 20 steps, short segments) killed mid-epoch-2; PASS = resumes from last.pt, i_seg continues, histories contiguous, LR does not rewind.
- **Where:** Polaris debug (harness); the change itself is one line in Weidong's script
- **Cost (node-h):** 0.3
- **Prediction:** P=0.9 the harness passes; P=0.5 the parse shows >= 1 restart making the gain > 0 (1.3 / 6.7 / 26.9 h at 1x2 / 2x5 / 4x10; increment over item 5 only 0.17-3.4 h).
- **Gate:** The guard (finite AND <= 1.5x best) is mandatory - it preserves the divergence protection best.pt-resume was presumably for. Gain quoted to the owners only from the measured restart count, never the hypothetical table.
- **Expected gain:** 0 h if the run never restarted; up to ~27 h/run if reading (a)'s overhead is restart-dominated.
- **Changes what is computed:** no for an unrestarted run; for a restarted run it moves training toward the uninterrupted trajectory (procedure change)
- **Decides:** Weidong (best.pt-resume is his deliberate convention)

### 8. L6 (+replication pair): Compressed OneCycle on stored frames: two 1/4-schedule seeds (the project's FIRST true replication) + one 1/2-schedule arm, each with a free EMA readout; gated on the offline val-history precursor

- **Why:** Step count multiplies every path (online cost is linear in steps; offline 21 h likewise). Near a floor where ~half the residual is unpredictable target noise, late steps mostly average noise, and a compressed anneal averages it too. Equally important: no two-seed pair exists anywhere in this program - every falsifier in items 2-7 is denominated in 'seed spread' or 'late-epoch std' that nobody has measured. The 2 x 5.25 h seeds buy that calibration for everything, and serve as the shared baseline for the L9 (y_LP) and L11 (trajectory-time) contingency arms if items 1-2 reopen them.
- **Test:** Gate from item 1's parse: if offline v125 - v249 <= max(2 sigma_late, 2% v249) run 1/4 (seeds 42, 43) + 1/2 (seed 42); if only v200 passes, 1/2 only; else 1/2 only with prior 0.2 (the gate is one-directional: mid-LR val carries iterate noise any anneal removes). Whole-schedule compression, never truncation (peak LR, pct_start, div factors kept). Score on item 2's card: >= 2,000 val pairs, per band vs the floor table, rollouts; EMA (0.9995) evaluated at the end free.
- **Where:** Science machine, 7 H200, offline trainer (Polaris possible if the 46 GB m16 store is staged, ~2.1-2.8x slower, user's OK)
- **Cost (node-h):** 21
- **Prediction:** P=0.6 the 1/4-schedule seed spread is <= 2% of final val; P=0.5 the 1/2 arm matches the 250-epoch run within max(2 x seed spread, 2%) on val, every band within 10% of (error_b - A_b), and rollout low-k energy within the seed spread.
- **Gate:** Fixed now: the falsifier set above; EMA is reported, never used for selection without owner sign-off; the 1/4 arms run only if the precursor gate passes. It breaks o1.2's 'differs ONLY in data' control unless any online comparison is compressed identically.
- **Expected gain:** If 1/2 passes: every path halves its step term - online 168 -> ~84 h now, 64-74 -> ~43-48 h after lean32ek, replay 21 -> 10.5 h (-588 H200 GPU-h on today's run). The replication pair alone is worth the spend: it calibrates every other item's falsifier.
- **Changes what is computed:** yes: model/training (step count; EMA = model selection)
- **Decides:** jesswan/Weidong

## Combined projection (R5, verbatim)

Baseline (user's numbers, handoff L70-72): online = 168 h on 7 H200 = 1,176 GPU-h for 195,250 steps; offline on stored frames = 21 h. CHANGELOG L449 records ~142 h / ~120 h solver; item 1 pins which is real - until then every figure is dual-quoted as (D=120 | D=147 solver-hours). MEASURED (A100, Polaris, delta_fidelity.log + notes Parts VII-XI): production solver 9.29 ms/step; lean_u1 6.40 ms BITWISE (1.45x); lean32ek 3.35 ms (2.77x, e_delta 8.8e-6); lean32_u1_ring3 2.52 ms (3.68x, bitwise to lean32); trainer 0.39 s/step; DDP ~5 ms/step. ASSUMED: A100 ratios carry to H200 within +-25% (checked by item 3's Hopper timing); fast-form 2.7 ms (~3.44x) is an UNVERIFIED estimate; owner acceptance at every non-bitwise rung. Ladder, using T = 21 + D/s: (1) Bitwise-only, no sign-off needed: lean_u1 gives T = 21 + 120/1.45 = 104 h | 21 + 147/1.45 = 122 h (from 142|168; -27% wall, -266 to -322 GPU-h/run). MEASURED except H200 carry-over. (2) + lean32ek if item 3's taxonomy is clean and the owners sign item 4's gates: T = 21 + 120/2.77 = 64 h | 74 h; fast form 56 | 64 h. Measured speed + measured e_delta; ASSUMED: owner acceptance, taxonomy transfer from one synthetic state to archive states (that is what item 3 tests). (3) + restart/refill recovery (items 5+7) if item 1 finds reading (a): up to ~26 h/run more, ASSUMED pending the parse. (4) Pivot outcome - if item 2 finds offline-o1.2 parity at the floor (P~0.6; floor table MEASURED and out-of-sample validated: lo 0.00222 / central 0.00377 vs val 0.0043 = 1.93x | 1.14x): generation is paid ONCE per stream via the tee (item 5), and every subsequent run is 21 h; + compile 1.2-1.4x (ASSUMED, gated) -> 15-17.5 h; + 1/2 schedule if item 8 passes (ASSUMED, P=0.5) -> ~8-10.5 h, i.e. ~60-75 GPU-h per run vs 1,176 today (a ~15x reduction for repeat runs, with the one-time generating run at 56-122 h depending on rungs 1-2). Plan cost to find all this out: ~26 node-h total, dominated by item 8's 21 h; items 1-5 (the decisive ones) cost ~4 node-h against the 840-1,029 GPU-h per production run at stake.

## Do not pursue (R5, verbatim)

- **N6 - 2048^2 coarse generator restarting from archived 4096^2 states:** It re-generates trajectories already stored on disk at 0.02 (and 0.0025 for m1-4) - information available at zero solver cost; it beats replay only if 2e-4 density has value, which the verified floor table makes unlikely; and k_max*l_nu = 0.88 drift risk returns on long runs. Sole legitimate role: cheap NEW trajectory time if the o1.1/L11 readout comes back unsaturated - a contingency, not a program.
- **N7 - densify training pairs from the 0.0025 archive:** Density is not the information axis: m16 increment C(0.02) = -0.065, so 0.02-spaced targets are already near-independent and 0.0025 pairs add ~1.0x effective samples; the tee (item 5) gives the EXACT 2e-4 stream for less than this test costs. The dense members' possible extension past t=520 transfers to the L11 contingency.
- **N8 - custom f64 FFT engineering (2-3 weeks + 5 node-h):** Withdrawn by its own proposer: its 3.1-3.5 ms target matches lean32ek's measured 3.35 ms which item 3 may admit for ~0 engineering; with a tee the saving is paid once (27-33 h, once); and the production box's cuFFT callback support is UNVERIFIED. Revisit only if the owners refuse every f32 generator AND multiple f64 stream generations are planned.
- **N10 - sub-cell offset views of the 4096^2 archive:** Expected value < 1 by the proposer's own arithmetic; the premise (online beats offline via draw diversity) is the outcome the floor math bets against; needs an ~11 TB archive pass + 720 GB materialization with unverified I/O. Re-raise iff item 2 finds a diversity gain.
- **L7 - online-ahead-of-offline exploitation designs:** Explicitly conditional on item 2 finding online > offline, which both agents and the verified floor arithmetic predict against. Design stays on file; re-enters above item 8 only if item 2's paired per-band test surprises.
- **L8 standalone / L11 step-B campaign (15.75 node-h) now:** o1.1 ('3500 x 0.5', level with offline, o1.2 docstring L1-2) is an existing production-scale SEG_T=0.5 / reuse-10x data point that item 1 extracts for free; funding offline arms before reading it would buy evidence we may already have. L11 step B runs only if the parse leaves the annealed endpoint open; L8's knob-separation arms fold into item 8's shared baseline.
- **Known dead ends (standing):** Plain float32 state (systematic 3.6e-3 bias at k1-2, verified); STRIDE thinning (confounded overfit report, and it cuts pairs/segment 10x); channels_last (+38.5% step time, measured); fp16/bf16 FFTs; Pallas (no complex in jax 0.4.28); cuFFT LTO callbacks; editing the hash-pinned A-SQG_data.py; loosening any gate after seeing the number.

## Questions for jesswan / Weidong (R5, verbatim)

1. Tolerances before data (item 4): will you sign the per-band acceptance gates (variance < 1% of the alias floor F1_b; bias < 0.1/sqrt(n_b) x the model's own per-band error) BEFORE the taxonomy job is unblinded - and do you want rollout/climate criteria added to them now? Without a signed yardstick the 2.77-3.44x generator lever stays parked regardless of what we measure.
2. Will future runs reuse these 3,281 trajectories (more seeds, ablations, o1.3)? If yes, the 4.30 TB tee makes the solver a one-time cost and dominates every precision/dt proposal for repeat runs - and we need the free-space number on the H200 box's /data to proceed.
3. Is lean32ek acceptable IN PRINCIPLE for generating runs if it passes the signed gates (its delta error is 8.8e-6 overall / 2.2e-5 at k1-2, random-walk-like, vs the f64 floor 2.3e-7 - all verified), or is float64 generation a hard requirement whatever the tests say?
4. May we change the resume convention to last.pt with a finite-and-<=1.5x-best guard (best.pt is your deliberate convention; last.pt already carries identical fields every epoch)? We will bring the measured restart count from the logs, not a hypothetical.
5. Can you point us at (or send) the o1.2 and o1.1 stdout logs and the *_best.pt/*_last.pt files, and confirm whether the 7-day o1.2 run finished its schedule? Every projection divides by what that parse finds (142 vs 168 h).
6. Is GPU 0 of the production H200 box idle during o1.2 (the launcher uses GPUs 1-7)? If so we would host the tee writer, a 2,048-pair per-band validation watcher, and the Hopper timing there at zero marginal hardware - and separately: would you consider switching best.pt selection to the larger val set (a model-selection change), keeping the 200-pair series for continuity?
7. Two pre-registered science options we propose to TEST, not adopt: dt = 2e-4/30 for generating runs (1.33x fewer steps; its per-band bias measured by the dt ladder with a CFL<=0.6 guard; it kills the first-segment bitwise assert) and y_LP alias-free targets (same minimizer iff E[y_alias|x]=0; stage-0 rides free in item 2). Do you want their results on your desk, and at what tolerance?
8. Do the dense 0.0025 members extend past t=520 with the archive's ICs and numerics (files were still being written on 11-02)? If yes, that is up to ~2,300 TU of NEW trajectory time at zero solver cost - the cheapest possible answer if trajectory time turns out unsaturated.

## Dissent (R5, verbatim)

Three overrides, with the strongest counter-arguments on record. (1) vs NUMERICS on dt = 2e-4/30: by NUMERICS' own (saving x p)/cost ~ 60 it is the best ratio on either list, and if the owners refuse every f32 generator it is the ONLY lever beyond lean_u1's 1.45x that keeps float64. I still demoted it to a free rider inside item 3: it changes the data distribution, kills the first-segment bitwise assert (o1.2 L283-287/L336-339), needs per-dt decay tables + variable-step AB3 start + guard-redo semantics, and with a tee its saving is paid once per stream - but NUMERICS' counter (it also multiplies lean32ek, 56 -> 47 h, and its bias lands in bands the floor says are unlearnable anyway) is sound, which is why the dt arms stay in the job at ~0 marginal cost. (2) vs NUMERICS on the pivot criterion: I adopted LEARNING's rule - key the replay-pivot on DIRECT paired per-band o1.2-vs-offline differences, with the floor table used only for the pre-registered predictions P1-P4. NUMERICS' counter is real: val/F = 1.14x central is an absolute statement that needs no finished o1.2 and is already near-decisive, whereas the paired test is hostage to whether o1.2 completed; if item 1 finds o1.2 stopped mid-anneal AND the equal-epoch fallback is ambiguous, the floor ratio becomes the primary evidence after all. (3) vs LEARNING on funding order: I fund item 8's 21 node-h (compressed schedule + the program's first two-seed replication) while deferring L11's 15.75 node-h trajectory-time campaign entirely to the o1.1 natural experiment. LEARNING's counter: o1.1 is confounded (half buffer, ~10x reuse, unknown S and schedule, possibly never annealed), so 'saturated' read off it is weak evidence, and the annealed endpoint at controlled TU is the actual science question - if item 1 shows o1.1 stopped mid-schedule, L11's arms deserve item 8's slot more than the 1/2-schedule arm does. I accept that explicitly: the plan's standing instruction is that item 8's third arm (1/2 schedule) is swapped for L11's 1,750-TU annealed arm if the o1.1 parse comes back mid-schedule or discrepant. One self-correction for the record: my round-3 recomputation of Var(y_alias) (0.0024) contained the slope bookkeeping slip NUMERICS identified (per-mode vs 1D slope in corner_integral); the adopted 0.00222 table is LEARNING's, independently reproduced by NUMERICS and re-verified by me this round from the saved out-of-sample fits.

