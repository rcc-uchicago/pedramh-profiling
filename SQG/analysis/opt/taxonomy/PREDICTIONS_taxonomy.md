# Predictions for the generator-error taxonomy job (item 3, `run_taxonomy_polaris.pbs`)

Written 2026-11-06 by the main session, before any taxonomy job was submitted. Do not edit after submission. Scored in
the notes (`polaris_sqg_notes.md`, next Part) after the job; the sealed predictions are scored only after unsealing.

## Why this job

Item 3 of the 2026-11-06 team plan (`../team_2026-11-06/TEAM_REPORT.md`), at the user's request ("do item 3").

- **What the delta job left open (Part XI, `../delta/MATH_delta.md`).** Its numbers come from one synthetic state
  (rms(Δ) 0.825 vs the archive's ≈ 0.27).
  - lean32ek's error looked like near-white noise.
  - lean32's looked like a systematic large-scale bias.
- **This job's question.** On ARCHIVED states, is each faster generator's error in o1.2's 0.02 delta noise (zero
  conditional mean given the input) or bias (learnable)? It answers band by band, with positive controls that show the
  test can see a bias.
- **Sealing (the user's choice, 2026-11-06).** Item 4's acceptance gates have not been signed by jesswan/Weidong. The
  candidates' bias statistics are therefore sealed until they are:
  - which numbers: lean32ek (and its fast form), and dt = 2e-4/30 and 2e-4/32;
  - where: `$MEMBER_ROOT/sqg_profile/taxonomy_<job>/sealed/sealed_taxonomy.json`, with only its sha256 committed.
  - Everything else is reported.
  - Control b's pass/fail leaks about ±1 SE of information on lean32ek's x-coefficient. That is accepted.

## What runs

- **States.** 16 archived A-SQG checkpoints, loaded with o1.2's own recipe (`GD.load_checkpoint` under each member's
  config hash): members 1–4 × t = 100, 200, 300, 400. GPU w takes member ((w + i) mod 4) + 1 at the i-th time, a Latin
  square.
- **Pairs.** From each state, 4 consecutive 0.02 pairs along float64 `lean_u1` (bitwise production). That is 64 pairs,
  clustered by state.
- **Arms** (`taxonomy.py` docstring):

  | which pairs | arms |
  |---|---|
  | all 64 | `lean_u1` (reference), `lean32ek`, `inj` (lean32ek + the planted shift) |
  | first pair of each state (16) | `dquant` (control a), `inj64` (control b's float64 truth), `lean32`, `f64ic32`, `lean32sr` (stochastic rounding) |
  | first state of each GPU (4) | `prod` (production, bitwise check), `lean32ek_fast` |
  | first pair of 2 states per GPU (8) | dt ladder n = 40 (control), 30, 32, 80 |

- **Frames.** o1.2's own `sub` frame (float32 θ[::16, ::16] / 8) and the alias-free spectrum `af` (|k_x|, |k_y| ≤ 127).
- **The planted shift (control b).**
  - Every 4096² mode in the alias-free square gets its damping rate raised by drate_b.
  - drate_b is set from each GPU's first reference pair so that drate_b · 0.02 · rms(x_b)/rms(Δ_b) = 3 × MDE_PRED_b.
    MDE_PRED is the team's R4 power analysis: 8.0e-6, 8.8e-7, 2.6e-7, then 1.2e-7 of Δ from k 8 up.
  - **Why 3× and not the PS's 2×.** The static estimate ignores the flow's effect on the response, which may weaken it
    at high k. The pass criteria below are the PS's.

## The analysis (`analyze_taxonomy.py`; thresholds fixed in its constants)

- **Quantities.** x = F_ref(0) is the learner's input. d = F_ref(L) − F_ref(0) is the reference delta.
  e = F_v(L) − F_ref(L) is the paired error. L = 4,000 steps.
- **Level 1, per band.** Regression e = α x + β d, with real coefficients and cluster-robust SE (16 states).
  - B_x = α · rms(x)/rms(d) and B_d = β, both in units of the band's Δ.
  - MDE = 2.8 SE.
- **Level 2, per shell** (one kx² + ky²).
  - The same regression per shell, with a Wald p-value from F(2, S−1).
  - Per band: the fraction of shells with p < 0.0027, and the debiased learnable bias √(Σ(θᵀAθ − tr VA)/Σ|d|²).
  - Float32's damping error is a per-shell bias with random sign, so level 1 alone would under-read it
    (`test_analyze_taxonomy.py` shows this).
- **The MDE table** comes from lean32ek's own scatter. It is printed before any z-score.

## Validity gates (pre-set; any failure voids the bias conclusions, no post-hoc rescue)

- **V1.** `prod` == `lean_u1` bit for bit, at every lag, on every GPU. Otherwise the reference is not production.
- **V2 (control a, mechanism A).**
  - What it measures: `dquant`'s per-shell x-coefficient against the static prediction −drate_n · 0.02, where drate_n
    is the rate error of the float32-rounded decay table actually used.
  - The test statistic: κ_b, the energy-weighted slope over the band's shells.
  - Gate: κ_b ∈ [0.8, 1.2] in k 1–2, 2–4 and 4–8. Higher bands are reported, because there 0.02 is not short against
    the flow.
- **V3 (control b, power and calibration).** Uses o1.2's `sub` frames and regressors scaled per GPU by its drate.
  - Gate: in every band, `inj` gives z_x ≥ 3 and α_inj/α_inj64 ∈ [0.7, 1.3].
  - The ratios themselves are sealed.
- **V4 (dt arms only).** n = 40 through the ladder's shim must equal `lean_u1` bit for bit (frame and full state) on
  every ladder state. Otherwise the dt arms are discarded.

## Predictions

| # | prediction | P | sealed? |
|---|---|---|---|
| 1 | V2 and V3 both pass | 0.9 | no |
| 2 | **lean32ek has no detectable learnable bias**: every band has \|z_x\| < 3 and \|z_d\| < 3 (sub); ≤ 1% of shells at p < 0.0027 in each band up to k 64; level-2 learnable bias ≤ the band's MDE | 0.8 | **yes** |
| 3 | **lean32 is detected at k ≤ 4**: the learnable bias at k 1–2 is ≥ 1e-4 of Δ (mechanism A predicts ≈ 7e-4), and > 1% of shells have p < 0.0027 in k 2–4 or k 4–8 | 0.85 | no |
| 4 | **Mechanism B** (lean32 − dquant, af frames): NUMERICS predicts no learnable part at k ≤ 16 (≤ 1% of shells flagged, learnable bias ≤ 3 × its noise level); LEARNING predicts there is one. This job decides; no prior is recorded. | — | no |
| 5 | The dt order ratio D(2.5e-6)/D(2e-4/30) per band (sub) lies in [−1.6, −0.55] (−0.64 = third order, AB3; −1.50 = first order, splitting) | 0.6 | **yes** |
| 6 | `lean32ek_fast` is bitwise equal to lean32ek at 40/400/4,000 steps and runs at ≤ 2.9 ms/step | 0.5 | no |
| 7 | Alias split: lean32ek's k 1–2 e_Δ on alias-free frames ≤ 0.3 × on subsampled frames | 0.6 | no |
| 8 | Stochastic rounding (lean32sr): no detectable bias (\|z_x\| < 3 in every band; ≤ 1% shells flagged); k 1–2 e_Δ ∈ [3e-5, 1e-4]; overall e_Δ within 2× of lean32ek's | 0.6 | no |
| 9 | MDE from lean32ek's scatter is within 3× of MDE_PRED in every band | 0.6 | no |
| 10 | The states are archive-like: median rms(Δ) at 4,000 steps ∈ [0.2, 0.35] in o1.2's units (archive 0.27) | 0.85 | no |
| 11 | Overall e_Δ at 4,000 steps: lean32ek ∈ [1e-5, 4e-5] (synthetic 8.8e-6 × magnification 3.9/1.83 ≈ 1.9e-5); lean32 ∈ [5e-4, 3e-3] | 0.6 | no |
| 12 | f64ic32 floor, overall e_Δ ∈ [1e-7, 2e-6] | 0.7 | no |
| 13 | ms/step (median of 4 GPUs): lean_u1 6.40 ± 3%, lean32ek 3.35 ± 3%, lean32sr ≤ 5 | 0.7 | no |

What is NOT decided here:
- **Admission of lean32ek, or of a larger dt, for generation.** That is item 4's owner-signed gates. Gate A: per-band
  error variance < 1% of the alias floor. Gate B: |bias| < g_b · eps_model,b, with eps_model,b from item 2. This job
  supplies the inputs, sealed.

## Harness smoke (before the archive run)

- **How it runs.** `SMOKE=1`: the synthetic spun-up state from job 7819310, 4 copies offset by 0, 250, 500 and 750
  lean_u1 calls, 2 pairs each, dt ladder on each GPU's state.
- **What it checks.** That the harness runs end to end: tokens, bitwise controls, timing, analysis.
- **What it is not.** It is not scored against this file. Its candidate statistics are sealed in the same way, and its
  4 states share one trajectory.

## Limits

- **Sample.** 16 states from 4 members. The k 1–2 band holds only 4 complex modes (2 shells).
- **Paired errors only.** A distributional bias (spectra or PDFs over ≥ 1 time unit) is gate C of item 4, not this
  job.
- **Hardware.** Timings are A100. The H200 carry-over is unmeasured.

## Staging (needed once, by the user)

- **What.** `ckpt_t0{100,200,300,400}.00.npz` of members 1–4, 16 files of ≈ 0.4 GB.
- **From.** Midway `/project/pedramh/weidong/SQG/SQG_deterForcing/AllTrainData/<label>/checkpoint/`.
- **To.** `/eagle/projects/lighthouse-uchicago/members/mehta5/sqg_stage/AllTrainData/<label>/checkpoint/`.
- **The label.** `trainingdata_m0<m>_frozen_ring_NU_B_v2_nu0.0002_B1_mu1_kf3.5_dkf0.5_seed1_N4096_dt5e-06_Tstart0_Tspin20_Tprod1100_save0.0025_pair0.02`.
- **The copy must be byte-for-byte.** The loader checks a config hash, a forcing hash and a qh sha256.

## Addendum (2026-11-06, after the harness smoke 7823653, BEFORE the archive run)

The smoke (`SMOKE=1`) is unscored, as stated above. It exposed three problems in this file's rules. They are fixed
here, before any archive state is run. The text above is left as written. Nothing sealed was opened: the smoke's
sealed files are unread.

1. **V3's z must be signed toward the truth.**
   - The planted shift adds damping, so its coefficient is negative. "z_x ≥ 3" failed in all 8 bands only because
     z ≈ −4.5 to −59.
   - Corrected: z in the direction of α_inj64.
2. **V3's ratio α_inj/α_inj64 contained the candidate's own coefficient.**
   - Why: α_inj = α_planted + α_lean32ek, so the ratio would void the run exactly when lean32ek is biased.
   - What happened: in the smoke, 4 bands failed it while z exceeded the planted-only expectation. At k 2–4, z = 25.5
     against ≈ 10; at k 4–8, 58.7 against ≈ 24.
   - What that hints: an x-aligned component in lean32ek's own error on those 4 near-duplicate synthetic states. That
     is more leakage than the "±1 SE" accepted above. It is not evidence about archive states: the 4 smoke states lie
     on one trajectory, 0.05 apart, so any state-independent error projects onto x.
   - **V3 is now paired, and no candidate quantity enters it:**
     - **recovery:** α(e_inj − e_lean32ek) / α(e_inj64), on the same first pairs, must lie in [0.7, 1.3] in every
       band;
     - **power:** |α(e_inj64)| ≥ 3 × lean32ek's SE (all 64 pairs) in every band.
   - On the smoke this gives recovery 0.987–1.079 and power z 5.3–30.6, a PASS. The ratios are no longer sealed.
3. **Negative-control calibration (reported).**
   - The two arms with no learnable bias by construction (f64ic32, lean32sr) flagged 2.1% and 2.3% of shells on the
     smoke (nominal 0.27%). That is with 4 clusters and near-duplicate states; the 16-cluster synthetic test gave
     0.37%.
   - Reading rule for P2 and P8, fixed now: if the pooled negative-control shell rate on the archive run exceeds 1%,
     a candidate's "≤ 1% of shells" criterion is read as "not above the pooled negative-control rate by more than
     2 binomial SE".
   - Band-level |z| < 3 is unchanged.

**Smoke facts** (synthetic state; for the record, not scoring):
- **Bitwise:** prod == lean_u1, lean32ek_fast == lean32ek (40/400/4,000 steps), dt40 == lean_u1.
- **ms/step:** lean_u1 6.405, lean32ek 3.340, lean32ek_fast 2.694, lean32sr 2.789, lean32 3.190.
- **Control a:** κ = 1.08 / 1.00 / 0.99 at k < 8, falling to 0.93 / 0.45 / 0.17 / 0.10 / −0.08 above.
- **Alias split:** 0.084.
- **Production's own step error,** |D(2.5e-6)|/|Δ|: on o1.2's subsampled frames, 2.3e-4 at k 1–2. On alias-free
  frames it is 9.1e-7 at k 1–2, rising to 1.3e-4 in the corner. The large-scale dt sensitivity of the targets is
  aliased small-scale content.
