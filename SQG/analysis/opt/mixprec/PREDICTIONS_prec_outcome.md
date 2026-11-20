# Predictions: does training the network in bf16 change the trained model's outcome, against ieee?

Written 2026-11-20 by the main session, before the job was submitted. Do not edit after submission. Scored in the notes
(`polaris_sqg_notes.md`, Part XVII L) after the job.

## Why

Mixed-precision probe 2 (job 7854382, notes Part XVII K) timed o1.2's step with the network in three precisions:

| network | s / opt step | vs ieee | peak GPU memory |
|---|---|---|---|
| ieee (o1.2's strict FP32) | 0.8888 | — | 20.9 GiB |
| tf32 | 0.8088 | −9.0% | 21.0 GiB |
| bf16 autocast | 0.7882 | −11.3% | 12.2 GiB |

The ML step alone shrinks ~23% under bf16. o1.2 pins strict FP32 on purpose, so the write-up concluded that adopting
either needs a training-outcome comparison, not the solver gates. The user (2026-11-20): "build and test a Cheapest: a
training-outcome comparison of bf16 (or tf32) against ieee. bf16 is the better candidate: −11% time and 42% less
memory. on the preemptible queue".

## What runs (`run_prec_outcome_polaris.pbs`, one Polaris node on `preemptable`, 4 A100s)

**The harness is gate D's dose-0 training** (`gateD/train_dose.py`, `rollout_dose.py`, PREDICTIONS_gateD.md):
- o1.2's network, loss, optimiser and OneCycle;
- 195,250 steps (o1.2's count), batch 16, one A100 per arm;
- trained offline on the stored production frames of members 1–3 (12 segments, 58,812 pairs);
- validated on 512 fixed held-out member-4 pairs;
- rolled out for 2,500 steps (50 TU) from 16 held-out member-4 frames, with best.pt and with last.pt.

**The ieee side already exists and is not retrained.** Gate D trained two dose-0 arms in each of its three jobs, six
in all:

| job | run | seed 1 (a0_sA) | seed 2 (a0_sB) |
|---|---|---|---|
| 7835786 | gateD_v1 | 0.005540 | 0.005592 |
| 7843532 | gateD_tr_v1 | 0.005534 | 0.005620 |
| 7854209 | gateD_inrow_v1 | 0.005612 | 0.005688 |

These are best val_clean values. All six share the same code: `train_dose.py`, `rollout_dose.py` and `model_UViT.py`
are unchanged since 22b8617c, and the later edits to `dose_common.py` / `analyze_dose.py` are dose plumbing that
dose 0 does not use. They also share the same frames, and identical configs: segments, pair counts, n_val, and
persistence 0.08149451017379761 bit for bit. Every arm keeps 16/16 rollouts alive for 50 TU.

**The candidate side: four bf16 arms**, seeds 1, 1, 2, 2 (`bf16_sA_r1`, `bf16_sA_r2`, `bf16_sB_r1`, `bf16_sB_r2`).
- `train_dose.py --prec bf16` is `O12_PREC=bf16` exactly:
  - strict FP32 flags stay on;
  - the training forward and loss run under `torch.autocast("cuda", dtype=torch.bfloat16)`;
  - weights, AdamW state and gradients stay float32. bf16 has float32's exponent range, so there is no loss scaling;
  - validation (and so best.pt's choice) runs in float32, as in o1.2's patched loop.
- **Every rollout is strict-FP32 inference.** Only the training precision differs, so this compares training outcomes,
  not bf16 inference.
- The same seed gives the same initial weights and batch sequence as the ieee arm of that seed. Same-seed training
  is not bitwise reproducible (Part XVII H), so the 3 ieee realizations per seed already differ.
- `--prec` defaults to ieee, and that path keeps gate D's ops and RNG order. Additions: one first-step dtype census
  through forward hooks, which draw no random numbers, and a peak-memory query per val row.

**Proof that bf16 is on** (probe 2's M2 check could not see it through `x + net(x)`). On the first step, every Conv /
Linear output dtype is counted (`PREC_FORWARD`). bf16 must have some bfloat16 outputs and ieee none, or the arm stops.
The count is stored in config.json (`forward_census`), and the analysis requires it.

**Stage 0 self-test** (`test_prec_outcome.py --gpu`; the debug slot is held by another job, so there is no separate
smoke):
- 20-step ieee and bf16 runs on the real frames, with the census and peak memory (bf16 < ieee);
- a resume, and a refused precision-mismatch resume;
- a 5-step strict-FP32 rollout of the bf16 checkpoint.

## Decision rule (fixed now; `analyze_prec.py`)

**Validity (else INVALID).**
- All 10 arms finished 195,250 steps at dose 0.
- The ref arms are ieee and the candidates bf16, with a non-zero bf16 census.
- All arms share segments, pair count, val pairs and rollout initial conditions.
- Each seed has arms of both kinds.

**M1 = best val_clean** (the val of best.pt; decides). M2 (final val_clean) and the final train loss are reported.
- Δ = mean(bf16) − mean(ieee). σ_ref = SD of the six ieee arms (ddof 1): ieee's own retraining spread.
- p = exact permutation p-value of |Δ|. Labels are shuffled within each seed: 10 × 10 = 100 splits, the observed one
  included.

**Rollout family** (gate D's quantities and windows):
- R(w) = E(rollout)/E(climatology) − 1 for E_low and each of the 8 bands, per 250-step window, over the usable time.
- Cell statistic |mean_bf16 − mean_ieee| / s_c, with s_c the SD over all 10 arms.
- T = max over cells; p_FW = its exact stratified permutation p-value (family-wise).

**Verdict** (best.pt; last.pt reported), first match:

| verdict | condition |
|---|---|
| INVALID | a validity check fails |
| UNSTABLE | a bf16 arm blows up more initial conditions than the worst ieee arm (0) |
| WORSE | p(M1) < 0.05 and Δ > 0 |
| ROLLOUT_DIFFERS | p_FW < 0.05 |
| BETTER | p(M1) < 0.05 and Δ < 0 |
| **PASS_P** | \|Δ(M1)\| ≤ σ_ref |
| INCONCLUSIVE | otherwise: an effect above ieee's spread that the test cannot confirm |

**What PASS_P means.** bf16 training reaches ieee's validation loss within one ieee retraining SD (~1%). Its
strict-FP32 rollouts are statistically indistinguishable from ieee's, at gate D's training scale. Adopting bf16 for
o1.2 stays the user's call.

## Power (`power_prec.py`, 2,000 normal draws per row; σ = σ_ref ≈ 1.0% of val)

| true shift | WORSE | PASS_P | INCONCLUSIVE | BETTER |
|---|---|---|---|---|
| 0 | 0.015 | 0.821 | 0.143 | 0.020 |
| 1 σ (~1%) | 0.234 | 0.471 | 0.293 | 0.001 |
| 2 σ (~2%) | 0.698 | 0.068 | 0.234 | 0 |
| 3 σ (~3%) | 0.959 | 0.004 | 0.037 | 0 |

The test sees shifts of ≥ 2–3% in val MSE. It cannot resolve shifts below ~1%.

## ieee-only harness check (done before submission; no bf16 data exist)

`analyze_prec.py --prec ieee`: each job's ieee pair as the "candidate" against the other four (9 splits, so p ≥
0.11):

| candidate pair | Δ M1 | z | p | p_FW | verdict |
|---|---|---|---|---|---|
| 7835786 | −0.85% | −0.76 | 0.33 | 0.11 | PASS_P |
| 7843532 | −0.55% | −0.50 | 0.44 | 0.56 | PASS_P |
| 7854209 | +1.41% | +1.88 | 0.11 | 0.22 | INCONCLUSIVE |

Job 7854209's pair is the highest at both seeds. Nothing in the software differs between these jobs. That is chance
(1 in 3), or a per-job effect this design cannot separate from bf16, since all four bf16 arms share one job.

## Predictions

| # | prediction | P |
|---|---|---|
| 1 | Stage 0 passes first time (PREC_TEST_OK; in the self-test, bf16 peak memory ≤ 0.7 × ieee's) | 0.75 |
| 2 | Every Conv / Linear output of the training forward is bfloat16 (census bf16 = conv_linear_modules) in all 4 arms | 0.85 |
| 3 | All 4 bf16 arms finish 195,250 steps with finite losses | 0.9 |
| 4 | bf16 median s_step / ieee's (0.42 s) ∈ [0.70, 0.85] (probe 2: ML step −23%) | 0.6 |
| 5 | bf16 peak memory (val rows, max over arms) ≤ 0.65 × 20.9 GiB = 13.6 GiB | 0.6 |
| 6 | **Verdict PASS_P** (best.pt) | 0.7 |
| 7 | \|Δ M1\| ≤ 1% of the ieee mean | 0.7 |
| 8 | All 4 bf16 best.pt rollouts keep 16/16 initial conditions for 50 TU | 0.85 |
| 9 | p_FW ≥ 0.05 for best.pt and for last.pt | 0.85 |
| 10 | No bf16 arm overfits (final / min val_clean < 1.10) | 0.95 |
| 11 | Median one-step prediction RMS bf16–ieee within ±10% of ieee–ieee's (≈ 4.5e-2) | 0.8 |
| 12 | The bf16 arms' SD of M1 is 0.5–2× σ_ref | 0.7 |

Prior for 6 and 7: the output is x + net(x), with net(x)'s RMS ≈ 0.28 (persistence MSE 0.081). net(x)'s last conv
returns bf16 (unit roundoff u = 2⁻⁸), so its rounding RMS is ≈ u/√3 × 0.28 ≈ 6e-4. That adds ~4e-7 to an MSE of
5.6e-3 (< 0.01%). Gradient rounding sits below minibatch noise. So a real effect, if any, would come from the
optimisation path, not from output rounding.

## Limits

- **One job for all four bf16 arms.** A per-job effect (above) is confounded with bf16.
- **Training scale.** As gate D: one A100 per arm, batch 16, 12 TU of data; about 1/14 of o1.2's samples. o1.2's
  online FIFO, DDP across 4 GPUs and bf16 inference are not tested.
- **Model selection.** best.pt is chosen on fp32 val, as o1.2 would.
- **tf32** is not run. The same job runs it with `-v PREC_CAND=tf32,RUN_NAME=prec_tf32_v1`; the rule is unchanged.

## Cost and scoring

- **Cost:** one node, `preemptable`, walltime 30 h. Expected ~18–23 h: training at ~0.33–0.42 s/step, plus ~15 min of
  rollouts and a few minutes of tests and analysis. A preemption rerun resumes each arm from its last 2,500-step
  checkpoint.
- **Token:** `PREC_JOB_DONE failures=0`. Results go to `SQG/analysis/results/polaris_prec_<job>/{best,last}/prec_summary.json`
  and logs.
- **Re-score** (login node, numpy only; `SP=/eagle/projects/lighthouse-uchicago/members/mehta5/sqg_profile`):
  ```
  python3 SQG/analysis/opt/mixprec/analyze_prec.py --prec bf16 --which best --frames $SP/gateD_frames --out <dir> \
      --ref $SP/gateD_v1/a0_s{A,B} $SP/gateD_tr_v1/a0_s{A,B} $SP/gateD_inrow_v1/a0_s{A,B} \
      --cand $SP/prec_bf16_v1/bf16_s{A,B}_r{1,2}
  ```
