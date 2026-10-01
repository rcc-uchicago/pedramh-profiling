# M5 pre-registration — training equivalence (Part 0: slot-3 pre-tests)

*Part 0 is committed before the slot-3 gate job (HANDOFF_worker.md §3a, order panel 7682039) and
covers only the **old-venv measurements** that job takes for M5. The full M5 prereg (Parts 1+)
is written after the operator rules on N2 from these numbers, and before the slot-5 job. Nothing
here is loosened afterwards (CLAUDE.md #1, #6, #11). Every Part 0 measurement is valid only if
`PORT_GOLDEN_OK` and `PORT_GOLDEN_EXT_OK` are green **in the same job** (the old venv is unchanged
at the M2 sha).*

## P0.1 N2-PRE — forced clipping, pin arithmetic vs main's (api_delta N2, `a14185d`)

**Why.** The golden trace never clips (`trace_train_r1.json`: `clipped=false` on all 20 steps),
so it cannot see N2. The review's probe at `max_grad_norm 1.0` would not clip either (max grad
norm 0.567) and is dropped (panel fact 3).

**Threshold, and where it comes from.** Read from `makani_port/golden/trace_train_r1.json`, all
20 steps: min grad norm **6.396964e-03** (`0x1.a33b400000000p-8`, step 1), max 5.664676e-01
(step 2). Pre-registered `optimizer_max_grad_norm = 1e-3`, 6.4× below the minimum, so every step
should clip.

**Run.** Golden trace #3 exactly (`m0_golden_prereg.md`: warm start from A e243, 20 steps, batch 8,
bf16, seed 1234, deterministic flags) with only `optimizer_max_grad_norm = 1e-3`, old venv, two arms:

- **(a) pin** — stock `clip_grads` of `c9704308`: per-parameter `sum(square(abs(g)))`, stacked sum,
  `sqrt`; every grad multiplied by `clamp(max/(norm+1e-6), max=1)`.
- **(b) main arithmetic** — the same run with `clip_grads` swapped for a verbatim copy of main's
  (`a0aa4c4f` `training_helpers.py:123-180`, model-parallel size 1 so one group):
  `torch._foreach_norm(grads, 2)` → `stack().pow(2).sum()` → `sqrt`; `_foreach_mul_` only when
  the factor is < 1.

Each arm runs **twice** in-process.

**Tokens.** Per arm: `N2_ARM arm=<a|b> clipped=<k>/20 repeat_bitwise=<yes|no>`. Then
`N2_PRETEST clipped=<min k>/20 bitwise=<yes|no> max_rel=<x> at=<step>:<field>` comparing (a) with
(b) over loss and grad norm (first differing step and field).

**Expected.** `clipped=20/20` in both arms; `repeat_bitwise=yes` in both. (a) vs (b): **predicted
`bitwise=no`**, because the reduction order differs. That is a measurement, not a failure. The
**unclipped** trace stays bitwise in M5 whatever is ruled. `clipped < 20/20` means the threshold
did not bind; the arm is reported and the probe is re-specified, not re-run with a new number.

**Amendment (tolerance panel 7697688, `equivalence_tolerance.md` §2b, row G3).** Both arms
**must** show `clipped=20/20` and `repeat_bitwise=yes`; otherwise P0.1 is red. The per-step a-vs-b
relative differences of loss and grad norm are written as **`n2_null.json`** (`null_k` per step).
That file is the null for M5's clipped probe (row G3): step 1 loss bitwise and gn rel ≤ 1e-5;
step k ≥ 2 loss rel ≤ max(3·null_k, 1e-6) and gn rel ≤ max(3·null_k, 1e-5), capped at 1e-4,
with steps at and past the cap report-only. This is a pre-registered formula with an old-venv
null, not a widening.

## P0.2 `_foreach_norm` on complex gradients (review 7681379 #3)

A's SFNO carries complex parameters (87 tensors, `api_delta.md` §5). One backward pass of A e243
on one batch (bf16 autocast, as the trace), old venv, GPU: compare `torch._foreach_norm(grads, 2)`
per tensor with `torch.linalg.vector_norm(g)` (= `sqrt(sum(|g|^2))`, which is what the pin
computes).

**Token.** `FOREACH_NORM_COMPLEX_OK n=<tensors> n_complex=<c> max_rel=<x> total_rel=<y>` when every
result is real, finite, and within **max_rel ≤ 1e-5** of the reference. Otherwise
`ERROR FOREACH_NORM_COMPLEX <first tensor> <reason>`. That is an **M5 STOP** and a candidate
upstream bug, reported to the operator.

**Amendment (tolerance panel 7697688, §2b, row G4).** The 1e-5 bound stays, with three changes.
(1) The reference is fp64, `torch.linalg.vector_norm(g.double())`. (2) `n_complex ≥ 1` is required
and printed; 0 complex tensors is red, because the check would then prove nothing. (3) The total
norm, `sqrt(sum of squares)` over all tensors, is also compared, at rel ≤ 1e-5. Zero-gradient
tensors are compared absolutely at 1e-12. Recorded limitation: a norm cannot see a conjugated
complex gradient. Row T1-g (the step-2 loss, bitwise) catches that.

## P0.3 CRPS 2-member reference (`nf4_crps_b4_r1`, review 7681379 #4)

Warm start from `nf4_crps_b4_r1/training_checkpoints/best_ckpt_mp0.tar` (model weights only),
its own config (`polaris/e3sm_alldata_crps.yaml`: `ensemble_size 2`, `ensemble_crps`), ensemble
trainer, old venv. Torch and numpy are seeded with 1234 **after** construction, then one
optimizer step on the first 4 training samples of the seeded shuffle. Record the loss and grad
norm (`float.hex`) and the sha256 of the updated parameters. Two processes.

**Token.** `GOLDEN_MATCH crps_in_job` when both processes are bitwise equal. It becomes M5's
CRPS-probe reference (`crps_ref.json`). If the two are not equal, the CRPS path is
non-deterministic on the pin: M5's CRPS probe becomes report-only, by operator ruling, before
Parts 1+.

## P0.4 clipped reference

Arm (a) of P0.1 is also written as `clipped_ref.json`. `GOLDEN_MATCH clipped_in_job` = its two
processes bitwise equal. M5's clipped probe (new venv) compares against this file, under the
P0.1 ruling.

## P0.6 Per-tensor gradient reference and N1 training pre-test (panel 7697688 §3 item 6)

- **`trace_pertensor_ref.json`.** The slot-3 old-venv regression run of golden trace #3 also
  writes, at step 1, each parameter's gradient L2 norm in fp64 (`vector_norm(g.double())`), keyed
  by parameter name (A: 87 tensors). It is valid **only if** that run's trace matches 7676860
  bitwise (`GOLDEN_MATCH train_vs_ref`); otherwise it is not written. It is the reference for M5
  row G2: rel ≤ 1e-5 per tensor, and an identical tensor set.
- **`N1_TRAIN_PRETEST`.** Golden trace #3 rerun on the old venv under `TORCH_COMPILE_DISABLE=1`,
  compared with `golden/trace_train_r1.json`: `N1_TRAIN_PRETEST bitwise=<yes|no> first=<step>:<field>`.
  `bitwise=no` means N1 (main dropped `@torch.compile` from the dhconv contraction) moves training
  on its own. Row T1-g (unclipped trace bitwise) would then be impossible by construction. That is
  a **STOP before Parts 1+**, reported to the operator.

## P0.5 HB-0 — hydrostatic residual on TRAIN truth: **definition pending, not run in slot 3**

The pack's upper-air channels `{T,Z3,RELHUM,…}_l00..l17` are on **terrain-following hybrid
levels**, not pressure surfaces (`convert_e3sm_to_makani_alldata.py:30-45`: Z3_l17 tracks
topography, corr 0.979). `data.json attrs.level_table` records each level's reference-pressure
suffix and a nominal hPa label only. It has **no hybrid coefficients** (hyam/hybm), so the
per-column pressure a hydrostatic residual needs cannot be rebuilt from the pack. Treating the
labels as isobaric, or assuming pure sigma (`p = (lev/1000)·PS`), would each define a different
residual, and that is a science choice. HB-0 therefore runs only after one of these lands as an
amendment here: (i) hyam/hybm/P0 from the E3SM archive, recorded with their source, or
(ii) a written ruling on the approximation. Dry relation only (operator relay, 2026-10-01: no
RH→q formula; that is jesswan's, row 2b). Optional, measurement only, when it runs: a per-level
RELHUM range line on TRAIN truth (min, max, fraction < 0, fraction > 100).
