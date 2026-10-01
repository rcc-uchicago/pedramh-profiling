# Equivalence tolerance for the makani port: M4 (inference) and M5 (training)

**Status (2026-10-01):** proposed. The operator asked for it to be published. Two Opus critics and a moderator are
reviewing the thresholds in a debug job; the panel's ruling will be recorded here. The ruled values are committed into
`m4_infer_prereg.md` and `m5_train_prereg.md` before the jobs that measure them, and are not changed after results are
seen (CLAUDE.md #1, handoff §2 rule 5).

## Why a tolerance is needed

The port moves our SFNO code from makani `c9704308` to `a0aa4c4f`. The checkpoints, channels, normalisation statistics
and loss definition do not change. The new makani does change how some of the same operations are evaluated
(`api_delta.md` §N):

| id | change in makani main | commit | where it acts | status |
|---|---|---|---|---|
| N1 | the dhconv contraction `_contract_lwise` is no longer compiled | `6922a56` | inference and training | measured inert on the 5 original golden checkpoints (7681949, `N1_PRETEST bitwise=5/5`) |
| N2 | gradient-norm arithmetic in `clip_grads` (`_foreach_norm`) | `a14185d` | logged gradient norm; the update only on a clipped step | inert for unclipped steps; clipped steps measured in slot 3 (N2-PRE) |
| N3 | loss terms compiled by default | `b4e9c6a`, `74ba136` | training and validation loss | held off by `compile=False` (c7) |
| N4 | MLP layers built before initialisation, which changes the random draws | `f9b6e787` | fresh initialisation only | avoided: the M5 trace starts from a checkpoint |
| N5 | `DistributedInstanceNorm2d` normalises in fp32 (was bf16) | `18c4582` | spatial model parallelism (h·w > 1) only | M7 |
| N6 | the same commit on single-GPU paths (bias cast, `GeometricInstanceNormS2`, position embedding) | `18c4582` | bf16 runs that use those layers | measured inert for every golden checkpoint |

Floating-point addition is not associative. A sum evaluated in a different order, or at a different precision, can
differ in the last bits. Under a bitwise requirement any such difference fails the gate, even though the model, its
weights and its inputs are the same. A threshold separates round-off from a real defect.

The model is applied autoregressively: each six-hourly prediction is the input to the next step. The emulated atmosphere
is chaotic, so a round-off difference at the first step grows with lead time. After some days the two runs are
different but equally valid realisations of the weather. Over the 56-step (14-day) rollout, a point-by-point comparison
is meaningful only at short lead times. At longer lead times the two runs are compared by their error against the
verifying truth. One-year runs are compared by their outcome.

The tolerance does not address the grid check in makani main. That check rejects our latitude coordinates when the
configuration is loaded, before any output exists (`grid_declaration.md`).

## Proposed thresholds

Differences are in normalised units: the field minus the channel mean, divided by the channel standard deviation, using
the statistics each checkpoint was trained with. Golden outputs are those of the old venv (7676860, 7681949).

| gate | criterion |
|---|---|
| 1-step inference, all 10 checkpoints | per-channel max \|Δ\| ≤ 1e-4 |
| rollout leads 1–4 (first day) | per-channel max \|Δ\| ≤ 1e-3 |
| rollout leads 5–56 | per-lead global RMSE against truth within 2 % of the golden value; reported, not pointwise |
| one-year screens | same outcome: A e243 truncates within ±30 days of day 595; nf4p_r1 survives the year |
| 20-step training trace | per-step loss relative difference ≤ 1e-4; gradient norm relative difference ≤ 1e-3 |
| checkpoints written by the new venv | load strict in both venvs, with identical keys, shapes and dtypes (no tolerance) |

A result outside a threshold is a STOP: the change is bisected to an upstream commit and reported to the operator. A
threshold is never widened to pass a result.
