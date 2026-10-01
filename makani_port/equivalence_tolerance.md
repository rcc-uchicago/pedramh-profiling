# Equivalence tolerance for the makani port: M4 (inference) and M5 (training)

**Status (2026-10-01):** ruled by the threshold panel, debug job 7697688 (two Opus 5.5 critics and a Fable 5.1
moderator), verdict **accept with changes**. Full ruling, with evidence for each point:
`$MEMBER_ROOT/runs/makani_port/review/tolerance_panel/7697688/ruling.md`. The table below is the ruling. It goes
verbatim into `m4_infer_prereg.md` and `m5_train_prereg.md`, committed before the jobs that measure it, and is not
changed after results are seen (CLAUDE.md #1, handoff §2 rule 5). Five operator decisions are open (§4 below).

## Why a tolerance is needed

The two makani versions evaluate the same model, weights and inputs, but some sums are accumulated in a different
order or at a different precision: the gradient-norm reduction, the no-longer-compiled contraction, the compiled loss
terms. Floating-point addition is not associative, so such a change can move a result by one unit in the last place of
the working precision. Inference runs in bfloat16, whose last place is 0.4 to 0.8 % of the value, so a single changed
rounding appears as a difference of up to about 8e-3 in a standardised field and cannot be told from a code defect by
its size at that point. Each six-hourly prediction is fed back as the next input, and the emulated atmosphere amplifies
such a difference with lead time; after one to two weeks two runs from identical states are different but equally
valid weather realisations, and after a year only their statistics are comparable. A bitwise requirement would fail on
a harmless change in summation order, while a fixed small number such as 1e-4 is below one bfloat16 unit and is
bitwise in disguise. The tolerance is therefore stated in units of the output precision at the first step, as a
measured round-off envelope at later leads, and as reproducible climate statistics for one-year runs, with the bitwise
result as the expected outcome and any non-bitwise result reported for attribution.

Sources of arithmetic difference in makani main (`api_delta.md` §N):

| id | change in makani main | commit | where it acts | status |
|---|---|---|---|---|
| N1 | the dhconv contraction `_contract_lwise` is no longer compiled | `6922a56` | inference and training | measured inert on the 5 original golden checkpoints (7681949, `N1_PRETEST bitwise=5/5`) |
| N2 | gradient-norm arithmetic in `clip_grads` (`_foreach_norm`) | `a14185d` | logged gradient norm; the update only on a clipped step | inert for unclipped steps; clipped steps measured in slot 3 (N2-PRE) |
| N3 | loss terms compiled by default | `b4e9c6a`, `74ba136` | training and validation loss | held off by `compile=False` (c7) |
| N4 | MLP layers built before initialisation, which changes the random draws | `f9b6e787` | fresh initialisation only | avoided: the M5 trace starts from a checkpoint |
| N5 | `DistributedInstanceNorm2d` normalises in fp32 (was bf16) | `18c4582` | spatial model parallelism (h·w > 1) only | M7 |
| N6 | the same commit on single-GPU paths (bias cast, `GeometricInstanceNormS2`, position embedding) | `18c4582` | bf16 runs that use those layers | measured inert for every golden checkpoint |

## Thresholds (panel ruling)

Tier 1 is bitwise and is the expected result. Tier 2 rows apply only when the matching Tier 1 row fails; the job
still prints the first differing lead, channel and magnitude, a bisect is queued, and a non-bitwise result must be
attributed to a source before M8.

ulp(v) = 2^(⌊log2 v⌋−7). z = (phys − μ_golden)/σ_golden, both fields rounded to bf16 before flip counts.
w = equiangular band weights. rel = |new − ref| / |ref|.

| gate | quantity compared | statistic | threshold | units/precision | defect it is designed to catch |
|---|---|---|---|---|---|
| T1-a 1-step + K=56, 10 ckpts (slot 4) | `lead_sha256[k]`, `full_sha256` vs `infer.json`/`infer_ext.json` | equality | bitwise, 56/56 leads | fp32 physical of bf16 z, 0 ulp | everything; PASS branch |
| T1-b flags + dtype (slots 4, 5) | `amp_mode`, both `allow_tf32`, `cudnn.benchmark/deterministic`; pred dtype before the fp32 cast; fed-back state dtype | exact | equal to golden flags; both dtypes `bfloat16` | – | TF32 or fp32-feedback switch that looks like round-off |
| T1-c non-finite (slots 4, 5) | count of non-finite per lead and channel | equality with golden | 0 for K=56; any new non-finite = STOP | – | NaN/Inf |
| T1-d identity (slots 3, 4, 5) | `params_sha256`, `params_struct_sha256`, `state_struct_sha256` of every legacy load; params sha after new-venv save → old-venv load | equality | equal to `infer*.json` / in-memory sha; none of these nor `flags.*` in `--ignore` | – | wrong epoch or run, EMA-for-model swap, weight in wrong same-shape layer |
| T1-e one-year screens (slot 4) | member `.nc`, old vs new venv at the same repo sha, `compare_member_nc.py` | `MEMBER_NC_EQUIV_OK`; A e243 non-finite at lead 595 on PS; nf4p_r1 1460/1460 finite | bitwise | file hand-off at lead 368 (new HDF5 path), long-horizon drift of any kind |
| T1-f val-only loss (slot 5) | `trace_val_r1.json` `valid_loss` | equality | bitwise with `0x1.b3af2p-7` | fp32 hex | loss weights, data order; quadrature defect 70× over |
| T1-g unclipped 20-step trace (slot 5) | per-step `loss`, `train_loss_epoch`, `valid_loss` | equality | bitwise, 20/20 | fp32 hex | any forward, backward, optimizer or data change |
| T1-h metrics (slot 5) | `metrics_ep0000.h5` per-lead per-channel RMSE/ACC | equality vs 7676860 | bitwise | – | scorer change |
| T1-i EMA (slot 5) | shadow-param sha256 after 5 steps on `ema_smoke` config, old vs new venv in-job | equality | bitwise | – | EMA update on new-makani parameters |
| G1 trace grad norm (slot 5) | per-step `grad_norm` vs `trace_train_r1.json` | rel, each step | ≤ 1e-5 | fp32 scalar; N2 reordering ≲ 1e-6 | N2 beyond a reordering; grads not bitwise |
| G2 per-tensor grads (slot 5, ref from slot 3) | 87 per-tensor fp64 L2 norms at step 1, by name | rel each; tensor set identical | ≤ 1e-5 | fp64 of fp32 grads | dropped or zero gradient on a small tensor; wrong param group |
| G3 clipped probe, P0.1 (slot 5) | per-step loss and grad norm vs `clipped_ref.json` | step 1: loss bitwise, gn rel ≤ 1e-5; step k ≥ 2: loss rel ≤ max(3·null_k, 1e-6), gn rel ≤ max(3·null_k, 1e-5); cap 1e-4, above it steps ≥ k report-only | `clipped=20/20` required | fp32; null_k = N2-PRE arm (a) vs (b) at step k | clip not applied, Adam ε/β change |
| G4 `_foreach_norm` complex, P0.2 (slot 3) | per-tensor `_foreach_norm` vs fp64 `vector_norm`; total norm | rel each, and total | ≤ 1e-5 both; n_complex ≥ 1; all real and finite; zero-grad tensors compared absolutely at 1e-12 | fp64 reference | real-part-only norm, NaN on complex |
| G5 CRPS probe, P0.3 (slot 5) | 1-step loss, grad norm vs `crps_ref.json` | rel | loss ≤ 1e-4; gn ≤ 1e-3; params sha report-only | fp32 scalars from bf16 GEMMs of changed shape | wrong member count or CRPS weights |
| I1 Tier 2, 1-step, per channel ×101, 10 ckpts | Δz vs golden | finite mask equal; max\|Δz\|; RMS_w(Δz); \|mean_w Δz\| | max ≤ 4·ulp(max_c\|z_ref\|); RMS_w ≤ 2·ulp(RMS_w z_ref); \|mean_w\| ≤ 1e-4 | bf16 ulps in golden z | bias, dtype path, decoder bias, double normalisation; round-off passes |
| I2 Tier 2, leads 2…L*, per channel per lead | RMS_w(Δz) new vs golden | envelope | ≤ 3 × max over 4 null twins at the same lead; L* = first lead where null-median RMS_w > 0.5 | z; null = old venv, IC flipped 1 bf16 ulp at a seeded random 50 % of points | forcing or history off by one lead; any systematic pre-saturation change |
| I3 Tier 2, leads L*…56 | per-channel per-lead RMSE_w vs truth, both venvs, numpy scorer | report-only | flag channels where new is outside null [min, max] | physical units | late-appearing truth or forcing shift (reported, not gated) |
| I4 Tier 2 screens | A e243: non-finite before lead 1460, on PS, lead reported; nf4p_r1: finite 1460/1460, n>3σ = 0, PS drift@1460 vs −62.9 | outcome + drift | PS within ±5 hPa | hPa, `climate_screen_summary.py` | doubled mass loss, blow-up regression |

Part 0 of `m5_train_prereg.md` (215b4ec4): P0.1 N2-PRE needs `clipped=20/20` and `repeat_bitwise=yes` in both arms;
the measured per-step a-vs-b differences become `n2_null.json`, the null in row G3. P0.2 keeps 1e-5 with an fp64
`vector_norm(g.double())` reference, a printed `n_complex ≥ 1`, and the total norm at ≤ 1e-5.

The comparison scripts this needs (z-space comparator, null twins, NaN-aware npy diff, trace and h5 comparators,
per-tensor gradient reference, `N1_TRAIN_PRETEST`) are listed in ruling §3.

## Open operator decisions (defaults from the ruling)

1. A Tier 2 pass: tag m4-green with a mandatory `PORT_INFER_NONBITWISE_REPORT` and a queued bisect. **Default: tag;
   attribution required before M8.**
2. Spend slot-4 compute on the null twins and in-job old-venv screens. **Default: yes.**
3. nf4p_r1 PS band ±5 hPa; its own spread is unmeasured. **Default: ±5 hPa.**
4. G5 CRPS thresholds have no null. **Default: as tabled; report-only if `crps_in_job` is not bitwise.**
5. A has `weight_decay 0.0`, so M5 cannot see main's weight-decay change. **Default: record the gap.**
