# G4D after 63 epochs: does more training fix G4D16's short-lead gap, and does it change the pick? — pre-registration (2026-11-02)

Written before any G4D36/43/56/63 forecast exists; no job for this test has been submitted. Operator
2026-11-02: "yes score it" (the K=56 + 5-yr scoring of the 7771394 checkpoints proposed in the 11-02 CHANGELOG
entry), with "Free scored sweeps" for disk.

## Why

- **10-21 (`depth_vs_length_2026-10-21/results.txt`):** G4D16's 6 h RMSE is 1.385× A243's. About 3/4 of the log
  gap is the 5-step loss and 1/3 is A's extra 227 epochs, with the recipe giving back 0.15. The whole 24 h gap
  (1.082) is training length. For A, 227 more epochs bought 14–18 % RMSE at every lead.
- **11-02 (7771394):** G4D was resumed in place from v22 to epoch 63 (code `faf610d2`, unchanged recipe). Its EMA
  val at G4D16's SGDR position fell 0.020900 (e16) → 0.018900 (e36) → 0.018188 (e56). Val is not evidence
  for rollouts: for A it understated the K=56 length gain about 2×, and within a run it does not rank 5-yr
  climate (write-up Result 4).

## Arms (new)

All are EMA weights of `f_nosoil_4n_b32_e23_scratch_fp32_dryair_nf4` (G4D: from scratch, 99 ch, fp32, EMA 0.9995,
5-step loss n_future 4, dry-air constraint trained in, batch 32 on 4 nodes, SGDR T_0 20 / lr 2e-3 / 3 warmup).
Version N holds epoch N+1. SGDR cycles are e4–23, e24–43, e44–63; position t = (e − 4) mod 20.

| arm | checkpoint | t | role |
|---|---|---|---|
| **G4D36** | `ckpt_mp0_v35` | 12 | G4D16's cycle position, +20 epochs |
| **G4D56** | `ckpt_mp0_v55` | 12 | G4D16's cycle position, +40 epochs — **the primary arm** |
| **G4D43** | `ckpt_mp0_v42` | 19 | cycle end, pairs with G4D23 |
| **G4D63** | `ckpt_mp0_v62` | 19 | cycle end, run minimum of EMA val |

References, all already scored on eval-protocol `5848643d` (A243 rolled at `65f5e40` and shown bitwise
identical at `5848643d` on 10-21, `A_REPRO_BITWISE`): **G4D16** (K=56 + 5-yr), **G4D23** (5-yr), **A243**,
**A16** (K=56).

## Gates (before any number is read)

- **Checkpoint pins** (`polaris_g4d63_ckpt_prep.pbs` → `G4D63_CKPT_OK`): v(E−1) must rebuild the trainer's
  `best_ckpt_ema_mp0.tar` bitwise (E = that file's own epoch, expected 63). Each of v35/v42/v55/v62 must
  report epoch 36/43/56/63.
- **Code:** eval-protocol HEAD `5848643d`. No file under `src/`, `polaris/` or `scripts/` was modified after
  2026-10-21 21:00Z (checked 11-02). This is the tree that rolled G4D16's K=56 and 5-yr runs.
- **Drift census** uses the stride-1 truth (`window_truth/truth_annual_gm_2015_2047_stride1.npz`), as tag `g4d` did.

## Runs

- **K=56 monthly sweep** (`polaris_eval_inference.pbs`, the 24 test ICs, K = 56) of each arm's extracted EMA
  weights → `polaris_score_k56.pbs`.
- **5-yr protocol** (`polaris_climate_run.pbs`, 8 members, `WEIGHTS_<arm>=ema`, rollout defaults → fp32 +
  trained-in dry air, exactly as G4D15–23 were rolled) → `polaris_protocol_score.pbs`, tag **`g4d63`**.
- **Disk:** after each K=56 score passes, its `inference/nwp/*.nc` are deleted (operator's word) and logged to
  `freed_2026-11-02.txt`; `scores/` is kept. The 5-yr NetCDFs (~52 GB) are kept.

## Reported (`compare_g4d_e63.py` → `results.txt`, token `G4D_E63_OK`)

**Part 1, K=56**, on the channels common to all arms (99), at leads 6, 12, 24, 36, 48, 72, 126, 168, 336 h:
- the median over channels of RMSE ratios G4D36/G4D16, G4D56/G4D16, G4D43/G4D16 and G4D63/G4D16 (length within
  G4D);
- G4D16/A243, G4D56/A243 and G4D63/A243 (gap to A);
- A16/A243 for context;
- channels where the numerator is worse;
- 14-day median NRMSE and ACC for every arm;
- **gap closed** = channel-mean ln(G4D16/G4D56) / channel-mean ln(G4D16/A243), printed only at leads where
  G4D16's median gap exceeds 1.02 (where the gap crosses 1 the share is meaningless, 10-21).

**Part 2, 5-yr protocol:**
- for each new arm: health, gates 1–3, A, W, drifting groups and median |dPS|;
- paired dA (by start date, healthy in both) for G4D36 − G4D16, G4D56 − G4D16, G4D43 − G4D23 and
  G4D63 − G4D23;
- the parent §5 rule re-run on the pool fp32 + f32a + g4e + g4fix + g4d + **g4d63**, with current pick G4D16.

## Read-out rules (fixed now)

1. **Does training longer help at matched cycle position?** At 6 h and 24 h separately, take G4D56/G4D16:
   - ≤ 0.97 **and** better on ≥ 66/99 channels → **HELPS**;
   - ≥ 1.03 **and** worse on ≥ 66/99 → **HURTS**;
   - otherwise **FLAT**.
2. **How much of G4D16's 6 h gap to A243 does it close?** gap-closed share at 6 h:
   - ≥ 1 → **CLOSED**;
   - ≥ 0.5 → **MOSTLY**;
   - ≥ 0.25 → **PARTLY**;
   - < 0.25 → **NO**.
   The same rule is reported at 12 h and 24 h.
3. **Pick:** the parent §5 rule, unchanged. The leader is the gate 1–3 passer with the lowest A. The tie set is
   every passer not worse than the leader by more than 3.3 paired SE. Tie-break: fewest drifting groups, then
   lowest W. A new arm replaces G4D16 only through this rule. Part 1 does not override it (K=56 is
   descriptive, as on 10-19).

## Predictions (before any data)

- **Rule 1, 6 h:** G4D56/G4D16 ≈ **0.90** (0.85–0.95) → **HELPS**.
  - Basis: EMA val −13.0 % in squared loss = −6.7 % RMSE; A's val understated its K=56 length gain about 2×.
  - G4D36/G4D16 ≈ 0.94.
- **Rule 1, 24 h:** ≈ 0.90 → HELPS.
- **Rule 2, 6 h:** ≈ 0.3 → **PARTLY**. The depth share (≈ 3/4 of the log gap) is not a length effect, so
  G4D56/A243 at 6 h stays ≈ 1.25 (1.18–1.32).
- **Rule 2, 24 h:** **CLOSED** (G4D56/A243 ≈ 0.97).
- **5-yr** (low confidence):
  - all four arms pass gates 1–3. Every G4D at e15–23 did, including late-cycle e20–23. Depth-4 removed the
    late-cycle collapse.
  - G4D36/G4D56 within G4D16's tie set (|dA| < 3.3 SE), so the pick does not change on A alone.
  - No prediction for drifting groups.

## Caveats

- One training seed.
- The K=56 scorer stores only IC mean/std, so there is no IC-paired interval for Part 1. Channel counts are
  indicative only (channels are correlated).
- The 5-yr numbers are on the 2045–2047 selection window, and many arms have now been scored on it (winner's
  curse). The sealed 2048–2049 window stays closed.
- G4D43/G4D63 vs G4D23 compare cycle ends. G4D36/G4D56 vs G4D16 compare matched mid-cycle (t = 12).

## Cost

All on debug / debug-scaling, 1 node each (pre-authorized):
- checkpoint prep: CPU, ≤ 30 min;
- 5-yr run (4 arms): ~30 min;
- protocol score: ~5 min;
- 4 K=56 sweeps: 40–47 min each;
- 4 K=56 scores: ~25 min each.

Total ≈ 5.3 node-hours.

Disk: peak ≈ 52 GB (5-yr) + 2.4 GB (EMA tars) + 2 × 58 GB (two sweeps in flight) ≈ 170 GB. That is against
≈ 278 GB free under the 50 TB soft quota after the 157.6 GB freed today.
