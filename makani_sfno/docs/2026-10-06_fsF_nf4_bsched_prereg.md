# Pre-registration — depth-4 fine-tune of soil-free F on B's exact recipe (2026-10-06)

Committed before submission. Operator request (2026-10-06): "finetune F with n_future = 4 once this F
is trained". Operator choices: base = F-scratch, schedule = B's exact (`T_max 100`), queue =
`capacity` after 7718436.

## Why

The 2026-10-03 claim "dropping soil destabilizes the depth-4 recipe" compared **single-step** F
with **depth-4** B, so it is retracted (`makani_sfno/KNOWLEDGE.md` §9a). This arm is the depth-4
test that F3 was not: B's fine-tune recipe applied to a soil-free base.

## The arm

| | this arm | B (`nf4_prod_b16_r1`), the comparator |
|---|---|---|
| base | F-scratch `f_nosoil_2n_b32_e23_scratch/training_checkpoints/best_ckpt_mp0.tar`, raw, as written when 7718436 ends (expected epoch 23, val ≈ 0.0143–0.0144, 99 channels) | A `prod1n_b32_sgdr` best, epoch 243, val 0.01284, 101 channels |
| depth | `MULTISTEP=5` (`n_future` 4) | same |
| LR / schedule | 4e-4, `CosineAnnealingLR`, **`SCHED_TMAX=100`**, 1-epoch warmup from `LR_START` 0.01, min 1e-6 | same (`scheduler_T_max 100`) |
| batch / nodes | global 16 = 2 nodes × local 2 | global 16 |
| epochs / checkpoints kept | 24 / 25 | 24 |
| optimizer / counters / loss state | fresh (`LOAD_*=0`, `OVERRIDE_LR=1`) | fresh |
| corrector flags | none | none |
| code | pinned worktree `arm-fsF-nf4-bsched` @ `a1b27d79`, the tree that ran DRYAIR 7709268 and F4 7709269 | — |

**The confound, stated up front:** the base differs in maturity as well as channel set. F-scratch at
epoch 23 is about A-at-epoch-23 quality, ~11 % worse single-step than A e243. So a worse outcome
than B cannot by itself be attributed to soil.

## Read-out (in this order, all `debug`)

1. **Stage-0 1-yr screen at both starts** (2044 f1092, f1156) on epochs 1, 21, 22, 23, 24 and best,
   with the `_tmq` truth files and `DRY_AIR_FIX` off.
2. **The 5-yr protocol on e22 and e24**, run regardless of the screen (a 1-yr screen cannot predict
   5-yr survival). 8 members, the same Oct-2044 starts as B22/B24 in 7707597.
3. **Climate fidelity** (time-mean pattern RMSE vs `climate_fidelity_7709968/true_climatology_b.npz`)
   on every surviving member, compared with B22/B24 **on the 99 shared channels**.

## Outcomes (pre-stated)

- **S — soil not needed for stability (from this base):** F e22 and F e24 both finish 8/8, like
  B22/B24. Then "dropping soil destabilizes depth-4" is refuted outright.
- **D — worse than B:** either checkpoint below 8/8. Then soil **or** base maturity. Separating the
  two needs the same arm on a mature soil-free base (F-warm e43, val 0.01306). Report the counts;
  do not attribute.
- Survival ties, so accuracy (step 3) decides "better / worse than B". It is a measurement, not a
  verdict: no pre-registered 5-yr accuracy rule exists yet (`KNOWLEDGE.md` §11).

PS drift is reported as dDRY / dPS from the budget readout (`scripts/dryair_budget_readout.py`), in
hPa. Mind the `readout.log` Pa units.

## Amendment (2026-10-07, after training, before any rollout of this arm)

7719538 hit its 12 h wall after **16 of 24 epochs** (≈ 2670 s/epoch at `MULTISTEP=5`; `ckpt_mp0_v0..v15`;
Exit -29). Epochs 21–24 do not exist, so read-out steps 1–2 cannot run as written. Operator (2026-10-07):
"screen e16 now, then decide" on a resume.

What training already shows (logged before this amendment, so not a rollout result). Single-step val is
best at e1 (raw 0.015667, EMA 0.015267, so `best_ckpt_mp0.tar` and `best_ckpt_ema_mp0.tar` are both e1)
and flat at 0.01583–0.01588 after that. Per-lead, median over 99 channels vs the base: 24 h RMSE −8.3 % raw
and −12.6 % EMA at e16, 6 h +5.5 % / +3.9 %.

**Amended step 1 (descriptive, Stage-0 1-yr screen, both starts, `DRY_AIR_FIX=off`, `_tmq` truth):**
`fsF_e01` (v0), `fsF_e08` (v7), `fsF_e14` (v13), `fsF_e15` (v14), `fsF_e16` (v15), `fsF_ema`
(`best_ckpt_ema_mp0.tar`, expect `ckpt_epoch` 1), `fsF_best` (`best_ckpt_mp0.tar`, gate: `ckpt_epoch` 1, equal
to `fsF_e01`).

**Not amended:** outcomes S/D stay defined on e22/e24 at 5 yr. A 1-yr screen of e14–e16 does not decide
them; it only informs whether to resume to e24. The "worse than B" attribution rule still applies.
