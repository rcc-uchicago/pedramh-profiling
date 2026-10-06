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
