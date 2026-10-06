# Prereg: extend F-scratch to epoch 43, plus two descriptive debug jobs (2026-10-06)

Written before submission. Operator, 2026-10-06: continue training FSEMA, no dry-air fix for now, and
"can we just do 2 nodes for F instead of a spare".

## 1. The extension (capacity)

- **Resume in place:** `RUN_NUM=f_nosoil_2n_b32_e23_scratch` with `EPOCHS=43`. Every other variable is
  identical to 7718436's submit line. makani resumes from the newest versioned checkpoint (`ckpt_mp0_v22.tar`,
  epoch 23) with optimizer, scheduler, counters and the EMA shadow/step/best-loss (`plasim_trainer.py:675-778`).
  New epochs are written as v23 and up. Same tree as the e23 run: `makani-ace2-ports` @ `a530bf65`.
- **Nodes:** `-l select=2`, `TARGET_NODES=2`, no spare (operator). A sick GPU now ends the job at
  preflight (`NOT_ENOUGH_HEALTHY_NODES`) instead of falling back to a spare.
- **Walltime:** 7 h, for ~5.2 h of training. Both F runs hung in `mpiexec` after training finished.
- **Protected:** `best_ckpt_ema_mp0_e23_stable.tar` (= FSEMA) and `best_ckpt_mp0_e23_stable.tar`
  (= 7719538's base) are byte-verified copies. The live `best_*` files will be overwritten as
  validation improves.
- **Extension rule** (`submit_f_nosoil.sh`, 2026-09-23): extend by one 20-epoch cycle only if the last
  cycle improved val by > 0.5 %. Cycle 1 went 0.02754 (e3) → 0.01438 (e23), so it is met.

**Read-out:**
- val at e43 vs A's e43;
- 1-yr screen at both starts on raw e43, EMA e43, raw e41/e42 and A e43;
- 5-yr protocol on EMA e43;
- fidelity on the 99 shared channels vs FSEMA e23 (0.234 σ).

**Outcomes:**
- **M-OK:** EMA e43 is 8/8 at 5 yr, median |dPS| ≤ 1 hPa, and fidelity ≤ 0.254 σ. Longer
  single-step training does not cost F its stability.
- **M-DEGRADE:** EMA e43 is < 8/8 at 5 yr, or e43 collapses in the 1-yr screen the way A e22/e23 did.
  Maturity hurts F as it hurt A.
- Anything else: mixed. Report it, no ranking.

**Prediction:** EMA e43 survives the 1-yr screen at both starts, but is < 8/8 at 5 yr. A's
stability fell with training (e21 fine, e22/e23 collapsed, e200+ non-finite).

## 2. Debug, descriptive (no ranking)

- **DRYAIR24 fidelity** (`climate_protocol_7711720`, `ARMS=DRYAIR24`, 101-ch truth reused). What does
  a trained-in fix cost? Compare with B22 0.251 / B24 0.317 / B22+fix 0.256 (median, 101 ch).
  "Cost" means Δ > +0.02 σ vs B22.
- **A's stability by epoch:** 1-yr screen at f1092 of A e43, e63, e83, e103, e123, e143, e163, e183
  (`prod1n_b32_sgdr/ckpt_mp0_v{N-1}.tar`). Known: e21 is finite (14 channels past 3σ), e22/e23 collapse
  finite, e200/e220/e243 go non-finite at 490–664. Report the first epoch that goes non-finite
  within 1 yr. This says which F epochs to watch.
