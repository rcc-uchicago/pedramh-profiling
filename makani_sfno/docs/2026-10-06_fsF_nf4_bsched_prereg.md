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

## Amendment 2 (2026-10-07, before submission): resume in place to epoch 63

Operator (2026-10-07), after the f1092 screens of this arm (7725803) and of F-scratch e43 (7725618):
"submit the 23 epoch F with n_futures=4 and let go to 63 epochs on capacity".

**What the screens showed before this amendment.**
- This arm at f1092: all 7 checkpoints are finite with 0 channels past 3σ. dPS at 1460 is
  e1 −72, e8 +50, e14 −4.5, e15 −17, e16 −40, EMA −78 hPa.
- The `fsF_best` gate passes: `fsF_best` = `fsF_e01`, ckpt_epoch 1.
- The single-step F-scratch e43 is M-DEGRADE: raw e42/e43 are non-finite at 1098/1060.

**The resume.** Same `RUN_NUM` (`fsF_anneal_nf4_b16_rtmax100_scratch`) and same tree
(`arm-fsF-nf4-bsched` @ `a1b27d79`). It starts at epoch 17 from `ckpt_mp0_v15.tar`, with `capacity`,
`select=2` (no spare) and 40 h wall. 47 epochs at 44–46 min is about 36 h. Every other variable is
7719538's submit line, except:
- `EPOCHS` 24 → **63**.
- `LOAD_OPTIMIZER/SCHEDULER/COUNTERS/LOSS` 0 → **1**. makani applies the `load_*` flags on the resume
  path too (`deterministic_trainer.py:278-287`). At 0, the resume would restore the weights only and
  restart the optimizer, the warmup and the epoch counter.
- **`OVERRIDE_LR`, `PRETRAINED` and `PRETRAINED_CKPT` are dropped.**
  - The resume ignores them anyway (`:237`, `plasim_trainer.py:430`).
  - The path now holds F e43, not this arm's base. The base survives as F's `best_ckpt_mp0_e23_stable.tar`.
  - If resume detection failed, the run would therefore start from random init, which is visible in
    the loss, instead of silently warm-starting from F e43.
- `CKPT_VERSIONS` 25 → **250**. Rotation is modulo (`:395`), so at 25 the run would overwrite v0 (e1) at
  epoch 26. At 250 every epoch v0..v62 is kept (≈ 2.2 GiB each).

**The schedule is unchanged.** `T_max 100` counts from the end of the 1-epoch warmup, so the LR at
epoch N does not depend on `max_epochs`. Stopping early yields the same trajectory. The LR is about
3.5e-4 at e24 and about 1.3e-4 at e63, so the run is **not annealed** at e63.

**Protected.** The e1 bests are byte-verified copies `best_ckpt_{ema_,}mp0_e1_stable.tar`. The f1156
screen (`next_screens_2026-10-07.sh both1156`) reads those copies, so both starts screen the same weights.

**Read-out.**
- Outcomes S/D (5-yr protocol on e22 = v21 and e24 = v23) become reachable and stay as stated.
- Epochs 25–63 are **descriptive; no outcome is pre-stated for them.**
- Planned: a 1-yr screen at both starts on e24, e32, e40, e48, e56, e63, EMA-best and best. Then the
  5-yr protocol on e63 and on EMA-best. Then fidelity vs B22/B24 on the 99 shared channels.

**Expected failure mode.** Every F run so far has hung in `mpiexec` after training (Exit -29, no OK
token). Completion is keyed on `ckpt_mp0_v62.tar` plus the `Total training time` log line.

## Amendment 3 (2026-10-08, before submission): the 5-yr protocol on e22 and e24, plus e14

The operator, 2026-10-08: "continue with F-depth4". Single-step F-scratch e63 is cancelled, so this arm
is now the F track. Resume 7725884 is running. It had logged e17–e21 by 15:18 UTC (v20 written 14:50,
about 45 min per epoch). Expected write times: e22 = `ckpt_mp0_v21.tar` at about 15:35, e24 = `v23`
at about 17:05.

**One debug job, deferred start 17:45 UTC** (`qsub -a`). That leaves a 40-min margin for `v23` to finish
writing; the run script has no partial-file guard. If `v23` is missing or truncated at start, the members
error and no `CLIMATE_RUN_OK` prints. That is a timing failure, not a result. Arms:
- `FD22` = `ckpt_mp0_v21.tar` (e22) and `FD24` = `ckpt_mp0_v23.tar` (e24). This is read-out step 2 as
  pre-registered. **Outcomes S/D are decided by these two arms only.**
- `FD14` = `ckpt_mp0_v13.tar` (e14), **descriptive**. It is the only checkpoint of this arm within ±10 hPa at
  both 1-yr starts (−4.5 / −2.4 hPa; 7725803, 7725885). It shows whether that 1-yr ranking holds at 5 yr.
  It does not enter S/D.

8 members each, 2044 f1092 + 16i, `DRY_AIR_FIX` off (no flags), the same protocol as B22/B24 (7707597)
and FSEMA (7720705). The job imports `src/` from the `b-continuation-dryair` tree when it starts, so do not
edit that tree's `src/` until it ends. Follow-ups, each on debug: readout, budget (dPS / dDRY in hPa), then
fidelity on the 99 shared channels for every 8/8 arm. No prediction is made.

**Trap found:** the resume reset the raw best-val tracker. `best_ckpt_mp0.tar` was rewritten at 14:05 with
e20 (0.015758). That is the best since the resume, not the run's best (e1, 0.015667). The EMA best is
unchanged. The e1 bests survive as `best_ckpt_{ema_,}mp0_e1_stable.tar`.

**Job-id note (2026-10-08 15:31 UTC, before any rollout):** 7727527 was deleted while still waiting (state
W). That freed the single per-user `debug` queue slot for the F-scratch by-epoch job 7727625 (prereg
`8c8f9048`). The identical arm list is resubmitted by `runs/makani_eval/next_5yr_2026-10-08.sh fd` once
7727625 is running. The arms, files and outcomes above are unchanged.
**Resubmitted (2026-10-08 15:37 UTC):** **debug-scaling 7727731**, the same arms and files, still deferred
to 17:45 UTC. That queue has its own one-queued-job-per-user limit. Operator: "can't you put F-depth4 on scaling".
