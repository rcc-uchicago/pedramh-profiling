# HANDOFF — makani: re-base the stability fine-tune on Port F (the soil-free model)

Written 2026-09-28 by the monitor session, on the operator's first-hand decision. **Scope:** build and
queue the three Stage-1 fine-tune arms (T-anneal, `anneal_dryair`, T-d8) so they start from **Port F's
checkpoint** (105 in / 99 out, soil removed) instead of checkpoint A (107 / 101). Integrate the code they
need, prove it, queue the arms behind F, then screen every epoch. A monitor watches this work and will
audit each gate against its primary. Message it at every commit, `qsub` and result.

Read in this order: this file → CHANGELOG entries of 2026-09-23 (port F decision), 2026-09-24 and after
(the climate driver, the Stage-0 screen, S1a, the isolation incident, dry-air, negativity) →
`polaris_makani_finetune_stability_handoff.md` (on `feat/makani-climate-driver`) §2–§5 →
`polaris_makani_ace2_ports_handoff.md` §6a (port F). Do not re-derive anything in §1.

---

## 0. Why, in one paragraph

A is single-step (`n_future 0`) and goes non-finite in long rollouts: every A checkpoint screened died
within 490–664 leads. A one-epoch `n_future=4` fine-tune of A (checkpoint B) survives a year at two start
dates. That is why the Stage-1 arms exist. Separately, the operator decided on 2026-09-23 to drop
`SOILWATER_10CM` and `TSOI_10CM`, and Port F (job 7660250) is that model: A's recipe, soil removed,
warm-started from A's sliced epoch-243 weights, **also single-step** (`e3sm_alldata_nosoil.yaml:128`
`n_future: 0`). As queued, the Stage-1 arms started from A and kept soil. The two lines would never have
produced a model that is both soil-free and stable. **Operator ruling (2026-09-28, first-hand): "replace
T-anneal, anneal_dryair, T-d8 with F."** The three A-based jobs were cancelled the same day, before any
of them ran.

---

## 1. Facts you inherit (verified 2026-09-28; re-open the primary before relying on one)

| fact | value | primary |
|---|---|---|
| cancelled A-based arms | 7650020 T-anneal, 7650769 `anneal_dryair`, 7650770 T-d8 — all `F` with 0 s walltime (never ran); deleted in reverse order because the chain used `afterany` | `qstat -x` |
| Port F | job **7660250**, `capacity`, `select=3` (2 nodes + 1 spare), 12 h, run `f_nosoil_2n_b32_e43_warm`, **still queued** (capacity at its 32-node cap). Recipe = A's (batch 32, LR 2e-3, β₂ 0.95, clip 32, cosine warm restarts T₀=20, 3-epoch warmup), EMA 0.9995, 43 epochs, pre-registered extension rule (extend one 20-epoch cycle only if the last cycle improved single-step val loss > 0.5 %) | `…/runs/makani_mn_scaling/f_nosoil_2n_b32_e43_warm.warmstart_provenance.txt`; `submit_f_nosoil.sh` header |
| F's run dir (once it runs) | `$MEMBER_ROOT/runs/makani_mn_scaling/e3sm_mn_scaling/f_nosoil_2n_b32_e43_warm/` | launcher `RUN_NUM` |
| soil-free stats are FULL-WIDTH | a soil-free run's `global_means.npy`/`global_stds.npy` are `(1, 101, 1, 1)` while `N_out_channels = 99` and `out_channels` has 99 entries | `surgical_nosoil_7646690/`, `smoke_nosoil_4n_b32_r2/` (npy headers + `config.json`) |
| channel positions | dropped = 101-list indices **8, 9**; `PS` = 0 and `TMQ` = 5 in both lists; `PRECT` = 100 → 98 | both `config.json` `channel_names` |
| the three code lines | **soil-free support** on `worktree-makani-ace2-ports` @ `a530bf65` (`e3sm_alldata_nosoil.yaml`, `channel_subset_gate.py`, `plasim_trainer.py` `_plasim_channel_indices`, subset-aware `rollout_driver.py` / `eval_inference.py` / `score_rollout_nc.py`, `submit_f_nosoil.sh`). **Fine-tune + driver + screen + dry-air** on `feat/makani-dryair-negativity` @ `82a80043` (contains `feat/makani-finetune-stability` and `feat/makani-climate-driver`). Common ancestor `a5b83ae2` | `git log`, `git diff --stat a5b83ae2 <branch>` |
| files changed on BOTH lines | `makani_sfno/polaris/polaris_makani_multinode_scaling.pbs` (ports +104 lines; fine-tune +2 additive `_sched`/`_bools` lines) and `makani_sfno/scripts/eval_inference.py` (ports +18, driver line +19) — expect merge conflicts only there | the two `--stat` lists |
| Stage-1 recipe (unchanged, now applied to F) | all arms: global batch 16, LR 4e-4, `CosineAnnealingLR`, 1-epoch warmup, `LR_START=0.01`, `SCHED_MIN_LR=1e-6`, **`SCHED_TMAX=22`** (S1a: epoch e trains at cosine t = e−2, so 22 puts epoch 24 at the minimum), 24 epochs, `CKPT_VERSIONS=25`, `LOAD_OPTIMIZER/SCHEDULER/COUNTERS/LOSS=0`, `OVERRIDE_LR=1`, a **new `RUN_NUM`** per arm. T-anneal `MULTISTEP=5` 2 nodes × local 2; `anneal_dryair` = T-anneal + `CONSERVE_DRY_AIR=1` (DIAGNOSTIC, jesswan before any default); T-d8 `MULTISTEP=9` 4 nodes × local 1 (probe 7650039: 19.66 GB peak). T-d16 stays blocked (OOM, 7650263) | `submit_finetune_stability_arm.sh`; CHANGELOG 2026-09-24 Stage-1 entry |
| launcher hard-wiring | `submit_finetune_stability_arm.sh:35` `CKPT=…/prod1n_b32_sgdr/…/best_ckpt_mp0.tar`; `:70` `CONFIG_YAML=e3sm_alldata_full.yaml` | the script |

Process facts: no Python/torch/h5py on the login node (operator ruling 2026-09-24; cgroup pid cap 256,
threads count). Tests run as `debug` jobs. Object-only git where possible (`-c core.preloadIndex=false -c
index.threads=1`); working-tree git can wedge on Lustre. **A worktree with a queued or running job is
read-only**: the harness imports `PBS_O_WORKDIR/src` at start and at every preemption restart.
`.claude/worktrees/makani-ace2-ports` is frozen until 7660250 ends. Do all work in a new worktree.

---

## 2. What to build

Create a new worktree and branch, e.g. `git worktree add .claude/worktrees/f-finetune -b
feat/makani-f-finetune feat/makani-dryair-negativity`, then `git merge worktree-makani-ace2-ports`.
Resolve the two conflicting files by keeping **both** sides' additions. None of the ports lines and none
of the `SCHED_TMAX`/`CONSERVE_DRY_AIR` lines may be lost.

1. **Subset-aware stats in the climate driver and screen.** `climate_driver.load_stats_f64` requires
   `len(global_means) == N_out` and raises `STATS_CHANNEL_MISMATCH` on a 99-channel model with 101-wide
   stats. That is loud, not silent, but it blocks every screen of F. Make it select rows **by
   `eval_params.out_channels` (or by name)** when the widths differ, mirroring what the ports branch did
   in `rollout_driver._load_run_norm_stats`. Full-width paths must stay bitwise unchanged. Same for
   `load_time_means_z`, `negativity_rollout.py` and anything else calling `load_stats_f64`.
2. **Name-based stats in the dry-air fix.** `_build_dry_air_fix` passes the 99-name `channel_names`
   together with the 101-wide `global_means`, and `DryAirFix` indexes the stats by the position in the
   99-list. It is correct for F **only because `PS` (0) and `TMQ` (5) precede the dropped indices 8, 9**.
   That is coincidence, not construction. Index the full-width stats through `out_channels` / names, and
   refuse a width that matches neither.
3. **A launcher that starts from F.** Add a sibling of `submit_finetune_stability_arm.sh` (do not edit
   it; it is the record of the cancelled arms), or make `CKPT` and `CONFIG_YAML` env-overridable with
   the old values as defaults. It takes `PRETRAINED_CKPT=<F's selected checkpoint>` and
   `CONFIG_YAML=e3sm_alldata_nosoil.yaml`, uses new tags (e.g. `fsF_anneal_nf4_b16_r1`), and refuses
   to run if the checkpoint's stored channel count is not 99. The launcher must go through the channel
   subset gate the ports branch added (`CHANNEL_SUBSET_GATE_OK 99 of 101`).
4. **Tests** (each shown red on a seeded fault first, as in the driver handoff):
   - stats subset: 99-channel fixture with 101-wide stats → correct rows by name; seeded wrong-row
     selection → red; full-width fixture unchanged;
   - dry-air on a subset: fixture where `PS`/`TMQ` sit **after** a dropped channel → correct; the
     current positional code → red;
   - launcher: a checkpoint with 101 output rows is refused;
   - all existing suites stay green (`CLIMATE_DRIVER_TEST_OK` 25/25, `CLIMATE_SCREEN_TEST_OK`,
     mass-fix tests, `test_channel_subset_gate.py` and the other `makani_sfno/polaris/test_*.py`).

---

## 3. Sequencing — nothing trains before F exists

1. **Wait for F** (7660250). At epochs 23 and 43 apply F's own pre-registered extension rule. Extensions
   are resumes from the ace2-ports worktree (operator decides the queue). F's checkpoint for the arms =
   **the raw `best_ckpt_mp0.tar` at the end of F**, labelled with its epoch. *Default*; the EMA
   checkpoint is the alternative, and that choice is the operator's.
2. **Screen F first** (debug, pre-authorised). One-year rollouts of F's snapshot checkpoints from 2044
   f1092 and f1156 with the Stage-0 screen tool. This is F's stability baseline and the reference the arms
   are compared against. It also answers port F's own question: does the ~500-lead blow-up persist with
   no soil in the state? Report it either way.
3. **Smoke the F-based launcher** (debug): a short `lrcheck`-style run (3 epochs × 20 steps,
   `SCHED_TMAX=1`) from F's checkpoint with the nosoil config. It must print `CHANNEL_SUBSET_GATE_OK`,
   render `N_out_channels: 99` and `scheduler_T_max: 1`, load F's weights (first loss near F's, not near
   initialisation), and give LRs 4e-4 → 1e-6 → 4e-4.
4. **Queue the arms on `preemptable`, one at a time, chained `afterany`**: T-anneal-F → `anneal_dryair`-F →
   T-d8-F. The operator approved Stage 1 on `preemptable` one-at-a-time on 2026-09-24. **Re-confirm
   with the operator before the first `qsub`**, because the arms and their starting point changed.
   Each arm from a **frozen** code tree: its own worktree pinned at the commit it runs, read-only until
   the chain ends.
5. **Screen every epoch of every arm** with the Stage-0 ranking rule. Pre-register it for the
   99-channel set first (a dated addendum), before the first arm's screen `stime`. Validation loss is
   reported beside the rank and never used to select.

---

## 4. Gates — key on the token, never on `rc`

| # | gate | green = |
|---|---|---|
| F0 | merge + all existing suites | every suite's own OK token in one `debug` job; the two conflict files contain both sides' lines |
| F1 | subset-aware stats + dry-air by name | new tests green, each seeded fault shown red |
| F2 | full-width paths unchanged | `CLIMATE_DRIVER_EQUIV_OK` (A, 2048 f1092, K=56, bitwise) and `DRYAIR_OFF_EQUIV_OK` re-run on the merged tree, **tolerance bitwise, stated in a commit before the job** |
| F3 | F baseline screen | `CLIMATE_SCREEN_OK` on F's checkpoints at both starts; recorded in CHANGELOG |
| F4 | F-based launcher smoke | the §3.3 checks, all printed |
| F5 | arms queued | operator's word first; `FINETUNE_ARM_QUEUED`-style lines; chain verified with `qstat -f` (`depend`) |
| F6 | arms screened | per-epoch table, winner by the pre-registered rule, compared against F's baseline and B's |

---

## 5. Do NOT

- ❌ Edit `.claude/worktrees/makani-ace2-ports` while 7660250 is queued or running (it imports that `src/`).
- ❌ Edit `submit_finetune_stability_arm.sh`, `submit_f_nosoil.sh`, `submit_nfuture_ladder.sh` in place; add siblings.
- ❌ Repack the pack or rewrite the stats files to 99 entries. The ports decision keeps them full-width and reversible.
- ❌ Loosen a bitwise gate, `xfail`/skip a test, or compare F-based numbers to A's 101-channel numbers.
  Use the common-99 rebaseline (`prod1n_b32_sgdr_K56/scores/k56_readout_common99.json`) and F's own screen.
- ❌ Make the dry-air fix a default or select a checkpoint from it (jesswan).
- ❌ Submit to `preemptable`/`capacity` without the operator; resubmit a job on `queue_tags` (CLAUDE.md #12).
- ❌ Chain with `afterany` and then delete an upstream job first; delete downstream-first.
- ❌ Run Python/torch/h5py on the login node.

## 6. Open for the operator (state the default, do not block)

1. F's checkpoint for the arms: raw `best_ckpt` (default) or EMA-best.
2. Queue for the arms: `preemptable` one at a time (default, re-confirm), or `capacity` after F.
3. Whether to also keep one A-based T-anneal as the like-for-like comparison with B and C1. *Default:* no;
   the operator chose replacement.

## 7. Definition of done

F0–F6 green on `feat/makani-f-finetune`, pushed; draft PR against `feat/makani-dryair-negativity`; the
F baseline screen and each arm's per-epoch screen in CHANGELOG with job ids and paths; the winning
soil-free fine-tuned checkpoint named by the pre-registered rule, n and start dates stated.
