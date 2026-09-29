# HANDOFF — makani: port G, the 2020–2044 split, the F fine-tune base, and spatial parallelism

Written 2026-09-29 at the end of a long session (named `makani_mulit_jobs`), to continue in a
fresh one. It picks up `polaris_makani_f_finetune_handoff.md` (on branch
`worktree-monitor-ace2`), which is still the plan for the F-based fine-tune arms; this file
records what was built on top of it today and what is left.

**Read in this order:** this file → CHANGELOG entry **2026-09-29 (makani)** on
`feat/makani-f-finetune` (top of the log) → `polaris_makani_f_finetune_handoff.md` §3–§5 →
`makani_sfno/docs/2026-09-29_spatial_cxi_prereg.md` (on `feat/makani-spatial-cxi`) →
`makani_sfno/docs/2026-09-10_fcn3_recipe_vs_ours.md` §4a, §6.

Status keys: ✅ **done** (evidence named) · 🟡 **needs discussion** (an operator/science decision,
default stated) · 🔵 **open** (work to do, no decision needed).

---

## 1. Where everything is

| item | value |
|---|---|
| main branch of this work | `feat/makani-f-finetune`, worktree `.claude/worktrees/f-finetune`, pushed (head after this file) |
| spatial branch | `feat/makani-spatial-cxi` @ `caafdbf7`, **pinned** worktree `.claude/worktrees/spatial-cxi` — **read-only while 7669001 is queued/running** |
| base of both | merge `c51a90be` = `worktree-makani-ace2-ports` @ `a530bf65` + `feat/makani-dryair-negativity` @ `82a80043` |
| G config | `makani_sfno/polaris/e3sm_alldata_ace2vars.yaml` — 83 in / 77 out (76 state + PRECT) |
| train view (2020–2044) | `$MEMBER_ROOT/data/e3sm_makani_alldata_train2020_2044` (per-year symlinks into the production pack + its own stats/metadata) |
| F (soil-free, 99 out) | **job 7660250**, `capacity`, still queued, `select=3` = **2 training nodes + 1 spare**, 43 epochs, 12 h |
| G launcher | `makani_sfno/polaris/submit_g_ace2vars.sh <QUEUE> <NODES> <EPOCHS> <WALL> [scratch\|warm]`, env `DEPEND=`, `SPARE=0\|1`, `DRY_RUN=1` |
| arm launcher (F or G base) | `makani_sfno/polaris/submit_subset_finetune_arm.sh <F\|G> <arm> [rep]`, env `PRETRAINED_CKPT=` (required), `DRY_RUN=1` |
| G smoke | `makani_sfno/polaris/polaris_g_smoke.pbs` (debug, 2 nodes) |
| spatial matrix | `makani_sfno/polaris/polaris_spatial_cxi_matrix.pbs` (debug, 2 nodes) — on the spatial branch |
| unit suites (F0/F1) | `makani_sfno/polaris/polaris_f_finetune_tests.pbs` → `F_FINETUNE_TESTS_OK 6/6` |

## 2. Jobs of this session

| job | queue | what | result |
|---|---|---|---|
| 7668600 | debug 1n | build the 2020–2044 view + stats | ✅ `TRAINVIEW_OK`, 36,500 samples, stats pass 572 s |
| 7668627 | debug | unit suites | qdel'd before running (debug holds one queued job per user); folded into 7668637 |
| 7668637 | debug 2n | G smoke + unit suites | ✅ `G_SMOKE_OK 3/3`; unit suites **5/6** (see §3, T6) |
| **7669001** | debug 2n | **spatial matrix (task #7 phase 1)** + the six unit suites after the arms | 🔵 **RUNNING** at hand-off — read `makani_spatial_cxi.o7669001` in `.claude/worktrees/spatial-cxi/makani_sfno/` |

Nothing was submitted to `capacity` or `preemptable` this session.

## 3. Task status

### ✅ Completed

| # | task | evidence |
|---|---|---|
| T1 | merge the ports line into dryair-negativity (handoff §2 prerequisite) | `c51a90be`, done in the object DB; conflicts only `CHANGELOG.md`, `scripts/eval_inference.py`, both sides kept, every added line checked present |
| T2 | **port G** channel set (operator; jesswan approved, relayed): drop SOILWATER_10CM, TSOI_10CM, Z3 ×18, U10, RHREFHT, PSL, TMQ; keep RELHUM, TREFHT | `07d022dd`; subset gate test; the smoke's trainer line `CHANNEL_SUBSET in=76 out=77 … dropped pack idx [2, 3, 4, 5, 8, 9, 64..81]` |
| T3 | **train split 2020–2044**, valid 2045–47, test 2048–49 (2015–19 = spin-up) | `e7d0175f`; 7668600. Stats moved little: means ≤ 1.4 % σ (`T_l00`), stds ≤ 2.9 % (`RELHUM_l03`) |
| T4 | subset-aware stats in the climate driver / screen / negativity; dry-air stats by `out_channels` (handoff §2.1–2.2) | `ec5ccfb7`; 7668637: climate-driver suite 29 passed, screen + dry-air + negativity 48 passed |
| T5 | launchers: G base (`submit_g_ace2vars.sh`), F/G arms (`submit_subset_finetune_arm.sh`), torch-free base check | `c611badb`, `c3e4988c` (DEPEND), `b7101753` (`SPARE`, capacity nodect ≤ 4); login dry runs against real run dirs (A refused for F, F refused for G, G `anneal_dryair` refused) |
| T6 | G smoke | 7668637: 2-node train on CXI, 380-lead rollout machinery, the arm launcher's own `lrcheck` vars on G (`LR_SCHEDULE_OK`) |
| T7 | the one failing unit test traced | `test_regional_scores`: `lat_weights` is float32-rounded (Σw = 1 − 1.49e-8), `region_weights` renormalises, `rmse_lat_weighted` does not ⇒ gap 1 − √Σw = 7.4505806e-9 = observed. Test fixed at unchanged 1e-12 (`dd99055e` / `caafdbf7`); **re-verification is in 7669001** |
| T8 | analysis/docs: A/F/G sizing table; vs the makani papers; vs FCN3 (paper-stated compute); the 128-node run beside FCN3's 512-A100 stage | CHANGELOG 2026-09-29; `2026-09-10_fcn3_recipe_vs_ours.md` §6 |

### 🟡 Needs discussion (operator / science decisions — default stated, nothing blocks on them yet)

| # | decision | what we know | default / recommendation |
|---|---|---|---|
| D1 | **G's release to `capacity`** — operator: *"G should run after F … don't queue it to capacity yet"* | chain with `DEPEND=7660250` (afterany: G starts when F's current job ends, **before** any F extension resume) | hold until the operator says so |
| D2 | **G's nodes, epochs, walltime** | A = 473 ms/step, 46.3 h for 243 epochs **of a 48 h allocation**. G ≈ 575 s/epoch at 1 node ⇒ ~39 h for 243 epochs (unmeasured). 2-node CXI step time unknown until 7669001's h1w1 arm | 1 node, 243 epochs (A's count; 283 ≈ A's update count), **60–72 h** walltime; revisit 2 nodes if h1w1 < 365.4 ms |
| D3 | **`debug-scaling` for task #7 phase 2** (8 nodes) | held by the ACE2 batch-16 LR sweep, cron-chained 1-h segments (`max_queued 1` per user) — taking the slot delays that sweep | wait for the sweep to finish unless the operator chooses to interleave |
| D4 | **ensemble / probabilistic training** (operator asked: "can we consider A as pretraining or do we need the ensemble function?") | A = makani's shipped deterministic recipe (single-step l2 pretrain → multi-step fine-tune, like the ICML SFNO paper and ACE2); B (1 epoch at n_future 4 from A) already survived a year at two starts; K=56 readout VR ≈ 1.0, CRPS branch not indicated. `PlasimEnsembleTrainer` exists on `wt-perlead-metrics`, never run, not merged, noise only in `perturb` mode. Input noise was ruled out by jesswan (2026-09-10) | **A/F/G count as pretraining; ensemble not needed now.** Revisit only if the deterministic arms fail stability, or if a probabilistic emulator becomes the goal (needs jesswan) |
| D5 | FSNT / FSNTOA stay **prognostic** in G | ACE2 keeps every radiative flux diagnostic; moving them needs a repack (the pack's diagnostic group holds PRECT only) | science owner's call; not part of G |
| D6 | **no dry-air arm on G** | G has no TMQ; `DryAirFix` refuses by name; launcher refuses `anneal_dryair` for G | accept (the fix stays a diagnostic arm on F) |
| D7 | F-arm details from the F handoff §6 | F's checkpoint for the arms (raw `best_ckpt` default vs EMA); queue for the arms (`preemptable` one at a time, re-confirm); keep one A-based T-anneal as a comparison (default no) | defaults as written there |

### 🔵 Open (work, no decision needed)

| # | item | how / gate |
|---|---|---|
| O1 | **read 7669001** | per-arm `SPATIAL_CXI_ARM_OK/_HANG/_FAILED`, `SPATIAL_CXI_MATRIX_DONE`, the P1–P5 table (prereg), then `F_FINETUNE_TESTS_OK 6/6` from the post-matrix unit run. Record in CHANGELOG on **both** branches; score each prediction as written |
| O2 | **F2 — bitwise equivalence of the full-width paths on the merged tree** (F handoff §4) | `CLIMATE_DRIVER_EQUIV_OK` (A, 2048 f1092, K=56) + `DRYAIR_OFF_EQUIV_OK`; state the tolerance (bitwise) **in a commit before the job**; `polaris_climate_equiv.pbs`, `polaris_dryair_equiv.pbs` from `feat/makani-f-finetune` |
| O3 | **F3 — F baseline screen** once F (7660250) has run | `CLIMATE_SCREEN_OK` at 2044 f1092 and f1156; answers whether the ~500-lead blow-up persists without soil |
| O4 | **F4 — F-based `lrcheck` smoke** from F's real checkpoint | `PRETRAINED_CKPT=<F best> bash polaris/submit_subset_finetune_arm.sh F lrcheck` (debug) |
| O5 | F5/F6 — queue and screen the F arms | operator's word first (D7); pre-register the 99-channel screen addendum before the first arm's screen |
| O6 | **G's rollout across the view's train→valid file boundary** is still unexercised | the smoke model went non-finite at step 264, before the step-368 handoff. Covered automatically by G's first real screen; or re-run the smoke rollout from a better checkpoint |
| O7 | **common-77 rebaseline** before any G number is quoted | `restrict_readout.py --drop` the 24 G channels on A's K=56 readout, as was done for F (`k56_readout_common99.json`); note G's split differs from A's too |
| O8 | **task #7 phase 2** (after O1, and D3) | (a) equivalence across layouts: same checkpoint, validation loss per layout — first read how makani loads a `legacy` (per model-parallel-rank) checkpoint into a sharded model; (b) T-d16 memory at **8 nodes × h2w2, batch 16** (0.5 sample-equivalents/GPU — at 4 nodes the split does not reduce per-GPU memory) |
| O9 | draft PRs (F handoff §7) | `feat/makani-f-finetune` → `feat/makani-dryair-negativity`; `feat/makani-spatial-cxi` after O1. Solo session cannot self-approve; leave open, note in CHANGELOG |

## 4. Facts learned this session (so they are not re-derived)

- **`debug` allows ONE queued job per user** (`qsub: would exceed queue generic's per-user
  limit of jobs in 'Q' state`); `debug-scaling` likewise (`max_queued 1`). Fold checks into one
  job rather than queueing several.
- **`capacity` caps `nodect` at 4** (`qstat -Qf`): 4 training nodes + a spare is refused
  (`SPARE=0`).
- **F allocates 3 nodes but trains on 2** (`select=NODES+1`); the spare is charged.
- **The 2-node / 4-node CXI step time at batch 32 is not measured** beyond a 50-step 4-node
  smoke (521.1 ms, 7647798); short smokes overstate (a 20-step 1-node smoke read 1587.6 ms vs
  473 steady). 7669001's h1w1 arm is the first real 2-node number.
- **The `w=4` hang is application-level** (`makani_bench_report.md` §5b), not a transport
  failure; the fabric fix is predicted not to cure it. **Sharding overhead** +41 % to +81 % at
  equal per-GPU work (§5c), including +81 % at 1 node — sharding is a memory tool here.
- **FCN3 (arXiv:2507.12144, quoted):** stage 1 1024 H100 / 78 h / 208,320 steps; stage 2 512
  A100 (Perlmutter) / 15 h / 5,040 steps / 4-step rollout; fine-tune 256 H100 / 8 h. A's
  332,424 steps = 1.60× FCN3 stage 1. The paper reports no efficiency figures.
- **Our 128-node run (7566145)**: 512 A100, batch 512, 8,500 steps, 1.68 h, 863 GPU-h, valid
  0.018297, 10 % of the 1-node per-GPU rate, **over TCP**.
- One measured GPU-efficiency figure exists: **56 % kernel-busy at 1 node** (nsys 7591822).

## 5. Traps (each cost time today)

- **Worktree isolation:** git in this session must target the current worktree with plain
  commands (no `cd …/..`, no computed paths before `git`). Use
  `git -c core.preloadIndex=false -c index.threads=1` for working-tree git; `git grep` needs
  `-c grep.threads=1` when the login node's pid cap is tight (`pids.current` was 172/256).
- **A queued job freezes its worktree** (the harness imports `PBS_O_WORKDIR/src` at start).
  `spatial-cxi` is frozen until 7669001 ends; `f-finetune` is free.
- **No venv Python on the login node** (operator ruling). Syntax checks: `bash -n` and
  `/usr/bin/python3.11` compile-only; tests run in debug jobs.
- **Bash `${VAR:?…}` with an apostrophe in the message** is an unterminated quote.
- The **`polaris/test_*.py` glob** in the unit-suite job picks up any new `test_*.py` added
  while a job that runs it is queued.

## 6. Definition of done for this line

O1–O2 green and recorded; D1/D2 decided and G queued behind F (or explicitly deferred); F's
baseline screen (O3) in CHANGELOG; phase 2 of #7 either run or explicitly parked with D3's
answer; PRs open (O9).
