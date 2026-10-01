# HANDOFF (worker) — port our makani fork to upstream makani `main`, keeping every checkpoint

Written 2026-10-01 from the f-finetune session. Operator request: *"create a handoff to port to the
latest branch of makani, we should be able to keep the same checkpoints … small commits to make sure
everything works in a milestone, and then if needed if something breaks revert to previous green
commit."* A **monitor** session audits you (`HANDOFF_monitor.md`) and messages you. It never commits
to your branch.

**Every session, before anything else:** read `makani_port/_papercuts.md` (mistakes not to repeat),
then the newest 3 entries of `makani_port/PROGRESS.md`. **Before every `qsub`:** re-read
`_papercuts.md`. **Before stopping:** add a dated `PROGRESS.md` entry and any new papercut.

---

## 0. The job in one paragraph

Our jobs import **makani 0.2.0 @ `c9704308`** (2026-04-23, pip-installed from git into
`$MEMBER_ROOT/conda-envs/sfno-venv`). Upstream `main` is **`a0aa4c4f`** (2026-09-30), **184 commits /
209 files ahead**, still labelled `__version__ = "0.2.0"`. Port `makani_sfno/src`, `scripts/` and
`polaris/` so they run on `a0aa4c4f`, in a **new venv beside the old one**, such that **every existing
checkpoint (A, B, C1, the proxies, the surgical soil-free one, F when it exists) loads and produces
the same outputs**. "Same" is **bitwise** unless a milestone's pre-registered tolerance says
otherwise *and* the operator has signed off. The old venv stays the default until the operator
switches (M8). Science is unchanged: nothing here may change what a model computes (CLAUDE.md #1).

## 1. Facts you inherit (verified 2026-10-01 — re-open the primary before relying on one)

| fact | value | primary |
|---|---|---|
| installed makani | `makani-0.2.0`, `direct_url` commit `c97043086e60d44a3adc3bede9a6b3dc71f5005d` | `sfno-venv/.../makani-0.2.0.dist-info/direct_url.json` |
| target | `a0aa4c4f` (main head 2026-09-30); pin the **full sha** you build from and record it | `gh api repos/NVIDIA/makani/compare/c9704308...main` |
| other venv pins | torch 2.8.0 (from base conda, `--system-site-packages`), torch-harmonics `0.9.2.dev75+g2edb24ed`, nvidia-physicsnemo `2.2.0a0` **editable** from `${REPO}/physicsnemo_sfno`, zarr 2.18.7, DALI cuda120 2.2.0, h5py from `makani-h5py-overlay` | dist-info in the venv; `polaris_setup_sfno_venv.sh` |
| upstream main's requirements | `nvidia-physicsnemo>=1.3.0` (1.x and 2.x normalised in `makani.models.physicsnemo_compat`), **`zarr>=3`**, `torch-harmonics>=0.9.0`, `torch>=2.4`, DALI ≥ 2.0 (dev extra) | `pyproject.toml` @ main |
| what we import from makani | `models.model_registry`, `models.preprocessor.Preprocessor2D`, `models.stepper.{MultiStepWrapper,SingleStepWrapper}`, `mpu.helpers.sync_params`, `utils.{argument_parser,comm,logging_utils,profiling}`, `utils.YParams.{YParams,ParamsBase}`, `utils.dataloader.init_distributed_io`, `utils.dataloaders.{data_helpers,data_loader_multifiles}` (+`MultifilesDataset`, `get_climatology`, `get_data_normalization`), `utils.driver.Driver`, `utils.loss.get_time_diff_stds`, `utils.metric.MetricsHandler`, `utils.parse_dataset_metada.parse_dataset_metadata`, `utils.profiling.Timer`, `utils.training.{deterministic_trainer.Trainer, ensemble_trainer.EnsembleTrainer}` | `from makani` lines under `makani_sfno/{src,scripts,polaris}` |
| what we patch / subclass | `sfno_training/compat.py` (py3.12 shim for `get_timedelta_from_timestamp`, installed at import); `PlasimTrainer(Trainer)` overrides (logger-on-every-rank fix, metric-handle rebuild for E3SM names, `_plasim_channel_indices`, EMA, per-lead metric save, channel subset), `_stock_sync_params` wrapper; `MetricsHandler` rebuilt | `src/sfno_training/{compat.py,trainer/plasim_trainer.py}` |
| upstream changes that touch us (from the commit list) | `sfnonet.py` +416/−44 (incl. `f9b6e787` "retiring fin/fout matmul for plain fork-join matmul" — **numerics risk**), `preprocessor.py` +594/−37, `stepper.py` +201, `model_package.py` +266 (`fa93cc70` "model package rework"), `8cf889ec` save/restore **pname checksum**, `4693db43` dataloader save-restore, dataloader stack rewritten (`93113887` pluggable backends, `1b6e0084` metadata rework, `798245b1` "require a dataset to declare its grid"), `fbe141fb` warmup start 1e-5, `3fe70fb9` weight-decay param groups, AMP/compile/fp8 (`18c4582c`, `0b884515`), `c673c2c8` multistep checkpointing, **`4c40a0e0` "Guard spectral weight splitting on spatial_distributed"** and **`4af60539` "removing deadlocked collective"** (both relevant to our `w=4` problem, O8c) | `makani_port/api_delta.md` (you write it in M0) |
| checkpoints to keep | A `prod1n_b32_sgdr` (`best_ckpt_mp0.tar` = e243, `ckpt_mp0_v<i>` = epoch i+1), B `nf4_prod_b16_r1` (`ckpt_mp0_v1.tar` = e22 — rotating slot, **copy, never resume**), C1 `c1_rollout_full_b16`, proxies `nf{1,3,4}_proxy_b8_r*`, `nf4_crps_b4_r1`, `surgical_nosoil_7646690` (99 ch), F `f_nosoil_2n_b32_e43_warm` (when it exists; EMA on), `fsurg_nf4_proxy_b8_r1` (job 7671977). EMA files (`best_ckpt_ema_mp0.tar`) exist only for EMA runs — not A, B or C1 | `$MEMBER_ROOT/runs/makani_mn_scaling/e3sm_mn_scaling/`; `polaris/climate_screen_stage0.list` |
| all checkpoints are | **legacy** format, model-parallel size 1, saved by makani `c9704308` (no pname checksum) | `driver.py:515` at the pin |
| equivalence tools that already exist | `polaris_f2_equiv.pbs` (climate driver vs `rollout_one_ic`, bitwise; dry-air-off vs pre-fix tree), `polaris_climate_screen.pbs` (one-year screens), `polaris_f_finetune_tests.pbs` (six suites), `polaris_makani_multinode_scaling.pbs` (training, `SKIP_TRAIN=1` validate-only), `compare_member_nc.py` (bitwise NetCDF) | `makani_sfno/polaris/` |

## 2. Hard rules for this port

1. **Never modify `sfno-venv`** (pip install/upgrade/uninstall into it). F (7660250), every screen and
   every arm import it. Build `$MEMBER_ROOT/conda-envs/sfno-venv-main` as a **sibling**, from a sibling
   script `polaris_setup_sfno_venv_main.sh` (do not edit `polaris_setup_sfno_venv.sh`).
2. **Never edit `physicsnemo_sfno/`** (editable install shared by both venvs — CLAUDE.md coupling).
   In the new venv install it **non-editable** from a recorded commit, so the two venvs cannot be
   changed by one edit. If makani main needs a different physicsnemo, that is a STOP → operator.
3. **Work in your own worktree/branch:** `feat/makani-port-main`, cut from `feat/makani-f-finetune`
   at a recorded commit, worktree `.claude/worktrees/makani-port`. Never edit `f-finetune`,
   `spatial-cxi`, `makani-ace2-ports` (F's frozen tree) or any worktree with a queued/running job.
4. **Old and new must both work from the same source tree** until M8: feature-detect, don't fork the
   code. Every commit must leave the **old venv's** suites green too (regression gate).
5. **Equivalence tolerance is pre-registered in a commit before the job** that measures it, and never
   loosened afterwards (CLAUDE.md #1, #6, #11). Default: **bitwise**. A non-bitwise result is a STOP:
   bisect upstream to the commit that moved it, report it, and let the operator decide.
6. **Checkpoints are read-only.** Never resave over an existing file; new-venv test saves go to a new
   run dir under `$MEMBER_ROOT/runs/makani_port/`.
7. Queues: everything here fits `debug` (≤ 1 h, ≤ 2 nodes; one queued job per user — fold checks into
   one job). `debug` is pre-authorised; `debug-scaling`, `preemptable`, `capacity` need the operator.
8. No Python from either venv on the login node (operator ruling): tests run as `debug` jobs.
   Login-node checks: `bash -n`, `/usr/bin/python3.11 -m py_compile`, stdlib `python3` (3.6!).

## 3. Milestones — small commits, a green tag per milestone, revert on red

Each milestone = a few small commits (one logical change each), closed by **one `debug` job whose
PASS token is the gate**. On green: tag `makani-port/mN-green` on the commit the job ran (`git tag` +
`git push origin <tag>`), PROGRESS entry. On red: diagnose from the `.err`/log first; if the fix is
not obvious within the milestone, **`git revert`** the commits since the last green tag (never
`reset --hard`, never force-push), record the failure in PROGRESS **and** `_papercuts.md`, and retry
smaller. The commit a job ran is the sha it prints — never edit the worktree while a job from it is
queued (`_papercuts.md`).

| M | goal | small commits | gate (one debug job, token) |
|---|---|---|---|
| **M0** | **Inventory + golden baselines** (no code change) | (a) `makani_port/api_delta.md`: for every row of §1 "what we import / patch", the upstream diff `c9704308..a0aa4c4f` (signature, behaviour, removal) and the action; read `4c40a0e0`, `4af60539`, `f9b6e787`, `8cf889ec`, `fa93cc70` in full. (b) prereg `makani_port/m0_golden_prereg.md` listing the golden artefacts. (c) `polaris_makani_port_golden.pbs` | `PORT_GOLDEN_OK` on the **old** venv: (1) per checkpoint {A e243, B e22, C1 e24, nf4p_r1, Fsurg, and the only EMA file on disk, `smoke_nosoil_4n_b32_r2/best_ckpt_ema_mp0.tar` — A/B/C1 have none; add F's raw + EMA when F exists}: 1-step prediction + K=56 rollout from 2048 f1092 saved as `.npy` under `$MEMBER_ROOT/runs/makani_port/golden/` + sha256 manifest (JSON in repo); (2) 20-step training loss trace, A config, 1 node, fixed seed, deterministic flags; (3) validate-only loss of A on fixed 64 samples (`SKIP_TRAIN=1`); (4) the six suites. Run it **twice** and require the two runs bitwise equal — otherwise the baseline is not deterministic and every later gate needs that fixed first |
| **M1** | **New venv beside the old** | `polaris_setup_sfno_venv_main.sh` (makani @ full target sha, `--no-deps`; torch-harmonics pinned to the **same** `2edb24ed` unless makani main refuses it; physicsnemo non-editable from a recorded `physicsnemo_sfno` commit; `zarr>=3`; DALI ≥ 2.0). Build on a compute node via the ALCF proxy if pip needs it (`_papercuts.md`: login pid cap) | `VENV_MAIN_OK`: provenance (every package resolves from the new venv, versions printed), `import` of every §1 symbol listed per module as OK/MISSING/MOVED, **old venv's makani dist-info sha256 unchanged** |
| **M2** | **Our code imports and the suites pass on both** | one commit per broken symbol/patch from M1's MISSING/MOVED list, feature-detected; `compat.py` shim kept only if still needed | `PORT_SUITES_OK old=<n> new=<n>`: six F0/F1 suites + `test_per_lead_metrics` + `test_trainer_ci` + `test_e3sm_port` on **both** venvs, identical pass counts |
| **M3** | **Every checkpoint loads** | loader changes only if needed (pname checksum: old files have none — must load without it, not by disabling the check globally) | `PORT_CKPT_LOAD_OK n=<k>`: each §1 checkpoint builds its model from its own `config.json` on the new venv and loads **strict**; key set and every tensor shape/dtype identical to the old venv's load; parameter sha256 identical |
| **M4** | **Inference is the same** | prereg `m4_infer_prereg.md` (bitwise, committed before the job) | `PORT_INFER_EQUIV_OK`: new-venv 1-step + K=56 outputs vs M0 golden, bitwise, every checkpoint; then `polaris_f2_equiv.pbs` on the new venv; then a one-year screen of A e243 (must truncate at **595** from 2044 f1092) and nf4p_r1 (must survive) |
| **M5** | **Training is the same** | prereg `m5_train_prereg.md` | `PORT_TRAIN_EQUIV_OK`: 20-step loss trace vs M0 golden (bitwise); EMA run; `MULTISTEP=5` warm start (`LOAD_LOSS=0`); resume from an old checkpoint; 2-node DDP smoke on CXI (`FABRIC_CXI_CONFIRMED`, `MAKANI_MN_SCALING_OK`); a checkpoint saved by the new venv loads in the old venv (record if not — compatibility note, not a blocker) |
| **M6** | **Launchers can choose the venv** | `SFNO_VENV` override in the launch path, **default = old venv**; prove the default is bitwise unchanged (F2-style) | `PORT_LAUNCH_OK`: every launcher with the default prints the old venv; with `SFNO_VENV=…-main` prints the new one; F2 bitwise under default |
| **M7** | **Spatial `w=4`** (O8c) | none unless needed | spatial matrix (`polaris_spatial_cxi_matrix.pbs` from `feat/makani-spatial-cxi`, ported) on the new venv: does h2w4's epoch-2 loss now match h1w1 (7669001: 0.1394 vs 0.1112)? Report; read with `4c40a0e0` |
| **M8** | **Switch-over** | — | **operator decision**, not yours: default venv, when, and whether queued work restarts. Present the M0–M7 evidence |

## 4. Bookkeeping (token-efficient by design)

- **`makani_port/PROGRESS.md`** — append-only, newest first, one dated entry per working block:
  `## YYYY-MM-DD HH:MMZ — <one-line state>` then four short headings **Progress / Surprises /
  Decisions / Next** (bullets, job ids, tokens, shas; no prose paragraphs). This is the hand-off:
  a fresh session reads the top 3 entries, not the whole history.
- **`makani_port/_papercuts.md`** — one line per mistake that cost time: date · what bit · the rule
  that avoids it. Add it the moment it happens; read the file at session start and before every
  `qsub`. Never delete an entry; mark obsolete ones `~~struck~~` with the reason.
- **CHANGELOG.md** — one entry per milestone (green or reverted), pointing at PROGRESS.
- Draft PR `feat/makani-port-main` → `feat/makani-f-finetune` after M2; leave open (no self-approve).

## 5. STOP and ask the operator

A non-bitwise M4/M5 result; makani main needs a physicsnemo, torch or torch-harmonics change; any
checkpoint fails strict load; anything that would touch `sfno-venv`, `physicsnemo_sfno/` or a frozen
worktree; any queue beyond `debug`; any change to loss, channels, normalisation or outputs (also
jesswan). State the default you would take, then wait.

## 6. Definition of done

M0–M7 green and tagged (or a milestone explicitly parked by the operator), PROGRESS and papercuts
current, CHANGELOG entries, draft PR open, and an evidence summary for the operator's M8 decision.
