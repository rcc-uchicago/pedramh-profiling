# /work Quota Cleanup Plan — 2026-05-16

**Trigger:** Stampede3 login banner warning — `/work` at 938.9 / 1024 GB (91.7%).

**Scope:** Reduce `$WORK` (`/work2/11114/zhixingliu/stampede3/SFNO_Climate_Emulator/`) below 50%
without touching anything load-bearing for the active SFNO own-track (v10/v11)
or SFNO-5410 eval pipelines.

## Pre-cleanup snapshot

| Path | Size | Status |
|---|---|---|
| `artifacts/derecho_glade/PanguWeather/v2.0/` | 449 G | mostly upstream cruft (see A, B) |
| `results/sfno_eval/` | 391 G | 28 dated run dirs (see C) |
| `results/sfno_eval_5410/` | 63 G | mostly load-bearing prod, 5 G tmp (see D) |
| `artifacts/derecho_runtime/aires_env_20260509/` | 14 G | **KEEP** — packed Python env for `submit_eval_inference_5410_packed.slurm` |
| `artifacts/derecho_{blocking,exact,fingerprints,vendor}/` | 3.6 G | leave |
| `envs/`, `miniforge3/` | 7.4 G | leave |

## Bucket A — Pangu upstream cruft (~357 G freed)

`PanguWeather/v2.0/` was imported wholesale from upstream (Derecho/Glade). Our
own scripts only consume:
- the source tree (Python, configs, HPC_scripts, env_files)
- `results/SFNO/5410/checkpoints/ckpt_epoch_{48,50}.tar` (ref in
  `submit_eval_5410.sh`, `score_5410.py`, `submit_eval_score_5410.slurm`,
  `submit_eval_report_5410.slurm`)
- `results/SFNO/5410/{hyperparams.yaml, out.log}` (referenced in
  `plasim_sim52_zgplev_group_clone_gb32.yaml` header comment)

Delete:
- `PanguWeather/v2.0/results/PLASIM/` (135 G)
- `PanguWeather/v2.0/results/SFNO/<every dir except 5410/>` (~201 G; 17 dated dirs)
- `PanguWeather/v2.0/wandb/` (21 G)

Keep:
- `PanguWeather/v2.0/logs/` (32 G) — user-requested retain for upstream training history reference.

## Bucket B — Redundant 5410 checkpoints (~51 G freed)

`results/SFNO/5410/checkpoints/` has 50 ckpts × 1.28 G each.

Keep epochs 41–50 (10 ckpts, ~12.8 G).
Delete epochs 1–40 (40 ckpts, ~51 G).

## Bucket C2 — Superseded own-track eval runs (~97 G freed)

Drop pre-`v11_clip` dirs and paused `gb16/gb32` HPO siblings. Keep gb4
(winning HPO per `project_zgplev_gb_decision`) and May-10..May-16 active runs.

Sized dirs to delete:
- `20260504_eval-8377f46_data-ba0796a_train-8377f46_ckpt-best_ckpt_mp0` (16 G)
- `20260505_e25-snapshot_noncanonical` (4.5 G)
- `20260508_eval-adb71c4_data-e3c934b_train-8ba5f3d` (16 G)
- `20260508_eval-e3beb57_data-e3c934b_train-e3c934b` (16 G)
- `20260509_gb16_lr1e4_vs_gb4` (16 G)
- `20260509_gb16_lr2e4_retry1_vs_5410` (16 G)
- `20260509_gb32_ema` (13 G)

Empty stubs (sweep):
- `20260505_eval-e3c934b_data-e3c934b`
- `20260509_gb16_lr1e4_ema`
- `20260513_v11_surface_holdout_best_ckpt_mp0`
- `20260515_eval-867fead_data-e3c934b_train-8b395eb_family-sfno_zgplev_group_clone_v11_clip_warmstart_ckpt-best_ckpt_ema_mp0`
- `v10_zgplev_full_n96`

## Bucket D — Smoke/tmp in sfno_eval_5410 (~5.5 G freed)

- `tmp_5410_valid_writer_smoke/` (4.8 G)
- `tmp_boundary_template_preflight/` (20 K)
- `20260509_one_step_probe_awikner_local/` (68 M)
- `20260509_inverse_replay_y121s0_derecho_env_*/` (6 dirs, ~90 M)
- `20260509_derecho_env_block0_deep_*/` (2 dirs, ~578 M)

## Projected post-cleanup state

| Bucket | Freed |
|---|---|
| A | ~357 G |
| B | ~51 G |
| C2 | ~97 G |
| D | ~5.5 G |
| **Total** | **~510 G** |

/work after: ~429 / 1024 G (~42%).

## What is explicitly NOT touched

- `derecho_runtime/aires_env_20260509/` (packed Python env)
- `PanguWeather/v2.0/{ckpts 41-50, source tree, logs/}`
- `sfno_eval_5410/20260509_blocking_96ic_h100_packed_derecho_env_valid` (34 G — 5410 production)
- `sfno_eval_5410/tas_no_ice_20260514_1415_5410_prod` (24 G — 5410 tas-fix overlay)
- `sfno_eval/` runs from May-10 → May-16 (active v11/gb32 comparisons + warmstart family)
- `sfno_eval/20260509_gb4_ema`, `20260509_y11valid_gb4_k60` (winning GB decision)
- `/scratch`, `/home1`, `envs/`, `miniforge3/`
