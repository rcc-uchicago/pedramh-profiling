# HPO distill index

Generated: 2026-05-23T17:42:37

Plan: [docs/2026-05-23_hpo_prune_plan.md](../2026-05-23_hpo_prune_plan.md)

## Per-group notes

- **[G1](G1_legacy_gb16_gb32_pre-v11_partial-clone_era.md)** — Legacy GB16/GB32 (pre-v11 partial-clone era) (0 keep, 6 prune)
- **[G2](G2_early_group_clone_exploration_pre-v11___non-noise.md)** — Early group_clone exploration (pre-v11 / non-noise) (1 keep, 4 prune)
- **[G3](G3_v11_clip_a_b_concluded.md)** — v11 clip A/B (concluded) (0 keep, 3 prune)
- **[G4](G4_v11_gb32_peak-lr_sweep.md)** — v11_gb32 peak-LR sweep (1 keep, 7 prune)
- **[G5](G5_v11_gb32_lr8e4_min-lr_sweep.md)** — v11_gb32_lr8e4 min-LR sweep (1 keep, 1 prune)
- **[G6](G6_v11_gb32_lr8e4_noise_sweep_first_round.md)** — v11_gb32_lr8e4 noise sweep (first round) (0 keep, 2 prune)
- **[G7](G7_v11_gb32_lr8e4_minlr1e5_β₁_sweep_null_result.md)** — v11_gb32_lr8e4_minlr1e5 β₁ sweep (null result) (0 keep, 2 prune)
- **[G8](G8_v11_gb32_lr8e4_minlr1e5_noise_sweep_second_round.md)** — v11_gb32_lr8e4_minlr1e5 noise sweep (second round) (1 keep, 1 prune)
- **[G9](G9_v11_gb32_lr8e4_minlr1e5_epochs_extension.md)** — v11_gb32_lr8e4_minlr1e5 epochs extension (1 keep, 0 prune)

## Distilled tables

- `inventory.csv` — every discovered training + eval dir with verdict and bytes
- `train_scores.csv` — per-epoch (train_loss, val_loss, val_loss_ema, grad_norm, …)
- `train_summary.csv` — per-run (best val loss, final epoch, wall time)
- `eval_scores.csv` — per (eval, section, channel, model, metric, lead) scorecard rows
- `prune_manifest.csv` — every path the prune subcommand will delete
- `prune_audit.jsonl` — append-only log of actual deletions (written by `prune --apply`)

## Archived per-eval records

`runs/<eval_name>/` — verbatim copy of `report.md`, `provenance.txt`, `scores/`, `figures/`, `diagnostics/` for every eval (winner + loser). These survive deletion of the eval's `inference/` and `baselines/` NetCDFs.

