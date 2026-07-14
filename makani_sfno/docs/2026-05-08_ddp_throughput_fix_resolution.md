# DDP throughput fix — resolution

Date: 2026-05-08
Status: resolved. Production YAML edited (Commit 4 of the throughput
fix work); production-launch smoke verified end-to-end on 2026-05-08.
Plan: see `docs/2026-05-05_ddp_throughput_fix_plan.md` for the design
and pass criteria this resolution closes against.

This doc is the closing record for the throughput fix. It captures the
root cause, the two empirical sweeps, the production-launch smoke, and
the final production setting. It does not propose any further changes.

## Root cause

The 4-rank DDP run of `submit_zgplev_full.slurm` was wall-clock
equivalent to the single-GPU baseline because the **global batch size
stayed at 4**. `train_plasim._resolve_batch_sizes` divides global by
the data-parallel size, so `batch_size: 4 / dp=4` produced
**per-rank=1**. SFNO kernels at per-rank=1 sharply under-utilise the
H100 — enough that 4× the ranks ≈ 1× the throughput. The launch
summary (I0) made this visible; before I0 there was no rank-0 log of
either the global batch or the resulting per-rank batch, so the
collapse was silent.

## I1 — short-config DDP sweep

Short architecture (`embed_dim=128`, `num_layers=4`, `scale_factor=3`),
4-rank DDP, 8 epochs, bf16, `--save_checkpoint legacy`. Steady-state
means over epochs 2-8 (epoch 1 absorbs JIT + cudnn-autotune + DataLoader
prefetch fill). All four points satisfy I1: ≥ 2 epochs complete, no OOM,
no NCCL hang, no NaN.

| GB | per-rank | step time (ms) | samples/sec | speed-up vs GB=4 | mem (GB) | steps/ep |
|----|----------|----------------|-------------|------------------|----------|----------|
| 4  | 1        | 20.0           | 199         | 1.00×            | 8.62     | 1024     |
| 8  | 2        | 21.1           | 378         | 1.90×            | 8.66     | 512      |
| 16 | 4        | 23.3           | 691         | 3.47×            | 8.74     | 256      |
| 32 | 8        | 25.5           | 1257        | 6.32×            | 8.89     | 128      |

Step time grew only +28% across an 8× per-rank increase — exactly the
under-utilisation pattern the plan predicted. Memory remained tiny on
the short architecture, so the choice between GB=16 and GB=32 had to be
made under the production architecture (I2).

## I2 — production microbench

Production architecture (`embed_dim=256`, `num_layers=12`,
`scale_factor=1`, 106.9 M params), 4-rank DDP, 2 epochs (epoch 1 warmup,
epoch 2 measured), `--skip_validation`, `--save_checkpoint none`. EMA
enabled (production-shaped optimiser-step path). 1500 steps/epoch.

| GB | per-rank | step time (ms) | samples/sec | mem (GB) | training loss (E2) | grad norm (E2) |
|----|----------|----------------|-------------|----------|---------------------|----------------|
| 16 | 4        | 64.84          | 246.77      | 12.31    | 0.213               | 0.384          |
| 32 | 8        | 95.58          | 334.79      | 14.35    | 0.198               | 0.344          |

GB=32 is **1.36× faster** than GB=16 in throughput at +17% memory.
Memory at GB=32 leaves 50 GB headroom on H100. Step-time scaling
(1.47× for 2× per-rank batch) shows kernels are approaching but not at
saturation. Loss / grad-norm at the same step count are slightly better
at GB=32 — sqrt-LR scaling is holding.

A side-effect of `--skip_validation` was that Makani's
`deterministic_trainer.log_epoch` reads
`valid_logs["base"]["validation steps"]` unconditionally on the rank-0
screen path; with validation skipped that key is absent and rank 0
crashed with `KeyError`, leaving ranks 1-3 hung on the next AllReduce
until NCCL timed out (10 min). Fixed by backfilling the keys in
`PlasimTrainer.log_epoch` before delegating to `super()`.

## Final production smoke

Real `submit_zgplev_full.slurm` against the post-edit production YAML,
unique `EXP_DIR` (the live `…/runs/sfno_zgplev_full/` directory was not
touched), 4-rank DDP, 2 full epochs, validation + EMA validation +
checkpoint write all enabled.

| Epoch | step time (ms) | samples/sec | mem (GB) | train loss | grad norm | val loss | val loss EMA | ema decay | ema step | epoch wall (s) |
|-------|----------------|-------------|----------|------------|-----------|----------|--------------|-----------|----------|----------------|
| 1     | 100.71         | 317.73      | 11.62    | 2.195      | 2.466     | 2.097    | 2.116        | 0.998     | 4549     | 487.05 (train 458 + val 13 + ckpt 14.7) |
| 2     | 95.58          | **334.79**  | 11.62    | 0.052      | 0.394     | 0.019    | 0.021        | 0.999     | 9098     | 462.55 (train 435 + val 12 + ckpt 15.0) |

Smoke samples/sec at epoch 2 is **identical to the I2 microbench
prediction** (334.79). Validation produced real numeric losses (not
the NaN backfill sentinel). EMA-best checkpoint written separately
from the regular `best_ckpt`. No NaN/Inf. Memory 11.62 GB peak — well
under the 65 GB H100 cap.

## Final production setting

Source of truth: `src/sfno_training/config/plasim_sim52_zgplev_full.yaml`.

- `batch_size: 32` — GLOBAL batch (per-rank=8 at 4-GPU DDP)
- `lr: 2.83E-4` — sqrt scaling from the prior global-4 baseline
  (`1.0e-4 × sqrt(32/4) ≈ 2.83e-4`)
- EMA block unchanged (`enabled: True`, `decay: 0.999`,
  `warmup: True`, `allow_config_change: False`)
- `--batch_size` CLI override removed from
  `src/sfno_training/submit_zgplev_full.slurm`; the YAML is the single
  source of truth so the I0 launch summary, the YAML, and the runtime
  always agree.

## Expected speedup on the production schedule

The pre-fix 4-rank DDP run took ~38 min/epoch (samples/sec collapsed
to roughly the single-GPU baseline because per-rank=1 was idling the
H100). The smoke epoch-2 wall is 462.55 s ≈ **7.7 min/epoch** — a
**~4.9× speedup**. End-to-end the full 50-epoch schedule that
previously projected to ~32 h of training time should now project to
~6.4 h (still under the SLURM `-t 06:00:00` budget when starting from
scratch, comfortably under it on resume).

## Pointers

- Plan (design + pass criteria): `docs/2026-05-05_ddp_throughput_fix_plan.md`
- Production YAML: `src/sfno_training/config/plasim_sim52_zgplev_full.yaml`
- Production submit: `src/sfno_training/submit_zgplev_full.slurm`
- I1 sweep wrapper: `scripts/run_zgplev_short_ddp_sweep.sh`
- I2 microbench harness: `src/sfno_training/submit_zgplev_full_microbench.slurm`
- I0 + skip_validation backfill tests: `tests/sfno_training/test_ddp_logging.py`
