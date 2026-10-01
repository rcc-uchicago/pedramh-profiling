# M0 pre-registration — golden baselines on the OLD venv

*Committed before the golden job is submitted (the monitor checks this commit's time against the
job's `stime`). Tolerance is **bitwise** and is not a parameter; it is never loosened afterwards
(CLAUDE.md #1, #6, #11). Inputs: `api_delta.md` §3.3 (N1–N6), `golden_checkpoints.json`.*

## What is produced

Job `polaris/polaris_makani_port_golden.pbs`, one `debug` node, the **old** venv (`sfno-venv`,
makani `c97043086e60…`; the job refuses any other commit and prints `PORT_GOLDEN_MAKANI` with the
dist-info sha256s and `PORT_GOLDEN_PROVENANCE` with the repo sha and venv path). Outputs under
`$MEMBER_ROOT/runs/makani_port/golden/<jobid>/`; the JSON manifests are copied into the repo at
`makani_port/golden/` after the gate.

| # | artefact | definition |
|---|---|---|
| 1 | `infer.json` (+ `npy/<tag>_K56.npy` on the write run) | for each of the 5 checkpoints in `golden_checkpoints.json` (A e243, B e22, C1 e24, nf4p_r1 e1, ema_smoke e1): sha256 of the file, of the restored parameters (names+bytes in `named_parameters()` order), of the state dict and of its structure (names/shapes/dtypes); `rollout_one_ic` from test **2048**, frame **1092**, **K = 56**, physical units — sha256 per lead (lead 1 = the 1-step prediction) and of the whole tensor; a second in-process rollout must be bitwise equal |
| 2 | `trace_val_r{1,2}.json` | validate-only (`--skip_training`) of A e243 warm-started into a fresh run dir; first 64 samples of the valid split (unshuffled), batch 8, `valid_autoreg_steps 3` as shipped; validation loss as `float.hex` |
| 3 | `trace_train_r{1,2}.json` | **warm start** from A e243 (model weights only — the fork's `pretrained_checkpoint_path` path; optimizer/scheduler/counters fresh), 20 steps = 160 samples, batch 8, 1 GPU, bf16, `lr_warmup_steps 0` (so every step updates at A's `lr 2e-3`), shuffle seeded with `GOLDEN_SEED=1234` **after** construction; per step: loss, grad norm (`float.hex`), `clipped` = grad norm > `optimizer_max_grad_norm` (32); then validation as in #2 |
| 4 | `suites.out` | `polaris_f_finetune_tests.pbs` → `F_FINETUNE_TESTS_OK 6/6` (CPU, concurrent) |

Config for #2/#3 = A's own rendered YAML (`e3sm_mn_scaling.prod1n_b32_sgdr.yaml`) with **only**
`exp_dir`, `n_train_samples_per_epoch 160`, `max_epochs 1`, `n_eval_samples 64`,
`lr_warmup_steps 0`, `log_to_wandb False` changed; model, loss, optimizer and data untouched.
cuDNN `benchmark=False, deterministic=True`; TF32 flags as the entrypoint sets them (recorded).

## Why the trace has this shape (api_delta §3.3)

- **Warm start, not fresh init (N4).** `f9b6e787` reorders RNG draws at MLP initialisation, so a
  fresh-init trace would differ on the new venv by construction. Reseeding after construction makes
  the data order independent of how many draws initialisation consumed.
- **Grad norm and `clipped` recorded (N2).** `a14185d` changes the grad-norm arithmetic. The norm is
  observed, never fed back unless it exceeds 32; the shipped runs peak at 0.300 per epoch, so a
  difference **only** in `grad_norm` with every `loss` bitwise equal and `clipped` all false reads as
  N2 and not as a model difference. Both versions multiply unclipped grads by exactly 1.0 or skip —
  bitwise the same.
- **Hooks are observation only:** a forward hook on `loss_obj` and a wrapper returning
  `clip_grads`' own result; the call sites (`deterministic_trainer.py:493/515` at the pin,
  `:552/574` on main) are the same shape in both versions.

## PASS criteria

- **Write run** (no `GOLDEN_REF`): `PORT_GOLDEN_INFER_OK n=5` (every control rollout bitwise);
  `GOLDEN_TRACE_SHAPE_OK` (20 steps each with a grad norm; validation non-empty);
  `GOLDEN_MATCH val_in_job` and `GOLDEN_MATCH train_in_job` (two processes, bitwise);
  `F_FINETUNE_TESTS_OK 6/6` → `PORT_GOLDEN_WRITTEN`.
- **Gate run** (`-v GOLDEN_REF=<write run dir>`, a separate job): all of the above **and**
  `GOLDEN_MATCH infer_vs_ref`, `val_vs_ref`, `train_vs_ref` → **`PORT_GOLDEN_OK`**. Only then is
  the baseline deterministic enough to gate M3–M5; tag `makani-port/m0-green` on the sha the gate
  printed.
- Any non-bitwise result is reported with the first differing field and is a STOP, not a retry.

## Not covered here (by design)

Fsurg (`fsurg_nf4_proxy_b8_r1`) has no checkpoint — job 7671977 hit its 2880 s timeout
(`rc=124`) before the first epoch end; add it when it exists. F (`f_nosoil_2n_b32_e43_warm`) is
added when it exists. B's file is a rotating slot: it is only read (sha256 recorded), never resumed.
