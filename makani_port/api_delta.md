# makani API delta — pin `c9704308` → main `a0aa4c4f`

*M0(a), written 2026-10-01 from a bare clone `$MEMBER_ROOT/external/makani-main.git` (184 commits,
209 files, +45339/−7578). Every row was read in the upstream diff, not inferred from commit titles.
Full shas: old `c97043086e60d44a3adc3bede9a6b3dc71f5005d`, new `a0aa4c4fe5c40207d4fcc3da61d4c656a4ea0346`.*

Legend — **BREAK**: fails on the new venv as-is · **NUMERICS**: runs, but outputs predicted to move
off bitwise · **OK**: no action · **STOP**: needs the operator (and jesswan where marked) before M2.

## 1. Every symbol we import still exists

Import-level check over the 30 `from makani …` sites under `makani_sfno/{src,scripts,polaris}`: all
resolve at both shas. Nothing is MISSING at import time — the breaks below are behavioural.

## 2. What we import / patch — upstream change and action

| our use | upstream change | verdict | action (milestone) |
|---|---|---|---|
| `utils.dataloaders.data_loader_multifiles.MultifilesDataset` (subclassed by `PlasimForcingDataset`) | rewritten onto storage backends (`9311388`, `4f0af12`, `1b6e008`, `798245b`). Removed: `dataset_path` → `dataset_name`, `file_suffix`, `file_driver(_kwargs)`, `self.files`, `_open_file`, `_get_stats_h5` (no longer called). `__init__` takes `**kwargs`, so the old names are **silently swallowed** and the backend reads `/fields` | **BREAK** (silent if it ran) | M2: port `PlasimForcingDataset` to `self.backend` (`read(file_idx, slice, channels)`, `read_anchor/read_shape`, `metadata.files`, `file_driver` on the backend); feature-detect on `hasattr(MultifilesDataset, "_get_stats_h5")` |
| `PlasimForcingDataset._get_stats_h5` (tolerates PlaSim timestamp resets across splits) | not called any more; `order_files_by_time` **raises** on overlapping ranges | **BREAK** for PlaSim split-spanning subsets; E3SM packs are monotonic | M2: keep the reset tolerance in the port (override `_get_files_stats` or pre-sort); E3SM unaffected |
| `utils.dataloader.get_dataloader` (replaced by `_plasim_get_dataloader`) | returns `(loader, data_shapes, sampler)` instead of `(loader, dataset, sampler)`; trainer reads `valid_data_shapes.lat_lon_local` | OK by duck-typing (our dataset has every attr) | M2: return `dataset.data_shapes` when present |
| `training.deterministic_trainer.Trainer` → `PlasimTrainer.save_checkpoint(…)` override | trainer now calls `save_checkpoint(…, dataloader_state=…)` | **BREAK** — `TypeError` at the first epoch-end save | M2: accept `dataloader_state=None`, forward it only if `Driver.save_checkpoint` takes it |
| `Trainer.__init__` internals | `self.autocast = AutocastManager(amp_mode)` replaces `amp.autocast(...)` (`18c4582`); `self.amp_enabled/amp_dtype` kept; `step_timer`; `_setup_visualizer` (rank 0 only — our logger-on-every-rank shim stays harmless); `get_train_dataloader_state()` returns None for a torch `DataLoader` | OK — bf16 path is the same `amp.autocast(bfloat16)` | none |
| `Driver.restore_from_checkpoint` (warm start) and `Driver._restore_checkpoint_legacy` (inference `checkpoint_loader.py:232`) | new `dataloader=` and `log_to_screen=` params (our kwargs calls stay valid); **file now read by `load_checkpoint(...)` with `torch.load(weights_only=True)`** + allowlist {`ParamsBase`, `YParams`} (`c8bdd6f`) | **BREAK** — see §3.1 | M3: allowlist exactly the two `ruamel` types (§3.1) |
| `makani.utils.comm.init(model_parallel_names=…)` — `train_plasim.py:256`, `checkpoint_loader.py:297` (from `config.json`), `scripts/preflight.py:122`, `polaris_makani_env_probe.pbs:177` all pass `["h","w","fin","fout"]` | `fin`/`fout` retired for one `matmul` group (`f9b6e787`) | **BREAK** — `comm.init` has no `fin` node | M2: map `fin×fout → matmul` when `comm.init`'s default names lack `fin` |
| `utils.argument_parser.get_default_argument_parser` (`train_plasim.py:221` reads `args.fin_parallel_size`) | `--fin/--fout_parallel_size` → `--matmul_parallel_size`; `--enable_odirect` → `--odirect_config` | **BREAK** (`AttributeError`) | M2: `getattr(args, "matmul_parallel_size", None)` fallback. No launcher passes the removed flags (searched) |
| `utils.parse_dataset_metada.parse_dataset_metadata` | `coords.grid_type` required, and **verified against `coords.lat`** (`798245b`) | **BREAK + STOP** — §3.2 | operator + jesswan |
| `utils.loss.LossHandler` (built by the trainer, not by us) | every non-SHT loss term is now `torch.compile(dynamic=False)`'d by default (`b4e9c6a`, `74ba136`); our `l2` is non-SHT | **NUMERICS** (fused reductions) + new inductor dependency in every job | M2: rebind `deterministic_trainer.LossHandler`/`ensemble_trainer.LossHandler` to `partial(LossHandler, compile=False)` in `_install_plasim_patches` (feature-detect the kwarg). Turning compile on is an optimisation, gated by DESIGN §4 — not a port |
| `utils.training.training_helpers.clip_grads` (stock training step) | grad norm now `_foreach_norm` per tensor → `pow(2)` → sum → sqrt, grouped by sharding (`a14185d`); old: per-param `sum(square(abs(g)))` → sum → sqrt | **NUMERICS** whenever clipping fires (A clips at 32; later runs at 1.0) | M5: predicted non-bitwise source #1 for training; pre-register it, do not loosen |
| `Driver.get_optimizer` | param groups via `get_parameter_groups(model, wd, "full")` = one group of `requires_grad` params (`3fe70fb`) | OK if every param is trainable (true for SFNO); optimizer-state layout unchanged | M5 resume test covers it |
| `Driver.get_scheduler` | `lr_start` default `0.0 → 1e-5` (`fbe141f`) | OK for A (`lr_start: 0.01` explicit) | M5: assert `lr_start` is explicit in every config we resume |
| `models.preprocessor.Preprocessor2D` (subclassed) | docstrings; `torch._check` batch guards; noise run eager; `cache_unpredicted_features` refactored, same semantics | OK | none |
| `models.stepper.{Single,Multi}StepWrapper` (subclassed) | `update_internal_state(batch_size=…)`; `_preprocess`; optional `multistep_checkpoint` (off) | OK | none |
| `models.model_registry` (wrapper rebinding) | `SingleStepWrapper`/`MultiStepWrapper` still module-scope names; `ensure_resampled_shapes` | OK — patch still lands | none |
| `mpu.helpers.sync_params` (`_serialized_sync_params` mirrors it) | formatting only | OK | none |
| `utils.dataloaders.data_helpers.get_timedelta_from_timestamp` (py3.12 shim, `compat.py`) | still `timedelta(seconds=np.int64)`; **new second import site** `backends/base.py:74` used by `timestamp_converter` (we read with `relative_timestamp=True`) | **BREAK** on py3.12 unless the shim also patches `backends.base` | M2: add that module to `compat.py` |
| `utils.metric.MetricsHandler` (rebuilt by us) | +67/−21 | unreviewed beyond import | M2 `test_per_lead_metrics` |
| `training.ensemble_trainer.EnsembleTrainer` (`PlasimEnsembleTrainer`) | +202/−130 (`d6bdc89` ensemble dim folded into batch) | unreviewed beyond import | M2 suites; the CRPS proxy is the one consumer |
| `utils.YParams` | `YAML(typ="safe")` | OK — new checkpoints stop pickling `ruamel` objects | none |
| `utils.driver.Driver.init_visualizer` | eval-functor → op registry | OK (we have no viz channels) | none |

## 3. STOP items — need a decision before the milestone that hits them

### 3.1 Legacy checkpoints carry `ruamel` objects the new safe unpickler rejects (M3)

Measured with a stdlib unpickler on A `best_ckpt_mp0.tar` (e243): besides tensors it pickles
`ruamel.yaml.scalarfloat.ScalarFloat` and `ruamel.yaml.anchor.Anchor`, under `optimizer_state_dict`
and `scheduler_state_dict` (YAML-parsed `lr`/`eps` values). `torch.load(weights_only=True)` unpickles
the whole file, so **even a model-only load raises** on the new venv.
Options: (a) `torch.serialization.add_safe_globals([ScalarFloat, Anchor])` in our compat layer — two
named types, our own files; (b) `MAKANI_ALLOW_UNSAFE_CHECKPOINT_LOAD=1` — disables the check
globally (the handoff rules this shape out). **Default I would take: (a).**

### 3.2 makani now refuses our grid declaration (M2 — blocks every job) — **science-owned**

`data.json` declares `coords.grid_type: equiangular` with `lat = 89.5 … −89.5` (180 cell centres,
`convert_e3sm_to_makani.py`: `LAT = np.arange(89.5, -90.0, -1.0)`). makani's `equiangular` means
`linspace(−90, 90, nlat)` — nodes **on** the poles; worst disagreement 0.5° vs tolerance 1e-3°.
`verify_grid_type` raises in `parse_dataset_metadata` (config load) and again in the HDF5 backend.
No makani grid type matches cell-centred latitudes ("they match no grid type makani knows").

Upstream's reason (verbatim): *"a wrong declaration silently mis-weights every area averaged loss
and metric."* That makes it a **hypothesis** about our existing checkpoints — that they were trained
and scored with `naive` quadrature over nodes that are not where the data lives. Its size is
**unmeasured**: nobody has compared makani's equiangular weights at `linspace(−90, 90, 180)` with
cell-centred weights at 89.5…−89.5, or propagated the difference into the `l2` loss and metrics.
State that magnitude before calling it a defect.
- **To keep "same outputs" (the port's contract)** keep `equiangular` and opt out of the check for
  **our dataset only** — never a global monkeypatch of `verify_grid_type` (same rule as the
  checkpoint bypass): `PlasimForcingDataset` passes the file's own coordinates as `lat_lon` to its
  backend (coordinates unchanged, verification not reached), and a narrowly-scoped catch of the
  `parse_dataset_metadata` check for our `data.json`. One greppable line per job:
  `GRID_VERIFY_OPTOUT equiangular_cellcentred`. Bitwise-preserving.
- **To fix the weighting** is a loss/metric change → not a port; jesswan.
**Default I would take:** preserve as above, and send jesswan the measured magnitude separately.
**Not implemented — STOP until the operator answers** (the monitor concurs on the scoping).

### 3.3 Predicted non-bitwise sources (pre-register in M4/M5 preregs; never loosen)

| # | where | commit | hits | mitigation |
|---|---|---|---|---|
| N1 | `_contract_lwise` (dhconv, our `operator_type`) lost `@torch.compile` | `6922a56` | **inference + training** | none in our code; if M4 is non-bitwise, bisect here first |
| N2 | grad-norm arithmetic in `clip_grads` | `a14185d` | the **logged** grad norm always; the update only on a clipped step | none (stock step). Measured inert for the update: max per-epoch grad norm 0.300 (A, 243 ep) / 0.187 (B) / 0.043 (C1) vs `optimizer_max_grad_norm 32`. Unclipped: old multiplies by `clamp(…, max=1) = 1.0` (bitwise identity), new skips — same grads |
| N3 | compiled loss terms | `b4e9c6a`, `74ba136` | training + validation loss | `compile=False` rebind (§2) |
| N4 | MLP `fc1`/`fc2` constructed before init → different RNG draws | `f9b6e787` | fresh-init training only (not checkpoint loads) | M5 trace must warm-start from a checkpoint |
| N5 | `DistributedInstanceNorm2d` normalises in fp32 (was bf16) | `18c4582` | **spatial `h·w > 1` only** | none — see §4 (M7) |
| N6 | same commit, h1w1 paths: `SpectralConv` adds `bias.to(x.dtype)` (was fp32-promoting `x + bias`); `GeometricInstanceNormS2` in fp32; `pos_embed.to(x.dtype)` (monitor, 2026-10-01) | `18c4582` | inference + training under bf16, **only** for a SpectralConv bias / `instance_norm_s2` / a pos embed | **measured inert for every golden checkpoint**: 0 `filter*bias` keys in all six model states, all `normalization_layer: instance_norm`, all `pos_embed: none`. Kept as a bisect suspect next to N1 |

## 4. The five commits the handoff named, read in full

- **`4c40a0e0` "Guard spectral weight splitting"** — only `SpectralCoherenceLoss` and
  `SpectralRegularization` (`energy_score.py`, `regularization.py`). We train an `l2` loss. **Not our
  `w=4` problem.**
- **`4af60539` "removing deadlocked collective"** — tests only. **Not our `w=4` problem.**
- **Instead, `18c4582` is the M7 lead:** at our pin `DistributedInstanceNorm2d` (used when spatial > 1
  with `normalization_layer: instance_norm`, i.e. our config) casts mean/var to bf16 and normalises
  in bf16 under autocast, while the h1w1 path's `nn.InstanceNorm2d` runs fp32 under autocast. main
  normalises in fp32. A precision gap that exists only when spatially sharded fits "h2w4 trains but
  its loss is ~26 % high" (7669001). Hypothesis, not a result — M7 tests it.
- **`f9b6e787` fin/fout → matmul** — removes `fin`/`fout` comm nodes and CLI flags; non-TE MLP keeps
  the same `nn.Sequential(fc1, act, drop, fc2, drop)` (state-dict keys `…fwd.0.weight` unchanged) but
  reorders RNG consumption at init (N4). At `matmul = 1` the forward is unchanged.
- **`8cf889ec` "save restore pname checksum"** — a crc32 of the *gather plan* checked across the
  model group before a **flexible save** under model parallelism. Nothing is written into files and
  nothing is checked on load. The handoff's "old files have no checksum" concern does not arise.
- **`fa93cc70` "model package rework"** — we import nothing from `model_package` (searched).

## 5. Checkpoint facts measured for M3

A `best_ckpt_mp0.tar`: keys `model_state, comm_grid, loss_state_dict, optimizer_state_dict,
scheduler_state_dict, iters, epoch`; epoch 243, iters 332424; 87 model tensors (incl. complex);
`comm_grid` = {model, spatial, matmul, w, h, fin, fout}, all size 1. main validates only its own
model comm names {model, spatial, matmul, h, w} (a subset) → the legacy comm check passes.
A's `config.json`: `amp_mode bf16`, `pos_embed none` (so `sfnonet`'s new `pos_embed.to(dtype)` is
inert), `optimizer_max_grad_norm 32`, `lr_start 0.01`, `model_parallel_names [h,w,fin,fout]`.

### Golden checkpoint files (stored `epoch` read from each file, not inferred from the name)

| tag | file under `e3sm_mn_scaling/` | epoch | note |
|---|---|---|---|
| A | `prod1n_b32_sgdr/training_checkpoints/best_ckpt_mp0.tar` | 243 | |
| B | `nf4_prod_b16_r1/training_checkpoints/ckpt_mp0_v1.tar` | 22 | rotating slot — read-only, never resume |
| C1 | `c1_rollout_full_b16/training_checkpoints/ckpt_mp0_v23.tar` | 24 | `best_ckpt_mp0.tar` there is e18 |
| nf4p_r1 | `nf4_proxy_b8_r1/training_checkpoints/best_ckpt_mp0.tar` | 1 | |
| ema_smoke | `smoke_nosoil_4n_b32_r2/training_checkpoints/best_ckpt_ema_mp0.tar` | 1 | 99 ch; no optimizer/scheduler state |
| — Fsurg | `fsurg_nf4_proxy_b8_r1/training_checkpoints/` | — | **empty**: job 7671977 `train rc=124 wall=2880s` (its own timeout) before the first epoch end; log stops at "Starting Training Loop". Not in M0 |

## 6. Requirements delta (for M1)

main: `nvidia-physicsnemo>=1.3.0` (2.x via `models/physicsnemo_compat`), `zarr>=3`,
`torch-harmonics>=0.9.0`, `torch>=2.4`, DALI ≥ 2.0. Ours: torch 2.8.0, th `0.9.2.dev75+g2edb24ed`,
physicsnemo 2.2.0a0, zarr 2.18.7. Only zarr must move (new venv only).
