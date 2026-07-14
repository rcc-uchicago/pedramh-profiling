# Phase-1 efficiency implementation plan (low/medium-risk items only)

Date: 2026-05-04
Status: plan only — no code changes in this commit.
Scope: implements the high-confidence subset of
[2026-05-04_makani_pipeline_efficiency_review.md](2026-05-04_makani_pipeline_efficiency_review.md).

**Revision history**

- v1 (2026-05-04 morning): initial draft.
- v2 (2026-05-04, post-review): see "v2 changes" below.
- **v3.2 (2026-05-04, fourth review):** four small fixes, no
  plan-level changes:
  - **P5 source path uses `REPO_ROOT`, not `$(dirname "$0")`.**
    Under `sbatch`, `$0` is the spooled batch-script path, not
    the repo path. The existing convention at
    `src/sfno_training/submit_zgplev_full.slurm:56,61` defines
    `REPO_ROOT="$HOME/projects/SFNO_Climate_Emulator"` and sources helpers via
    `source "$REPO_ROOT/src/sfno_training/slurm_helpers.sh"`.
    P5's source line is updated to follow that convention.
  - **P2 handle-count bound updated** to count both train and
    eval persistent worker pools, and to use the actual
    `num_data_workers: 4` from the full zgplev config (not 2).
    The bound is now `ranks × workers × 2 (train+eval pools) ×
    3 (datasets) × N_files`.
  - **P2 lsof invocation strengthened.** `pgrep -f "sfno_training"`
    can miss worker children whose process titles differ.
    Replaced with a `pstree -p`-based enumeration of all
    descendants of the SLURM step root pid before feeding to
    `lsof`.
  - **P6 profile-dir creation moved out of the rank-filter
    branch.** Today the only configured rank is 0, but the
    `os.makedirs(out_dir, exist_ok=True)` should run on every
    rank (or before the `world_rank in args.capture_ranks`
    branch) so the SLURM script does not need to also `mkdir
    -p`, and so a future change to capture on nonzero ranks
    does not silently lose the trace. The profile SLURM still
    `mkdir -p`s as a defense-in-depth.
  - **P1 validation-bias explanation tightened.** When the
    sampler's `drop_last=False` pads duplicates, Makani's
    metric handler increments `valid_steps` for each *processed*
    batch (`makani-src/makani/utils/metric.py:604`) and divides
    the accumulated metric by that count at `:636` — i.e. the
    bias is "padded duplicates averaged over the duplicate-
    inflated step count", not raw double-counting. The fix
    (sampler `drop_last=True`) eliminates the duplicates at
    source, so the bias-shape detail is moot once P1 lands.
- v3.1 (2026-05-04, third review): wording-only corrections,
  no plan-level changes:
  - **P4 reasoning corrected.** Skipping EMA metric keys on
    non-EMA epochs is safe because Makani iterates the metrics
    dict dynamically at
    `makani-src/makani/utils/training/deterministic_trainer.py:709`
    (`list(valid_logs["metrics"].keys())`), not because consumers
    use `.get(..., default)` — that earlier framing was wrong.
  - **P6 trace handler symbol corrected.** Makani exports Chrome
    traces via `profiling.trace_handler` →
    `prof.export_chrome_trace(...)`
    (`makani-src/makani/utils/profiling.py:21-24`), **not**
    `tensorboard_trace_handler`. The `mkdir -p` and full-path
    `--capture_prefix` requirements are unchanged.
  - **P1 "same seed" wording softened.** The current entrypoint
    does not establish a deterministic global seed before
    training; the 1-GPU vs 4-GPU loss comparison is therefore
    *statistical*, not seed-matched. Either we accept that and
    drop "same seed", or we add explicit seed plumbing as a
    separate small item. Plan now drops the wording and notes
    seed plumbing as an optional follow-up.
  - **P1 validation-count formula** restated using floor notation:
    expected per-rank validation steps =
    `floor(len(eval_dataset) / global_batch_size)`. Numerically
    identical to the previous `len // G` (Python integer division
    is floor division for positive operands), but more explicit.
  - **P2 handle-count check** is now informational only.
    Inspecting rank 0's `lsof` misses worker/rank fan-out by
    construction (other ranks have their own worker processes
    with their own file descriptors), so the check is a
    directional guardrail, not a proof. Treat as a sanity
    snapshot for the PR description.
- v3 (2026-05-04, second review): see "v3 changes" below.

**v3 changes** (kept for the next reviewer's reference):
  - **P1 sampler unit test redesigned.** `_plasim_get_dataloader`
    calls `init_distributed_io(params)` at
    `src/sfno_training/trainer/plasim_trainer.py:97`, which resets
    `params.data_num_shards = 1` whenever `torch.distributed` is not
    initialized
    (`makani-src/makani/utils/dataloader.py:30-44`). A plain unit
    test that just sets `data_num_shards=4` therefore exercises the
    `sampler is None` branch, not the DistributedSampler branch.
    Plan now factors sampler construction into a small helper that
    can be unit-tested in isolation, plus a `torchrun`-driven smoke
    that exercises the real path.
  - **P1 validation-count formula fixed.** With sampler
    `drop_last=True` symmetric across both ends, expected per-rank
    validation steps is `len(eval_dataset) // G` (where `G` is the
    *global* batch), not the previously-stated `len // (G * R) * R`.
  - **P6 must replace the `trainer.train()` call**, not insert after
    it. Plan now points at `src/sfno_training/train_plasim.py:218`
    explicitly and requires the profile directory to exist before
    `torch.profiler` runs. `--capture_prefix` must be passed as a
    full path stem.
  - **P7 performance claim weakened.** `non_blocking=True` does
    *not* by itself create a separate copy stream or guarantee
    GPU-side H2D / compute overlap; it lets the host thread proceed
    so the prefetched-pinned data path can fill the pipeline. Test
    revised to drop the "visible parallel activity in trace" claim.
    P7 is also explicitly an exception to "do not edit Makani core",
    requires the local-patches doc as a merge gate, and stays last
    in the merge order pending the weakened claim.
  - **P3 risk list trimmed.** Makani's legacy store dict at
    `makani-src/makani/utils/driver.py:540-572` has only
    `model_state`, `comm_grid`, `loss_state_dict`,
    `optimizer_state_dict`, `scheduler_state_dict`, `iters`, `epoch`
    — no `train_steps` or best-loss keys. `sharded_dims_mp` is a
    tensor attribute attached inside `state_dict`, not a top-level
    key.
  - **P2 / P4 wording corrected.** P4 changed "byte-for-byte" →
    "behaviour-equivalent" (`plasim_trainer.py:521`). P4 period=2
    example tightened: 5 EMA epochs = {0,2,4,6} from the period plus
    {7} from the final-epoch override. P2 memory estimate uses
    float32 (DataLoader-side dtype) not bf16 (autocast-side). P2
    prefetch-factor assertion uses "is None or default" rather than
    "not present".

**v2 changes** (kept for the next reviewer's reference):
  - **P1 batch policy is no longer conditional.** Verified that
    `src/sfno_training/train_plasim.py:132-140` already treats
    `params.batch_size` as the *global* batch and divides by
    `comm.get_size("data")` before constructing the DataLoader. So
    `torchrun --nproc_per_node=4 ... --batch_size 4` gives per-rank=1,
    global=4. The "verify before merge" branch is removed.
  - **P1 picks up a real bug to fix.** The train/eval samplers are
    `DistributedSampler(...)` without `drop_last=True`
    (`src/sfno_training/trainer/plasim_trainer.py:156-165`), while the
    DataLoader uses `drop_last=True` (`:172`). Under DDP this means the
    sampler pads/duplicates the tail and the DataLoader then drops the
    incomplete per-rank batch. We fix the sampler.
  - **P1 smoke target is the `short` config, not `smoke`.** The smoke
    config uses `batch_size: 1` and 4 train samples; both violate
    multi-rank divisibility and per-rank slicing. `short` is the
    smallest config that exercises 4-rank DDP cleanly.
  - **P3 (single-write checkpoint) is demoted to "defer or last".**
    Makani's `Driver.save_checkpoint`
    (`makani-src/makani/utils/driver.py:515`) builds a non-trivial
    store dict (sharded_dims_mp, comm_grid, loss/optimizer/scheduler,
    counters); a local one-shot reimplementation is high refactor risk
    for a small wallclock saving. Land only with golden key-set and
    bidirectional resume tests, after profiling shows it's worth the
    risk.
  - **New P7: `non_blocking=True` on the H2D copy.** Smaller, safer
    win than P3. Two-line patch into the vendored
    `makani-src/makani/utils/training/deterministic_trainer.py` at
    `:471` (training) and `:609` (validation).
  - **P6 ports Makani's existing `--capture_*` CLI** rather than
    inventing a parallel profiler interface. The block is ready to
    lift from `makani-src/makani/train.py:147-169`.
  - **`set_epoch` risk note removed from P1.** Already handled at
    `makani-src/makani/utils/training/deterministic_trainer.py:351-356`
    for both train and valid samplers.
  - **P5 wording corrected.** `NCCL_ASYNC_ERROR_HANDLING=1` changes
    failure behavior (forces an abort on async error rather than
    hanging), not just logging.
  - **Tests tightened.** P2 adds a two-epoch persistent-worker test;
    P4 checks validation call-counts and metric keys instead of
    log-line matching.

**Explicit non-goals for this plan** (deferred to a Phase-2 plan after
profiling lands):

- MPI launcher / `srun --mpi=pmi2` bootstrap
- NVIDIA DALI dataloader
- Async / sharded / distributed checkpointing
- Model / tensor / spatial parallelism
- Moving normalization to GPU
- Precomputing the Legendre-Gauss-grid dataset
- In-memory dataset cache
- `torch.compile` / Inductor
- `channels_last`

---

## Verified facts (used by the items below)

- `src/sfno_training/train_plasim.py:132-140` reads
  `params.batch_size` as **global** batch, asserts divisibility by
  `comm.get_size("data")`, and rewrites `params.batch_size` to the
  per-rank value before the DataLoader is built. So `--batch_size N`
  on `world_size = K` gives per-rank `N // K`, global `N`.
- `src/sfno_training/trainer/plasim_trainer.py:156-174` constructs the
  sampler **without** `drop_last`, then the DataLoader **with**
  `drop_last=True`. This is the asymmetry P1 fixes.
- `src/sfno_training/trainer/plasim_trainer.py:379` tolerates checkpoints
  with no `ema_config` key (`ckpt.get("ema_config")` returns `None` and
  the warmup branch is taken). So pre-EMA checkpoints already load; any
  format change in P3 must preserve this tolerance for *future* old →
  new resumes too.
- `src/sfno_training/trainer/plasim_trainer.py:502-511` is the
  EMA-append double-write. The four extra top-level keys are
  `ema_state`, `ema_step`, `ema_config`, `ema_best_loss`.
- `src/sfno_training/trainer/plasim_trainer.py:562-589`
  (`_save_best_ema_checkpoint`) is **not** a double-write; it builds
  its own `store_dict` once. P3 leaves it alone.
- `makani-src/makani/utils/training/deterministic_trainer.py:351-356`
  already calls `train_sampler.set_epoch(epoch)` and
  `valid_sampler.set_epoch(epoch)` when distributed. P1 needs no work
  here.
- `makani-src/makani/utils/training/deterministic_trainer.py:471` and
  `:609` are blocking host→device copies (`x.to(self.device)`).
  These are the two lines P7 patches.
- `makani-src/makani/train.py:147-169` already implements a full
  `torch.profiler` capture branch driven by `--capture_ranks`,
  `--capture_type`, `--capture_mode`, `--capture_range_start`,
  `--capture_range_stop`, `--capture_prefix`. P6 ports this.
- HDF5 chunking is `(1, C, H, W)` per dataset
  (`src/plasim_makani_packager/packager.py:392`) — i.e. one chunk per
  timestep — so per-sample reads are already chunk-aligned. The risk
  P1+P2 introduce is the *handle count*: 4 ranks × `num_data_workers`
  × 3 datasets (state/diagnostic/forcing) × N files. Worth a one-time
  inspection; not blocking.
- `src/sfno_training/trainer/plasim_trainer.py:97`
  (`_plasim_get_dataloader`) calls `init_distributed_io(params)`
  before the sampler is built; that helper, at
  `makani-src/makani/utils/dataloader.py:30-44`, **resets
  `params.data_num_shards = 1` whenever `torch.distributed` is not
  initialized**. So a unit test that constructs a fresh `params`
  with `data_num_shards = 4` and calls `_plasim_get_dataloader`
  directly will get back `sampler is None`, not a
  `DistributedSampler`. Tests must either bypass
  `init_distributed_io` (factor sampler creation into a helper) or
  run under `torchrun --nproc_per_node=N`.
- `makani-src/makani/utils/driver.py:540-572` is the legacy save
  store dict. Top-level keys produced are: `model_state`,
  `comm_grid`, `loss_state_dict`, `optimizer_state_dict`,
  `scheduler_state_dict`, `iters`, `epoch`. **No** `train_steps` or
  best-loss keys. `sharded_dims_mp` is a tensor attribute attached
  inside the `state_dict`, not a separate top-level key. P3's risk
  list and golden-fixture must reflect this exact set.
- `src/sfno_training/train_plasim.py:218` is the `trainer.train()`
  call. P6 replaces this line with the Makani-style profiler-or-not
  conditional block.

---

## Shared scaffolding (touched by multiple items)

YAML keys to add at the bottom of every
`src/sfno_training/config/plasim_sim52_zgplev_*.yaml`. Defaults preserve
current behaviour so adding the keys is a no-op:

```yaml
# Phase-1 efficiency knobs.
prefetch_factor: 4              # used only when num_data_workers > 0
persistent_workers: True        # used only when num_data_workers > 0
ema_validation_period: 1        # 1 = every epoch (current behaviour)
```

One new shell file `src/sfno_training/env_nccl.sh` (item P5), sourced
from each multi-GPU SLURM script.

No new Python helper file. All Python changes are in:
- `src/sfno_training/trainer/plasim_trainer.py` (P1 sampler fix, P2
  knobs, P4 EMA period; eventually P3 if it lands)
- `src/sfno_training/train_plasim.py` (P6 profiler arg plumbing)
- `makani-src/makani/utils/training/deterministic_trainer.py` (P7
  vendored patch — record the patch in `docs/`)

---

## P1. Switch full training to multi-GPU DDP via `torchrun --nproc_per_node=4`

### Files / functions to change

- `src/sfno_training/submit_zgplev_full.slurm` — SLURM header (1–10),
  body / launch block (87–109).
- `src/sfno_training/trainer/plasim_trainer.py:156-165` — fix the
  `DistributedSampler` `drop_last` asymmetry (see §"Sampler fix").
- `src/sfno_training/submit_zgplev_full.dsi.slurm` and
  `submit_zgplev_full.dsi_1epoch.slurm` — **leave alone** (DSI variants
  use single-A100 nodes).
- No code change to `train_plasim.py`. The
  `_should_skip_distributed_init` fast path
  (`train_plasim.py:61-78`) only triggers when `--disable_ddp` is
  passed, which we are dropping.

### Sampler fix (must land before or with the launcher change)

Replace the current sampler construction at
`plasim_trainer.py:156-165` with explicit `drop_last=True` on
the `DistributedSampler` for both `train` and `eval` modes:

- Today: `DistributedSampler(loader_dataset, shuffle=..., num_replicas=..., rank=...)`
- Proposed: same call with `drop_last=True` added.

Why this matters: the DataLoader at `:172` already has
`drop_last=True`. With the sampler defaulting to `drop_last=False`,
the sampler **pads** the tail by duplicating samples, then the
DataLoader **drops** any per-rank incomplete batch. Two consequences:

1. Train: some samples are silently duplicated within an epoch, biasing
   gradients toward the duplicated tail. Small effect; still wrong.
2. Validation: the validation loss is computed on a slightly padded
   set, then averaged over the un-padded count. Small bias; still
   wrong, and undermines comparability against the single-GPU baseline.

Setting `drop_last=True` on the sampler makes both ends symmetric: the
sampler trims the tail to a multiple of `num_replicas`, and the
DataLoader's `drop_last=True` is then a no-op except on the genuine
last partial batch.

### Launch block change

`submit_zgplev_full.slurm:101-109` becomes:

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=4 \
    -m sfno_training.train_plasim \
    --yaml_config "$FULL_YAML" \
    --config plasim_sim52_zgplev_full \
    --run_num 0 \
    --batch_size 4 \
    --multistep_count 1 \
    --amp_mode bf16 \
    --checkpointing_level 2
```

`--disable_ddp` is dropped. **`--batch_size 4` is correct as-is**: it
is the *global* batch (verified above), divided to per-rank=1 by
`train_plasim.py:140`. Same effective batch as today's single-GPU run,
so loss curves should be directly comparable modulo DDP averaging
noise.

### SLURM header / env

Add explicit CPU pinning (avoids worker oversubscription with 4 ranks):

```bash
#SBATCH --cpus-per-task=64        # tune to actual cores/node on h100
```

In the body, **before `torchrun`**:

```bash
export OMP_NUM_THREADS=$(( SLURM_CPUS_PER_TASK / 4 ))
source "$REPO_ROOT/src/sfno_training/env_nccl.sh"     # P5
```

### Tests / smoke checks

A plain unit test of the sampler does not work as drafted in v2 (see
"Verified facts" — `init_distributed_io` resets `data_num_shards=1`
when `torch.distributed` isn't initialized). The redesigned plan:

1. **Refactor for testability**: extract the sampler construction
   out of `_plasim_get_dataloader` into a small pure helper
   `_make_train_eval_sampler(loader_dataset, mode, num_replicas,
   rank)` that does **not** call `init_distributed_io`. Then the
   unit test constructs a tiny in-memory dataset and calls the
   helper directly with `num_replicas=4`, asserting
   `sampler.drop_last is True` and `sampler.num_replicas == 4` for
   both train and eval modes.
2. **Refactor regression test**: assert that
   `_plasim_get_dataloader` still returns the same sampler shape it
   does today when `data_num_shards == 1` (i.e. `sampler is None`
   on the single-rank path). This is the "did the refactor break
   the single-rank fast path?" guard.
3. **`torchrun`-driven distributed smoke**
   (`tests/sfno_training/distributed/run_sampler_smoke.sh`,
   invoked under `torchrun --standalone --nproc_per_node=2`):
   builds the actual `_plasim_get_dataloader` against a tiny smoke
   dataset, prints `sampler.drop_last`, `sampler.num_replicas`,
   `len(dataloader)`, and exits non-zero on the asymmetric case.
   This is the only path that exercises the *real* sampler
   construction with `dist.is_initialized() == True`. Run manually
   pre-merge; not in the CI default.
4. **4-GPU short-config functional test.** **Use the `short`
   config, not `smoke`.** The smoke config has `batch_size: 1` and
   four train samples per epoch — neither survives 4-rank
   divisibility or per-rank slicing. Concretely: copy the short
   SLURM into `submit_zgplev_short.4gpu.slurm`, switch the launcher
   to `torchrun --nproc_per_node=4`, drop `--disable_ddp`, keep
   `--batch_size 4`. Pass criterion: 2 epochs complete with no
   NCCL hang and no NaN.
5. **Loss-curve compare.** Run the short config 2 epochs single-GPU
   (today's path) and 2 epochs 4-GPU DDP (new path). Plot
   train/valid loss; expect agreement within ~5% of the single-GPU
   curve. **This is a statistical comparison, not seed-matched** —
   the current entrypoint does not establish a deterministic
   global seed before training, so we explicitly do not claim
   "same seed" or "bitwise match". If determinism becomes
   important, plumb a `--seed` arg into `train_plasim.py` calling
   `torch.manual_seed`, `numpy.random.seed`, and (under DDP)
   per-rank-offset seeding — track as a separate optional
   follow-up, not a P1 blocker.
6. **Validation-count assert.** With sampler `drop_last=True`
   symmetric across both ends and global batch `G =
   params.batch_size` (the YAML/CLI value before division),
   expected per-rank validation steps is **`floor(len(eval_dataset)
   / G)`** (equivalently `len(eval_dataset) // G` in Python; floor
   division for positive operands). Reasoning: the sampler shards
   `len(eval_dataset)` into `R = world_size` pieces of size `len
   // R` each (tail dropped at the sampler). Each rank's
   DataLoader then iterates over `len // R` samples in batches of
   per-rank size `B = G / R`, yielding `(len // R) / B = floor(len
   / G)` steps per rank. Log both sides and `assert ==`.
7. **EMA shadow consistency.** Add a debug assert (gated on a CLI
   flag, off by default): once at the end of validation,
   `dist.all_reduce(ema_norm)` and check max-min ≈ 0 across ranks.
   Catches any silent divergence between rank-local EMA shadows.
8. **No regression on existing GH200 baseline.** The
   `submit_zgplev_baseline.slurm` already runs `torchrun
   --nproc_per_node=4`; rerun it post-merge and confirm the loss
   curve matches a recent reference run.

### Correctness risks

- **Sampler asymmetry** (the bug we are fixing). Resolved by the fix
  above; the unit test guards against regression.
- **EMA shadows across ranks.** The trainer keeps per-rank EMA
  shadows; DDP keeps parameters identical, and EMA is a deterministic
  function of (params, step count), so shadows agree as long as ranks
  step the same number of times — which is what the sampler fix
  guarantees. The all-reduce assert in test (5) is a soak.
- **Effective batch size.** Stays at 4 globally. No LR re-tuning
  needed.

### How to verify speedup

On the `short` config (8 epochs, single H100 node):

- Single-GPU baseline: wallclock for 1 epoch, mean per-step time.
- 4-GPU DDP: same. Compute scaling efficiency
  `t_1gpu / (4 · t_4gpu)`. Target ≥ 0.75. Below 0.6 → strongly
  suggests we are dataloader-bound, which is exactly what P6's
  profiler will then characterize.

### What is kept unchanged

- Model, loss, optimizer, scheduler.
- The dataset class.
- The DSI SLURM scripts.
- The smoke / tiny / baseline SLURM scripts (smoke and tiny stay
  single-GPU; baseline already runs DDP on GH200).
- The `--disable_ddp` flag (still useful for short/smoke runs).

---

## P2. DataLoader: `persistent_workers=True`, `prefetch_factor=4`

### Files / functions to change

- `src/sfno_training/trainer/plasim_trainer.py:166-174` — the
  `DataLoader` construction inside the `get_dataloader`-equivalent
  helper.
- All `src/sfno_training/config/plasim_sim52_zgplev_*.yaml` — add the
  three YAML keys from the shared scaffolding (the `prefetch_factor`
  and `persistent_workers` keys are used here; `ema_validation_period`
  belongs to P4 but ships in the same scaffolding edit).

### Proposed code-level change

Replace the current `DataLoader(...)` call with one that adds two
guarded kwargs. Both `persistent_workers=True` and `prefetch_factor=N`
raise `ValueError` when `num_workers == 0`, so:

- `persistent_workers = bool(params.get("persistent_workers", True)) and (params.num_data_workers > 0)`
- Build a `loader_kwargs` dict and only insert
  `prefetch_factor=int(params.get("prefetch_factor", 4))` when
  `params.num_data_workers > 0`.

This applies to both train and eval modes (single edit covers both —
the helper is mode-parameterized).

### Config / SLURM changes

YAML keys (shared scaffolding). No SLURM change. Smoke YAML keeps
`num_data_workers: 0`; the guards above ensure the new keys are no-ops
there.

### Tests / smoke checks

1. **Unit test** `tests/sfno_training/test_dataloader_kwargs.py`:
   - `num_data_workers=0` → `persistent_workers is False`, and
     `prefetch_factor` is `None` (PyTorch's default for the
     workerless path). Assert `dl.prefetch_factor is None` (or
     however the installed PyTorch version exposes the default —
     the test pins on the installed-version default rather than on
     a fixed integer).
   - `num_data_workers=2, persistent_workers=True, prefetch_factor=4`
     → both reflected on the DataLoader.
2. **Smoke** (`submit_zgplev_smoke.slurm`, `num_data_workers=0`): pass
   identically — the new keys must be no-ops.
3. **Two-epoch short-config soak** (single-GPU, `num_data_workers=2`):
   confirm the second epoch completes correctly (this is the only
   meaningful test for stale-handle / persistent-worker HDF5
   behaviour — workers are torn down at end of epoch only when
   `persistent_workers=False`). Pass criterion: epoch 2 loss matches
   epoch 2 of the pre-change run within stochastic noise.
4. **Handle-count sanity** (informational only, run once): on a
   4-GPU short run with persistent workers on, take a single
   `lsof` snapshot mid-epoch and eyeball that open HDF5 handles
   are within an order of magnitude of the upper bound
   `ranks × workers × 2 × 3 × N_files` (see "Correctness risks"
   above for the `2 ×` factor accounting for coexisting train +
   eval persistent worker pools). The snapshot must walk the
   **full process tree** — DataLoader worker children's process
   titles may not match `sfno_training.train_plasim`, so a
   `pgrep`-only enumeration silently undercounts. Recommended
   invocation, run on the SLURM compute node mid-epoch:

   ```
   # Find the SLURM step root pid (or the torchrun pid), then
   # walk all descendants via pstree -p:
   ROOT_PIDS=$(pgrep -f "torchrun.*sfno_training" || pgrep -f "sfno_training.train_plasim")
   ALL_PIDS=$(for p in $ROOT_PIDS; do pstree -p "$p" \
                | grep -oE '\([0-9]+\)' | tr -d '()'; done | sort -u | paste -sd,)
   lsof -p "$ALL_PIDS" | grep '\.h5$' | wc -l
   ```

   Falls back to `pgrep` if `pstree` is missing on the node.
   Recorded in the PR description as a directional guardrail;
   not a CI check, not a hard pass criterion.

### Correctness risks

- **HDF5 handle persistence.** `persistent_workers=True` keeps the
  dataset object alive across epochs in each worker, which keeps the
  cached `h5py.File` handles in `_diag_files` / `_forcing_files`
  (`plasim_forcing_dataset.py:74-75, 192-210`) open. The data files
  are read-only for the duration of training, so this is the desired
  behaviour. Worth one comment in the YAML.
- **Memory.** `prefetch_factor=4` doubles per-worker prefetch from 2
  to 4 batches. DataLoader-side tensors are float32 (numpy-side
  normalization → `torch.from_numpy` produces float32; AMP/bf16
  casting happens later, inside the model). For 64×128 float32
  batches with 4 workers this is on the order of low MB per worker;
  negligible. With larger grids in a future config, revisit.
- **Concurrent open handles after P1+P2.** The
  `_plasim_get_dataloader` helper at
  `src/sfno_training/trainer/plasim_trainer.py:166` is invoked
  for **both `train` and `eval` modes**, and with
  `persistent_workers=True` both pools stay alive across epochs.
  Production zgplev full has `num_data_workers: 4` (not 2). So
  the upper-bound estimate is `ranks × workers × 2 (train+eval
  pools) × 3 (state/diag/forcing datasets) × N_files`. With 4
  ranks × 4 workers × 2 × 3 × ~50 files, that is on the order of
  ~5000 file descriptors total across the job — distributed
  across 4 rank processes plus their worker children, so ~1250
  per rank-tree, well below Stampede3's per-process
  `RLIMIT_NOFILE` (typically 4096+). Confirm with the lsof
  snapshot below; raise `ulimit -n` in the SLURM body only if
  needed.

### How to verify speedup

- Per-epoch wallclock for the **second** epoch on the `short` config
  (where the persistent-workers savings materialize). Should drop by
  ~the first-epoch dataloader warmup time.
- Per-step time variance should narrow; visible in the per-step time
  log lines if we keep the existing tqdm/log cadence.

### What is kept unchanged

- Dataset implementation, `pin_memory=True`, `drop_last=True`.
- `num_data_workers` defaults per config.

---

## P3. (Deferred) Collapse the EMA-key checkpoint double-write

**Status: deferred until P6 measures whether the savings warrant the
refactor risk.** The reviewer flagged this as the highest-risk / lowest-
gain item in the original plan; we agree.

### Why deferred

The double-write at
`src/sfno_training/trainer/plasim_trainer.py:502-511` re-opens the
just-written `.tar`, adds four keys (`ema_state`, `ema_step`,
`ema_config`, `ema_best_loss`), and re-writes. The wallclock cost is
~hundreds of ms per epoch — small relative to a 47.5h job. Eliminating
it requires reimplementing
`makani-src/makani/utils/driver.py:540-572` (legacy mode) locally.
The actual top-level keys produced there are:

- `model_state` (with canonical key-prefix stripping applied; tensors
  inside may carry a `sharded_dims_mp` *attribute* used at load-side
  for model-parallel scatter — the attribute lives on the tensor,
  not as a separate dict key).
- `comm_grid` (per-comm-name `{size, rank}` map).
- `loss_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`
  (each conditional on the object existing).
- `iters`, `epoch` (from the `counters` dict, when passed).

That is the full set — there are no `train_steps` or best-loss keys
in the legacy store dict. Anything we reimplement must produce
exactly those keys (plus our four EMA keys), in a way that the
existing load path at `makani-src/makani/utils/driver.py` and our
EMA-resume path at `plasim_trainer.py:372-379` accept unchanged.

There is no clean upstream hook to inject extra state into Makani's
save path without overriding the entire method. The risk is silently
losing one of the above keys, mis-applying the prefix-strip, or
losing the `sharded_dims_mp` attribute, any of which silently
breaks resume.

### If/when we land it, gates

1. **Golden key-set fixture**: commit the exact set of top-level keys
   produced by the current two-write code (against a tiny smoke
   trainer state). Reimplementation must match exactly.
2. **Bidirectional resume tests**: train 1 epoch with old code, save,
   resume with new code → loss matches. And: train 1 epoch with new
   code, save, resume with **old** code (still on disk in git
   history) → loss matches. This guards against an in-flight rollback.
3. **Pre-EMA checkpoint tolerance preserved**: the load path at
   `plasim_trainer.py:379` (`ckpt.get("ema_config")`) treats missing
   EMA keys as a fresh-EMA resume. Whatever new save path we write
   must continue to produce checkpoints that load cleanly under this
   `.get()` tolerance — so older deployments can step forward into
   new checkpoints without a code update.
4. **Best-EMA writer untouched.** `_save_best_ema_checkpoint`
   (`plasim_trainer.py:562-589`) builds its own `store_dict`
   directly; it is not a double-write and is out of scope.

### Until then

Keep the current two-write pattern. Document the per-epoch cost from
P6's profiler trace and revisit only if it shows up as a meaningful
fraction of epoch time.

---

## P4. EMA validation every K epochs

### Files / functions to change

- `src/sfno_training/trainer/plasim_trainer.py:513-560` —
  `validate_one_epoch` override.
- `src/sfno_training/trainer/plasim_trainer.py:549-551` — best-EMA
  tracking branch (must be guarded so it only fires on epochs where
  the EMA pass actually ran).
- All `plasim_sim52_zgplev_*.yaml` — add `ema_validation_period: 1`
  (default = current behaviour).

### Proposed code-level change

Inside `validate_one_epoch(self, epoch, profiler=None)`:

- Read `period = int(self.params.get("ema_validation_period", 1))`,
  with `period >= 1` enforced (assert at trainer init).
- Compute `should_run_ema = self.ema_enabled and (
    period == 1
    or (epoch % period == 0)
    or (epoch == self.params.max_epochs - 1)   # always on final epoch
  )`.
- When `should_run_ema is False`, return the raw validation tuple
  unchanged. Skip the entire block at lines 527-558. In particular:
  - do not touch `params.log_video`,
  - do not write the best-EMA checkpoint (best_ema_loss is unchanged),
  - do not add `validation loss ema` / `ema decay effective` / `ema
    step` / `ema best loss` to `valid_logs["metrics"]` for that
    epoch.

When `should_run_ema is True`, behaviour is identical to today.

With `period=1`, every epoch is an EMA epoch and the function is
**behaviour-equivalent** to the pre-change code at
`plasim_trainer.py:521`. (Strict byte-for-byte is not a goal — call
ordering and timer values may differ marginally; what matters is that
the same `valid_logs` dict shape and the same EMA-best-checkpoint
sequence are produced.) That is the default shipped in every config
so the merge is behaviour-preserving.

### Config / SLURM changes

YAML keys (shared scaffolding). No SLURM change at merge time. Production
flip to `ema_validation_period: 5` ships in a follow-up commit so the
plumbing change can be reverted independently of the policy change.

### Tests / smoke checks

1. **Unit test** `tests/sfno_training/test_ema_validation_period.py`:
   - `period=1` → `should_run_ema(epoch)` is True for all epochs.
   - `period=5` → True for epochs 0, 5, 10, …, and for the last epoch
     regardless.
   - `period=0` or negative → init-time assertion.
2. **Functional check on the `short` config** (8 epochs, epochs
   indexed 0..7):
   - Run with `period=1` (default). Expected: 8× raw + 8× EMA = 16
     `super().validate_one_epoch` calls. Assert metric keys
     `validation loss ema`, `ema decay effective`, `ema step`,
     `ema best loss` present in `valid_logs["metrics"]` on all 8
     epochs.
   - Run with `period=2`. Expected EMA epochs: **{0, 2, 4, 6} from
     `epoch % 2 == 0` plus {7} from the final-epoch override = 5
     EMA passes**, not 4. Total `super().validate_one_epoch` calls
     = 8 raw + 5 EMA = 13. Assert the EMA metric keys are present
     on `{0, 2, 4, 6, 7}` and absent on `{1, 3, 5}`.
3. We do **not** rely on log-line matching. Brittle to timings.
   Tests check call-counts via a `Mock` wrapping
   `super().validate_one_epoch` and metric-key presence/absence.

### Correctness risks

- **Best-EMA lag.** With `period=K`, the best-EMA checkpoint is
  selected from a coarser sample; worst-case miss is K-1 epochs.
  Acceptable for K ≤ 5 on a 50-epoch run. Document in YAML comment.
- **Logger / wandb consumers.** Any downstream consumer that expects
  `valid_logs["metrics"]["validation loss ema"]` to be present every
  epoch must tolerate its absence. Verified: Makani's screen logger
  at
  `makani-src/makani/utils/training/deterministic_trainer.py:709`
  builds its print list from
  `list(valid_logs["metrics"].keys())` and iterates whatever is
  present, and the wandb upload at `:731` calls
  `wandb.log(valid_logs["metrics"], ...)` with the dict as-is.
  Both consume the metrics dict *dynamically*, so missing keys on
  non-EMA epochs are silently absent rather than triggering a
  `KeyError`. Confirm in test (2) by running with `period=2` and
  observing no `KeyError`.
- **`log_video` invariant.** The skip path must not touch
  `params.log_video` (currently zeroed inside a try/finally during
  the EMA pass). The proposed implementation skips the entire block,
  so this invariant is preserved trivially.

### How to verify speedup

- Per-epoch validation wallclock on the `short` config with `period=1`
  vs `period=2`. Expect roughly `1/period` reduction on EMA-skip
  epochs. On a 50-epoch full run with `period=5`, the steady-state
  validation cost drops by ~80%.

### What is kept unchanged

- The EMA implementation in `ema.py`.
- The raw validation pass and best-raw-checkpoint logic.
- The on-disk EMA shadow format (still written every epoch by the
  ema-update post-step hook; only the *validation* of the shadow is
  skipped).

---

## P5. NCCL diagnostics for multi-GPU runs

### Files / functions to change

- New file: `src/sfno_training/env_nccl.sh`.
- `src/sfno_training/submit_zgplev_full.slurm` — `source` line in the
  body (also added by P1). Use the existing `REPO_ROOT`
  convention from line 56-61: `source
  "$REPO_ROOT/src/sfno_training/env_nccl.sh"`. Do **not** use
  `$(dirname "$0")` — under `sbatch`, `$0` is the spooled batch
  script path, not the repo path.
- `src/sfno_training/submit_zgplev_baseline.slurm` — same `source`
  pattern. (If `REPO_ROOT` is not yet defined there, define it
  using the same `"$HOME/projects/SFNO_Climate_Emulator"` value first.)

### Proposed contents of `env_nccl.sh`

```sh
# env_nccl.sh — sourced by multi-GPU SLURM scripts.
# NCCL_DEBUG: log level only; no behavioural change. Default WARN is
#   quiet; set NCCL_DEBUG=INFO at submit time for collective hangs.
# NCCL_ASYNC_ERROR_HANDLING / TORCH_NCCL_ASYNC_ERROR_HANDLING:
#   *behavioural*. With these set to 1, NCCL aborts the process on an
#   async error (e.g. a peer rank dying) instead of hanging
#   indefinitely. This is what we want for SLURM jobs — failures
#   surface as job exits within minutes rather than walltime exhausts.
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_ASYNC_ERROR_HANDLING=${NCCL_ASYNC_ERROR_HANDLING:-1}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}
# export NCCL_DEBUG_SUBSYS=COLL    # uncomment when debugging collectives
```

Using `${VAR:-default}` lets a user override at submit time
(`NCCL_DEBUG=INFO sbatch ...`).

### Config / SLURM changes

As above. Limited to the multi-GPU scripts (`_full.slurm` and
`_baseline.slurm`); single-GPU scripts unchanged.

### Tests / smoke checks

1. Run `submit_zgplev_baseline.slurm` (4× GH200) for 1 epoch and
   confirm SLURM stdout contains `NCCL INFO Bootstrap : Using …` only
   when `NCCL_DEBUG=INFO`, and is silent at default `WARN`.
2. Manual fault-injection: start a 4-rank short-config run; `kill -9`
   one rank mid-training; confirm the surviving ranks abort within
   ~2 minutes rather than hanging until walltime. (Recorded in the
   PR description.)

### Correctness risks

- `NCCL_ASYNC_ERROR_HANDLING=1` changes failure semantics (force-abort
  on async error). This is the desired behaviour, but is **not**
  purely diagnostic — flagged accurately in the YAML comment above.
- `NCCL_DEBUG=INFO` is verbose enough to slow startup logging by a few
  seconds. Default at `WARN`.

### How to verify speedup

This change does not alter steady-state speed. Its purpose is fault
visibility on multi-GPU runs.

### What is kept unchanged

- All training code.
- The single-GPU SLURM scripts.

---

## P6. Port Makani's existing `--capture_*` profiler block

This item produces the measurement we use to verify everything else.
It does not itself produce a speedup.

### Files / functions to change

- `src/sfno_training/train_plasim.py:215-218` — **replace** the
  current
  ```
  trainer = PlasimTrainer(params, world_rank)

  if not params.get("skip_training", False):
      trainer.train()
  ```
  with the Makani-style profiler-or-not conditional ported from
  `makani-src/makani/train.py:147-169`. The new shape is roughly:
  ```
  trainer = PlasimTrainer(params, world_rank)

  if params.get("skip_training", False):
      pass
  elif world_rank in args.capture_ranks:
      # build profile output dir, normalize capture_prefix to a
      # full path stem under that dir, then enter
      # torch.profiler.profile(...) (or CUDAProfiler for cupti),
      # and call trainer.train(training_profiler=profiler) /
      # trainer.train(validation_profiler=profiler).
      ...
  else:
      trainer.train()
  ```
  Important: this is a *replacement* of the bare
  `trainer.train()` call at line 218, not an addition after it.
  Otherwise we'd double-train (once outside the profiler context,
  once inside).
- The CLI args (`--capture_ranks`, `--capture_type`, `--capture_mode`,
  `--capture_range_start`, `--capture_range_stop`, `--capture_prefix`)
  come from Makani's
  `argument_parser.get_default_argument_parser()`. Verify they
  survive our local argparse customizations in
  `train_plasim.py`; if the local parser is built from scratch
  rather than inheriting Makani's, port the args explicitly.
- The PlaSim trainer's `train(...)` method must accept
  `training_profiler=` / `validation_profiler=` kwargs. Makani's
  signature already supports them; our subclass shadows
  `validate_one_epoch` but probably not `train` — verify before
  merge.
- **Profile output directory must exist before the profiler
  enters its context.** Makani's trace handler is
  `profiling.trace_handler` at
  `makani-src/makani/utils/profiling.py:21-24`; it calls
  `prof.export_chrome_trace(export_trace_prefix + "_" +
  str(prof.step_num) + ".json")`, which writes to whatever
  directory the prefix points at and assumes that directory
  exists. (Chrome trace export, not TensorBoard.) Add an
  `os.makedirs(out_dir, exist_ok=True)` call **before** the
  `world_rank in args.capture_ranks` filter (or, equivalently,
  on every rank with `exist_ok=True`). Gating the `mkdir` on
  `world_rank == 0` is fragile if a future change captures on
  ranks other than 0 — those ranks would silently fail to
  produce a trace because their target directory does not
  exist. The profile SLURM additionally `mkdir -p`s the dir as
  defense-in-depth.
- **`--capture_prefix` is a full path stem, not a basename.** The
  Makani port passes
  `f"{args.capture_prefix}_rank{world_rank}"` straight into the
  trace exporter, which writes to wherever that prefix points (cwd
  if relative). The new SLURM script (below) sets `--capture_prefix
  "$PROFILE_DIR/plasim_sfno"` so the exporter writes inside
  `PROFILE_DIR`.
- New SLURM script `src/sfno_training/submit_zgplev_profile.slurm`,
  copied from `submit_zgplev_short.slurm` with:
  - Single-GPU (we want to characterize the production codepath
    independently of DDP overhead first).
  - `PROFILE_DIR="$WORK/SFNO_Climate_Emulator/profiles/$(date +%Y%m%d_%H%M%S)"`,
    created via `mkdir -p` before launch.
  - `--capture_ranks 0 --capture_type torch --capture_mode training
    --capture_range_start 5 --capture_range_stop 35 --capture_prefix
    "$PROFILE_DIR/plasim_sfno"`.
  - 30-minute walltime.

### Why port rather than reinvent

Makani's block already handles:

- both `torch` and `cupti` capture types,
- training-mode and validation-mode capture,
- multi-rank capture (rank-0 only by default; configurable),
- Chrome trace export via `profiling.trace_handler` with stats print,
- NVTX range emission via `torch.autograd.profiler.emit_nvtx` for the
  cupti path,
- file-prefix per rank.

A homegrown profiler interface would lose these. The port is ~25 lines.

### Config / SLURM changes

The profile SLURM is new and isolated. No existing config is modified.
Default behaviour (no `--capture_ranks`) is byte-identical to today.

### Tests / smoke checks

1. Run the new `submit_zgplev_profile.slurm` end-to-end on the `short`
   config. Verify:
   - One Chrome trace file produced under `--capture_prefix`.
   - Stats are printed to stdout (Makani's `trace_handler(...,
     print_stats=True)`).
   - Without `--capture_ranks`, behaviour matches a stock short run.
2. Open the trace in `chrome://tracing` (or perfetto.dev). Visually
   confirm the gaps between train steps reflect dataloader stalls.
3. Capture a second trace **after** P2 has landed and compare the
   inter-step gap distribution. This is the empirical confirmation
   that `persistent_workers` + `prefetch_factor=4` materially helps.

### Correctness risks

- Profiler overhead while the capture window is active is ~5–15%.
  Read the breakdown as **ratios**, not absolute step times. Document
  in the SLURM script header.

### How to verify "speedup"

Measurement, not speedup. Success criterion: we can answer "is the
next bottleneck dataloader, CPU preprocessing, HDF5 I/O, or GPU
compute?" from a single 30-minute profile run, and that answer
reorders the Phase-2 priority list.

### What is kept unchanged

- All non-profile codepaths.
- Production SLURM scripts.

---

## P7. `non_blocking=True` on the H2D copy (vendored Makani patch)

A two-line patch into the vendored Makani trainer. Small, low-risk,
but **the performance claim is modest** — see "Why this is safe and
what it does (and doesn't) buy us" below.

### Vendored-Makani exception (explicit)

This item is the single exception to the general "do not edit
`makani-src/` core" rule. Justification: there is no clean upstream
hook to inject `non_blocking=True` into the inner training/validation
loops without overriding the entire method, and the alternative
(subclassing and copy-pasting Makani's ~150-line train loop) is much
worse for maintainability. Treating this as a vendored patch is the
right shape.

**Merge gate**: a new file `docs/2026-05-04_makani_local_patches.md`
must exist and list this patch (file:line, the unified diff, the
date the patch was applied, and the upstream Makani commit hash we
are patched against). The same file is the canonical
re-apply-after-bump source. P7 does not merge until this doc lands
in the same commit.

### Files / functions to change

- `makani-src/makani/utils/training/deterministic_trainer.py:470` and
  `:608` — change
  `gdata = map(lambda x: x.to(self.device), data)` to
  `gdata = map(lambda x: x.to(self.device, non_blocking=True), data)`.

### Why this is safe and what it does (and doesn't) buy us

What it does:

- **Frees the host (Python) thread.** With `pin_memory=True` already
  on the DataLoader (`plasim_trainer.py:173`), the H2D copy is
  issued asynchronously: the host thread does not block in `.to()`
  and continues to the next line. This lets PyTorch's prefetcher
  pipeline (next batch fetch + pin) overlap with the GPU work
  scheduled by the *previous* iteration.

What it does **not** do:

- **It does not, by itself, place the H2D copy on a separate CUDA
  stream from the compute.** The next GPU op on `gdata` runs on
  the same default stream as the copy and serializes behind it via
  CUDA stream ordering. True kernel-level H2D / compute overlap
  requires a dedicated prefetcher stream with explicit
  `stream.wait_stream(...)` synchronization (the NVIDIA "data
  prefetcher" pattern). That is **out of scope** for this plan and
  is the kind of change we'd consider only if P6 profiles still show
  a copy-bound stall after P2 + P7 are in.

So the framing of P7 is: a small, safe correctness-of-pattern fix
that lets the existing prefetched-pinned data path realize whatever
host-side asynchrony is available to it. The marginal kernel-level
overlap on the GPU side is opportunistic, not guaranteed.

### Tests / smoke checks

1. Run the `short` config 2 epochs before and after the patch. Loss
   curves should match within stochastic noise.
2. Confirm via the post-P6 profiler trace that the host-side
   `aten::to` event no longer dominates the inter-step gap (i.e.
   the host thread proceeds past the copy quickly). We **do not**
   require visible kernel-level parallel activity in the trace —
   that's a separate, more-invasive change (see above).

### Correctness risks

- Pattern is standard. Risk is forgetting that the patch lives in
  vendored upstream and silently dropping it on a future
  `makani-src/` bump — addressed by the
  `docs/2026-05-04_makani_local_patches.md` merge gate.

### How to verify speedup

Per-step time on the `short` config. Expected: low single-digit
percent at most, possibly within noise. The headline value is
unblocking the host thread so the existing prefetcher can fill the
pipeline — not a guaranteed wallclock win on its own. **Treat this
as plumbing**, not as the next throughput lever.

### What is kept unchanged

- Everything else in the Makani trainer.
- Our PlaSim subclass.

---

## Suggested merge order (v3)

1. **P5** (NCCL env file). Trivial, isolated, useful as a soak prior
   to any DDP work.
2. **P6** (port Makani's `--capture_*`, with the *replacement* call
   at `train_plasim.py:218` and the profile-dir / capture-prefix
   path-stem fixes). Get baseline evidence early so the rest of the
   order is data-driven. Run once on the `short` config single-GPU
   as a control.
3. **P2** (DataLoader knobs). Low risk, contained, gives an easy
   wallclock signal we can compare against on the next P6 trace.
4. **P4** (EMA-every-K plumbing). Behaviour-equivalent with default
   `period: 1`. Production flip to `period > 1` ships in a follow-up.
5. **P1** (multi-GPU DDP for full). Lands only after P5 (NCCL
   diagnostics) is in **and** the sampler `drop_last` fix is merged
   in the same commit. Functional smoke uses the `short` config,
   not `smoke`. The sampler unit test goes via the
   factored-out helper; the real DistributedSampler path is
   exercised by the `torchrun --nproc_per_node=2` smoke from §"Tests
   / smoke checks" item 3.
6. **P7** (`non_blocking=True`). Small vendored Makani patch.
   **Last** in the order, gated on the
   `docs/2026-05-04_makani_local_patches.md` merge gate. Treat as
   plumbing: do not land it expecting a measurable wallclock win on
   its own, and do not block other items on it.
7. **P3** (single-write checkpoint). **Deferred**. Land only if P6
   shows it is worth the refactor risk, and only with golden
   key-set + bidirectional resume tests against the exact key set
   from `makani-src/makani/utils/driver.py:540-572`.

Each item except P3 is independently revertable. P1 has a hard
dependency on the sampler fix; both must land together.

---

## What this plan deliberately does **not** address

(Re-stated for the next reviewer's convenience.)

- We do not move normalization or the GridConverter to GPU.
- We do not precompute a Legendre-Gauss-grid dataset variant.
- We do not switch to DALI.
- We do not turn on `torch.compile` or `channels_last`.
- We do not enable model / spatial / tensor parallelism.
- We do not touch the loss function or rollout policy.
- We do not change the `.tar` checkpoint format or move to async/sharded
  checkpointing (P3 is deferred, not adopted).
- We do not adopt an MPI-based launcher.

Re-evaluate after the P6 profile run.
