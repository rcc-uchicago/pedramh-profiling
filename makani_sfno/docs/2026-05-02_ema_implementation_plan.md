# EMA implementation plan for SFNO emulator training

**Status:** Proposed v7 (Codex-review-resolved 2026-05-02). No code changed yet.
**Author:** Zhixing Liu / Claude (interview-resolved)
**Scope:** Add an Exponential Moving Average (EMA) of model weights to
`sfno_training/` as an **explicit experimental option**. EMA is **not**
inherited from the group-convention / PanguWeather-v2.0 baseline — those
configs do not specify EMA. We are adding it as a deliberate experiment
on top of the translated Makani-SFNO trainer.

This plan covers behavior, configuration, checkpointing, validation,
logging, and tests. It does not modify any code; it specifies what the
implementation PR should contain so it can be reviewed against intent.

### Revision history
- **v1 (2026-05-02)** — initial draft post-interview.
- **v2 (2026-05-02)** — addresses Codex review. Five red-risk fixes:
  (1) complex64 spectral weights handled in EMA shadow dtype;
  (2) EMA resume reads `self.checkpoint_version_current`, not `0`;
  (3) `best_ckpt_ema_mp0.tar`'s `model_state` is the FULL canonical
  state_dict so `strict=True` inference load works;
  (4) checkpoint includes `ema_config` (decay, warmup) and validates it
  on resume;
  (5) EMA validation pass runs on **all** ranks (barriers/reductions in
  `validate_one_epoch`); only the EMA-best file write is rank-gated.
  Plus four importants: viz suppressed on the EMA pass; `best_ema_loss`
  persisted across resume; `loss_state_dict` included in EMA-best
  (resolving an internal inconsistency); `save_checkpoint == "flexible"`
  hard-errored when EMA enabled (gathered/scattered EMA state deferred).
- **v3 (2026-05-02)** — addresses Codex follow-up clarifications.
  Four fixes: (1) Goal #1 wording corrected — shadow dtype is dtype-aware,
  not blanket fp32; (2) `update()` step indexing made explicit
  (increment-then-compute, so first decay is `2/11 ≈ 0.182`, not
  `1/10 = 0.1`); (3) §6 logging contract pinned to literal Makani-style
  `valid_logs["base"]` keys (e.g. `"validation loss ema"`) rather than
  slash-namespaced aliases that aren't actually emitted; (4) EMA-best
  path derivation made explicit about formatting `mp_rank=comm.get_rank("model")`
  on the modified template.
- **v4 (2026-05-02)** — addresses Codex tightening pass. Five fixes:
  (1) nested YParams access uses dict `.get` (e.g. `ema_cfg.get(
  "allow_config_change", False)`), not attribute access — YParams
  exposes attribute access only for top-level keys; (2) flexible-mode
  hard-error broadened to also reject `load_checkpoint == "flexible"`
  (per-rank EMA file restore would silently re-seed); (3) validation
  override now populates `ema decay effective`, `ema step`, `ema best
  loss` keys AFTER the EMA-best update so the logged best-loss is
  current; (4) §7.1 `EMAModel.load_state_dict` spec spelled out:
  in-place `shadow.copy_(incoming.to(device=shadow.device,
  dtype=shadow.dtype))` with strict key/shape/complex-vs-real
  checks (CPU-loaded ckpt tensors must land on the live shadow's
  device, never rebound); (5) §10 stale "log key naming" question
  removed since v3 resolved it.
- **v5 (2026-05-02)** — stale-wording cleanup pass. Five edits, no
  design changes: (1) §4.4 `params.ema` / `params.ema.allow_config_change`
  references replaced with the dict-access form (already prescribed
  in v4 but textually missed in this passage); (2) §7.2(a)
  `__init__` sample now rejects flexible mode on **both**
  `save_checkpoint` and `load_checkpoint` (was save-only); (3) §3.5
  bullet now says save OR load flexible mode; (4) §8 risk #7
  retitled "Flexible mode rejected on save AND load"; (5) §7.1
  `load_state_dict` validations reordered — complex-vs-real and
  shape checks now run on the **raw incoming tensor BEFORE** the
  `.to(device, dtype)` cast (a real → complex cast silently
  succeeds and would defeat the check). Plus minor: Goal #4
  updated to list all four embedded EMA keys.
- **v7 (2026-05-02)** — final cleanup pass for v6 logging-routing
  spillover. Two textual fixes, no design changes: (1) §3.4 "Merge
  logs" step 4 now routes EMA scalars into `valid_logs["metrics"]`
  (matching §6 / §7.2(d)) and drops the redundant
  `validation loss raw` alias; (2) §10 resolved-in-v3 note updated
  to also credit v6's routing decision and stop saying EMA scalars
  live in `["base"]`.
- **v6 (2026-05-02)** — addresses Codex's logging-routing finding.
  Stock `Trainer.log_epoch` does NOT mirror `valid_logs["base"]` to
  screen — only `["base"]["validation loss"]` is printed; everything
  else under `["base"]` is wandb-only (`deterministic_trainer.py:696-723
  vs 725-731`). Since `zgplev_full.yaml` ships with `log_to_wandb:
  false`, base-only EMA scalars would be invisible during the run we
  actually plan to babysit. **Routing fix**: §6 and §7.2(d) now route
  the four EMA scalars (`validation loss ema`, `ema decay effective`,
  `ema step`, `ema best loss`) and per-channel breakdowns into
  `valid_logs["metrics"]` so they print on screen AND log to wandb.
  Dropped the redundant `validation loss raw` alias. Plus two
  implementation nits: (1) the per-epoch `torch.load` rewrite path
  in §7.2(c) uses explicit `weights_only=False, map_location="cpu"`
  to match stock's load semantics and silence the PyTorch ≥ 2.4
  default-flip warning; (2) integration tests in §7.5 now cover
  BOTH `save_checkpoint="flexible"` and `load_checkpoint="flexible"`
  reject paths separately, and a new test asserts the §6
  `["metrics"]` vs `["base"]` routing contract.

---

## 1. Goals & non-goals

### Goals
1. Maintain a per-rank EMA shadow of `self.model`'s **trainable parameters**,
   updated after every successful optimizer step (post-`gscaler.step`). Shadow
   dtype is dtype-aware: `complex64`/`complex128` for complex spectral weights
   so imaginary components are preserved, `float32` for half-precision real
   parameters (fp16/bf16) for accumulation precision, and the parameter's own
   dtype otherwise. Full rules in §3.2.
2. Run validation **twice** at end of each epoch — once with raw weights,
   once with EMA weights — and log both losses.
3. Maintain **two best checkpoints** with disjoint semantics:
   - `best_ckpt_mp0.tar` — selected by **raw** val loss (existing semantics
     preserved exactly; `model_state` = raw weights).
   - `best_ckpt_ema_mp0.tar` — selected by **EMA** val loss; `model_state`
     = EMA snapshot. Drop-in compatible with the existing
     `sfno_inference/checkpoint_loader.py` (no inference-side change).
4. Embed `ema_state`, `ema_step`, `ema_config`, and `ema_best_loss` in
   **per-epoch** checkpoints so that resume is exact (full key list
   and semantics in §4.1).
5. Default-off knob; existing runs (tiny/smoke/short) unaffected.
6. Enable EMA on `plasim_sim52_zgplev_full.yaml` and
   `plasim_sim52_zgplev_baseline.yaml` only.
7. EMA checkpointing scoped to legacy mode. Hard-error at `__init__`
   if EMA is enabled and **either** `params.save_checkpoint == "flexible"`
   **or** `params.load_checkpoint == "flexible"`. The latter matters
   because `_maybe_restore_ema_state()` reads the per-rank legacy
   file (`ckpt_mp{mp_rank}_v{version}.tar`); if the run was started
   in flexible-load mode, no per-rank EMA file exists to restore
   from, and the per-rank EMA shadow on each MP rank would be
   silently re-seeded from the freshly-loaded (already-flexibly-
   gathered) model. Flexible support requires gathered/scattered
   EMA tensors mirroring `gather_model_state_dict` /
   `scatter_model_state_dict`; deferred.

### Non-goals
- No change to `sfno_inference/checkpoint_loader.py`. For EMA inference,
  callers point the existing loader at `best_ckpt_ema_mp0.tar`. An
  `--ema` flag on the loader can come later if EMA proves useful.
- No multi-EMA (Karras-style multiple decays in parallel).
- No EMA for the loss module, optimizer state, or scheduler state.
- No average over persistent buffers in this PR (see §3.2 for rationale).
- No `flexible`-mode EMA save/restore (see goal #7).

---

## 2. Assumptions baked in

Verified against current code on `zgplev-migration-dsi-bootstrap` (2026-05-02):

- `PlasimTrainer` extends stock `makani.utils.training.deterministic_trainer.Trainer`,
  which extends `Driver`. `Trainer` holds exactly one model handle:
  `self.model` (also bound as `self.model_train` and `self.model_eval`;
  validation uses the same module with `.eval()` flag).
- `normalization_layer: "instance_norm"` ⇒ no BatchNorm running stats.
- `sfno_training/models/preprocessor.py` has **no** `nn.Parameter` and **no**
  `register_buffer` calls — preprocessor is not in EMA scope.
- Stock checkpoint format (`Driver._save_checkpoint_legacy`,
  `makani-src/makani/utils/driver.py:515`) writes:
  `model_state`, `optimizer_state_dict`, `scheduler_state_dict`,
  `loss_state_dict`, `iters`, `epoch`, `comm_grid`. Inference reads
  `model_state` only.
- Per-epoch save site: `deterministic_trainer.py:399`. Best-ckpt save
  site: `deterministic_trainer.py:404-406`, gated on
  `valid_logs["base"]["validation loss"] <= best_valid_loss`.
- `gscaler.step(optimizer)` may **skip** the optimizer step on inf/nan
  gradients. EMA must not update on skipped steps.
- DDP via custom `init_gradient_reduction_hooks` keeps params bcast-identical.
  Model parallel possibly enabled (`sharded_dims_mp` attached at save in
  `driver.py:540-543`). `torch.compile` may wrap the model
  (`_orig_mod.` prefix stripped at save time, `driver.py:529-533`).

---

## 3. Behavior specification

### 3.1 Decay schedule

- **Default decay:** `0.999`.
- **Warmup (default on):** effective decay at step `t` (1-indexed by
  EMA-update count, i.e. successful optimizer steps since EMA enabled):

  ```
  decay_t = min(decay_max, (1 + t) / (10 + t))
  ```

  Yields `decay_1 ≈ 0.182`, `decay_10 ≈ 0.55`, `decay_100 ≈ 0.918`,
  `decay_1000 ≈ 0.991`, asymptote = `decay_max = 0.999`. Karras-style
  convention; gives meaningful EMA behavior at the start of training
  rather than ~initial weights for many epochs.
- **Warmup off:** `decay_t = decay_max` from `t = 1`.

### 3.2 What gets averaged

- **Trainable parameters only.** Iterate `model.named_parameters()` and
  shadow every `p` with `p.requires_grad`.

- **Shadow dtype** depends on the parameter dtype:
  - **Complex parameters** (`p.is_complex()` ⇒ `complex64` /
    `complex128`): the SFNO spectral convolution weights are
    `complex64` (`makani-src/makani/models/common/spectral_convolution.py:90,93,94`).
    Shadow is **`torch.complex64`** (or `complex128` if the source is
    `complex128`). Casting these to fp32 would silently drop the
    imaginary component — must not happen.
  - **Half-precision real parameters** (`float16` / `bfloat16`):
    shadow is **`torch.float32`** for accumulation precision (the
    classic reason for an fp32 EMA shadow under AMP).
  - **Other real parameters** (`float32` / `float64`): shadow matches
    the parameter dtype.

- **Update math** (works uniformly across real and complex shadows
  since `decay_t` is a Python float and complex × float is well-defined):
  ```python
  shadow.mul_(decay_t).add_(p.detach().to(shadow.dtype),
                             alpha=(1.0 - decay_t))
  ```

- **Persistent buffers excluded** in this PR. Rationale: with
  `instance_norm` we have no running stats, and SFNO's other
  registered buffers are precomputed transforms (Legendre tables,
  spectral filters) that do not drift during training; averaging
  them would be a no-op. If a future model swap reintroduces buffers
  that change during training (BatchNorm, EMA-style buffers in some
  attention variants), revisit this decision; the EMA module is
  structured so adding `named_buffers()` is a one-line change.

- **Implication for EMA-best `model_state`** (see §4.3): because
  buffers are NOT shadowed but ARE present in the live model's
  state_dict, the EMA-best `model_state` is built by starting from
  the **live** `model.state_dict()` (canonical, prefix-stripped) and
  substituting EMA values only for tracked-trainable-parameter keys.
  Buffers and any non-trainable parameters in the state_dict are
  copied through unchanged. This guarantees `strict=True` load
  succeeds in `Driver.restore_from_checkpoint`.

### 3.3 Update cadence

- Update **after** `self.gscaler.step(self.optimizer)` and only when the
  step actually executed (i.e. not skipped by `GradScaler` due to inf/nan).
- Hook mechanism: register a post-step hook on the optimizer via
  `optimizer.register_step_post_hook`. This hook fires only on actual
  `optimizer.step()` invocations — `GradScaler.step()` calls into
  `optimizer.step()` only on non-skipped batches, so the hook fires
  correctly. This avoids overriding `train_one_epoch` wholesale.
- Hook reads current `model.named_parameters()` and updates the EMA
  in-place.
- **Step indexing — increment-then-compute.** The first update must
  observe `t = 1`, not `t = 0` (the warmup formula at `t = 0` gives
  `1/10 = 0.1`, while the intended first-step decay is
  `2/11 ≈ 0.182`). The `update()` implementation increments
  `self._step` to its post-update value first, then computes
  `decay_t` from that value:
  ```python
  def update(self, model):
      with torch.no_grad():
          self._step += 1                       # post-update count, t >= 1
          d = self.decay_t                       # uses self._step
          for name, p in model.named_parameters():
              if not p.requires_grad:
                  continue
              shadow = self._shadow[strip(name)]
              shadow.mul_(d).add_(p.detach().to(shadow.dtype),
                                   alpha=(1.0 - d))
  ```
  Equivalent: compute `decay_t` from `self._step + 1` and increment
  after. Either is fine; pick one and document it in the
  `EMAModel.update` docstring.

### 3.4 Validation

Override `validate_one_epoch` in `PlasimTrainer`:

1. Run `super().validate_one_epoch(epoch, profiler)` on **all ranks**
   → `(valid_time_raw, viz_time_raw, valid_logs_raw)`.
2. If EMA disabled, return as-is.
3. **EMA pass — runs on all ranks** (`validate_one_epoch` contains
   `dist.barrier` at `deterministic_trainer.py:588, 681` and metric
   all-reduces inside `metrics.update`/`finalize`; running it on
   only a subset of ranks would deadlock):
   a. Suppress visualization on the EMA pass: stash `params.log_video`,
      set it to `0` so the `visualize_data` predicate at
      `deterministic_trainer.py:593` evaluates `False`. (Restore in
      a `try/finally` so any exception in the EMA pass doesn't
      leave the param mutated.)
   b. All ranks: enter `self.ema.applied_to(self.model)` context
      manager — snapshot live params, copy EMA shadow into
      `model.parameters()`, yield. The same context manager
      restores live params on exit.
   c. All ranks: call `super().validate_one_epoch(epoch, profiler=None)`.
   d. Restore `params.log_video`.
4. Merge logs (routing per §6 — EMA scalars go in `["metrics"]` so
   they print on screen even when `log_to_wandb=False`; `["base"]` is
   wandb-only on screen except for the literal `"validation loss"` key):
   - `valid_logs["base"]["validation loss"]` ← raw (unchanged — keeps
     the stock raw-best save flow at `deterministic_trainer.py:404-406`
     pointing at raw weights).
   - `valid_logs["metrics"]["validation loss ema"]` ← EMA val loss.
   - All other `valid_logs_ema["metrics"]` entries copied into
     `valid_logs["metrics"]` under suffixed keys (e.g. `"<name> ema"`).
   - The remaining three EMA scalars (`ema decay effective`,
     `ema step`, `ema best loss`) are written into
     `valid_logs["metrics"]` AFTER the rank-gated EMA-best write in
     step 5, so that `ema best loss` reflects the just-updated value
     when an improvement happened this epoch.
5. **Rank-gated EMA-best write**: only on `data_parallel_rank == 0`,
   if EMA val loss strictly improves over `self.best_ema_loss`:
   write `best_ckpt_ema_mp0.tar` (see §4.3) and update
   `self.best_ema_loss`. Other ranks do nothing here.

Notes:
- All ranks compute identical EMA shadows (deterministic from
  bcast-identical params), so the swap-and-restore yields consistent
  results across ranks without any extra communication.
- `eval` dataloader has `shuffle=False` (enforced in
  `plasim_trainer.py:153`), so the second pass sees the same data
  as the first. Validation wall-time roughly doubles end-of-epoch.
  On `full`, this is small relative to training time per epoch.
- `self.metrics.zero_buffers()` is called at the top of stock
  `validate_one_epoch` (`deterministic_trainer.py:591`), so the
  second pass starts with a clean metrics state.

### 3.5 Distributed / compile correctness

- **DDP:** all ranks have bcast-identical params, run identical EMA
  updates ⇒ identical EMA shadows. **No extra all-reduce.**
- **Model parallel** (`comm.get_size("model") > 1`): each rank holds
  its shard of params; EMA shadow is per-rank shard. At save time
  for `best_ckpt_ema_mp0.tar`, copy `sharded_dims_mp` from the live
  parameter onto the EMA tensor (mirror `driver.py:540-543`).
  **Hard-error if `params.save_checkpoint == "flexible"` OR
  `params.load_checkpoint == "flexible"`** with EMA enabled (see
  goal #7); flexible mode requires gather/scatter of EMA shadows
  on save, and the per-rank EMA file restore in
  `_maybe_restore_ema_state` cannot recover from a flexibly-loaded
  init. Both deferred to a follow-up.
- **`torch.compile`:** Strip the `_orig_mod.` prefix when iterating
  named_parameters for shadow construction so EMA keys match
  canonical (prefix-stripped) state-dict keys, exactly as
  `_save_checkpoint_legacy` does (`driver.py:529-533`).

---

## 4. Checkpointing (Layout C, legacy-mode only)

Three checkpoint files live in `training_checkpoints/` per run.

### 4.1 Per-epoch — `ckpt_mp{mp_rank}_v{checkpoint_version}.tar`
Existing keys + four new (all under `ema.enabled` only):
```
+ ema_state     : OrderedDict[name -> tensor]    # EMA shadow; dtype per §3.2
+ ema_step      : int                             # update count for warmup
+ ema_config    : {"decay": float, "warmup": bool, "version": 1}
+ ema_best_loss : float                           # self.best_ema_loss at save time
```
Used for **resume**. See §4.4 for the resume contract and ema_config
validation.

### 4.2 Raw best — `best_ckpt_mp{mp_rank}.tar`
**Semantics unchanged.** `model_state` = raw weights, selected by raw
val loss as today. We additionally write `ema_state`, `ema_step`,
`ema_config`, and `ema_best_loss` for symmetry / convenience — no
code reads these from the raw best file, so this is a free addition
that does not change the meaning of the file.

### 4.3 EMA best — `best_ckpt_ema_mp{mp_rank}.tar` (NEW)
Drop-in compatible with `sfno_inference/checkpoint_loader.py`, which
calls `Driver.restore_from_checkpoint` with `strict=True`
(`driver.py:356`, `checkpoint_loader.py:212`). The `model_state` MUST
therefore contain the FULL canonical state_dict — buffers and any
non-trainable params present in the live model — with EMA values
substituted only where we have a shadow.

```
model_state           : full canonical state_dict; for each tracked
                        trainable param, its value is the EMA shadow;
                        for buffers / untracked params, the live value
                        is copied through unchanged (see §3.2 final
                        bullet).
loss_state_dict       : self.loss_obj.state_dict()  (matches raw best)
iters, epoch          : counters at save time
comm_grid             : rebuilt from comm.get_model_comm_names()
# NO ema_state / ema_step / ema_config / ema_best_loss — EMA weights
# ARE model_state here; the shadow itself is redundant in this file.
# NO optimizer_state_dict / scheduler_state_dict — not needed for
# inference; resume always comes from per-epoch ckpt, not from a best
# file.
```
Written when EMA val loss improves (§3.4 step 5). `sharded_dims_mp`
attached on each tensor as in `driver.py:540-543`.

### 4.4 Resume contract

`Trainer.__init__` restores from per-epoch checkpoint via
`restore_from_checkpoint` at `deterministic_trainer.py:278-287` after
calling `get_latest_checkpoint_version` at line 270 to populate
`self.checkpoint_version_current`. We extend this with EMA restore.

After `super().__init__()` returns:

1. Construct `self.ema` if EMA enabled — shadow seeded from live
   `self.model.named_parameters()`. (Live model already holds the
   restored raw weights at this point.)
2. Call `_maybe_restore_ema_state()`:
   - Path: format `params.checkpoint_path` with
     `checkpoint_version=self.checkpoint_version_current` and
     `mp_rank=comm.get_rank("model")`. **This is the key fix vs v1**
     (which read `version=0` and would silently pair latest raw
     weights with stale EMA shadow after rotation.)
   - If `not self.params.resuming` ⇒ skip (no per-epoch ckpt to read).
   - `torch.load` the file (CPU map).
   - **Validate `ema_config`** against the current `ema_cfg` (read
     via `ema_cfg = self.params.get("ema", {}) or {}` — YParams
     attribute access is top-level only; nested blocks remain
     dict-like, so `self.params.ema.allow_config_change` would
     raise `AttributeError`):
     - `decay`, `warmup` — if they differ, raise
       `RuntimeError` unless `ema_cfg.get("allow_config_change",
       False)` is True. Message names the keys that differ.
     - `version` — currently `1`; raise on mismatch (forward-compat
       hook).
   - On hit: `self.ema.load_state_dict(ema_state)`,
     `self.ema.step = ema_step`,
     `self.best_ema_loss = ema_best_loss`.
   - On any missing key: log a `WARNING` naming the missing keys,
     keep the freshly-seeded shadow (effective re-warmup), and
     leave `self.best_ema_loss = +inf`. This is the
     pre-EMA-checkpoint back-compat path.
3. **`self.best_ema_loss` IS persisted** across resumes (this is the
   v1 → v2 fix). Without persistence, the first post-resume epoch
   could overwrite a genuinely better `best_ckpt_ema_mp0.tar` with a
   worse weight snapshot. Note: stock's `best_valid_loss` is still
   reset to `1.0e6` at line 350 — that's an upstream bug we don't
   touch, but we don't replicate it for EMA.

---

## 5. Configuration surface

New nested block in YAML, default-off:

```yaml
ema:
  enabled: true                # bool; when False, all of EMA is a no-op
  decay: 0.999                 # float in (0, 1)
  warmup: true                 # bool; when False, decay_t = decay always
  allow_config_change: false   # bool; if True, mismatched ema_config on
                               # resume is logged as warning instead of raising.
                               # Use only when intentionally changing decay or
                               # warmup mid-run; the EMA shadow may carry
                               # stale dynamics after the change.
```

Read via `params.get("ema", {})` to keep back-compat with configs that
don't have the block (treated as disabled).

**Config rollout:**
- `plasim_sim52_zgplev_full.yaml`     — **enable** with defaults above.
- `plasim_sim52_zgplev_baseline.yaml` — **enable** with defaults above.
- `plasim_sim52_zgplev_short.yaml`    — leave EMA disabled.
- `plasim_sim52_zgplev_smoke.yaml`    — leave EMA disabled.
- `plasim_sim52_zgplev_tiny.yaml`     — leave EMA disabled.
- Non-zgplev configs (`plasim_sim52_*.yaml`) — leave EMA disabled (legacy paths).

Rationale for tiny/smoke/short staying off: too brief for EMA to be
meaningful, doubling validation wall time hurts smoke iteration speed,
and we don't want EMA logs cluttering smoke comparisons with
pre-EMA baselines.

---

## 6. Logging

Stock `Trainer.log_epoch` (`makani-src/makani/utils/training/deterministic_trainer.py:688`)
treats the two sub-dicts of `valid_logs` differently:

- **Screen** (`log_to_screen`, lines 696-723): prints exactly one
  base entry — `valid_logs["base"]["validation loss"]` (line 718) —
  plus iterates `valid_logs["metrics"]` and prints scalar entries
  (lines 719-722). Other keys in `valid_logs["base"]` are silently
  dropped on screen.
- **Wandb** (`log_to_wandb`, lines 725-731): logs the entire
  `valid_logs["base"]` and `valid_logs["metrics"]` dicts.

Since `plasim_sim52_zgplev_full.yaml:151` sets `log_to_wandb: false`
for the initial launch, EMA scalars placed only in `["base"]` would
be invisible on screen — exactly the run we plan to babysit.

**Routing decision: EMA scalars go in `valid_logs["metrics"]`** (screen
+ wandb). The EMA-pass per-step/per-channel breakdowns also live
there with `" ema"` suffix. The `["base"]["validation loss"]` key
stays raw-only (it drives the stock raw-best save at
`deterministic_trainer.py:404-406` — must not be overwritten).

| valid_logs key                                  | Set by                          | Visibility |
|-------------------------------------------------|---------------------------------|------------|
| `valid_logs["base"]["validation loss"]`              | stock raw pass (unchanged)  | screen + wandb (drives raw-best save) |
| `valid_logs["metrics"]["validation loss ema"]`       | EMA override                | screen + wandb |
| `valid_logs["metrics"]["ema decay effective"]`       | EMA override                | screen + wandb |
| `valid_logs["metrics"]["ema step"]`                  | EMA override                | screen + wandb |
| `valid_logs["metrics"]["ema best loss"]`             | EMA override                | screen + wandb |
| `valid_logs["metrics"]["<name> ema"]`                | EMA override                | screen + wandb (per-channel breakdowns from `valid_logs_ema["metrics"]`, suffixed) |

Notes:
- `validation loss raw` alias dropped — redundant with the existing
  `["base"]["validation loss"]` line at `deterministic_trainer.py:718`.
- All EMA-named keys are added only when `self.ema_enabled` is True;
  existing runs see `valid_logs` unchanged.
- Screen prints only **scalar** `["metrics"]` entries (line 721:
  `if np.isscalar(value)`). The four EMA scalars qualify; per-channel
  breakdowns mirrored under `"<name> ema"` follow whatever scalar/
  non-scalar shape the source had — same visibility as the un-suffixed
  raw key.
- Stock has an off-by-one in `log_epoch` (line 719 starts iteration
  at index 3 of `print_list`, skipping the first metric); this is
  upstream behavior we do not patch. A simple defense is to ensure
  our four headline EMA scalars are ordered such that one
  pre-existing metric precedes them in dict insertion order.

If a future dashboard wants slash-namespaced wandb keys (`val/loss_ema`,
`ema/step`, …), that requires either modifying `log_epoch` in stock
(out of scope) or post-processing in the wandb sink (also out of scope
for this PR). The plan ships Makani-style spaced names only.

---

## 7. File-by-file change list

### 7.1 NEW: `src/sfno_training/trainer/ema.py` (~150 lines)

A small, self-contained EMA helper. Public surface:

```python
class EMAModel:
    def __init__(self, model, *, decay=0.999, warmup=True): ...
    @property
    def decay_t(self) -> float: ...
    def update(self, model) -> None: ...           # fires from optimizer post-step hook
    def state_dict(self) -> "OrderedDict": ...     # canonical (prefix-stripped) keys
    def load_state_dict(self, state_dict, *, strict=True) -> None: ...
    @property
    def step(self) -> int: ...
    @step.setter
    def step(self, value: int) -> None: ...

    # Validation swap helpers — all-rank, in-place, snapshot-based
    @contextmanager
    def applied_to(self, model):                   # snapshot, swap-in, yield, restore
        ...

    # For best_ckpt_ema_mp0.tar emission
    def export_model_state(self, model) -> "OrderedDict":
        """Return a FULL canonical state_dict (model.state_dict() with
        prefix stripped) where tracked-trainable-parameter values are
        replaced with the EMA shadow (cast to the parameter's dtype).
        Buffers and untracked params are copied through. sharded_dims_mp
        is attached on every tensor that has it on the live module."""
        ...
```

Implementation notes:
- Iterate `model.named_parameters()`, strip wrapper prefixes via
  `get_model_state_dict_prefix(model)` (already used by
  `driver.py:531`); shadow only `requires_grad=True` tensors on the
  same device as the parameter (same shard for MP).
- **Shadow dtype** (per §3.2):
  ```python
  if p.is_complex():
      shadow_dtype = p.dtype                    # complex64 / complex128
  elif p.dtype in (torch.float16, torch.bfloat16):
      shadow_dtype = torch.float32
  else:
      shadow_dtype = p.dtype                    # float32 / float64
  ```
- `update` runs under `torch.no_grad()`. Single line works for real
  and complex shadows uniformly (since `decay_t` is a Python float):
  ```python
  shadow.mul_(decay_t).add_(p.detach().to(shadow.dtype),
                             alpha=(1.0 - decay_t))
  ```
- `applied_to` snapshots live params (in their native dtype, on their
  native device — no CPU↔GPU transfer), copies `shadow.to(p.dtype)`
  in-place via `p.data.copy_(...)`, yields, then restores the snapshot
  in a `finally` block. Memory cost during validation: 1× model
  parameters (snapshot) + 0 (shadow already exists). Released on
  context exit.
- **`load_state_dict(state_dict, *, strict=True)`** semantics:
  - Inputs arrive from `torch.load(..., map_location="cpu")` so
    incoming tensors are on CPU. Existing shadows live on the model's
    device (matching the live parameter's shard).
  - **Strict key check** (when `strict=True`, default): the set of
    keys in `state_dict` must exactly equal the set of keys in
    `self._shadow`. On mismatch, raise `RuntimeError` listing both
    `missing_keys` and `unexpected_keys` (mirror PyTorch's
    `load_state_dict` error format). With `strict=False`, log a
    WARNING for each missing/unexpected key and copy whatever
    overlaps.
  - Then iterate the matching keys and **copy in-place**, with all
    validations on the **raw incoming tensor** BEFORE any `.to(...)`
    cast (a real → complex cast silently produces a complex tensor
    with zero imaginary parts and would defeat the check):
    ```python
    raw = state_dict[name]   # CPU, original dtype from the ckpt

    # Complex-vs-real check on the RAW tensor — must precede the cast.
    if raw.is_complex() != shadow.is_complex():
        raise RuntimeError(
            f"EMA load_state_dict: complex/real mismatch for {name!r}: "
            f"incoming dtype={raw.dtype}, shadow dtype={shadow.dtype}. "
            f"Refusing to cast (would corrupt complex spectral weights)."
        )

    # Shape check on the RAW tensor — cast can't change shape, but we
    # want a clean error before paying the device transfer.
    if raw.shape != shadow.shape:
        raise RuntimeError(
            f"EMA load_state_dict: shape mismatch for {name!r}: "
            f"got {tuple(raw.shape)}, expected {tuple(shadow.shape)}"
        )

    # Now safe to cast to the shadow's device/dtype and copy in-place.
    incoming = raw.to(device=shadow.device, dtype=shadow.dtype,
                       non_blocking=False)
    shadow.copy_(incoming)
    ```
    In-place copy preserves the shadow's storage and device; never
    rebind `self._shadow[name]` to the incoming tensor (would land
    the shadow on CPU and break subsequent updates).
- `export_model_state(model)`:
  1. `state_dict = model.state_dict()` then strip prefix via
     `consume_prefix_in_state_dict_if_present`.
  2. For each `(name, p)` in `model.named_parameters()` whose
     prefix-stripped name has a corresponding EMA shadow: replace
     `state_dict[stripped_name]` with `shadow.to(p.dtype).clone()`.
  3. For `(name, p)` in `model.named_parameters()` and
     `(name, b)` in `model.named_buffers()`: if `hasattr(..., "sharded_dims_mp")`
     and the prefix-stripped name is in `state_dict`, attach it.
     Mirrors `driver.py:540-543`.
  4. Return.

### 7.2 MODIFY: `src/sfno_training/trainer/plasim_trainer.py` (~+80 lines)

Three additions to `PlasimTrainer`, all surgical:

**(a) `__init__`** — after `super().__init__(...)`:
```python
ema_cfg = params.get("ema", {}) if params is not None else {}
self.ema_enabled = bool(ema_cfg.get("enabled", False))
self.ema = None
self.best_ema_loss = float("inf")
if self.ema_enabled:
    # Hard-error on flexible mode in EITHER direction (see §3.5, goal #7).
    # Save side: flexible-save would gather state across MP ranks but our
    #            EMA write path appends per-rank ema_* keys to per-rank
    #            files; mismatch.
    # Load side: _maybe_restore_ema_state reads per-rank legacy files; if
    #            the run was started with flexible-load, those per-rank
    #            EMA files don't exist for the current MP topology.
    flex_save = (self.params.save_checkpoint == "flexible")
    flex_load = (self.params.load_checkpoint == "flexible")
    if flex_save or flex_load:
        raise NotImplementedError(
            "ema.enabled=True is currently scoped to legacy save AND load "
            f"(got save_checkpoint={self.params.save_checkpoint!r}, "
            f"load_checkpoint={self.params.load_checkpoint!r}). "
            "Flexible-mode EMA support requires gather/scatter of EMA "
            "shadows on save and a flex-aware EMA restore path; deferred "
            "to a follow-up. Either switch both to 'legacy' or set "
            "ema.enabled: false."
        )
    self.ema = EMAModel(
        self.model,
        decay=float(ema_cfg.get("decay", 0.999)),
        warmup=bool(ema_cfg.get("warmup", True)),
    )
    self.optimizer.register_step_post_hook(
        lambda *_args, **_kwargs: self.ema.update(self.model)
    )
    # If we just resumed, super().__init__ already loaded model_state and
    # set self.checkpoint_version_current. Now load ema_* from that same
    # checkpoint version.
    self._maybe_restore_ema_state()
```

**(b) `_maybe_restore_ema_state`** (new private method):
- If `not self.params.resuming`, return immediately (fresh-start; live
  params are the EMA seed).
- Format the checkpoint path using **`self.checkpoint_version_current`**
  (set by stock `__init__` via `get_latest_checkpoint_version`,
  `deterministic_trainer.py:270`) and `mp_rank=comm.get_rank("model")`.
  This is the v1 → v2 fix — v1 hard-coded `version=0` and would have
  silently paired latest raw weights with stale EMA shadow after
  rotation.
- `torch.load(path, map_location="cpu", weights_only=False)`. Tensors
  arrive on CPU; `EMAModel.load_state_dict` is responsible for moving
  them onto the live shadow's device — see §7.1 spec note below.
- **Read `ema_cfg` via dict access**, NOT attribute access:
  ```python
  ema_cfg = self.params.get("ema", {}) or {}
  allow_config_change = bool(ema_cfg.get("allow_config_change", False))
  ```
  YParams gives attribute access only for top-level keys; nested YAML
  blocks remain dict-like. `self.params.ema.allow_config_change`
  would raise `AttributeError`.
- **Validate `ema_config`** against current `ema_cfg`:
  - If `ema_config` key absent ⇒ log WARNING, treat as missing
    (leave fresh seed; see below).
  - Else compare `decay`, `warmup`, `version`. On any mismatch:
    - If `allow_config_change` is True: log WARNING naming the
      differing keys, accept the loaded shadow.
    - Else: raise `RuntimeError` with a message that names the
      differing keys and tells the user to either revert the
      config change or set `ema.allow_config_change: true` in YAML.
- For each of `ema_state`, `ema_step`, `ema_best_loss`: if present,
  load (`self.ema.load_state_dict(...)`, `self.ema.step = ...`,
  `self.best_ema_loss = ...`); if absent, log WARNING naming the
  missing key. Missing `ema_state` ⇒ keep fresh seed
  (effective re-warmup); missing `ema_best_loss` ⇒ leave at +inf
  (tradeoff: first post-resume epoch may overwrite a better EMA
  best, but the missing-key path only fires for pre-EMA ckpts
  anyway).

**(c) `save_checkpoint`** (override the inherited static method as a
bound method on `PlasimTrainer`):
- **Reject flexible mode early**: if `checkpoint_mode == "flexible"`
  and `self.ema_enabled`, raise `NotImplementedError` (defense in
  depth; the `__init__` guard should already have prevented this).
- Always call `Driver.save_checkpoint(...)` first to write the stock
  legacy file.
- If `self.ema_enabled` AND `self.data_parallel_rank == 0`:
  open the just-written file with
  `torch.load(checkpoint_fname, map_location="cpu", weights_only=False)`
  (the explicit `weights_only=False` matches stock's load path at
  `driver.py:388, 453` and silences PyTorch ≥ 2.4's default-flip
  warning), set:
  ```
  ckpt["ema_state"]     = self.ema.state_dict()
  ckpt["ema_step"]      = self.ema.step
  ckpt["ema_config"]    = {"decay": self.ema.decay_max,
                            "warmup": self.ema.warmup,
                            "version": 1}
  ckpt["ema_best_loss"] = self.best_ema_loss
  ```
  then `torch.save` back. This adds the four keys to **both**
  per-epoch and raw-best files (raw best is harmless: nothing
  reads `ema_state` from there, and exact resume from raw best is
  now possible if ever needed).
- Read-then-rewrite costs ~O(model size) extra disk I/O at end of
  each epoch. Acceptable trade-off for not duplicating
  `_save_checkpoint_legacy` internals. Per-rank concern: only data-
  parallel rank 0 within the model-parallel rank performs this
  rewrite, but since `Driver._save_checkpoint_legacy` already
  writes one file per `mp_rank`, we apply the rewrite for
  whichever `mp_rank` this rank owns (i.e. `comm.get_rank("model")`).

**(d) `validate_one_epoch`** (override):
```python
def validate_one_epoch(self, epoch, profiler=None):
    # Pass 1: raw — runs on all ranks, viz follows params.log_video as today.
    raw = super().validate_one_epoch(epoch, profiler=profiler)
    valid_time, viz_time, valid_logs = raw

    if not self.ema_enabled:
        return valid_time, viz_time, valid_logs

    # Pass 2: EMA weights — runs on ALL ranks (validate_one_epoch has
    # dist.barrier and metric all-reduces; subset-rank execution would
    # deadlock). Viz suppressed by stashing log_video.
    ema_t0 = time.perf_counter()
    saved_log_video = self.params.get("log_video", 0)
    try:
        self.params["log_video"] = 0  # falsifies visualize_data predicate
                                       # at deterministic_trainer.py:593
        with self.ema.applied_to(self.model):
            _, _, valid_logs_ema = super().validate_one_epoch(
                epoch, profiler=None
            )
    finally:
        self.params["log_video"] = saved_log_video
    ema_t = time.perf_counter() - ema_t0

    # Merge: raw stays under valid_logs["base"] (preserves stock raw-best
    # save flow at deterministic_trainer.py:404-406). EMA scalars go in
    # valid_logs["metrics"] so they are visible on screen even when
    # log_to_wandb=False — see §6.
    ema_loss = valid_logs_ema["base"]["validation loss"]
    metrics = valid_logs.setdefault("metrics", {})
    metrics["validation loss ema"] = ema_loss
    for k, v in valid_logs_ema.get("metrics", {}).items():
        metrics[f"{k} ema"] = v

    # Rank-gated EMA-best write — file write only on data_parallel_rank 0,
    # but EMA validation itself ran on all ranks above.
    if (self.data_parallel_rank == 0) and (ema_loss < self.best_ema_loss):
        self._save_best_ema_checkpoint(epoch)
        self.best_ema_loss = ema_loss

    # Populate the remaining §6 metrics AFTER the EMA-best update so that
    # "ema best loss" reflects the just-written value when an improvement
    # happened this epoch.
    metrics["ema decay effective"] = float(self.ema.decay_t)
    metrics["ema step"] = int(self.ema.step)
    metrics["ema best loss"] = float(self.best_ema_loss)

    valid_time = valid_time + ema_t
    return valid_time, viz_time, valid_logs
```

**(e) `_save_best_ema_checkpoint`** (new private method):
- Compute path. `params.best_checkpoint_path` is set in
  `train_plasim.py:166-168` to the un-formatted template
  `".../best_ckpt_mp{mp_rank}.tar"`. Two-step derivation:
  1. Insert `_ema` before the `_mp{mp_rank}` placeholder, yielding
     `".../best_ckpt_ema_mp{mp_rank}.tar"` (a string `.replace` on the
     raw template, NOT on a formatted instance).
  2. Format with `mp_rank=comm.get_rank("model")` to produce the
     final on-disk path (e.g. `".../best_ckpt_ema_mp0.tar"`).
  Note this mirrors the formatting stock does at
  `deterministic_trainer.py:402` for the raw best path. Skipping
  step 2 would write a literal `{mp_rank}` filename — easy to miss.
- Build `store_dict` matching `_save_checkpoint_legacy`'s structure
  so that `Driver.restore_from_checkpoint(..., strict=True)` (the
  call used by `sfno_inference/checkpoint_loader.py:212`) succeeds:
  - `model_state` ← `self.ema.export_model_state(self.model)` —
    the FULL canonical state_dict with EMA values substituted only
    where we have a shadow (per §3.2 final bullet and §7.1
    `export_model_state` spec). `sharded_dims_mp` attached.
  - `loss_state_dict` ← `self.loss_obj.state_dict()` (matches the
    raw best file; v1 → v2 fix — earlier draft inconsistently
    excluded this).
  - `comm_grid`   ← rebuilt from `comm.get_model_comm_names()` /
    `comm.get_rank` / `comm.get_size` (mirror `driver.py:548-555`).
  - `iters`       ← `self.iters`.
  - `epoch`       ← `self.epoch`.
- `torch.save(store_dict, path)`. **Do not** include
  `optimizer_state_dict`, `scheduler_state_dict`, `ema_state`,
  `ema_step`, `ema_config`, or `ema_best_loss` — this file is for
  inference, not resume; resume always comes from the per-epoch ckpt.
- Log a single info line: `"saved EMA-best checkpoint @ epoch {N},
  val/loss_ema={:.4e}"`.

### 7.3 MODIFY: `src/sfno_training/trainer/__init__.py`
Export `EMAModel` for testing convenience.

### 7.4 MODIFY: configs
- `src/sfno_training/config/plasim_sim52_zgplev_full.yaml`
- `src/sfno_training/config/plasim_sim52_zgplev_baseline.yaml`

Append the `ema:` block from §5.

### 7.5 NEW: tests
- `tests/sfno_training/test_ema.py` (unit, pure CPU):
  - `test_decay_warmup_curve`: warmup-on, assert `decay_t` matches
    closed form at t = 1, 10, 100, 1000.
  - `test_decay_no_warmup`: warmup-off, assert `decay_t == decay_max`
    from t=1.
  - `test_update_math_real`: 1-param toy module (real fp32), run 3
    manual updates with fixed param values, compare shadow against
    hand-computed result.
  - **`test_update_math_complex`**: toy module with `nn.Parameter(...,
    dtype=torch.complex64)`, run updates, assert both real and
    imaginary components are averaged correctly. Critical regression
    guard against dropping imaginary parts on `.float()` (the v1 bug).
  - `test_dtype_invariant`: real param in fp16/bf16, shadow stays
    fp32; complex64 param ⇒ shadow stays complex64. Update precision
    matches fp32 / complex64 reference within tolerance.
  - `test_state_dict_roundtrip`: save → restore → shadow tensors
    match bit-exactly. Include both real and complex params.
  - `test_applied_to_restores`: in-place swap inside context manager
    leaves params unchanged on exit; complex params restored
    bit-exactly.
  - `test_export_model_state_full_dict`: build a toy `nn.Module` with
    a trainable param + a `register_buffer`. Assert that
    `export_model_state(model)` returns ALL keys from
    `model.state_dict()` (param + buffer), with the param value
    replaced by the EMA shadow (in the param's dtype) and the buffer
    value identical to the live model. Also assert the dict loads
    via `model.load_state_dict(..., strict=True)` without error
    (this is the §4.3 contract).
  - `test_sharded_dims_mp_preserved`: when source params carry
    `sharded_dims_mp`, exported tensors carry it too.
  - `test_resume_ema_config_mismatch`: build EMAModel, save state +
    `ema_config`; load with mismatched decay → `_maybe_restore_ema_state`
    raises `RuntimeError`; with `allow_config_change=True` → loads
    with WARNING.
- `tests/sfno_training/test_ema_smoke_integration.py`
  (integration; uses existing smoke fixtures):
  - Run 2 epochs of a synthetic-data smoke config with EMA enabled
    (override the smoke YAML in test scope, do NOT enable in the
    checked-in smoke YAML).
  - Assert per-epoch ckpt has all four EMA keys: `ema_state`,
    `ema_step`, `ema_config`, `ema_best_loss`.
  - Restart from that ckpt, run 1 more epoch, assert shadow matches
    what a continuous-run would have produced (within fp tolerance).
    Also assert resume picks up the ckpt at
    `self.checkpoint_version_current` (rotate by running 3+ epochs
    with `checkpoint_num_versions=2` and confirm the loaded ckpt is
    the correct rotation slot).
  - Assert `best_ckpt_ema_mp0.tar` exists. Its `model_state` differs
    from `best_ckpt_mp0.tar`'s `model_state`, has the SAME key set
    (full state_dict including buffers), and
    `sfno_inference.checkpoint_loader.build_wrapper_from_checkpoint`
    can load it without error (validates the strict=True contract).
  - **Two flexible-mode reject cases** (Goal #7 covers both directions):
    - `save_checkpoint="flexible"` + EMA enabled raises `NotImplementedError`
      at `PlasimTrainer.__init__`, error message names the rejected key.
    - `load_checkpoint="flexible"` + EMA enabled raises `NotImplementedError`
      at `PlasimTrainer.__init__`, error message names the rejected key.
    Both cases assert no checkpoint file is written.
  - Assert validation logging routing: after one validated epoch with
    EMA enabled, `valid_logs["metrics"]` contains the four §6 EMA
    scalars (`validation loss ema`, `ema decay effective`, `ema step`,
    `ema best loss`); `valid_logs["base"]` does NOT contain any
    `*_ema` / `ema *` keys (so stock raw-best save still triggers on
    the unmodified `validation loss`).
- Add to `submit_zgplev_smoke.slurm` flow only if existing test infra
  already wires test invocation; otherwise tests are dev-machine only.

### 7.6 Documentation
- Add a short note to `src/sfno_training/README.md` describing the
  `ema:` config block and how to use `best_ckpt_ema_mp0.tar` for
  inference (point the existing
  `sfno_inference/checkpoint_loader.py` at it; no flag change).

---

## 8. Risks & things to watch

1. **Validation wall-time roughly doubles** at end of epoch (two
   passes through the val loader). On `full` with
   `valid_autoreg_steps: 3`, this is small relative to training time
   per epoch, but worth measuring on first run.
2. **Loss-object internal state** (`self.loss_obj`) — running
   `super().validate_one_epoch` twice in succession might mutate
   loss-object state if it accumulates. Need to inspect
   `makani.loss` once during implementation to confirm idempotence;
   if any state accumulates per-call, reset between passes.
3. **Inf/nan-skipped step** (GradScaler): hook does not fire (per
   PyTorch semantics, `optimizer.register_step_post_hook` runs only
   after a real `optimizer.step()`). EMA is not corrupted by skipped
   batches. Add a sanity log line: count of EMA updates per epoch
   should equal count of non-skipped optimizer steps.
4. **Model-parallel save**: `sharded_dims_mp` attachment must be
   present on EMA tensors in `best_ckpt_ema_mp0.tar` so that any
   future flexible-mode restore at different MP topology works.
   Covered by `test_sharded_dims_mp_preserved` (§7.5).
5. **AMP / mixed precision shadow**: real shadows in fp32 (when
   parameter is fp16/bf16) and complex shadows in complex64 — see
   §3.2. `p.detach().to(shadow.dtype)` allocates a temp each step;
   `add_(..., alpha=1 - decay_t)` avoids an extra. Memory overhead:
   ~1× model params worth of shadow. For SFNO at embed_dim=256 /
   12 layers, on the order of ~150 MB — negligible vs. activation
   memory.
6. **Raw best file gains four EMA keys** (§4.2). Confirm no external
   tooling currently checksums or asserts the keyset of
   `best_ckpt_mp0.tar` — should be safe given inference reads
   `model_state` only and the file has not been published widely.
7. **Flexible mode rejected on save AND load**: hard-error at
   `__init__` (and again at save) when EMA is enabled and either
   `save_checkpoint == "flexible"` or `load_checkpoint == "flexible"`.
   The current zgplev configs use legacy in both directions so this
   does not block any in-flight runs. If a future run needs
   flexible-mode EMA, the follow-up needs gather/scatter for
   `ema_state` (save side) and a flex-aware per-rank EMA restore
   path (load side); deferred.
8. **`ema_config` change mid-run**: protected by `allow_config_change`
   knob (§5). Without the override, mismatched decay/warmup on
   resume raises rather than silently changing dynamics. Document
   this in the README addition (§7.6).
9. **Complex-weight regression risk**: an accidental `.float()` cast
   anywhere in the EMA path drops imaginary components on the
   `complex64` SFNO spectral weights. Guarded by
   `test_update_math_complex` and by the §7.1 implementation note
   forbidding `.float()` in favor of `.to(shadow.dtype)`.

---

## 9. Rollout plan

1. PR with §7.1 – §7.5 changes + tests. Land on a feature branch.
2. Local CPU unit tests pass.
3. Smoke integration test passes (auto-enables EMA in test scope).
4. Run `submit_zgplev_short.slurm` (which has EMA off) to verify
   no regressions on the EMA-off path — checkpoint format and
   training curve must match a pre-EMA short run modulo the new
   keys absent.
5. Run `submit_zgplev_baseline.slurm` (EMA on) to verify EMA
   metrics appear in logs, both ckpts emitted, both losses tracked.
6. Run `submit_zgplev_full.slurm` for the next serious emulator
   training.
7. Inference: point `sfno_inference/checkpoint_loader.py` at
   `best_ckpt_ema_mp0.tar`. If raw clearly beats EMA on
   downstream metrics, fall back to `best_ckpt_mp0.tar` — both
   are available.

---

## 10. Open questions deferred to implementation review

None blocking. Item to revisit during code review:
- Whether `ema decay effective` should be logged per-step (verbose)
  or per-epoch (terminal value). Plan defaults to per-epoch — the
  validation override populates it once at end of epoch from
  `self.ema.decay_t` (see §7.2(d)). Per-step logging would require
  a tap inside the optimizer post-step hook and a wandb call there,
  which adds non-trivial overhead per training step; not worth it
  unless we observe surprising warmup dynamics on the first run.

(Resolved in v3 / v6: log key naming convention — Makani-style spaced
names; EMA scalars routed to `valid_logs["metrics"]` for screen +
wandb visibility, pinned in §6.)
