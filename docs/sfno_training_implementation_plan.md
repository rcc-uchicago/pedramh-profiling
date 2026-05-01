# SFNO Training + Validation Subproject (Makani wrapper) — implementation plan

> **Plan v1** (2026-04-24). On approval, promote to `docs/sfno_training_implementation_plan.md` for Codex review before writing any code.
> **v4 (2026-04-24):** Codex round 3 review. Two fixes:
>   1. **`register_model` call signature was backwards.** Stock signature at `makani/makani/models/model_registry.py:83` is `register_model(model: Union[str, nn.Module], name: Optional[str] = None)`. When the first arg is a string, it routes to `_register_from_file` (`:64`), which splits on `:` and treats it as a `path/to/file.py:ClassName` filesystem path. v3's `register_model("plasim_test_recording_dummy", RecordingDummyModel)` would therefore raise (path does not exist). v4: `register_model(RecordingDummyModel, name="plasim_test_recording_dummy")` — class first, name as kwarg.
>   2. **Stale "Dummy `nn.Conv2d(58, 53, 1)`" labels removed.** The integration-test model is the `RecordingDummyModel` class (Codex round 2 fix #1, kept in v4), not a bare Conv2d. v4 updates the decisions table row and the verification success bullet to say `RecordingDummyModel`.
>
> **v3 (2026-04-24):** Codex round 2 review. Six fixes:
>   1. **Dummy-nettype registration must register a class, not a factory function.** Stock `register_model` → `_register_from_module` at `makani/makani/models/model_registry.py:49` does `issubclass(model, nn.Module)` and rejects callables that aren't classes. v2's `make_recording_dummy(...)` function would raise. v3: `RecordingDummyModel(nn.Module)` directly accepts `(inp_shape, out_shape, inp_chans, out_chans, **kw)` in `__init__`, and `register_model(RecordingDummyModel, name="plasim_test_recording_dummy")` registers the class. (Call signature corrected in v4.)
>   2. **`RecordingDummyModel` must include a trainable parameter on the grad path.** v2 returned constants — loss has no grad path, optimizer step never fires, the planned `optimizer.state[*]['step'] >= 1` assertion would fail. v3: add `self.dummy_param = nn.Parameter(torch.zeros(1))` and emit `pred = const + 0.0 * self.dummy_param.sum()` so backward populates a gradient.
>   3. **Wrapper test forcing-content index off-by-one.** v2's `test_wrappers.py` row says `inputs_seen[k][:, 52:58, ...] == tar_forcing_normalized[:, k:k+1, ...]` for `k ≥ 1`. The validation row got it right (`k-1:k`). v3 corrects the wrapper row to `k-1:k`.
>   4. **Aux-feature asserts incomplete.** v2 only locked `add_zenith`, `add_grid`, `input_noise`. Stock `Driver._set_data_shapes` at `driver.py:207-219` also injects static channels for `add_orography` (+1), `add_landmask` (+1 or +2), `add_soiltype` (+8). v3 adds `not params.get("add_orography", False)`, `not params.get("add_landmask", False)`, `not params.get("add_soiltype", False)`.
>   5. **`train_one_epoch()` does not trigger validation.** v2 said "or run `pt.train_one_epoch()` which triggers validation." It does not — `train()` (`deterministic_trainer.py:331`) drives `train_one_epoch()` then `validate_one_epoch()` from the outer loop. v3 calls `pt.validate_one_epoch(epoch=0)` directly (`deterministic_trainer.py:577`).
>   6. **Drop checkpoint-write assertion from trainer-CI test.** `train_one_epoch()` (`:445`) skips outer-loop behavior: validation, scheduler step, `log_epoch`, checkpoint write. v2's "completes + writes checkpoint" assertion is wrong for that entry point. v3 keeps the call to `train_one_epoch()` (right tradeoff for a fast wiring smoke) and drops the checkpoint-write assertion. Optimizer-step assertion remains via #2's grad path. If we ever want a checkpoint-write gate, that's a separate `test_trainer_full_train.py` calling `pt.train()` with `max_epochs=1`.
>
> **v2 (2026-04-24):** Codex round 1 review. Eight fixes:
>   1. **Scope renamed** to "training + validation only." Inference is out of scope (v9-locked); v2 adds an explicit hard gate: **no full emulator rollout / scoring / production inference until the follow-up `src/sfno_inference/` PR lands.** Block end users at the README + SKILL.md level too.
>   2. **Trainer integration test rewritten.** `Trainer.train()` (`makani/makani/utils/training/deterministic_trainer.py:331`) takes no `n_iters`. v2 uses `pt.train_one_epoch()` (`:445`) + a registered dummy nettype (`make_dummy_model(inp_shape, out_shape, inp_chans, out_chans)`) wired through stock `model_registry.get_model()` (`model_registry.py:121`).
>   3. **Forcing index off-by-one fixed.** v1 said `tar_forcing[:, step+1, ...]`; stock copies `unpredicted_tar[:, step:step+1, ...]` (`preprocessor.py:212`). At step k the next input gets forcing at `t+k+1`, indexed as `tar_forcing[:, k]` (since target sequences start at t+1). v2 corrects the invariant.
>   4. **Validation-rollout smoke added.** The critical bug site is `deterministic_trainer.py:661`, which calls `self.preprocessor.append_history(inpt, pred, idt)` directly. v1 only tested wrappers. v2 adds `tests/sfno_training/test_validation_rollout.py` with `valid_autoreg_steps >= 2` and content-level sentinels.
>   5. **Smoke tests check content, not only shapes.** All wrapper + trainer + validation tests use a `RecordingDummyModel` that records its inputs and emits a sentinel pr_6h, asserting (a) input channel count is exactly 58 at every step, (b) forcing appears exactly once, (c) forcing values match `tar_forcing[:, step:step+1]` (off-by-one fix #3), (d) the pr_6h sentinel never appears in the next-step state input.
>   6. **`history_normalization_mode == "none"` hard-asserted.** Stock history normalization (`preprocessor.py:270, :360`) computes stats on the 58-channel post-concat input and would denormalize a 53-channel target using the first 53 input stats — wrong if anyone enables it. Driver default is `"none"` (`driver.py:111-112`); v2 asserts in `PlasimTrainer._set_data_shapes`.
>   7. **Auxiliary stock features hard-asserted off.** Stock `Driver._set_data_shapes` adds zenith/grid/noise channels (`driver.py:178-200`). Any flip would break the 58-channel input contract. v2 asserts `not params.add_zenith`, `not params.get("add_grid", False)`, `params.get("input_noise") is None` in `PlasimTrainer._set_data_shapes`.
>   8. **Production loader avoids `self.out_channels` mutation.** Stub mutates `self.out_channels` temporarily (`tests/plasim_makani_packager/stub_forcing_loader.py:201`); production passes an explicit `channels` argument into the read helper.
>
> **Plan v1** (2026-04-24). Initial draft.
>
> Executes commit chunk 7 of `docs/plasim_makani_packager_plan.md` v9 (the "trainer-patch contract"). Inherits all v9 locked decisions; does not re-open them.

---

## Context

The PlaSim → Makani packager (v9, `docs/plasim_makani_packager_plan.md`) has produced a structurally-validated sim52 Makani dataset at `/scratch/11114/zhixingliu/AI-RES/data/makani/sim52_astro_64x128/` (98 train + 20 valid + 8 test files; commit `106d19d`). The dataset contract is asymmetric:

- **Input (58):** 52 state feedback (`/fields_state`) + 6 prescribed forcing (`/forcing`).
- **Output (53):** 52 state + 1 diagnostic `pr_6h` (`/fields_diagnostic`).
- **Rollout feedback:** `pred[:, :52]` only — `pr_6h` must never feed back; forcing at `step+1` comes from truth via `cache_unpredicted_features`.

Stock Makani cannot consume this contract: `MultifilesDataset` assumes a single `/fields` HDF5 key with `in_channels == out_channels`; `MultiStepWrapper._forward_train` at `makani/makani/models/stepper.py:112` blindly feeds the full predicted tensor back through `append_history`, which would yield `53 + 6 = 59` channels at the next forward while the model expects 58.

v9 §"Trainer-patch contract" specifies the fix in precise detail. This plan turns that one-paragraph chunk into two reviewable PRs, resolves implementation details v9 deferred (loss weighting, validation scope, CI test strategy, stub lifecycle, obsolete doc), and adds a CPU quick-check smoke alongside v9's mandatory GPU integration test.

**Why a new plan rather than "just execute chunk 7":**
- Chunk 7 is ~50 lines. The surface area here — stub relocation, Python 3.12 shim, real-SFNO smoke environment, obsolete extraction plan — deserves its own spec.
- v9 leaves open: loss weighting, v1 validation scope, CI vs GPU smoke split, commit chunking. Those are chosen below.

### Hard gate on full emulator rollout (v2)

The `src/sfno_training/` PRs (A and B) deliver a runnable **training + validation** path only. Stock `Inferencer` will silently load `inpz=tarz=None` and skip forcing; its predictions are physically wrong on this dataset. **No production scoring, long-rollout evaluation, or downstream emulator runs may use this checkpoint until the follow-up `src/sfno_inference/` PR lands and passes its own integration test.** Enforced at three layers:

1. `_plasim_get_dataloader` raises `AssertionError("PlaSim inference is out of scope (v8)")` on `mode == "inference"`.
2. `src/sfno_training/README.md` calls out the inference block prominently in its first section.
3. `skills/sfno-training/SKILL.md` flags it under "When NOT to use this skill."

---

## Interview decisions (locked)

| Topic | Decision | Source |
|---|---|---|
| Patch Makani core vs separate module | **Separate.** `src/sfno_training/`, two monkey-patches only. | v9 (locked) |
| Forcing normalization | **Separate** `forcing_bias`/`forcing_scale` before in-wrapper concat | v9 (locked) |
| Inference in v1 | **Out of scope.** Deferred to `src/sfno_inference/` PR. | v9 (locked) |
| `params.channel_names` | **53 target names** (state ‖ diagnostic) | v9 (locked) |
| `n_history > 0` | **Forbidden** by hard assert in `PlasimTrainer._set_data_shapes` | v9 (locked) |
| v1 validation metrics | **Loss + rollout shape only.** No ACC/RMSE/spectra. | This plan |
| `pr_6h` loss weight | **Equal** (`channel_weights="constant"`, 53-wide uniform) | This plan |
| Integration-test model (CI hard gate) | **`RecordingDummyModel`** (`nn.Module` with single `dummy_param`, sentinel pr_6h, records inputs) | This plan |
| Pre-training hard gate (real SFNO) | **Both:** CPU quick-check (`@pytest.mark.slow`) + GPU sbatch | This plan |
| Smoke-test location | **`tests/sfno_training/`** (repo convention) | This plan |
| Commit strategy | **Split into 2 PRs** (A: data-side; B: trainer-side) | This plan |
| Stub lifecycle | **Thin to re-export** from `src/sfno_training/*` after PR-A | This plan |
| Python 3.12 shim | **Vendor in `src/sfno_training/compat.py`** | This plan |
| Obsolete extraction plan | **Rewrite in place as v4** — pointer to v9 | This plan |

---

## Module structure (target tree)

```
src/sfno_training/
├── __init__.py
├── compat.py                    # Python 3.12 get_timedelta_from_timestamp shim
├── data/
│   ├── __init__.py
│   └── plasim_forcing_dataset.py   # PlasimForcingDataset(MultifilesDataset)
├── models/
│   ├── __init__.py
│   ├── preprocessor.py             # PlasimPreprocessor(Preprocessor2D)
│   └── stepper.py                  # PlasimSingleStepWrapper, PlasimMultiStepWrapper
├── trainer/                        # PR-B
│   ├── __init__.py
│   └── plasim_trainer.py           # PlasimTrainer(Trainer), _plasim_get_dataloader,
│                                   # _install_plasim_patches
├── train_plasim.py                 # PR-B — CLI entry; mirrors makani/train.py::main
├── config/                         # PR-B
│   ├── plasim_sim52_smoke.yaml     # tiny SFNO, 1 file, 1 epoch — GPU sbatch hard gate
│   └── plasim_sim52_baseline.yaml  # production SFNO for full training runs
├── submit_train.slurm              # PR-B — full training job template
├── submit_smoke.slurm              # PR-B — GPU smoke (hard gate)
└── README.md                       # PR-B — pointer to skills/

tests/sfno_training/
├── __init__.py
├── conftest.py                     # PR-B — RecordingDummyModel + dummy-nettype registration + 1-file fixture
├── test_data_loader.py             # PR-A — shape / channel-order / stats / n_future={0,1}
├── test_preprocessor.py            # PR-A — append_history strip + hard-assert
├── test_wrappers.py                # PR-A — 58-in/53-out single-step + two-step rollout + content sentinels
├── test_trainer_ci.py              # PR-B — PlasimTrainer.train_one_epoch(), dummy nettype, content sentinels
├── test_validation_rollout.py      # PR-B — validation rollout (deterministic_trainer.py:661), content sentinels
└── test_smoke_sfno_cpu.py          # PR-B — tiny-dim real SFNO, CPU, @pytest.mark.slow

skills/sfno-training/                # PR-B
└── SKILL.md                        # matches skills/plasim-makani-packager/ structure
```

---

## 1. Custom dataloader contract (PR-A)

**`src/sfno_training/data/plasim_forcing_dataset.py`** — production of the stub at `tests/plasim_makani_packager/stub_forcing_loader.py:51-232`. Minimal delta from stub:
- Move Python 3.12 shim to `src/sfno_training/compat.py`; import at top of this module.
- Inherit `MultifilesDataset` (`makani/makani/utils/dataloaders/data_loader_multifiles.py:41`).
- `__init__` args (v9 §"Custom dataloader PlasimForcingDataset"): `diagnostic_dataset_path`, `forcing_dataset_path`, `n_forcing_channels`, `forcing_bias`, `forcing_scale`. Passes `dataset_path="fields_state"` to `super().__init__` (stock default is `"fields"`).
- Override `get_sample_at_index` fully (stock's `_get_data` assumes `in_channels == out_channels`):
  - **Input state:** `/fields_state[in_channels]` (52 channels) → `(n_history+1, 52, H, W)`, normalized by `in_bias/in_scale[:,:52]`.
  - **Target:** `/fields_state[out_channels[:52]]` + `/fields_diagnostic[0]`, concat along channel → `(n_future+1, 53, H, W)`, normalized by full 53-wide `out_bias/out_scale`.
  - **Forcing (input and target):** `/forcing[0..5]` (6 channels), normalized by `forcing_bias/forcing_scale`.
- **Read helper takes explicit `channels` (Codex round 1 fix #8).** The stub's `_read_state` reuses `self.in_channels` / `self.out_channels` and the target-state path temporarily mutates `self.out_channels` (`tests/plasim_makani_packager/stub_forcing_loader.py:201-208`). Production refactors to `_read_state(global_idx, off_start, off_end, *, channels: np.ndarray) -> np.ndarray` — caller passes the explicit channel list. No mutation of instance attrs.
- Return 4-tuple `(inp_state, tar, inp_forcing, tar_forcing)`. All 4D `(T, C, H, W)` per stock `data_loader_multifiles.py:411` convention (T=1 when `n_history=0`/`n_future=0`, T>1 for multistep).
- Pass `relative_timestamp=True` to bypass the Unix-epoch interpretation at `data_helpers.py:158`.

**Stats files:** Reads from `{output_root}/stats/` produced by `plasim_makani_packager.stats`:
- `global_means.npy / global_stds.npy` — shape `(1, 53, 1, 1)` → sliced to `(1, 52, 1, 1)` for input bias/scale.
- `forcing_global_means.npy / forcing_global_stds.npy` — shape `(1, 6, 1, 1)` → forcing bias/scale.
- `time_means.npy` (1, 53, 64, 128) — loaded by Makani's `get_data_normalization` path for future ACC; not used by v1 validation.

---

## 2. Model-wrapper contract (PR-A)

**`src/sfno_training/models/preprocessor.py`** — production of stub `PlasimPreprocessor` at `stub_forcing_loader.py:238-264`. Subclass `Preprocessor2D` (`makani/makani/models/preprocessor.py:30`). Only `append_history` is overridden:

```python
def append_history(self, x1, x2, step, update_state=True):
    assert x2.dim() == 4, "..."
    assert x2.shape[1] in (self.n_state_channels, self.n_full_out_channels), "..."
    if x2.shape[1] == self.n_full_out_channels:
        x2 = x2[:, :self.n_state_channels, ...]
    return super().append_history(x1, x2, step, update_state=update_state)
```

v9 shape-asserts 52 or 53 — anything else raises. No silent slicing.

**`src/sfno_training/models/stepper.py`** — `PlasimSingleStepWrapper(SingleStepWrapper)` and `PlasimMultiStepWrapper(MultiStepWrapper)` each replace `self.preprocessor` with `PlasimPreprocessor(params)` in `__init__`. Stock `_forward_train` / `_forward_eval` are unchanged — the strip is inside `append_history`, so the stock loop bodies call the patched version transparently via the `self.model.preprocessor` linkage.

**Concat of 52 state + 6 forcing → 58 at model boundary** happens inside stock `Preprocessor.append_unpredicted_features` (`makani/makani/models/preprocessor.py:448`), called at `stepper.py:34` (single-step) / `:82` (multi-step step 0+) / `:125` (eval). Trainer must NOT manually concat.

---

## 3. Loss contract (PR-A)

Build `LossHandler(params)` (`makani/makani/utils/loss.py:58`) with:
- `params.channel_names` = 53 target names (state `pl, tas, ta1..ta10, ..., zg10` + diagnostic `pr_6h`). Sourced from `metadata/data.json`.
- `params.out_channels = list(range(53))`.
- `params.channel_weights = "constant"` → 53-wide ones. pr_6h gets the same weight as state channels; its small std (~2.8e-7 — flagged by stats with `--epsilon 1e-10`) is already accounted for by normalization.
- For `n_future > 0`, `multistep_weight` is 53·(n_future+1) wide; `channel_weights` stays 53 wide, forward tiles (`loss.py:437-440`). **Must rebuild `LossHandler(params)` whenever `n_future` flips** — v9 "Do NOT" list.

Loss contract:
- Prediction: `(B, 53·(n_future+1), H, W)` from wrapper (MultiStep concats per-step preds along channel).
- Target: `(B, 53·(n_future+1), H, W)` from `preprocessor.flatten_history(tar)`.
- v1 validation: pass to same `LossHandler`; assert `loss.isfinite()` and assert `pred.shape == (B, 53·(n_future+1), H, W)`. No ACC/RMSE.

---

## 4. Autoregressive rollout contract (PR-A)

Exact invariants the tests must pin down:

1. At each step `k ∈ [0, n_future]`, `wrapper(inp_state).shape[1] == 58` inside the model and `pred.shape[1] == 53` out of the model.
2. `preprocessor.unpredicted_inp_{train,eval}.shape[2] == 6` (six forcing channels) throughout rollout.
3. **Forcing-step indexing (Codex round 1 fix #3).** Stock `append_history` (`makani/makani/models/preprocessor.py:211-224`) copies `unpredicted_tar_*[:, step:step+1, ...]` into `unpredicted_inp_*` at the end of step `k`. Because `tar_forcing` is target-aligned (index 0 = forcing at `t+1`, index 1 = `t+2`, ...), this means: after predicting step `k`, the next forward sees forcing at physical time `t + (k+1)`, fetched as `tar_forcing[:, k:k+1, ...]`. Tests must assert this exact index match — not `step+1`.
4. `pred[:, :52]` feeds into next-step `inp_state` via `PlasimPreprocessor.append_history` strip. `pred[:, 52:53]` (pr_6h) is **loss-only**, never in feedback.

---

## 5. Rollout-path coverage

| Path | Call site | Covered? | How |
|---|---|---|---|
| Training multistep | `makani/makani/models/stepper.py:112` | ✅ | `PlasimMultiStepWrapper` installs `PlasimPreprocessor`; `append_history` strip fires |
| Validation rollout | `makani/makani/utils/training/deterministic_trainer.py:661` | ✅ | `self.preprocessor = self.model.preprocessor` at `:133` → shared instance |
| Stock single-step train | `stepper.py:34` | ✅ | `PlasimSingleStepWrapper` same pattern |
| Stock single-step eval | `stepper.py:125` | ✅ | Same |
| **Inference** | `makani/makani/utils/inference/inferencer.py:620` | ❌ **OUT OF SCOPE** | Stock `_inference_indexlist` unpacks only `add_zenith` tuples — forcing silently dropped. Covered by follow-up `src/sfno_inference/` PR. |

v1 blocks inference via a runtime error: `PlasimTrainer` reruns with `mode == "inference"` → the `_plasim_get_dataloader` assertion at v9 pseudocode raises `AssertionError("PlaSim inference is out of scope (v8)")`.

---

## 6. Trainer wiring (PR-B)

**`src/sfno_training/trainer/plasim_trainer.py`** — verbatim from v9 pseudocode (`docs/plasim_makani_packager_plan.md` §"Trainer wiring (exact insertion points — v8)"):

- `_plasim_get_dataloader(params, files_pattern, device, mode)` — constructs `PlasimForcingDataset`, wraps in `DataLoader` + optional `DistributedSampler`, attaches stock-compat attrs (`lat_lon`, `get_output_normalization`, `get_input_normalization`) and returns `(dataloader, dataset, sampler)` triple.
- `_install_plasim_patches()` — two rebindings:
  - `model_registry.SingleStepWrapper = PlasimSingleStepWrapper`
  - `model_registry.MultiStepWrapper = PlasimMultiStepWrapper`
  - `deterministic_trainer.get_dataloader = _plasim_get_dataloader`
- `PlasimTrainer(Trainer)`:
  - `__init__` calls `_install_plasim_patches()` BEFORE `super().__init__()`.
  - Overrides `_set_data_shapes(params, dataset)` to set `params.N_in_channels = 58` AFTER stock population, BEFORE model build. Asserts (v2/v3):
    - `params.n_history == 0` — locked by v9 (the 58-channel override only holds at history=0).
    - `params.history_normalization_mode == "none"` — stock history-normalization at `preprocessor.py:270, :360` would compute stats on the 58-channel post-concat input and try to denormalize a 53-channel target with the first 53 input stats (i.e. mismatch the diagnostic with a forcing channel's stats). Driver default is `"none"` (`driver.py:111-112`); assert here so config drift fails loudly.
    - **Auxiliary stock features all off** — any of these would inject extra channels via `Driver._set_data_shapes` (`driver.py:178-219`) and break the locked 58-channel input contract:
      - `not params.add_zenith` — adds 1 dynamic channel at `driver.py:178-189`. Solar insolation is already in `/forcing[rsdt]`.
      - `params.get("input_noise") is None` — adds N noise channels at `driver.py:185-194` if `input_noise.mode == "concatenate"`.
      - `not params.get("add_grid", False)` — adds 2+ static channels at `driver.py:199-205`.
      - `not params.get("add_orography", False)` — adds 1 static channel at `driver.py:207-208` (Codex round 2 fix #4).
      - `not params.get("add_landmask", False)` — adds 1 or 2 static channels at `driver.py:210-215` (Codex round 2 fix #4). Land/sea info is already in `/forcing[lsm]`.
      - `not params.get("add_soiltype", False)` — adds 8 static channels at `driver.py:217-218` (Codex round 2 fix #4).

Stock `Trainer.__init__` runtime order (from v9 verification): `_set_data_shapes` at `:106` → model build at `:132` → loss build at `:175`. The two monkey-patches + the `_set_data_shapes` override slot in cleanly.

**`src/sfno_training/train_plasim.py::main`** — mirrors stock `makani/train.py::main` (`train.py:77-146`) for runtime-injected params (v9 lists the 15 exact fields: `experiment_dir`, `checkpoint_path`, `best_checkpoint_path`, `resuming`, `amp_mode`, `jit_mode`, `skip_validation`, `skip_training`, `enable_synthetic_data`, `enable_s3`, `enable_odirect`, `checkpointing_level`, `multistep_count`, `n_future`, `disable_ddp`, `enable_grad_anomaly_detection`, `split_data_channels`, `print_timings_frequency`, `load_checkpoint`). Instantiates `PlasimTrainer(params, world_rank)` then calls `.train()` unless `--skip_training`.

---

## 7. Smoke tests

### PR-A unit tests (`tests/sfno_training/`, all CPU, all ~seconds)

All tests that exercise the wrapper or rollout share a `RecordingDummyModel` (defined in a `tests/sfno_training/conftest.py` fixture) that is itself an `nn.Module` subclass and registered via `register_model(RecordingDummyModel, name="plasim_test_recording_dummy")` (Codex round 3 fix #1: stock `register_model(model, name=None)` at `model_registry.py:83` routes a string first-arg to `_register_from_file`, which would parse `"plasim_test_recording_dummy"` as a filesystem path — class-first, name-kwarg is the right form. The class itself satisfies the `issubclass(model, nn.Module)` check at `model_registry.py:49`, per Codex round 2 fix #1):

```python
class RecordingDummyModel(nn.Module):
    def __init__(self, inp_shape, out_shape, inp_chans, out_chans, **kw):
        super().__init__()
        self.inp_chans = inp_chans
        self.out_chans = out_chans
        self.inputs_seen: list[torch.Tensor] = []
        # Codex round 2 fix #2: trainable parameter so loss has a grad path —
        # otherwise optimizer.step never fires and the optimizer-step assertion
        # in test_trainer_ci.py would fail.
        self.dummy_param = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        self.inputs_seen.append(x.detach().clone())
        out = torch.zeros(x.shape[0], self.out_chans, x.shape[2], x.shape[3], device=x.device)
        out[:, 52:53, ...] = -9999.0    # pr_6h sentinel — must never appear in next-step state input
        return out + 0.0 * self.dummy_param.sum()  # grad path through dummy_param
```

| File | Purpose | Lineage |
|---|---|---|
| `test_data_loader.py` | Shape asserts at `n_future ∈ {0, 1}`; stats dtype/shape; channel-order vs `metadata.data.json`; `relative_timestamp=True` path | Extracted from `tests/plasim_makani_packager/test_multifile_loader_smoke.py` steps 4–5, 10 |
| `test_preprocessor.py` | Positive strip `53 → 52`; hard-reject `60` / 3D via `pytest.raises(AssertionError)` | Steps 9 of existing smoke |
| `test_wrappers.py` | Single-step + two-step rollout. **Content asserts** with the `RecordingDummyModel`: (a) `model.inputs_seen[k].shape[1] == 58` for all `k`; (b) state portion `inputs_seen[k][:, :52, ...]` never contains the `-9999` pr_6h sentinel; (c) forcing portion `inputs_seen[k][:, 52:58, ...]` matches `tar_forcing_normalized[:, k-1:k, ...]` exactly for `k ≥ 1` (Codex round 2 fix #3 — `k-1:k`, not `k:k+1`; reasoning: at end of step `k-1`, stock `append_history` copies `unpredicted_tar[:, k-1:k]` into `unpredicted_inp`, which is what step `k` sees); (d) negative regression: stock `MultiStepWrapper` fails on two-step. | Steps 7, 11, 12 + new content checks |

**Stub relocation in the same PR:** `tests/plasim_makani_packager/stub_forcing_loader.py` becomes a 5-line re-export:

```python
from src.sfno_training.compat import *  # noqa: F401,F403  (install Python 3.12 shim)
from src.sfno_training.data.plasim_forcing_dataset import PlasimForcingDataset
from src.sfno_training.models.preprocessor import PlasimPreprocessor
from src.sfno_training.models.stepper import (
    PlasimSingleStepWrapper,
    PlasimMultiStepWrapper,
)
__all__ = ["PlasimForcingDataset", "PlasimPreprocessor",
           "PlasimSingleStepWrapper", "PlasimMultiStepWrapper"]
```

Phase 4b smoke (`test_multifile_loader_smoke.py`) continues to import from the stub file — zero test changes. Single source of truth is `src/sfno_training/`.

### PR-B integration tests

**Trainer-CI test setup (Codex round 1 fix #2 + round 2 fix #1).** Stock `Trainer.train()` (`makani/makani/utils/training/deterministic_trainer.py:331`) takes no `n_iters` argument; v1's `train(n_iters=1)` was not runnable. Stock `model_registry.get_model()` (`makani/makani/models/model_registry.py:121`) reads `params.nettype` from the registry and wraps the resolved class with `inp_shape, out_shape, inp_chans, out_chans, **model_kwargs` (`:187`). Stock `_register_from_module` at `model_registry.py:49` does `issubclass(model, nn.Module)` — registration takes a **class**, not a factory function. v3 strategy:

1. **Register the `RecordingDummyModel` class itself** in a test helper (`tests/sfno_training/conftest.py`):

   ```python
   from makani.models.model_registry import register_model
   # RecordingDummyModel(nn.Module) defined above accepts (inp_shape, out_shape, inp_chans, out_chans, **kw).
   # register_model(model, name=None) — string first-arg routes to _register_from_file (path:Class),
   # so we MUST pass the class first and the name as a kwarg.
   register_model(RecordingDummyModel, name="plasim_test_recording_dummy")
   ```

   Stock `get_model` then constructs it as `RecordingDummyModel(inp_shape=..., out_shape=..., inp_chans=58, out_chans=53, **model_kwargs)`.

2. **Synthetic 1-file fixture**: pytest fixture creates a tiny `MOST.0003.h5` (T=4, 64×128) under `tmp_path/{train,valid}/`, writes the six stats files, plus a `metadata/data.json` and YAML config inline.
3. **Run one epoch**, not one iteration: set `params.max_epochs = 1`, `params.batch_size = 1`, `params.train_year_start = 3, params.train_year_end = 3`. Then call `pt.train_one_epoch()` (`deterministic_trainer.py:445`) directly — sidesteps the full `train()` loop's outer behavior (validation, scheduler step, `log_epoch`, checkpoint write) while still exercising one forward + backward + optimizer step. **Codex round 2 fix #6:** v2 mistakenly asserted "writes checkpoint" — `train_one_epoch()` doesn't write one. v3 drops that assertion. If a checkpoint-write gate is wanted later, add a separate `test_trainer_full_train.py` calling `pt.train()` with `max_epochs=1`.

| File | Purpose | Env | Gating |
|---|---|---|---|
| `tests/sfno_training/test_trainer_ci.py` | `pt = PlasimTrainer(params, 0); pt.train_one_epoch()` with `params.nettype = "plasim_test_recording_dummy"`. Asserts: (1) `isinstance(pt, PlasimTrainer)`; (2) `isinstance(pt.train_dataset, PlasimForcingDataset)`; (3) `isinstance(pt.model, PlasimSingleStepWrapper)`; (4) `isinstance(pt.model.preprocessor, PlasimPreprocessor)`; (5) `pt.params.N_in_channels == 58`; (6) `pt.optimizer.state[*].step >= 1` (one optimizer step happened — relies on Codex round 2 fix #2's grad path through `dummy_param`); (7) **content sentinel** — gather all `RecordingDummyModel.inputs_seen` across the epoch's batches; assert every recorded input has `shape[1] == 58`. | CPU, <90s | Runs in every CI; blocks merge of any sfno_training change |
| `tests/sfno_training/test_validation_rollout.py` | The critical bug site is `deterministic_trainer.py:661`, where validation calls `self.preprocessor.append_history(inpt, pred, idt)` directly. Set `params.n_future = 0` for training but `params.valid_autoreg_steps = 2` (or whatever the stock validation rollout knob is). **Codex round 2 fix #5:** call `pt.validate_one_epoch(epoch=0)` directly (`deterministic_trainer.py:577`) — `train_one_epoch()` does NOT trigger validation. Use the `RecordingDummyModel`. Assert: (a) `len(model.inputs_seen) >= 2` per validation batch; (b) at every step `k ≥ 1`, the state portion `inputs_seen[k][:, :52, ...]` does **not** contain `-9999` (pr_6h sentinel correctly stripped); (c) at every step `k ≥ 1`, the forcing portion `inputs_seen[k][:, 52:58, ...]` matches `tar_forcing_normalized[:, k-1:k, ...]` exactly (off-by-one fix #3, validation-side); (d) `isinstance(pt.preprocessor, PlasimPreprocessor)` — the validation path's `self.preprocessor = self.model.preprocessor` linkage still binds the patched class. | CPU, <90s | Runs in every CI; blocks merge of any sfno_training change |
| `tests/sfno_training/test_smoke_sfno_cpu.py` | Tiny-dim real SFNO (e.g. `embed_dim=8`, `num_layers=1`, `scale_factor=4`). One forward + backward + step on a single batch. `@pytest.mark.slow`. | CPU, ~2 min | Developer self-check before pushing. Catches `torch_harmonics 0.6→0.8` / `RealSHT` breakage. |
| `src/sfno_training/submit_smoke.slurm` + `config/plasim_sim52_smoke.yaml` | Full SLURM job: tiny real SFNO (larger than CPU test), 1 epoch on 1 training file. Writes a checkpoint. | GPU sbatch, ~15 min | **Hard gate by v9:** no production training run launches until this completes cleanly. |

### Test commands

```bash
# PR-A CI (every push)
.venv/bin/python -m pytest tests/sfno_training/ -v --ignore=tests/sfno_training/test_smoke_sfno_cpu.py

# Pre-push self-check (developer)
.venv/bin/python -m pytest tests/sfno_training/ -v -m slow

# PR-B GPU hard gate (before any production training run)
sbatch src/sfno_training/submit_smoke.slurm
```

---

## 8. Minimally invasive

**Zero edits to `makani/`.** The entire wrapper relies on:

1. Subclassing four public stock classes (`MultifilesDataset`, `Preprocessor2D`, `SingleStepWrapper`, `MultiStepWrapper`, `Trainer`).
2. Two module-attribute rebindings in `PlasimTrainer.__init__`:
   - `makani.models.model_registry.{SingleStepWrapper, MultiStepWrapper}`
   - `makani.utils.training.deterministic_trainer.get_dataloader`

Both targets are `from X import Y` module-scope bindings in the importer, which are mutable Python module attributes. Rebinding them takes effect on subsequent accesses. Verified in v9's §"Verified Makani control flow" table (line references pinned to Makani git HEAD `c970430`).

**Upstream-bump risk:** if Makani changes the import form at `deterministic_trainer.py:36` or `model_registry.py:204-207` to `import X; X.Y(...)`, the monkey-patch path changes. The integration test (`test_trainer_ci.py`) catches this as a hard failure — no silent regression.

---

## 9. Files created / modified

### PR-A — Data-side production (no trainer wiring yet)

**Created:**
- `src/sfno_training/__init__.py`
- `src/sfno_training/compat.py` — Python 3.12 shim
- `src/sfno_training/data/__init__.py`
- `src/sfno_training/data/plasim_forcing_dataset.py`
- `src/sfno_training/models/__init__.py`
- `src/sfno_training/models/preprocessor.py`
- `src/sfno_training/models/stepper.py`
- `tests/sfno_training/__init__.py`
- `tests/sfno_training/test_data_loader.py`
- `tests/sfno_training/test_preprocessor.py`
- `tests/sfno_training/test_wrappers.py`

**Modified:**
- `tests/plasim_makani_packager/stub_forcing_loader.py` — thinned to re-exports.
- `skills/plasim-makani-packager/SKILL.md` — append cross-reference to `src/sfno_training/` (one section).

### PR-B — Trainer wiring + CLI + integration test

**Created:**
- `src/sfno_training/trainer/__init__.py`
- `src/sfno_training/trainer/plasim_trainer.py`
- `src/sfno_training/train_plasim.py`
- `src/sfno_training/config/plasim_sim52_smoke.yaml`
- `src/sfno_training/config/plasim_sim52_baseline.yaml`
- `src/sfno_training/submit_train.slurm`
- `src/sfno_training/submit_smoke.slurm`
- `src/sfno_training/README.md` — first section calls out the **inference block** (no production scoring until `src/sfno_inference/` lands)
- `tests/sfno_training/conftest.py` — `RecordingDummyModel` + dummy-nettype registration + synthetic 1-file fixture
- `tests/sfno_training/test_trainer_ci.py` — `train_one_epoch()` + content sentinels
- `tests/sfno_training/test_validation_rollout.py` — `valid_autoreg_steps ≥ 2` + content sentinels
- `tests/sfno_training/test_smoke_sfno_cpu.py`
- `skills/sfno-training/SKILL.md` — under "When NOT to use" flags inference block

**Modified:**
- `docs/sfno_training_extraction_plan.md` — rewrite in place as v4:
  > **v4 (2026-04-24): Superseded by `docs/plasim_makani_packager_plan.md` §"Trainer-patch contract" and `docs/sfno_training_implementation_plan.md` (this plan, once approved). v1–v3 described a fork of another group's `train.py` with a 4-group NetCDF model interface — that approach is obsolete. The current path is: subclass Makani proper + consume the 3-dataset HDF5 contract emitted by `src/plasim_makani_packager/`.
- `docs/plasim_makani_packager_plan.md` — append a v10 footnote: "Trainer-patch contract implemented in `docs/sfno_training_implementation_plan.md`; chunk 7 now split across PR-A (data side) and PR-B (trainer side)."

---

## 10. Codex review points (v4 — round 4)

**Round 1 (resolved in v2):** scope rename, trainer-test re-write around `train_one_epoch` + registered dummy nettype, forcing-step off-by-one in invariant statement, validation-rollout test added, content-sentinel assertions, history-norm + aux-features asserts, production loader explicit channels.

**Round 2 (resolved in v3):** dummy-nettype must be a class not a factory; `RecordingDummyModel` needs a trainable parameter; wrapper test forcing index `k-1:k` (not `k:k+1`); aux-feature asserts extended to `add_orography`/`add_landmask`/`add_soiltype`; validation test uses `pt.validate_one_epoch(epoch=0)` directly; trainer-CI test drops the checkpoint-write assertion (`train_one_epoch()` skips that).

**Round 3 (resolved in v4):** `register_model` call signature was backwards (string-first routes to `_register_from_file` and parses as `path:Class`); v4 uses `register_model(RecordingDummyModel, name="plasim_test_recording_dummy")`. Stale "Dummy `nn.Conv2d(58, 53, 1)`" labels in the decisions table and verification success bullet replaced with `RecordingDummyModel`.

**Open for round 4:**

1. **Validation rollout knob name.** v4's validation test assumes a `params.valid_autoreg_steps` (or equivalent) drives rollout depth at `deterministic_trainer.py:661`. Confirm the exact attribute name and where in `validate_one_epoch` (`:577`) it's read. If named differently in this Makani revision, update the test fixture.
2. **Content-sentinel value safety.** The sentinel `-9999.0` in `pred[:, 52:53]` must be distinguishable from any plausible normalized state value. Stats are unit-variance so `-9999σ` is well past any physical channel. Confirm.
3. **Forcing-content index final lock.** Wrapper test (training): `inputs_seen[k][:, 52:58] == tar_forcing_normalized[:, k-1:k]` for `k ≥ 1`. Validation test: same indexing with `idt` from `deterministic_trainer.py:661`. Trace once more in this Makani revision and confirm `cache_unpredicted_features(x, y, xz, yz)` initializes `unpredicted_inp ← xz` (so `inputs_seen[0]` shows the **input** forcing, not `tar_forcing[0]`).
4. **CPU smoke realism (carried).** Does `embed_dim=8, num_layers=1` SFNO exercise `torch_harmonics` enough to catch the 0.6 → 0.8 API break, or is the GPU smoke the only real gate? If GPU is the only gate, simplify to one smoke instead of two.
5. **v1 no-ACC risk (carried).** Is training-loss + content-sentinel rollout test sufficient to detect a rollout bug in a 1-epoch run? Content sentinels should catch `pr_6h` leakage and forcing-indexing bugs deterministically — much stronger than loss alone. Confirm this is enough for v1.
6. **Stub relocation transparency (carried).** Confirm the `from src.sfno_training.X import Y; __all__ = [...]` pattern keeps `test_multifile_loader_smoke.py` green with zero test changes — particularly that the Python 3.12 shim still loads before any `MultifilesDataset` instantiation reads the int64 `/timestamp`.
7. **`RecordingDummyModel` and stock `get_model` kwargs.** Stock `get_model` (`model_registry.py:187`) calls `partial(model_handle, inp_shape=..., out_shape=..., inp_chans=..., out_chans=..., **model_kwargs)`. `model_kwargs` may include keys our `__init__` doesn't accept (e.g. SFNO-specific knobs). Our `**kw` swallow should suffice but confirm by listing what stock `model_kwargs` contains for an SFNO config — anything our dummy can't accept must be filtered before registration, or the partial call will TypeError.

---

## Verification

```bash
# PR-A
.venv/bin/python -m py_compile $(find src/sfno_training/data src/sfno_training/models src/sfno_training/compat.py -name '*.py')
.venv/bin/python -m pytest tests/sfno_training/test_data_loader.py tests/sfno_training/test_preprocessor.py tests/sfno_training/test_wrappers.py -v
# Also: Phase 4b still passes
.venv/bin/python -m pytest tests/plasim_makani_packager/test_multifile_loader_smoke.py -v

# PR-B
.venv/bin/python -m py_compile $(find src/sfno_training -name '*.py')
.venv/bin/python -m pytest tests/sfno_training/ -v                         # full PR-A + trainer-CI + validation-rollout tests
.venv/bin/python -m pytest tests/sfno_training/test_validation_rollout.py -v   # validation-path content sentinels (Codex round 1 fix #4)
.venv/bin/python -m pytest tests/sfno_training/ -v -m slow                 # + CPU real-SFNO smoke
sbatch src/sfno_training/submit_smoke.slurm                                # GPU smoke — HARD GATE before any full training run
```

End-to-end success criteria:

- All PR-A tests pass locally, no GPU required.
- All PR-B tests pass locally (CPU, `RecordingDummyModel` for the trainer-CI / validation-rollout content sentinels + tiny SFNO for `test_smoke_sfno_cpu.py`).
- GPU sbatch smoke completes in <30 min, loss finite throughout, checkpoint written, reloadable.
- `tests/plasim_makani_packager/` is still green (stub relocation transparent).

Only after the GPU smoke passes: launch the full training run via `submit_train.slurm` with `plasim_sim52_baseline.yaml`.

---

## What NOT to do

Inherits v9's "Do NOT" list verbatim. Additions specific to this plan:

- **Do not** edit anything under `makani/`. Zero core edits. If a Makani API forces an edit, stop and escalate — the patch strategy is broken.
- **Do not** land PR-B before PR-A. PR-B's integration test imports from `src/sfno_training/data/` and `src/sfno_training/models/`; missing pieces crash the test.
- **Do not** add PlaSim-specific ACC/RMSE metrics in v1. v1 = loss + rollout shape. If a real-world signal is needed after the first training run, that's a v1.1 scope.
- **Do not** launch GPU-hours on full training before `submit_smoke.slurm` completes cleanly. v9 explicitly gates this.
- **Do not** duplicate the four classes in both `stub_forcing_loader.py` and `src/sfno_training/` — single source of truth. After PR-A, the stub is a re-export shim only.
- **Do not** embed the Python 3.12 shim in `plasim_forcing_dataset.py` directly — isolate in `compat.py` so its scope is obvious and removal is trivial when upstream Makani fixes the `int(t)` cast.
- **Do not** wire inference paths. Any `mode == "inference"` call into `_plasim_get_dataloader` must raise `AssertionError` loudly. Stock scoring tooling will otherwise silently produce wrong predictions (forcing = `None`).
- **Do not** flip `n_history > 0` in the YAML. The `_set_data_shapes` assert will trip; flipping it requires re-deriving `N_in_channels = n_state * (n_history + 1) + n_forcing` and auditing `PlasimForcingDataset` forcing-stacking behavior.

---

## Commit sequence

### PR-A (data side)

1. `src/sfno_training/__init__.py`, `compat.py`, `data/plasim_forcing_dataset.py`, `models/preprocessor.py`, `models/stepper.py`.
2. `tests/sfno_training/test_data_loader.py`, `test_preprocessor.py`, `test_wrappers.py`.
3. `tests/plasim_makani_packager/stub_forcing_loader.py` — thin to re-export.
4. `skills/plasim-makani-packager/SKILL.md` — append cross-reference.

Ship once all PR-A tests pass **and** `tests/plasim_makani_packager/test_multifile_loader_smoke.py` is still green.

### PR-B (trainer side)

5. `src/sfno_training/trainer/plasim_trainer.py`, `train_plasim.py`.
6. `src/sfno_training/config/plasim_sim52_smoke.yaml`, `plasim_sim52_baseline.yaml`.
7. `src/sfno_training/submit_train.slurm`, `submit_smoke.slurm`.
8. `tests/sfno_training/test_trainer_ci.py`, `test_smoke_sfno_cpu.py`.
9. `src/sfno_training/README.md`, `skills/sfno-training/SKILL.md`.
10. `docs/sfno_training_extraction_plan.md` — rewrite to v4 (pointer).
11. `docs/plasim_makani_packager_plan.md` — append v10 footnote pointing at this plan.

Ship once PR-A is merged, all PR-B unit + CI tests pass locally, and the GPU sbatch smoke completes cleanly.

Only after PR-B merges: full training run via `submit_train.slurm`.
