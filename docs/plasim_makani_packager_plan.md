# PlaSim → Makani packager — implementation plan

> - **v1 (2026-04-24):** initial plan, after user interview.
> - **v2 (2026-04-24):** v1 used asymmetric `in_channels=58 / out_channels=53` which the Makani metadata parser (`parse_dataset_metada.py:59-60`) forces to be equal. v2 adopted path C: `/fields` (53 predicted) + `/forcing` (6 prescribed) via a custom dataloader plus Makani's generic `unpredicted_*` hook.
> - **v3 (2026-04-24):** Codex round 2 flagged `pr_6h` in `/fields` as a feedback-leak risk. v3 split into `/fields_state` (52) + `/fields_diagnostic` (1) + `/forcing` (6); also fixed sic-validation quantification, forcing-std policy, and moved Phase 4b smoke into this PR using a dummy model + real `Preprocessor2D`.
> - **v9 (2026-04-24):** Codex round 8. Seven fixes. Codex round 8 approved the v8 architecture; these are cleanup before implementation.
>   1. **SIC validation reference is wrong (Major).** v8 Phase 1 step 3 hard-fails on `postproc.sic - adaptor.sic > 1e-3`, but adaptor clipping to `[0,1]` is intentional, so raw MOST vs clipped adaptor can differ by more than tolerance whenever a cell was out of range. v9 splits the check: **(a) hard-fail** `adaptor.sic != np.clip(MOST.sic, 0, 1)` (or abs diff > 1e-6; i.e., the adaptor did the right clip and nothing else); **(b) quantify** raw `MOST.sic - adaptor.sic` (`max_abs_diff`, `mean_abs_diff`, `fraction_cells_changed_by_clip`, NaN-mask parity) as a report only, no hard-fail.
>   2. **Integration test is a hard gate (Major).** v8's follow-up `src/sfno_training/` PR test is worded as "adds an integration test." v9 promotes it to a **blocking gate** before any real SFNO training run: the integration test must pass `PlasimTrainer(params, world_rank).train(n_iters=1)` to completion, with all four `isinstance` assertions. SFNO training (GPU-hours) is forbidden until this test lands and passes in CI.
>   3. **Preprocessor docstring claims inference coverage (Moderate).** v6/v7 PlasimPreprocessor docstring lists `inferencer.py:620` as a covered call site, but v8 explicitly declared inference out of scope. v9 drops the inference call-site entry; docstring now lists training + validation rollout only.
>   4. **Incorrect `Do not` note on LossHandler (Moderate).** "Do not reuse a LossHandler across n_future values" says `channel_weights` is tiled at construction — wrong (that's the v7 misread Codex already corrected for the test). `channel_weights` is 53-wide regardless of `n_future`; `multistep_weight` is 106-wide at n_future=1; forward tiles. v9 rewrites the note to reflect this.
>   5. **N_in_channels = 58 only valid for n_history=0 (Moderate).** `PlasimTrainer._set_data_shapes` sets `N_in_channels = n_state + n_forcing = 58` unconditionally. With `n_history>0`, stock driver multiplies by `(n_history+1)`; our override would silently mis-wire. v9 YAML locks `n_history: 0`; `PlasimTrainer._set_data_shapes` gets a hard `assert params.n_history == 0`.
>   6. **Fresh model per wrapper (Minor).** v8 Phase 4b shares one `nn.Conv2d` via `model_handle = lambda: dummy` — the same parameters get re-wrapped by both `SingleStepWrapper` and `MultiStepWrapper`, which is not what stock wrappers expect. v9 uses `model_handle = lambda: nn.Conv2d(58, 53, 1)` so each wrapper instantiates its own module.
>   7. **Stale wording (Minor).** Stats header still says "consolidated set for /fields" (pre-split naming); verification gate still says "v6". v9 cleans both.
>
> - **v8 (2026-04-24):** Codex round 7. Five fixes:
>   1. **Inference out of scope (Blocker).** v7's `PlasimInferencer` was never actually wired into the inference data path. Stock `Inferencer._inference_indexlist` (`inferencer.py:453-`) unpacks the token from `subset_dataloader` based on `params.add_zenith`: with `add_zenith=False` it does `inp, tinp = gtoken` at `:559` and `tar, ttar = gtoken` at `:584`, then sets `inpz=tarz=None` and calls `cache_unpredicted_features(None, None, None, None)` at `:589` — there is no slot for our 6 forcing channels. Supporting inference requires either overriding the whole `_inference_indexlist` method or adding a second unpacking branch. Rather than bloat this PR, v8 **drops `PlasimInferencer` entirely** and declares inference out of scope. Training is the sole target of this plan; a separate follow-up PR will add inference support with its own plan.
>   2. **CLI runtime fields (Blocker).** v7's "standalone YAML" is necessary but not sufficient: `Trainer` reads runtime-injected fields (`experiment_dir`, `checkpoint_path`, `best_checkpoint_path`, `resuming`, `amp_mode`, `jit_mode`, `skip_validation`, `skip_training`, `enable_synthetic_data`, `enable_s3`, `enable_odirect`, `checkpointing_level`, `multistep_count`, `n_future`, `disable_ddp`, `enable_grad_anomaly_detection`) that stock `makani/train.py::main` (`train.py:88-123`) injects into params *after* YAML load and *before* `Trainer(params, world_rank)`. v8 adds a full `train_plasim.py::main` skeleton that mirrors this setup and then instantiates `PlasimTrainer`.
>   3. **Multistep LossHandler assertion (Major).** v7's `loss_fn_multi.channel_weights.shape[1] == 53 * 2` is wrong. `LossHandler.__init__` (`loss.py:177`) stores `channel_weights` at width `ncw = n_channels = 53` regardless of `n_future`; it's the forward path (`loss.py:437-440`) that tiles via `torch.tile(chw, (1, n_future+1))` and multiplies by `multistep_weight` (`loss.py:194-195`, `(1, ncw*(n_future+1)) = (1, 106)`). v8 keeps the rebuild after `n_future` flip but replaces the assertion with (a) `channel_weights.shape[1] == 53` (stable), (b) `multistep_weight.shape[1] == 53 * 2 = 106`, and (c) a forward-pass check that `loss_fn_multi(pred_ms, tar_flat)` is finite.
>   4. **`_load_npy` undefined (Moderate).** v7's `_plasim_get_dataloader` pseudocode called an undefined `_load_npy`. v8 replaces it with `np.load(...)` directly (stock Makani stats loader pattern).
>   5. **Stale wrapper wording (Moderate).** "stock wrapper path" (§"At model forward time") and "patched MultiStepWrapper in src/sfno_training/" (§"At rollout step boundary") are v4/v5 leftovers. v8 updates both to the v6+ story: `PlasimSingleStepWrapper` / `PlasimMultiStepWrapper` with `PlasimPreprocessor`.
>
> - **v7 (2026-04-24):** Codex round 6. Five fixes:
>   1. **Production wiring (Blocker).** v6 specified `PlasimDriver(Driver)` but the real training entry is `Trainer(Driver)` (`deterministic_trainer.py:58`), and stock CLI instantiates `Trainer` directly. If the CLI builds stock `Trainer`, the `PlasimDriver.__init__` monkey-patch never runs. v7 replaces `PlasimDriver` with `PlasimTrainer(Trainer)` + `PlasimInferencer(Inferencer)`; the `model_registry` monkey-patch happens in their `__init__` *before* `super().__init__()`. Added an explicit CLI wiring note: `src/sfno_training/train.py` must instantiate `PlasimTrainer` (not stock `Trainer`).
>   2. **Dataloader override (Blocker).** Stock `Trainer.__init__` imports `get_dataloader` at `deterministic_trainer.py:36` (module-level import) and calls it at `:104` / `:105`. v6 never said how stock `get_dataloader` would return a `PlasimForcingDataset` instead of `MultifilesDataset`. v7 specifies a second monkey-patch: `deterministic_trainer.get_dataloader = plasim_get_dataloader` (inside `PlasimTrainer.__init__` before `super().__init__()`). `plasim_get_dataloader` wraps `PlasimForcingDataset` with the same `(dataloader, dataset, sampler)` triple stock `get_dataloader` returns (including `lat_lon`, `get_output_normalization`, `get_input_normalization` attrs). Same monkey-patch applied to `inferencer.get_dataloader` inside `PlasimInferencer`.
>   3. **Invalid hard-reject test (Major).** v6's `except AssertionError: pass` catches the test's own `AssertionError` from the `raise AssertionError(...)` inside `try`, so the test always passes. v7 uses `pytest.raises(AssertionError, match="channels must be")`.
>   4. **Stale LossHandler when `n_future` changes (Major).** v6 Phase 4b constructs `loss_fn = LossHandler(params)` while `params.n_future == 0`, then sets `params.n_future = 1` and reuses `loss_fn` for the two-step loss (`docs/plasim_makani_packager_plan.md:533` in v6). `LossHandler.channel_weights` tiling is frozen at construction, so shapes would be wrong. v7 re-instantiates `loss_fn = LossHandler(params)` after the `n_future` flip.
>   5. **YAML anchor (Moderate).** v6's rendered YAML uses `<<: *BASE_CONFIG` referencing an anchor defined in `makani/config/sfnonet.yaml`. Standalone load via `YParams(path, config_name)` fails without the anchor in the same file. v7 renders the file as self-contained (no anchor merge), inlining the required `BASE_CONFIG` keys.
>
> - **v6 (2026-04-24):** Codex round 5. Six fixes:
>   1. **Wrapper wiring (Blocker).** v5 defined `PlasimSingleStepWrapper` / `PlasimMultiStepWrapper` but never said how Makani's `model_registry.get_model()` (`model_registry.py:204-207`) — hard-coded to construct stock wrappers — would actually use them. v6 specifies a one-line monkey-patch in `PlasimDriver.__init__` that swaps `model_registry.SingleStepWrapper` / `MultiStepWrapper` for the PlaSim subclasses *before* `get_model` is called from `deterministic_trainer.py:132` / `inferencer.py:175`. Added to the trainer-patch contract.
>   2. **Dataloader shape convention (Major).** Stock `MultifilesDataset.get_sample_at_index` (`data_loader_multifiles.py:411`) returns `(n_history+1, C, H, W)` even when `n_history=0`. v5's Phase 4b smoke test used squeezed 3D shapes `(C, H, W)`. v6 rewrites Phase 4b to assert `(1, 52, 64, 128)` for inp_state etc., then batch to `(B, 1, C, H, W)` and call `preprocessor.flatten_history` before the wrapper.
>   3. **`flatten_history` in training loop (Major).** v5 pseudocode skipped the flatten step. Stock trainer (`deterministic_trainer.py:474-478`) does `inp, tar = preprocessor.cache_unpredicted_features(...); inp = preprocessor.flatten_history(inp); tar = preprocessor.flatten_history(tar)` before calling the wrapper. v6 updates the training-loop pseudocode to mirror this exactly.
>   4. **Hard-assert in `PlasimPreprocessor.append_history` (Major).** v5 silently sliced any `x2.shape[1] > 52`, which could hide real channel bugs. v6 asserts `x2.dim() == 4` and `x2.shape[1] in {52, 53}` before slicing; anything else raises.
>   5. **Removed stale contradictions (Moderate).** v5 said "stock SingleStepWrapper + patched MultiStepWrapper" (§Goal), "SingleStepWrapper + dummy model" (Phase 4b), but also "use PlasimPreprocessor" (§Trainer-patch contract). v6 collapses this to a single consistent story: PlaSim wrappers + PlaSim preprocessor everywhere; stock wrappers appear only in the negative multistep regression test.
>   6. **Positive rollout smoke test (Moderate).** v5 only had a negative test (unpatched multistep fails). v6 adds a positive two-step smoke test: `PlasimMultiStepWrapper` + `PlasimPreprocessor` succeed at rollout and produce `(B, 53·(n_future+1), H, W)`.
>
> - **v5 (2026-04-24):** Codex round 4. Six fixes:
>   1. **Rollout-path patch scope.** v4's `PatchedMultiStepWrapper` only covers `_forward_train`; stock `deterministic_trainer.py:661` and `inferencer.py:620` *also* call `self.preprocessor.append_history(..., pred, ...)` directly. Those callers set `self.preprocessor = self.model.preprocessor` (both files, lines 133 and 176 respectively), so v5 shifts the strip into a `PlasimPreprocessor(Preprocessor2D)` subclass whose `append_history` auto-slices `pred[:, :n_state_channels]`. Injected once at the custom stepper's `__init__`, propagates to trainer + inferencer automatically via the `self.model.preprocessor` linkage.
>   2. **Phase 4b forcing shape bug.** v4 had `xz` 4D and `yz` 5D, so `cache_unpredicted_features → append_history`'s `.copy_()` would fail on shape mismatch before reaching the 58-vs-59 channel error. v5: both `xz` and `yz` are 5D `(B, T, C, H, W)` consistently.
>   3. **n_future > 0 dataloader test.** v4 hand-built a 2-step target. v5 actually instantiates `PlasimForcingDataset(n_future=1)` and asserts returned shapes `inp_state=(1, 52, H, W)`, `tar=(2, 53, H, W)`, `inp_forcing=(1, 6, H, W)`, `tar_forcing=(2, 6, H, W)` — matching stock dataloader's `(n_future+1, C, H, W)` convention (`data_loader_multifiles.py:411`).
>   4. **Non-stock metadata warning + validator.** The metadata has `h5_path=fields_state` (52) with `coords.channel` = 53 names. Stock Makani dataloaders reading this will crash on channel-count mismatch. v5 adds an explicit "do not use stock dataloaders with this metadata" warning and a validator that checks `coords.channel[:52] == channel_state` and `coords.channel[52] == channel_diagnostic[0]`.
>   5. **Soften ACC claim.** v4 said the 53-wide `time_means.npy` makes ACC over all outputs "work." Stock metrics use ERA5 names (`makani/makani/utils/metric.py:239`) that won't select PlaSim channels. v5 says "shape-compatible for future PlaSim metrics," not active today.
>   6. **Timestamp wording.** v4 said "do not anchor to Gregorian epoch." Stock loader converts int64 seconds via `dt.datetime.fromtimestamp` (`data_helpers.py:158`, Unix-epoch interpretation) unless `relative_timestamp=True`. Synthetic seconds starting at 0 are fine either way because uniform spacing is what matters. v5 clarifies: custom loader passes `relative_timestamp=True` explicitly.
>
> - **v4 (2026-04-24):** Codex round 3 reviewed v3 and surfaced five trainer-side contract bugs. v4 fixes:
>   1. **Loss channel contract.** `LossHandler` sizes from `params.channel_names` (`loss.py:88`). v3's `channel_names=52` with a `(B,53,H,W)` target mismatches. v4 sets `channel_names = 53 target names` (state + diagnostic), stats files for loss are 53-wide (sliced by `out_channels`), and the "input" is a narrower slice handled by the custom dataloader.
>   2. **Manual forcing concat is wrong.** v3 said "concat forcing to inp before model forward" AND "use `cache_unpredicted_features`" — double-counting. Stock wrappers already call `append_unpredicted_features(inp)` (`stepper.py:34,82,125`). v4: trainer caches `xz/yz` via `cache_unpredicted_features` and passes **only `inp_state`** to the wrapper; the wrapper concats forcing internally.
>   3. **Diagnostic strip must happen in the rollout loop.** `append_history` at `stepper.py:112` returns raw `pred` when `n_history=0` (`preprocessor.py:237`). Feeding 53-channel pred back means next-step `append_unpredicted_features(pred)` = 53+6 = 59, mismatch. v4 trainer patch: subclass `MultiStepWrapper` to slice `pred[:, :52]` before `append_history`.
>   4. **N_in/N_out override timing.** `driver._set_data_shapes` (`driver.py:149`) reads `dataset.in_channels/out_channels`. Override must happen *after* dataloader shape setup, *before* model/loss/metric construction. v4 calls out exact insertion point in the trainer.
>   5. **Phase 4b smoke was under-specified.** Passing a plain dict to `Preprocessor2D` fails because it needs attribute access to many resampling/history config fields. v4 builds a full YParams object, uses `SingleStepWrapper`, and adds a negative test that exercises the multistep rollout mismatch without the strip patch (so the trainer PR can't silently regress).
>
>   Also fixed: `time_means.npy` is 53-wide for ACC over state + diagnostic; all YAML examples show concrete absolute paths (no ellipses); PlaSim time provenance captures original `time:units`, `calendar`, and raw days-since in-file.

---

## Goal

Produce a three-dataset HDF5 layout from PlaSim postprocessor + emulator-adaptor output under `sim52`, compatible with a minimally-patched Makani SFNO training path. The layout preserves the source SFNO emulator's input/output contract:

- **Model input at forward time:** 52 feedback state channels + 6 prescribed forcing = **58**.
- **Model output:** 52 feedback state + 1 diagnostic (`pr_6h`) = **53**.
- **Loss target:** 53 channels (state + diagnostic).
- **Rollout feedback state:** `prediction[:, :52]` — diagnostic stripped at the trainer's rollout loop.
- **Rollout forcing at next step:** `tar_forcing[t+1]` — from truth, never model output.

Prescribed forcing does not drift, `pr_6h` never feeds back, and the source emulator's 58-in/53-out tensor signature is preserved via `PlasimSingleStepWrapper` / `PlasimMultiStepWrapper` (both re-using `PlasimPreprocessor`, a `Preprocessor2D` subclass whose `append_history` auto-strips diagnostic channels from `pred` before feedback).

This is a **downstream packager** standalone at `src/plasim_makani_packager/`. It defines the data contract and the trainer-patch contract; the patched Makani wrappers live under `src/sfno_training/` per `docs/sfno_training_extraction_plan.md`.

---

## Interview decisions (locked)

| Decision | Choice |
|---|---|
| Makani integration | Path C — three HDF5 datasets consumed via `PlasimForcingDataset`; `PlasimSingleStepWrapper` / `PlasimMultiStepWrapper` (both installing `PlasimPreprocessor`) substituted into `model_registry` via monkey-patch inside `PlasimTrainer.__init__`; `deterministic_trainer.get_dataloader` rebound to `_plasim_get_dataloader` in the same `__init__`; generic `cache_unpredicted_features` hook carries the 6 prescribed channels. Inference path not patched — out of scope (v8). |
| `/fields_state` (52) | `pl, tas` + 50 upper-air. Feedback channels. |
| `/fields_diagnostic` (1) | `pr_6h`. Predicted + in loss, never fed back. |
| `/forcing` (6) | `lsm, sg, z0, sst, rsdt, sic`. Prescribed; never predicted. |
| `params.channel_names` (YAML) | **53 target names** = state + diagnostic. Sized for loss construction. |
| Stats files for `/fields_state` + `/fields_diagnostic` (loss side) | **53-wide** (`global_means.npy`, `global_stds.npy`, `time_means.npy`) spanning state[0..51] ‖ diagnostic[52]. Sliced by `out_channels = [0..52]`. |
| Stats files for `/forcing` | 6-wide (`forcing_*.npy`). Consumed by custom dataloader for forcing normalization. |
| State input normalization | Custom dataloader slices `global_means/stds[:, :52, ...]` for `in_bias/in_scale`. |
| `in_channels` (dataset attr) | `[0..51]` (52) — set by custom dataloader for correct `N_in_channels` wiring. |
| `out_channels` (dataset attr) | `[0..52]` (53). |
| `N_in_channels` (model build) | 58 — overridden after `_set_data_shapes`, before model build. |
| `N_out_channels` (model build) | 53 — unchanged from parser. |
| `rsdt_method` | astronomical |
| `sst` land-fill | 271.35 K |
| `sic` source | emulator_adaptor |
| Sigma ordering | `ta1 = TOA`, `ta10 = surface` |
| Train / valid / test | 3–100 / 101–120 / 121–128 |
| Output root | `/scratch/11114/zhixingliu/AI-RES/data/makani/sim52_astro_64x128/` |
| Timestamp | Synthetic monotonic int64 seconds; `/time_plasim` sibling carries original PlaSim days-since; file attrs carry original `time:units` + `calendar` |
| CLI | Standalone (`src/plasim_makani_packager/packager.py`) |

---

## Verified Makani control flow (ground for v4 design)

| Path | File | Behavior |
|---|---|---|
| Wrapper input concat | `makani/makani/models/stepper.py:34,82,125` | `SingleStepWrapper.forward` and both `MultiStepWrapper` paths call `preprocessor.append_unpredicted_features(inp)` first. Forcing concat happens inside the wrapper; the trainer must pass only `inp_state`. |
| Forcing carry across steps | `makani/makani/models/preprocessor.py:202-224` | `append_history` copies `unpredicted_tar[:, step+1:step+2, ...]` into `unpredicted_inp`. Already correct, no patch needed here. |
| Rollout feedback (the bug) — in-scope call sites | `makani/makani/models/stepper.py:112` (training rollout via `MultiStepWrapper`) and `makani/makani/utils/training/deterministic_trainer.py:661` (validation rollout). `makani/makani/utils/inference/inferencer.py:620` is the same bug but the inference path is out of scope in v8+ (see §"What is out of scope"). Both in-scope call sites do `self.preprocessor.append_history(..., pred, ...)`, which returns `pred` unchanged when `n_history=0` (`preprocessor.py:237`). Next iter: `append_unpredicted_features(pred)` = 53+6=59, model expects 58. `Trainer` sets `self.preprocessor = self.model.preprocessor` (line 133). v5 patch: ship `PlasimPreprocessor(Preprocessor2D)` that auto-slices `pred[:, :n_state_channels]` inside `append_history`. Injected at PlaSim stepper `__init__`; propagates to the trainer's validation rollout via `self.model.preprocessor`. |
| Loss channel sizing | `makani/makani/utils/loss.py:88, 101-108` | `n_channels = len(params.channel_names)`. `bias, scale` sliced by `params.out_channels`. If `channel_names=53` and `out_channels=[0..52]`, loss gets `(1, 53, 1, 1)` bias/scale. ✓ |
| Metric / ACC climatology | `makani/makani/utils/dataloaders/data_helpers.py:123` | `time_means_path` loaded and sliced by `params.out_channels`. Needs 53-wide `time_means.npy`. |
| N_in/N_out timing | `makani/makani/utils/driver.py:149-150` | `params.N_in_channels = len(dataset.in_channels)`. Runs right after dataset build; overrides must happen *after* this line and *before* model build. |
| Custom h5 dataset name | `makani/makani/utils/dataloader.py:85` | Stock multifile construction does **not** pass `dataset_path`; `MultifilesDataset` defaults to `"fields"`. Custom loader must set `dataset_path="fields_state"` explicitly in its own `__init__`. |
| Wrapper instantiation | `makani/makani/models/model_registry.py:204-207` | `get_model()` hard-codes `if multistep: MultiStepWrapper else: SingleStepWrapper`. Called from `deterministic_trainer.py:132` (inferencer is out of scope). `PlasimTrainer.__init__` monkey-patches `model_registry.SingleStepWrapper` / `MultiStepWrapper` before `super().__init__`. |
| Dataloader construction | `makani/makani/utils/training/deterministic_trainer.py:36, 104-105` | `Trainer.__init__` does `from makani.utils.dataloader import get_dataloader` (module-level) and calls it for train + valid paths. To substitute `PlasimForcingDataset` we rebind `deterministic_trainer.get_dataloader` (the module-level name, **not** `makani.utils.dataloader.get_dataloader`) in `PlasimTrainer.__init__` before `super().__init__`. |
| Inference-path forcing hole | `makani/makani/utils/inference/inferencer.py:554-589` | Stock `_inference_indexlist` unpacks `(inp, tinp)` / `(tar, ttar)` when `add_zenith=False`, sets `inpz=tarz=None`, and calls `cache_unpredicted_features(None, None, None, None)` — forcing would be silently dropped. Inference not patched in v8; declared out of scope. |
| Training entry class | `makani/makani/utils/training/deterministic_trainer.py:58` | Stock CLI instantiates `Trainer(Driver)` directly, so overrides must be on `Trainer`, not `Driver`. `PlasimTrainer(Trainer)` keeps all stock behavior and layers the two monkey-patches + `_set_data_shapes` override on top. |
| Dataloader sample shape | `makani/makani/utils/dataloaders/data_loader_multifiles.py:411` | `get_sample_at_index` returns `(n_history+1, C, H, W)` for input and `(n_future+1, C, H, W)` for target — always 4D, even when `n_history=0`. Our stub + production `PlasimForcingDataset` preserve this contract. |
| Flatten history before forward | `makani/makani/utils/training/deterministic_trainer.py:474-478` | Stock trainer does `inp, tar = preprocessor.cache_unpredicted_features(...); inp = preprocessor.flatten_history(inp); tar = preprocessor.flatten_history(tar)` before passing to the wrapper. The training-loop pseudocode must match this. |

---

## Channel list (locked, v4)

### `/fields_state` — 52 feedback channels, stock loader reads this via `dataset_path="fields_state"`

```
 0  pl
 1  tas
 2..11   ta1..ta10       (ta1 = TOA, ta10 = surface)
12..21   ua1..ua10
22..31   va1..va10
32..41   hus1..hus10
42..51   zg1..zg10
```

### `/fields_diagnostic` — 1 channel (loss-only; never in input)

```
 0  pr_6h
```

### `/forcing` — 6 prescribed channels

```
 0  lsm       (static, per-step repeated)
 1  sg        (static)
 2  z0        (varying — land-static, ocean-dynamic; Charnock/sea-state roughness 1.5e-5..1e-3 m)
 3  sst       (varying, land-filled 271.35 K)
 4  rsdt      (varying, astronomical method)
 5  sic       (varying, [0,1])
```

### Target construction in custom dataloader

```
tar = concat(/fields_state[t+dt], /fields_diagnostic[t+dt], dim=1)   # (53, H, W)
```

### At model forward time (PlasimSingleStepWrapper / PlasimMultiStepWrapper)

```
inp_state:    /fields_state[t]                         (52, H, W)
inp_forcing:  /forcing[t]                              (6,  H, W)

trainer:
    preprocessor.cache_unpredicted_features(
        x=inp_state, y=tar, xz=inp_forcing, yz=tar_forcing)
    pred = wrapper(inp_state)  # PlasimSingleStepWrapper.forward → append_unpredicted_features → (58,) at model input
                               # model → (53,) pred
loss(pred, tar)                # both (B, 53, H, W) → matches
```

### At rollout step boundary (PlasimPreprocessor.append_history strips diagnostic)

```
# Inside PlasimMultiStepWrapper._forward_train → self.preprocessor.append_history(inpt, pred, step):
#   PlasimPreprocessor.append_history asserts pred.shape[1] ∈ {52, 53}, slices pred[:, :52], calls super.
pred_state = pred[:, :52]                             # auto-stripped by PlasimPreprocessor
inpt       = preprocessor.append_history(inpt, pred, step)        # (52,)
# next step: append_unpredicted_features(inpt=52) → (58,) → model → (53,) ✓
```

---

## Source data

### Postprocess (MOST) — already staged

`/scratch/11114/zhixingliu/AI-RES/data/postproc/sim52/MOST.{YYYY:04d}.nc` for `YYYY ∈ 0001..0128`, 6-hourly proleptic Gregorian, 64 Gauss-Legendre lats × 128 lons. Each file has its own `time:units = "days since YYYY-MM-DD HH:MM:SS"` (verified: year 0001 = "0006-08-25", year 0002 = "0007-08-01"). All emulator-contract state vars present; `sst`/`rsdt` absent (come from adaptor).

### Boundary (adaptor) — to be produced (Phase 0)

`/scratch/11114/zhixingliu/AI-RES/data/boundary_astro/sim52/boundary.{YYYY:04d}.nc` via `src/emulator_adaptor/adaptor.py --rsdt-method astronomical`.

---

## Pipeline phases

### Phase 0 — Boundary prerequisite

```bash
python3 src/emulator_adaptor/adaptor.py --sims 52 --years 1 128 --count-tasks   # → 128
# Edit submit.slurm with the var block below, then:
sbatch --array=0-127 src/emulator_adaptor/submit.slurm

# Per-task invocation:
python3 src/emulator_adaptor/adaptor.py \
    --sims 52 --years 1 128 \
    --rsdt-method astronomical \
    --input-root  /scratch/11114/zhixingliu/AI-RES/data/postproc \
    --output-root /scratch/11114/zhixingliu/AI-RES/data/boundary_astro \
    --task-index $SLURM_ARRAY_TASK_ID
```

Gate: all 128 boundary files written and passing the adaptor's internal `_validate()`.

### Phase 1 — Package one sim-year

Per `YYYY ∈ 0001..0128`:

1. **Open both sources** with `xarray.open_dataset(..., decode_times=False)`.
2. **Cross-validate shared coords** (`time`, `lat`, `lon` byte-identical); `boundary.attrs["rsdt_method"] == "astronomical"`.
3. **Validate sic clipping** (v9 fix — split into hard-fail and quantify):
    - **Hard-fail (the adaptor must do clip-only, nothing else):** compare `adaptor.sic` against `np.clip(MOST.sic, 0.0, 1.0)` cell-wise; fail if `np.max(np.abs(adaptor.sic - np.clip(MOST.sic, 0, 1))) > 1e-6`, or if shape mismatch, or if NaN-mask parity with `MOST.sic` is broken.
    - **Quantify only (report, no hard-fail):** `max_abs_diff = np.max(np.abs(MOST.sic - adaptor.sic))`, `mean_abs_diff`, `fraction_cells_changed_by_clip = np.mean((MOST.sic < 0) | (MOST.sic > 1))`. Write to `validation/sic_clip_report_{YYYY}.json`. This can legitimately be > 1e-3 whenever PlaSim produced out-of-range sic — that is the adaptor's job to fix, not evidence of corruption.
4. **Fill sst over land**: `sst = np.where(np.isnan(sst), 271.35, sst).astype(np.float32)`. Record `sst_land_fill_fraction` per sim-year.
5. **Stack**:
   - `/fields_state`: `(T, 52, 64, 128)` float32
   - `/fields_diagnostic`: `(T, 1, 64, 128)` float32
   - `/forcing`: `(T, 6, 64, 128)` float32
6. **Resolve split**: train (3–100), valid (101–120), test (121–128); skip 1 and 2.
7. **Write `{output-root}/{split}/MOST.{YYYY:04d}.h5`**:

   ```
   MOST.YYYY.h5
   ├── /fields_state        (T, 52, 64, 128) float32  chunks (1, 52, 64, 128)
   ├── /fields_diagnostic   (T,  1, 64, 128) float32  chunks (1,  1, 64, 128)
   ├── /forcing             (T,  6, 64, 128) float32  chunks (1,  6, 64, 128)
   ├── /timestamp           (T,)             int64    — synthetic dataset-globally monotonic seconds
   ├── /time_plasim         (T,)             float64  — raw PlaSim days-since-reference from source NetCDF
   ├── /channel_state       (52,)            ascii
   ├── /channel_diagnostic  (1,)             ascii
   ├── /channel_forcing     (6,)             ascii
   ├── /lat                 (64,)            float64
   └── /lon                 (128,)           float64
   ```

   Dim scales attached on every 4D dataset: dim 0 → `timestamp`, dim 1 → `channel_*`, dim 2 → `lat`, dim 3 → `lon`.

   **File-level attrs (PlaSim timing provenance, v4 addition):**
   - `rsdt_method = "astronomical"`
   - `source_postproc`, `source_boundary` (absolute paths)
   - `packager_git_sha`
   - `sst_land_fill_K = 271.35`
   - `sst_land_fill_fraction = <float>`
   - `plasim_time_units = "days since 0006-08-25 00:00:00"` (copied from source NetCDF, per-file)
   - `plasim_calendar = "proleptic_gregorian"`

### Phase 2 — Stats (training split only)

**One consolidated set of stats spanning `/fields_state` + `/fields_diagnostic` (53-wide, channel order state[0..51] ‖ diagnostic[52])**, matching Makani's loss and ACC requirements — loss slices by `out_channels = [0..52]`, and the custom `_plasim_get_dataloader` slices `[:, :52, ...]` for state-only `in_bias/in_scale`.

- `stats/global_means.npy` — `(1, 53, 1, 1)` float32. Channels `[0..51]` = state means, channel `[52]` = diagnostic mean.
- `stats/global_stds.npy` — `(1, 53, 1, 1)` float32.
- `stats/time_means.npy` — `(1, 53, 64, 128)` float32. 53-wide so any future PlaSim-specific ACC metric over `out_channels = [0..52]` is shape-compatible. Stock Makani metrics (`makani/makani/utils/metric.py:239`) default to ERA5 names (`z500`, `t2m`, etc.) that do not match our channels, so ACC is effectively disabled for v1 unless the trainer PR registers PlaSim metric names.

**Forcing stats (6-wide, custom dataloader consumes):**

- `stats/forcing_global_means.npy` — `(1, 6, 1, 1)` float32.
- `stats/forcing_global_stds.npy` — `(1, 6, 1, 1)` float32.
- `stats/forcing_time_means.npy` — `(1, 6, 64, 128)` float32 (for optional forcing-side diagnostics; not required by Makani's stock metrics).

Single-pass Welford across training years 3–100 only. Hard-fail on any channel with `std < MIN_STD_EPSILON = 1e-6`, per-channel error message. `sst_land_sentinel_notes.txt` documents the 271.35 K fill's impact on `sst` zscore stats.

### Phase 3 — Metadata + Makani config

**`{output-root}/metadata/data.json`:**

```json
{
  "dataset_name": "plasim-sim52-astro-64x128",
  "h5_path": "fields_state",
  "diagnostic_h5_path": "fields_diagnostic",
  "forcing_h5_path": "forcing",
  "dims": ["time", "channel", "lat", "lon"],
  "dhours": 6,
  "coords": {
    "grid_type": "legendre-gauss",
    "lat": [<64 values>],
    "lon": [<128 values>],
    "channel":             [<53 target names: 52 state + "pr_6h">],
    "channel_state":       [<52 state names>],
    "channel_diagnostic":  ["pr_6h"],
    "channel_forcing":     ["lsm","sg","z0","sst","rsdt","sic"]
  },
  "attrs": {
    "description": "PlaSim sim52 postproc 64x128, astronomical rsdt, three-dataset layout for patched Makani",
    "source_postproc_root": "/scratch/11114/zhixingliu/AI-RES/data/postproc/sim52",
    "source_boundary_root": "/scratch/11114/zhixingliu/AI-RES/data/boundary_astro/sim52",
    "rsdt_method": "astronomical",
    "sst_land_fill_K": 271.35,
    "train_years": [3, 100],
    "valid_years": [101, 120],
    "test_years":  [121, 128],
    "packager_version": "<git sha>",
    "requires_patched_makani": true,
    "trainer_patch_contract_url": "docs/plasim_makani_packager_plan.md#trainer-patch-contract"
  }
}
```

**Makani YAML** (`{output-root}/config/plasim_sim52_astro_64x128.yaml`) — **standalone, concrete absolute paths, no ellipses, no external anchor merge** (v7 fix #5: stock `makani/config/sfnonet.yaml` defines `base_config: &BASE_CONFIG`; a merge via `<<: *BASE_CONFIG` requires the anchor in the same file. v7 inlines every required key explicitly so `YParams(path, "plasim_sim52_astro_64x128")` resolves without referencing stock YAMLs):

```yaml
plasim_sim52_astro_64x128:

    # --- Dataset paths ---
    metadata_json_path: "/scratch/11114/zhixingliu/AI-RES/data/makani/sim52_astro_64x128/metadata/data.json"
    train_data_path:   "/scratch/11114/zhixingliu/AI-RES/data/makani/sim52_astro_64x128/train"
    valid_data_path:   "/scratch/11114/zhixingliu/AI-RES/data/makani/sim52_astro_64x128/valid"
    inf_data_path:     "/scratch/11114/zhixingliu/AI-RES/data/makani/sim52_astro_64x128/test"
    exp_dir:           "/scratch/11114/zhixingliu/AI-RES/runs/sim52_astro_64x128"

    # --- Image / time ---
    img_shape_x: 64
    img_shape_y: 128
    dhours: 6
    dt: 1
    multifiles: true

    # --- Grids ---
    model_grid_type: "legendre-gauss"
    data_grid_type:  "legendre-gauss"
    sht_grid_type:   "legendre-gauss"

    # --- Channel names (53 loss-target names; state 52 + diagnostic 1) ---
    channel_names: ["pl","tas","ta1","ta2","ta3","ta4","ta5","ta6","ta7","ta8","ta9","ta10","ua1","ua2","ua3","ua4","ua5","ua6","ua7","ua8","ua9","ua10","va1","va2","va3","va4","va5","va6","va7","va8","va9","va10","hus1","hus2","hus3","hus4","hus5","hus6","hus7","hus8","hus9","hus10","zg1","zg2","zg3","zg4","zg5","zg6","zg7","zg8","zg9","zg10","pr_6h"]

    # --- PlaSim-specific plumbing (consumed by PlasimForcingDataset / PlasimPreprocessor) ---
    diagnostic_h5_path: "fields_diagnostic"
    n_state_channels: 52
    n_diagnostic_channels: 1
    forcing_h5_path: "forcing"
    n_forcing_channels: 6
    forcing_channel_names: ["lsm","sg","z0","sst","rsdt","sic"]
    forcing_global_means_path: "/scratch/11114/zhixingliu/AI-RES/data/makani/sim52_astro_64x128/stats/forcing_global_means.npy"
    forcing_global_stds_path:  "/scratch/11114/zhixingliu/AI-RES/data/makani/sim52_astro_64x128/stats/forcing_global_stds.npy"
    forcing_time_means_path:   "/scratch/11114/zhixingliu/AI-RES/data/makani/sim52_astro_64x128/stats/forcing_time_means.npy"

    # --- Normalization stats (53-wide; sliced by out_channels inside LossHandler) ---
    normalization: "zscore"
    global_means_path: "/scratch/11114/zhixingliu/AI-RES/data/makani/sim52_astro_64x128/stats/global_means.npy"
    global_stds_path:  "/scratch/11114/zhixingliu/AI-RES/data/makani/sim52_astro_64x128/stats/global_stds.npy"
    time_means_path:   "/scratch/11114/zhixingliu/AI-RES/data/makani/sim52_astro_64x128/stats/time_means.npy"

    # --- Architecture (stock SFNO keys; would otherwise come from BASE_CONFIG) ---
    nettype: "SFNO"
    filter_type: "linear"
    scale_factor: 3
    embed_dim: 384
    num_layers: 8
    complex_activation: "real"
    normalization_layer: "instance_norm"
    hard_thresholding_fraction: 1.0
    use_mlp: !!bool True
    mlp_mode: "serial"
    mlp_ratio: 2
    separable: !!bool False
    operator_type: "dhconv"
    activation_function: "gelu"
    pos_embed: "none"

    # --- Extra input channels (all off — prescribed forcing handles this) ---
    add_grid:      !!bool False
    add_zenith:    !!bool False
    add_orography: !!bool False
    add_landmask:  !!bool False

    # --- Training (stock BASE_CONFIG defaults inlined) ---
    losses:
    -   type: "l2"
        channel_weights: "constant"
        temp_diff_normalization: !!bool False
        parameters:
            squared: !!bool True

    lr: 1.0E-3
    n_eval_samples: 512
    n_train_samples_per_epoch: 8192
    max_epochs: 100
    batch_size: 8
    weight_decay: 0.0
    n_history: 0
    n_future: 0
    valid_autoreg_steps: 3
    prediction_type: "iterative"

    scheduler: "ReduceLROnPlateau"
    scheduler_T_max: 100
    scheduler_factor: 0.5
    scheduler_patience: 10
    scheduler_step_size: 5
    scheduler_gamma: 0.5
    lr_warmup_steps: 0
    optimizer_type: "AdamW"
    optimizer_beta1: 0.9
    optimizer_beta2: 0.95
    optimizer_max_grad_norm: 32

    num_data_workers: 2
    num_visualization_workers: 2
    crop_size_x: None
    crop_size_y: None

    ics_type: "specify_number"
    save_raw_forecasts: !!bool True
    save_channel:       !!bool False
    masked_acc:         !!bool False
    maskpath: None
    perturb:    !!bool False
    add_noise:  !!bool False
    noise_std:  0.0
    pretrained: !!bool False

    # --- Logging ---
    log_to_screen: !!bool True
    log_to_wandb:  !!bool False
    log_video: 0
    save_checkpoint: "legacy"
    verbose: !!bool False

    wireup_info: "mpi"
    wireup_store: "tcp"
```

### Phase 4 — Validation

**Phase 4a — Structural (login node, fast):**

- Every HDF5:
  - `/fields_state` (T, 52, 64, 128) float32 finite.
  - `/fields_diagnostic` (T, 1, 64, 128) float32 finite.
  - `/forcing` (T, 6, 64, 128) float32 finite.
  - `/timestamp` int64 monotonic, `diff == 21600`.
  - `/time_plasim` float64 strictly monotonic within file.
  - `/channel_*` match master lists in order.
  - Dim scales attached on all 4D datasets.
  - File attrs `plasim_time_units`, `plasim_calendar`, `rsdt_method`, `sst_land_fill_K` present and consistent.
- Across files: concat `/timestamp` across train+valid+test in chronological year order → strictly increasing.
- Stats: every `*_stds >= MIN_STD_EPSILON`; all `*_means` finite; static-channel staticness verified for `forcing_time_means[0..2]`.

**Phase 4b — Makani end-to-end smoke (v6, uses YParams + PlasimSingleStepWrapper + PlasimMultiStepWrapper + dummy model; positive *and* negative rollout tests):**

```python
# tests/plasim_makani_packager/test_multifile_loader_smoke.py
import json
import numpy as np
import torch
from torch import nn
from makani.utils.YParams import YParams
from makani.utils.parse_dataset_metada import parse_dataset_metadata
from makani.utils.loss import LossHandler
from makani.models.stepper import MultiStepWrapper  # stock — negative test only
from tests.plasim_makani_packager.stub_forcing_loader import (
    PlasimForcingDataset,
    PlasimPreprocessor,
    PlasimSingleStepWrapper,
    PlasimMultiStepWrapper,
)

# 1. Load YAML → full attribute-access params object
params = YParams(
    "/scratch/11114/zhixingliu/AI-RES/data/makani/sim52_astro_64x128/config/plasim_sim52_astro_64x128.yaml",
    "plasim_sim52_astro_64x128",
)

# 2. Populate data-derived fields (lat/lon/channel/...)
parse_dataset_metadata(params.metadata_json_path, params)
assert params.in_channels  == list(range(53))          # parser sees 53 (metadata has 53 names)
assert params.out_channels == list(range(53))
assert params.data_grid_type == "legendre-gauss"
assert params.dhours == 6

# 3. Metadata channel-list consistency validator (v5)
meta = json.load(open(params.metadata_json_path))
assert meta["coords"]["channel"][:52] == meta["coords"]["channel_state"], \
    "coords.channel[:52] must equal coords.channel_state"
assert meta["coords"]["channel"][52]  == meta["coords"]["channel_diagnostic"][0], \
    "coords.channel[52] must equal coords.channel_diagnostic[0]"

# 4. Build the custom dataloader. Single-step case: n_history=0, n_future=0.
ds = PlasimForcingDataset(
    location=params.train_data_path,
    dt=1,
    in_channels=list(range(params.n_state_channels)),                                    # 52
    out_channels=list(range(params.n_state_channels + params.n_diagnostic_channels)),    # 53
    n_forcing_channels=params.n_forcing_channels,
    dataset_path="fields_state",                                                         # explicit — stock default is "fields"
    diagnostic_dataset_path=params.diagnostic_h5_path,
    forcing_dataset_path=params.forcing_h5_path,
    relative_timestamp=True,
    data_grid_type=params.data_grid_type,
    model_grid_type=params.model_grid_type,
    bias=np.load(params.global_means_path),               # (1,53,1,1)
    scale=np.load(params.global_stds_path),               # (1,53,1,1)
    forcing_bias=np.load(params.forcing_global_means_path),
    forcing_scale=np.load(params.forcing_global_stds_path),
)
assert ds.in_channels.tolist()  == list(range(52))
assert ds.out_channels.tolist() == list(range(53))

# v6: stock dataloader returns (n_history+1, C, H, W) even when n_history=0
#     (data_loader_multifiles.py:411). inp/tar shapes include a singleton time dim.
inp_state, tar, inp_forcing, tar_forcing = ds[0]
assert inp_state.shape   == (1, 52, 64, 128), inp_state.shape
assert tar.shape         == (1, 53, 64, 128), tar.shape
assert inp_forcing.shape == (1,  6, 64, 128), inp_forcing.shape
assert tar_forcing.shape == (1,  6, 64, 128), tar_forcing.shape

# 5. Simulate driver._set_data_shapes + PlasimTrainer._set_data_shapes override (v7)
params.N_in_channels  = len(ds.in_channels)                # 52 from dataset
params.N_out_channels = len(ds.out_channels)               # 53
# PlasimTrainer._set_data_shapes override (the v4 fix, v7 relocated):
params.N_in_channels  = params.n_state_channels + params.n_forcing_channels  # 58

# Other Preprocessor2D-required attrs (v4 — replaces v3's plain-dict pass):
params.img_shape_x_resampled       = params.img_shape_x
params.img_shape_y_resampled       = params.img_shape_y
params.img_crop_shape_x            = params.img_shape_x
params.img_crop_shape_y            = params.img_shape_y
params.img_crop_offset_x           = 0
params.img_crop_offset_y           = 0
params.img_local_shape_x           = params.img_shape_x
params.img_local_shape_y           = params.img_shape_y
params.img_local_offset_x          = 0
params.img_local_offset_y          = 0
params.img_local_shape_x_resampled = params.img_shape_x
params.img_local_shape_y_resampled = params.img_shape_y
params.subsampling_factor          = 1
params.n_history                   = 0
params.n_future                    = 0
params.history_normalization_mode  = "none"

# 6. Dummy model: matches the real SFNO's in/out channel contract.
# v9 fix: fresh Conv2d per wrapper ctor — stock wrappers take ownership of the
# returned module, so sharing one `dummy` across SingleStepWrapper +
# MultiStepWrapper double-registers the same parameters and can drift state.
model_handle = lambda: nn.Conv2d(in_channels=58, out_channels=53, kernel_size=1)

# 7. PlasimSingleStepWrapper positive smoke (v6: PlaSim wrapper, not stock)
wrapper = PlasimSingleStepWrapper(params, model_handle)
wrapper.train()
assert isinstance(wrapper.preprocessor, PlasimPreprocessor), \
    "PlasimSingleStepWrapper must install PlasimPreprocessor"

# Batch-and-flatten to mirror stock trainer (deterministic_trainer.py:474-478):
#   inp, tar = preprocessor.cache_unpredicted_features(...)
#   inp = preprocessor.flatten_history(inp)
#   tar = preprocessor.flatten_history(tar)
# v6: all tensors now 5D (B, T, C, H, W). T_in = n_history+1 = 1; T_out = n_future+1 = 1.
B = 1
inp_b = inp_state.unsqueeze(0)                             # (1, 1, 52, 64, 128)
tar_b = tar.unsqueeze(0)                                   # (1, 1, 53, 64, 128)
xz    = inp_forcing.unsqueeze(0)                           # (1, 1,  6, 64, 128)
yz    = tar_forcing.unsqueeze(0)                           # (1, 1,  6, 64, 128)

inp_b, tar_b = wrapper.preprocessor.cache_unpredicted_features(inp_b, tar_b, xz, yz)
inp_b = wrapper.preprocessor.flatten_history(inp_b)        # (1, 52, 64, 128)
tar_b = wrapper.preprocessor.flatten_history(tar_b)        # (1, 53, 64, 128)

pred = wrapper(inp_b)
assert pred.shape == (1, 53, 64, 128), f"PlasimSingleStepWrapper output shape mismatch: {pred.shape}"

# 8. Loss construction smoke
loss_fn = LossHandler(params)
# channel_weights.shape[1] = len(channel_names) = 53 with channel_weights="constant"
assert loss_fn.channel_weights.shape[1] == 53
loss_val = loss_fn(pred, tar_b)
assert torch.isfinite(loss_val).item()

# 9. PlasimPreprocessor.append_history strip unit test (v6 — positive direct test)
x1 = torch.zeros(B, 52, 64, 128)       # (B, n_state, H, W)
x2 = torch.randn(B, 53, 64, 128)       # (B, n_state + n_diagnostic, H, W)
x_out = wrapper.preprocessor.append_history(x1, x2, step=0, update_state=False)
assert x_out.shape == (B, 52, 64, 128), f"expected (B, 52, H, W) after strip, got {x_out.shape}"

# append_history must hard-reject unexpected shapes (v6 fix #4 — no silent slicing)
# v7 fix: use pytest.raises so the test can't trivially self-pass on the
# `raise AssertionError(...)` inside its own try block.
import pytest
with pytest.raises(AssertionError, match="channels must be"):
    wrapper.preprocessor.append_history(x1, torch.zeros(B, 60, 64, 128), step=0)
# Also guard rank
with pytest.raises(AssertionError, match="4D"):
    wrapper.preprocessor.append_history(x1, torch.zeros(B, 53, 64), step=0)

# 10. n_future > 0 dataloader contract (v5: actually instantiate)
ds_multi = PlasimForcingDataset(
    location=params.train_data_path,
    dt=1,
    n_future=1,                                # two-step rollout
    in_channels=list(range(params.n_state_channels)),
    out_channels=list(range(params.n_state_channels + params.n_diagnostic_channels)),
    n_forcing_channels=params.n_forcing_channels,
    dataset_path="fields_state",
    diagnostic_dataset_path=params.diagnostic_h5_path,
    forcing_dataset_path=params.forcing_h5_path,
    relative_timestamp=True,
    data_grid_type=params.data_grid_type,
    model_grid_type=params.model_grid_type,
    bias=np.load(params.global_means_path),
    scale=np.load(params.global_stds_path),
    forcing_bias=np.load(params.forcing_global_means_path),
    forcing_scale=np.load(params.forcing_global_stds_path),
)
inp_state2, tar2, inp_forcing2, tar_forcing2 = ds_multi[0]
# stock dataloader convention: (n_history+1, C, H, W) for inp, (n_future+1, C, H, W) for tar
assert inp_state2.shape   == (1, 52, 64, 128), inp_state2.shape
assert tar2.shape         == (2, 53, 64, 128), tar2.shape
assert inp_forcing2.shape == (1,  6, 64, 128), inp_forcing2.shape
assert tar_forcing2.shape == (2,  6, 64, 128), tar_forcing2.shape

# 11. PlasimMultiStepWrapper positive two-step rollout smoke (v6 — NEW)
params.n_future = 1
# v7 (Codex round 6 fix #4): LossHandler's multistep_weight buffer is frozen at
# __init__ against (params.n_future + 1). Reusing the n_future=0 loss_fn would
# multiply by the wrong `multistep_weight`. Rebuild after flipping n_future.
loss_fn_multi = LossHandler(params)
# v8 (Codex round 7 fix #3): channel_weights is stored at width n_channels (53) regardless of
# n_future — the tile by (n_future+1) happens inside LossHandler.forward (loss.py:437-440).
# The per-step tiling lives in multistep_weight (loss.py:194-195).
assert loss_fn_multi.channel_weights.shape[1] == 53, \
    f"LossHandler.channel_weights is n_channels-wide regardless of n_future; got {loss_fn_multi.channel_weights.shape[1]}"
assert loss_fn_multi.multistep_weight.shape[1] == 53 * 2, \
    f"LossHandler.multistep_weight should be ncw*(n_future+1)=106 after rebuild; got {loss_fn_multi.multistep_weight.shape[1]}"

ms_patched = PlasimMultiStepWrapper(params, model_handle)
ms_patched.train()
assert isinstance(ms_patched.preprocessor, PlasimPreprocessor)

inp_b2 = inp_state2.unsqueeze(0)                           # (1, 1, 52, 64, 128)
tar_b2 = tar2.unsqueeze(0)                                 # (1, 2, 53, 64, 128)
xz2    = inp_forcing2.unsqueeze(0)                         # (1, 1,  6, 64, 128)
yz2    = tar_forcing2.unsqueeze(0)                         # (1, 2,  6, 64, 128)

inp_b2, tar_b2 = ms_patched.preprocessor.cache_unpredicted_features(inp_b2, tar_b2, xz2, yz2)
inp_flat  = ms_patched.preprocessor.flatten_history(inp_b2)   # (1, 52, 64, 128)
tar_flat  = ms_patched.preprocessor.flatten_history(tar_b2)   # (1, 2*53, 64, 128)

pred_ms = ms_patched(inp_flat)
# MultiStepWrapper concats per-step predictions along channel dim: (B, out_chans*(n_future+1), H, W)
assert pred_ms.shape == (1, 53 * 2, 64, 128), f"two-step rollout shape mismatch: {pred_ms.shape}"
loss_ms = loss_fn_multi(pred_ms, tar_flat)
assert torch.isfinite(loss_ms).item()

# 12. Negative regression guard: stock MultiStepWrapper (no PlasimPreprocessor) must FAIL on two-step
# (so the trainer-patch PR cannot silently regress away the strip.)
ms_stock = MultiStepWrapper(params, model_handle)
ms_stock.train()
# Re-create inputs (cache_unpredicted_features mutates preprocessor state)
inp_b3 = inp_state2.unsqueeze(0)
tar_b3 = tar2.unsqueeze(0)
xz3    = inp_forcing2.unsqueeze(0)
yz3    = tar_forcing2.unsqueeze(0)
inp_b3, tar_b3 = ms_stock.preprocessor.cache_unpredicted_features(inp_b3, tar_b3, xz3, yz3)
inp_flat3 = ms_stock.preprocessor.flatten_history(inp_b3)
try:
    _ = ms_stock(inp_flat3)
    raise AssertionError(
        "Stock MultiStepWrapper produced output — expected shape mismatch at step 2 "
        "because pred=53 feeding back into append_unpredicted_features gives 59, but model expects 58."
    )
except RuntimeError as e:
    assert "58" in str(e) or "53" in str(e) or "channel" in str(e).lower(), \
        f"Expected channel-mismatch runtime error, got: {e}"
```

**What this proves now:**

- YAML → YParams → parse_dataset_metadata → custom dataloader → PlasimPreprocessor → PlasimSingleStepWrapper → LossHandler single-step chain works end-to-end.
- `PlasimPreprocessor.append_history` strips 53→52 on the positive test, and hard-rejects unexpected shapes (no silent slicing).
- `PlasimMultiStepWrapper` two-step rollout produces the expected `(B, 53·(n_future+1), H, W)`.
- The multistep mismatch is reproducible with stock `MultiStepWrapper` — which means removing `PlasimPreprocessor` from the rollout path is a hard regression.
- Metadata channel-list self-consistency (`coords.channel` = `channel_state` ‖ `channel_diagnostic`).

**What Phase 4b does NOT exercise** (deferred to `src/sfno_training/` PR):

- The actual `model_registry` monkey-patch (Phase 4b instantiates `PlasimSingleStepWrapper` directly).
- `PlasimTrainer._set_data_shapes` hook inside a real `Trainer.__init__` (Phase 4b sets `N_in_channels` manually).
- The `get_dataloader` monkey-patch (Phase 4b instantiates `PlasimForcingDataset` directly; trainer PR's integration test exercises the rebinding path).
- A real SFNO model (Phase 4b uses `nn.Conv2d`).

---

## Trainer-patch contract (for `src/sfno_training/`)

### 1. Custom dataloader `PlasimForcingDataset`

Subclass `MultifilesDataset`. Additional args: `diagnostic_dataset_path`, `forcing_dataset_path`, `n_forcing_channels`, `forcing_bias`, `forcing_scale`.

- `__init__` passes `dataset_path="fields_state"` to `super().__init__` explicitly (stock loader otherwise defaults to `"fields"` and would fail to find it).
- Opens each HDF5 and stashes separate handles for `/fields_state` (inherited), `/fields_diagnostic`, `/forcing`.
- **Override `_get_data` or `get_sample_at_index`** fully: stock stock `_get_data` (`data_loader_multifiles.py:383`) uses `self.in_channels_sorted` for both input and target reads, which breaks when `in_channels ≠ out_channels`. Rewrite:
  - For input (`target=False`): read `/fields_state` columns `in_channels` (52), apply `in_bias/in_scale` sliced to 52.
  - For target (`target=True`): read `/fields_state` columns `out_channels[:52]` + read `/fields_diagnostic` column 0; concat along channel → 53; apply `out_bias/out_scale` (53-wide, sliced from 53-wide stats).
  - Forcing: read `/forcing` columns 0..5; apply `forcing_bias/forcing_scale` (6-wide).
- Return `(inp_state, tar, inp_forcing, tar_forcing)` matching the shape contract in §"At model forward time".

### 2. Trainer wiring (exact insertion points — v8)

v7 replaced v6's `PlasimDriver(Driver)` with `PlasimTrainer(Trainer)` + `PlasimInferencer(Inferencer)`. v8 **drops `PlasimInferencer`** (see §"What is out of scope" below). The training path is the only one patched.

Two monkey-patches inside `PlasimTrainer.__init__`, both installed *before* `super().__init__()`:

`src/sfno_training/train_plasim.py`:

```python
import numpy as np
import torch
from types import SimpleNamespace
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from makani.utils.training import deterministic_trainer
from makani.utils.training.deterministic_trainer import Trainer
from makani.models import model_registry
from makani.utils.dataloaders.data_helpers import get_data_normalization
from makani.utils.dataloader import init_distributed_io

from src.sfno_training.models.stepper import PlasimSingleStepWrapper, PlasimMultiStepWrapper
from src.sfno_training.data.plasim_forcing_dataset import PlasimForcingDataset


def _plasim_get_dataloader(params, files_pattern, device, mode="train"):
    """Drop-in replacement for makani.utils.dataloader.get_dataloader that
    constructs PlasimForcingDataset and returns the (dataloader, dataset, sampler)
    triple stock Trainer expects. Mirrors `dataloader.py:66-126` for the
    multifiles=True branch only (other branches hard-fail). Inference path is
    not supported — see §'What is out of scope'."""
    assert params.get("multifiles", False), "PlaSim path requires multifiles=True"
    assert mode != "inference", "PlaSim inference is out of scope (v8); stock Inferencer unpacks add_zenith-only tuples"
    init_distributed_io(params)

    bias, scale   = get_data_normalization(params)
    forcing_bias  = np.load(params.forcing_global_means_path)
    forcing_scale = np.load(params.forcing_global_stds_path)

    dataset = PlasimForcingDataset(
        location=files_pattern,
        dt=params.get("dt"),
        in_channels=list(range(params.n_state_channels)),                                    # 52
        out_channels=list(range(params.n_state_channels + params.n_diagnostic_channels)),    # 53
        n_forcing_channels=params.n_forcing_channels,
        n_history=params.get("n_history", 0),
        n_future=(params.get("valid_autoreg_steps") if mode == "eval" else params.get("n_future", 0)),
        dataset_path="fields_state",
        diagnostic_dataset_path=params.diagnostic_h5_path,
        forcing_dataset_path=params.forcing_h5_path,
        relative_timestamp=True,
        data_grid_type=params.get("data_grid_type", "legendre-gauss"),
        model_grid_type=params.get("model_grid_type", "legendre-gauss"),
        bias=bias, scale=scale,
        forcing_bias=forcing_bias, forcing_scale=forcing_scale,
        crop_size=(params.get("crop_size_x"), params.get("crop_size_y")),
        crop_anchor=(params.get("crop_anchor_x", 0), params.get("crop_anchor_y", 0)),
        subsampling_factor=params.get("subsampling_factor", 1),
        return_timestamp=False,
        return_target=True,
        file_suffix=params.get("dataset_file_suffix", "h5"),
        io_grid=params.get("io_grid", [1, 1, 1]),
        io_rank=params.get("io_rank", [0, 0, 0]),
    )

    sampler = (DistributedSampler(dataset, shuffle=(mode == "train"),
                                  num_replicas=params.data_num_shards,
                                  rank=params.data_shard_id)
               if params.data_num_shards > 1 else None)
    dataloader = DataLoader(dataset,
                            batch_size=int(params.batch_size),
                            num_workers=params.num_data_workers,
                            shuffle=((sampler is None) and (mode == "train")),
                            sampler=sampler,
                            drop_last=True,
                            pin_memory=torch.cuda.is_available())

    # Stock-compat attrs used elsewhere in Trainer
    dataloader.lat_lon = dataset.lat_lon
    dataloader.get_output_normalization = dataset.get_output_normalization
    dataloader.get_input_normalization  = dataset.get_input_normalization
    return dataloader, dataset, sampler


def _install_plasim_patches():
    """Installs two monkey-patches before Trainer.__init__ runs:
      (1) model_registry.{SingleStepWrapper, MultiStepWrapper} → PlaSim subclasses.
      (2) deterministic_trainer.get_dataloader → _plasim_get_dataloader.
    Stock Trainer does `from makani.utils.dataloader import get_dataloader`
    (deterministic_trainer.py:36), which binds the name into THAT module —
    so we patch the importer's binding, not the source module."""
    model_registry.SingleStepWrapper = PlasimSingleStepWrapper
    model_registry.MultiStepWrapper  = PlasimMultiStepWrapper
    deterministic_trainer.get_dataloader = _plasim_get_dataloader


class PlasimTrainer(Trainer):
    def __init__(self, params=None, world_rank=0, device=None):
        _install_plasim_patches()          # BEFORE super — patches take effect for get_dataloader and get_model calls inside Trainer.__init__
        super().__init__(params, world_rank, device)

    def _set_data_shapes(self, params, dataset):
        super()._set_data_shapes(params, dataset)
        # v9 hard assert: N_in = n_state + n_forcing only holds for n_history=0.
        # Stock driver multiplies N_in by (n_history+1) for history-stacked inputs, which
        # would mis-wire the SFNO conv if anyone ever flips n_history > 0.
        assert params.n_history == 0, (
            f"PlasimTrainer._set_data_shapes N_in_channels override assumes n_history=0, "
            f"got n_history={params.n_history}. Update PlasimForcingDataset forcing layout "
            f"and this override before enabling history stacking."
        )
        # Override N_in AFTER stock population, BEFORE model/loss construction. Trainer.__init__
        # calls _set_data_shapes at :106, then model at :132, then loss at :175 — this slots in between.
        params.N_in_channels = params.n_state_channels + params.n_forcing_channels   # 58
```

**CLI entry point** (`src/sfno_training/train_plasim.py::main`) — must mirror stock `makani/train.py::main` (`train.py:77-146`) for runtime-injected params (v8 fix #2):

```python
import argparse, os
import torch.distributed as dist
from makani.utils import comm
from makani.utils.YParams import YParams
from makani.utils.parse_dataset_metada import parse_dataset_metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_config", required=True)            # absolute path to plasim_sim52_astro_64x128.yaml
    parser.add_argument("--config",       required=True)           # "plasim_sim52_astro_64x128"
    parser.add_argument("--run_num",      default="0")
    parser.add_argument("--amp_mode",     default="none")
    parser.add_argument("--jit_mode",     default="none")
    parser.add_argument("--enable_synthetic_data", action="store_true")
    parser.add_argument("--enable_s3",             action="store_true")
    parser.add_argument("--enable_odirect",        action="store_true")
    parser.add_argument("--skip_validation",       action="store_true")
    parser.add_argument("--skip_training",         action="store_true")
    parser.add_argument("--disable_ddp",           action="store_true")
    parser.add_argument("--enable_grad_anomaly_detection", action="store_true")
    parser.add_argument("--checkpointing_level",   type=int, default=0)
    parser.add_argument("--load_checkpoint",       default="legacy")
    parser.add_argument("--multistep_count",       type=int, default=1)     # n_future+1
    parser.add_argument("--print_timings_frequency", type=int, default=0)
    parser.add_argument("--split_data_channels",   action="store_true")
    args = parser.parse_args()

    params = YParams(args.yaml_config, args.config, print_params=False)

    world_rank, world_size = comm.init(params=params, verbose=False)

    expDir = os.path.join(params.exp_dir, args.config, args.run_num)
    if world_rank == 0 and not os.path.isdir(expDir):
        os.makedirs(os.path.join(expDir, "training_checkpoints"), exist_ok=True)

    # Runtime-injected params — these are NOT in the YAML; stock makani/train.py:77-146 mirrors.
    params["experiment_dir"]       = os.path.abspath(expDir)
    params["checkpoint_path"]      = os.path.join(expDir, "training_checkpoints/ckpt_mp{mp_rank}_v{checkpoint_version}.tar")
    params["best_checkpoint_path"] = os.path.join(expDir, "training_checkpoints/best_ckpt_mp{mp_rank}.tar")
    params["load_checkpoint"]      = args.load_checkpoint

    resuming = True
    for mp_rank in range(comm.get_size("model")):
        ckpt = params.checkpoint_path.format(mp_rank=mp_rank, checkpoint_version=0)
        if params["load_checkpoint"] == "legacy" or mp_rank < 1:
            resuming = resuming and os.path.isfile(ckpt)
    params["resuming"] = resuming

    params["amp_mode"]       = args.amp_mode
    params["jit_mode"]       = args.jit_mode
    params["skip_validation"] = args.skip_validation
    params["skip_training"]   = args.skip_training
    params["enable_odirect"] = args.enable_odirect
    params["enable_s3"]      = args.enable_s3
    params["enable_synthetic_data"]   = args.enable_synthetic_data
    params["checkpointing_level"]     = args.checkpointing_level
    params["multistep_count"]         = args.multistep_count
    params["n_future"]                = args.multistep_count - 1
    params["disable_ddp"]             = args.disable_ddp
    params["enable_grad_anomaly_detection"] = args.enable_grad_anomaly_detection
    params["split_data_channels"]     = args.split_data_channels
    params["print_timings_frequency"] = args.print_timings_frequency

    params["log_to_wandb"]  = (world_rank == 0) and params["log_to_wandb"]
    params["log_to_screen"] = (world_rank == 0) and params["log_to_screen"]

    # Populate data-derived fields from metadata JSON (lat/lon/channel/...)
    parse_dataset_metadata(params["metadata_json_path"], params=params)

    trainer = PlasimTrainer(params, world_rank)
    if not args.skip_training:
        trainer.train()

if __name__ == "__main__":
    main()
```

The two module-level names we override (`deterministic_trainer.get_dataloader` bound at `deterministic_trainer.py:36`, and `model_registry.{SingleStepWrapper,MultiStepWrapper}` referenced at `model_registry.py:204-207`) are both module attributes — rebinding them takes effect for all subsequent calls. This is the full wiring; no other touch points.

### What is out of scope for this plan (v8)

- **Inference via `Inferencer`.** Stock `_inference_indexlist` (`inferencer.py:453-719`) unpacks tokens from `subset_dataloader` with `add_zenith`-gated branches: with `add_zenith=False` it does `inp, tinp = gtoken` / `tar, ttar = gtoken` (`:559`, `:584`) and sets `inpz=tarz=None`, so `cache_unpredicted_features(None, None, None, None)` at `:589` never loads forcing. Supporting PlaSim forcing at inference time requires either (a) a full `_inference_indexlist` override that adds a forcing-unpack branch, or (b) changing the dataloader's returned tuple to something stock Inferencer can ingest and adding a prescribed-forcing branch to Makani core. Both are non-trivial and have their own testing surface; defer to a follow-up PR.
- **Stock-Makani compatibility for `/fields_state` outside `PlasimTrainer`.** The metadata's `h5_path = fields_state` (52 channels) combined with 53 `coords.channel` names is deliberately non-stock; only `_plasim_get_dataloader` knows to split the target read between `/fields_state` and `/fields_diagnostic`.

### 3. Patched preprocessor (v5 — auto-strip inside append_history)

`src/sfno_training/models/preprocessor.py`:

```python
from makani.models.preprocessor import Preprocessor2D

class PlasimPreprocessor(Preprocessor2D):
    """Auto-strips diagnostic channels from `pred` before append_history's feedback logic.

    Covers two in-scope rollout call sites (v9 — inference explicitly out of scope):
      - makani/makani/models/stepper.py:112         (MultiStepWrapper training rollout)
      - makani/makani/utils/training/deterministic_trainer.py:661 (validation rollout)

    Trainer sets `self.preprocessor = self.model.preprocessor` (`deterministic_trainer.py:133`),
    so injecting this subclass at the PlasimSingleStepWrapper / PlasimMultiStepWrapper __init__
    covers both training and validation rollout paths.

    NOT handled: stock Inferencer._inference_indexlist — see docs/plasim_makani_packager_plan.md
    §"What is out of scope" and risks #14.
    """

    def __init__(self, params):
        super().__init__(params)
        self.n_state_channels = params.n_state_channels             # 52
        self.n_full_out_channels = (                                # 53
            params.n_state_channels + params.n_diagnostic_channels
        )

    def append_history(self, x1, x2, step, update_state=True):
        # v6 (Codex round 5 fix #4): hard-assert the only two shapes we
        # ever expect, so real channel bugs cannot hide behind silent slicing.
        assert x2.dim() == 4, f"expected x2 4D (B, C, H, W), got {x2.dim()}D shape {tuple(x2.shape)}"
        assert x2.shape[1] in (self.n_state_channels, self.n_full_out_channels), (
            f"PlasimPreprocessor.append_history: x2 channels must be "
            f"{self.n_state_channels} or {self.n_full_out_channels}, got {x2.shape[1]}"
        )
        if x2.shape[1] == self.n_full_out_channels:
            x2 = x2[:, :self.n_state_channels, ...]
        return super().append_history(x1, x2, step, update_state=update_state)
```

And the stepper subclasses simply install it:

```python
from makani.models.stepper import SingleStepWrapper, MultiStepWrapper
from src.sfno_training.models.preprocessor import PlasimPreprocessor

class PlasimSingleStepWrapper(SingleStepWrapper):
    def __init__(self, params, model_handle):
        super().__init__(params, model_handle)
        self.preprocessor = PlasimPreprocessor(params)  # replace stock

class PlasimMultiStepWrapper(MultiStepWrapper):
    def __init__(self, params, model_handle):
        super().__init__(params, model_handle)
        self.preprocessor = PlasimPreprocessor(params)
```

`n_state_channels` is read from `params.n_state_channels = 52`. No need to fork `_forward_train` or `_forward_eval` — the strip is inside `append_history` itself, so the stock loop bodies call the patched version transparently.

### 4. Training-loop sequence per batch

Mirrors stock `deterministic_trainer.py:474-478` (Codex round 5 fix #3 — `flatten_history` was missing from v5 pseudocode):

```python
# Dataloader returns 5D with a singleton time dim per stock contract
# (data_loader_multifiles.py:411). Batched: (B, n_history+1, C, H, W) and (B, n_future+1, C, H, W).
inp_state, tar, inp_forcing, tar_forcing = next(loader_iter)

# 1. Cache forcing. cache_unpredicted_features returns (inp_state, tar) passthrough
#    so we can chain flatten_history naturally (stock pattern).
inp_state, tar = wrapper.preprocessor.cache_unpredicted_features(
    inp_state, tar, inp_forcing, tar_forcing,
)

# 2. Flatten the history/future dims into channels BEFORE the wrapper call.
#    Stock SFNO forward expects 4D (B, C, H, W); flatten_history collapses
#    the T dim into C.
inp_state = wrapper.preprocessor.flatten_history(inp_state)    # (B, 52, H, W)
tar       = wrapper.preprocessor.flatten_history(tar)          # (B, 53*(n_future+1), H, W)

# 3. Forward. Wrapper's append_unpredicted_features concats forcing internally:
#    inp_state (52) + unpredicted_inp (6) → 58 at the model boundary.
pred = wrapper(inp_state)                                      # (B, 53*(n_future+1), H, W)

# 4. Loss. bias/scale sized to 53 (per out_channels); broadcast handles the
#    flattened time dim as long as channel stride matches.
loss = loss_fn(pred, tar)
loss.backward()
```

### 5. Rollout correctness invariants

- `wrapper(inp_state).shape == (B, 53 * (n_future+1), H, W)` after multistep concat.
- For every `step ∈ [0, n_future]`: `preprocessor.unpredicted_inp_*.shape[2] == 6` (six forcing channels).
- `tar_forcing[:, step+1, ...]` fed unchanged into next-step `inp_forcing` (verified via `append_history`'s internal copy).
- `prediction[:, :52]` fed into next-step `inp_state`; `prediction[:, 52:53]` (pr_6h) **never** feeds back.

---

## Target layout

```
src/plasim_makani_packager/
├── packager.py                        # per-(sim, year) writer, standalone CLI
├── stats.py                           # Welford over /fields (53 target) + /forcing (6)
├── validate.py                        # Phase 4a + Phase 4b smoke
├── submit.slurm
└── templates/
    └── plasim_64x128.yaml

scripts/
└── package_sim52_astro.sh             # optional orchestrator

tests/plasim_makani_packager/
├── test_channel_flatten.py
├── test_hdf5_writer.py                # dim scales + file attrs + /time_plasim
├── test_sst_land_fill.py
├── test_sic_provenance.py
├── test_stats.py                      # shapes (1,53,…) + (1,6,…) + epsilon policy
├── test_timestamp.py                  # synthetic + PlaSim provenance
├── test_metadata.py                   # 53 target names + abs paths
├── test_parse_dataset_metadata.py     # Makani parser produces in==out==[0..52]
├── stub_forcing_loader.py             # PlasimForcingDataset + PlasimPreprocessor + PlasimSingleStepWrapper + PlasimMultiStepWrapper stubs for Phase 4b
└── test_multifile_loader_smoke.py     # Full YParams chain + PlaSim wrappers (positive rollout) + stock MultiStepWrapper (negative regression)

docs/
└── plasim_makani_packager_plan.md     # this file

skills/plasim-makani-packager/
└── SKILL.md                           # after plan approval
```

Output data layout:

```
/scratch/11114/zhixingliu/AI-RES/data/makani/sim52_astro_64x128/
├── train/                             # years 3-100 (98 files)
│   └── MOST.{0003..0100}.h5
├── valid/                             # years 101-120 (20 files)
│   └── MOST.{0101..0120}.h5
├── test/                              # years 121-128 (8 files)
│   └── MOST.{0121..0128}.h5
├── stats/
│   ├── global_means.npy                    # (1, 53, 1, 1)
│   ├── global_stds.npy                     # (1, 53, 1, 1)
│   ├── time_means.npy                      # (1, 53, 64, 128)
│   ├── forcing_global_means.npy            # (1, 6, 1, 1)
│   ├── forcing_global_stds.npy             # (1, 6, 1, 1)
│   ├── forcing_time_means.npy              # (1, 6, 64, 128)
│   └── sst_land_sentinel_notes.txt
├── validation/
│   └── sic_clip_report_{YYYY}.json         # 128 files
├── metadata/
│   └── data.json
└── config/
    └── plasim_sim52_astro_64x128.yaml      # concrete absolute paths
```

---

## CLI

Same CLI as v3. All `--*-root` args absolute paths; `$SCRATCH`/`$HOME` not expanded.

```
packager.py --postproc-root … --boundary-root … --output-root …
            --sims 52 --train-years 3 100 --valid-years 101 120 --test-years 121 128
            --sst-land-fill-k 271.35
            [--task-index] [--count-tasks] [--overwrite] [--dry-run] [-v]

stats.py    --output-root … --train-years 3 100 [-v]

validate.py --output-root … --mode {structural, makani_smoke, full} [-v]
```

---

## Risks / open items

1. **Custom `_get_data` / `get_sample_at_index` override non-trivial.** The stock path assumes `in_channels == out_channels`; the subclass has to rewrite most of it. The stub in `tests/plasim_makani_packager/stub_forcing_loader.py` ships in this PR so Phase 4b can exercise the contract; the production loader in `src/sfno_training/` is a separate PR.
2. **Preprocessor subclass is a small surface area.** v5 shifts the diagnostic-strip into `PlasimPreprocessor.append_history`, so we no longer fork `_forward_train` / `_forward_eval`. Short method override (4 lines of assertion + 1-line slice + 1-line super call); upstream Makani changes to `append_history`'s signature (currently `(x1, x2, step, update_state=True)`) would require a quick re-sync but the surface is small. v6 fail-loud assertion converts any upstream signature/shape change into a pytest failure rather than silent wrong math.
13. **Monkey-patch targets are load-order / import-form sensitive.** `PlasimTrainer.__init__` must rebind both (a) `model_registry.SingleStepWrapper` + `model_registry.MultiStepWrapper`, and (b) `deterministic_trainer.get_dataloader` *before* `super().__init__()`. Stock code uses `from X import Y` at module scope (`deterministic_trainer.py:36`, `model_registry.py:204-207`), which binds a local name in the importing module — so we must patch the **importer's** name, not the source module's name. If upstream Makani switches to `import X; X.Y(...)` at the call site, the monkey-patch path changes and the integration test in the trainer PR will catch the regression.
14. **Inference is not supported in v8.** Any downstream work that drives rollout via stock `Inferencer` will get forcing = `None`, not our 6-channel `/forcing`. Stock scoring / inference tooling will silently produce wrong predictions. Document this prominently; add a follow-up PR placeholder for `src/sfno_inference/`.
3. **Permission-restricted source.** `/work2/09979/awikner/...data_loader_multifiles.py` still unreadable. May reveal a different forcing handling convention; could force a v5.
4. **`Preprocessor2D` config fields inventory.** v4's Phase 4b sets the fields we've found by reading Makani source; if a Makani upgrade adds new required fields, Phase 4b needs updating.
5. **`rsdt_method` lock.** Packager asserts `rsdt_method == "astronomical"` on every boundary file.
6. **sim52 boundary output doesn't exist yet.** Phase 0 blocks Phase 1.
7. **Disk footprint.** ~230 GB uncompressed across 128 years.
8. **Year 0002 warmup skipped.**
9. **Static-channel storage overhead.** `lsm, sg` repeated across time (~3% disk); kept for contract uniformity. `z0` also in /forcing but carries real time variation over ocean (Charnock).
10. **Stock Makani metadata parser tolerance.** Ignores unknown keys (`diagnostic_*`, `forcing_*`). Pin the Makani revision validated against.
11. **ERA5-style default Makani metrics off.** PlaSim-specific metric names future work.
12. **`add_zenith` off.** `rsdt` in `/forcing` covers insolation.

---

## What NOT to do

- **Do not** extend `src/plasim_postprocessor/` or `src/emulator_adaptor/` — audited contracts.
- **Do not** patch Makani core in-tree — patches live under `src/sfno_training/`.
- **Do not** include `pr_6h` in `/fields_state`. Diagnostic leaks into feedback otherwise.
- **Do not** manually concat forcing into the model input at the trainer. The wrapper's `append_unpredicted_features` already does this; double-concat produces 64-channel junk.
- **Do not** override `params.N_in_channels` *before* `driver._set_data_shapes` — it gets overwritten. Override *after*, before model build.
- **Do not** silently clamp zero-std channels. `MIN_STD_EPSILON` is a hard fail.
- **Do not** assert exact sic equality against MOST. Quantify.
- **Do not** recompute stats from valid/test years.
- **Do not** anchor timestamps to PlaSim's actual calendar days-since values — variable-length years (1359 vs 1459 steps) break Makani's uniform-dhours check. Synthetic int64 seconds starting at 0, step 21600, are fine: stock Makani interprets them via `dt.datetime.fromtimestamp` (i.e. Unix epoch + seconds) unless `relative_timestamp=True` is passed to the dataloader. The custom loader passes `relative_timestamp=True` explicitly, so time is treated as a timedelta from zero and the epoch anchor becomes irrelevant.
- **Do not** use `$SCRATCH` / `$HOME` in rendered YAMLs. Absolute paths only.
- **Do not** omit `dataset_path="fields_state"` from the custom loader's `super().__init__` — stock default is `"fields"`, file won't have that.
- **Do not** rely on `add_zenith` as a substitute for `rsdt`.
- **Do not** point stock Makani dataloaders at this metadata. `h5_path = "fields_state"` (52 channels) with `coords.channel` = 53 names is deliberately non-stock; stock `MultifilesDataset` would crash with a channel-count mismatch when normalization stats are applied. Only `PlasimForcingDataset` can consume this layout.
- **Do not** instantiate stock `SingleStepWrapper` / `MultiStepWrapper` anywhere in the PlaSim trainer path — rely on the `PlasimTrainer.__init__` monkey-patch. The only legitimate direct reference to stock `MultiStepWrapper` is the Phase 4b negative regression test.
- **Do not** instantiate stock `Trainer` from the PlaSim CLI — always use `PlasimTrainer`. Stock `Trainer` skips both monkey-patches and the `_set_data_shapes` override, producing a mis-wired model before any error surfaces.
- **Do not** drive inference through stock `Inferencer` on this dataset — v8 does not patch the inference path. Running stock inference will load `inpz=tarz=None` and skip forcing, yielding silently wrong predictions. Block inference until the follow-up PR.
- **Do not** patch `makani.utils.dataloader.get_dataloader` (the *source* module) — stock `Trainer` already bound the name into `deterministic_trainer` at import time, so patching the source has no effect. Patch `deterministic_trainer.get_dataloader` (the *importer's* binding) directly.
- **Do not** reuse a `LossHandler` across different `n_future` values — `channel_weights` stays `N_out`-wide (53) regardless of `n_future`, but `multistep_weight` is registered at construction as `(1, ncw * (n_future + 1))` (`loss.py:194-195`) and `forward` tiles `channel_weights` by `(n_future + 1)` before multiplying (`loss.py:437-440`). Reusing a `LossHandler` built at `n_future=0` with a `n_future=1` target produces a shape mismatch. Rebuild `LossHandler(params)` whenever `n_future` changes.
- **Do not** emit rendered YAMLs that merge `<<: *BASE_CONFIG` from an external file. `YParams` has no multi-file include; inline every key that Trainer/Inferencer read.
- **Do not** skip `flatten_history` between `cache_unpredicted_features` and the wrapper call — stock SFNO forward expects 4D tensors. Mirror `deterministic_trainer.py:474-478`.
- **Do not** add silent slicing to `PlasimPreprocessor.append_history` — v6 asserts `x2.shape[1] in {52, 53}` exactly; any other channel count is a bug that must surface.
- **Do not** flip `n_history > 0` without revising `PlasimTrainer._set_data_shapes`. Its `N_in = n_state + n_forcing = 58` override is only correct at `n_history=0`; otherwise stock driver multiplies by `(n_history+1)` and SFNO's input conv gets the wrong width. The v9 hard assert inside `_set_data_shapes` will trip immediately; remove it only after updating the `PlasimForcingDataset` forcing layout and re-deriving the correct `N_in`.

---

## Verification gate

Plan ready for Codex v9 review. Approval required before Phase 0.

v9 approval scope: approval applies to the **packager PR** (commit chunks 1–6). SFNO training (commit 7 and onward) is **not yet approved** — it is blocked on the follow-up PR's integration test (see chunk 7) passing in CI before any GPU-hours are spent. Codex explicitly flagged this gate in round 8.

Proposed commit chunks:
1. This plan (`docs/plasim_makani_packager_plan.md` v6).
2. `src/plasim_makani_packager/` scaffolding + CLI + stats + validate + `stub_forcing_loader.py` (contains `PlasimForcingDataset`, `PlasimPreprocessor`, `PlasimSingleStepWrapper`, `PlasimMultiStepWrapper` stubs) + Phase 4b smoke (positive + negative rollout tests).
3. `templates/plasim_64x128.yaml` + `submit.slurm`.
4. `skills/plasim-makani-packager/SKILL.md`.
5. Phase 0 execution artifacts (sim52 adaptor run).
6. Phase 1–4 packaging run over sim52, validation artifacts.
7. **Follow-up PR in `src/sfno_training/`** — production `PlasimForcingDataset`, `PlasimPreprocessor`, `PlasimSingleStepWrapper`, `PlasimMultiStepWrapper`, `PlasimTrainer`, `_plasim_get_dataloader`, `train_plasim.py::main` (stock `makani/train.py::main` mirror). **This chunk is a hard gate on SFNO training (v9).** It adds a mandatory integration test that runs `PlasimTrainer(params, world_rank).train(n_iters=1)` to completion (one full forward + backward + step + checkpoint write). The test must pass in CI before any real SFNO training run is launched; GPU-hours without this gate are forbidden. Required assertions:
   - `isinstance(pt, PlasimTrainer)` and `isinstance(pt.train_dataset, PlasimForcingDataset)` — dataloader monkey-patch engaged.
   - `isinstance(pt.model, PlasimSingleStepWrapper)` when `params.n_future == 0` (and `PlasimMultiStepWrapper` when `n_future > 0`) — wrapper monkey-patch engaged.
   - `isinstance(pt.model.preprocessor, PlasimPreprocessor)` — preprocessor install engaged.
   - `pt.params.N_in_channels == 58` after `_set_data_shapes`.
   - `train(n_iters=1)` completes without exception and writes a valid checkpoint.
8. **Follow-up follow-up PR in `src/sfno_inference/`** (separate plan): `PlasimInferencer(Inferencer)` with a full `_inference_indexlist` override that adds a forcing-unpack branch; or upstream a prescribed-forcing token-unpack hook into Makani core.
