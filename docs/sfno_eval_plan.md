# SFNO PlaSim emulator accuracy evaluation plan **v2.8** — implementation-ready (locked)

> **Sibling plan (2026-05-06).** Evaluation of the **group's SFNO-5410 emulator** lives in a separate, dated plan: `docs/2026-05-06_group_sfno_5410_eval_plan.md`. That plan reuses this plan's metric stack (`src/sfno_eval/`, Gauss–Legendre lat-weights, time-of-year-proleptic climatology schema, NWP scorecard CSV layout, sanity-gate thresholds) but has its own inference engine (upstream PanguWeather/v2.0 path-overridden), input data tree (`data/derecho_glade/sim52/...`), checkpoint (`ckpt_epoch_50.tar`), output tree (`results/sfno_eval_5410/<run_tag>/`), and an external climatology built from group post-processing (`docs/2026-05-06_group_sfno_5410_climatology_prompt_for_derecho.md`). The cross-emulator combined report is rendered by a third script described in that plan's §F.
>
> **Migration note (2026-04-30, updated 2026-05-04).** The v10 dataset
> contract (docs/plasim_zg_plev_migration_plan.md) replaced sigma-level
> `zg1..zg10` with pressure-level `zg150..zg925`; the v10.1 follow-up
> (docs/2026-05-04_zg1000hpa_migration_plan.md) shifted the subset to
> `zg200..zg1000` for ACE parity. The primary observable change in this plan is
> §D.6: the Z500 ACC gate now references the literal `zg500`
> channel rather than the v9 sigma proxy `zg5`. The scoring scripts
> (`scripts/score_nwp.py`, `scripts/render_eval_report.py`) detect
> the Z500-proxy channel from the inference NetCDFs at run time, so
> v9 (`zg5`) and v10 (`zg500`) checkpoints are both scorable
> against the same gate threshold; see
> `docs/plasim_zg_plev_migration_plan.md` §3.10 for the resolver
> contract.
>
> **Plan v2.8** (2026-04-29). Addresses Codex round-8 cleanup of v2.7.
> Four cleanup items: (1) deleted stale "torch_harmonics API plausible but not verified" note in §B.5 (already resolved at §J Q3); (2) refreshed §I device-mismatch risk row to describe the v2.7 safe type+index comparison instead of the old exact-equality test; (3) defined `out_bias` / `out_scale` explicitly from the run-dir stats at the top of §B.2 pseudocode; (4) added §A.4 pinning the **12 NWP ICs/year, monthly stride, no cross-file rollout** rule with a `s + K < n_samples` assertion. **Codex round-8 verdict: implementation-ready.**
> **Author:** Zhixing Liu (with Claude Code)
> **Depends on:**
> - `docs/sfno_full_training_plan.md` v1.1 — training shipped; best checkpoint at `/scratch/11114/zhixingliu/SFNO_Climate_Emulator/runs/sfno_full/plasim_sim52_full/0/training_checkpoints/best_ckpt_mp0.tar`.
> - `docs/sfno_training_implementation_plan.md` (the **"Hard gate on full emulator rollout (v2)"** section) — declares stock `Inferencer` out of scope and reserves `src/sfno_inference/` as the named deliverable.
> - `docs/plasim_makani_packager_plan.md` v9 — dataset contract.
> **Codebase deps:**
> - `src/sfno_training/data/plasim_forcing_dataset.py` (`PlasimForcingDataset`).
> - `src/sfno_training/models/preprocessor.py` (`PlasimPreprocessor`).
> - `src/sfno_training/models/stepper.py` (`MultiStepWrapper`'s `_forward_eval` → single-step eval forward).
> - `makani-src/` for the underlying SFNO module (loaded via `Trainer.restore_from_checkpoint`).
> - `external/earth2studio/` is **dropped** as a runtime dep — see §3. Kept only as a **reference clone** for naming/style.

---

## Changelog

**v2.8 — Codex round 8 cleanup (locked, implementation-ready):**

1. **Stale `torch_harmonics` "plausible but not verified" note in §B.5 deleted.** The API was already resolved in v2.2 (Codex round 2) as `torch_harmonics.quadrature.legendre_gauss_weights(64, -1.0, 1.0)`, with a numpy fallback `np.polynomial.legendre.leggauss(64)`. §J Q3 and the §I risk row already reflected this; only §B.5 had been overlooked. v2.8 §B.5 now states the resolution inline.
2. **§I device-mismatch risk row refreshed.** Previously described the v2.6 exact-equality assertion (`actual.device == torch.device(device)`), which v2.7 already replaced with a safe type+index comparison (`actual.type == expected.type and (expected.index is None or actual.index == expected.index)`). v2.8 row now describes the v2.7 mechanism, references the `cuda:<current_device>` explicit-index resolution in §B.2, and notes that `test_eval_params_load.py` exercises both the indexed and non-indexed expected forms.
3. **`out_bias` / `out_scale` defined explicitly in §B.2 pseudocode.** v2.7 used `predictions_phys = predictions * out_scale + out_bias` without showing where they came from. v2.8 §B.2 adds an explicit load block immediately before the `dataset[ic_global_idx]` call:
   - `out_bias_np = np.load(eval_params.global_means_path).astype(np.float32)`
   - `out_scale_np = np.load(eval_params.global_stds_path).astype(np.float32)`
   - shape assertion `(53,)`, then `.to(device).reshape(1, 53, 1, 1)` for broadcast against the `(K, 53, H, W)` z-scored predictions.
   - Source is the **run dir** (`runs/sfno_full/plasim_sim52_full/0/global_means.npy`), per Q5 (byte-identical to dataset-stats but run-dir is canonical).
4. **§A.4 added — NWP IC selection rule pinned.** v2.7 mentioned "8 yr × 12 IC = 96 files" without specifying how the 12 ICs/year are chosen or how to keep `K = 56`-step rollouts inside one file. v2.8 §A.4 defines a `nwp_ic_offsets(n_samples, K=56, n_ic=12)` helper:
   - Stride = `(n_samples - K) // n_ic` ≈ monthly cadence — works out to step=116 for both n=1455 and n=1459 at the locked K, n_ic.
   - IC list = `[0, 116, 232, ..., 1276]` (max IC=1276; max IC+K=1332 < 1455). Same indices for leap and non-leap files (uniform `nwp_scorecard.csv` layout).
   - Hard assertion `s + K < n_samples` for every IC. Cross-file rollout is **explicitly out of scope** because the dataset's `_get_indices` would silently mix file anchors and break `time_plasim` provenance.
   - Climate mode (§B.3) uses `s = 0, K = n_samples - 1`; the same guard holds by construction.

**v2.7 — Codex round 7 fixes (final, implementation-ready):**

1. **Device-equality false negative.** v2.6 §B.0 asserted `next(wrapper.parameters()).device == torch.device(device)` and §B.2 set `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`. After `.to(torch.device("cuda"))`, parameters land on `cuda:<current_device>` (typically `cuda:0`), but `torch.device("cuda") != torch.device("cuda:0")` under `==` because the indexed and non-indexed forms are distinct. The assertion would have raised on every GPU run. v2.7:
   - §B.2 resolves `device` with an explicit index when CUDA is available: `torch.device(f"cuda:{torch.cuda.current_device()}")` — what `.to()` actually produces.
   - §B.0 assertion now compares `actual.type == expected.type` and `expected.index is None or actual.index == expected.index`, so it accepts both indexed and non-indexed expected forms.
2. **Autocast hardcoded to `device_type="cuda"`.** v2.6 §B.2 inner loop used `torch.amp.autocast(device_type="cuda", ...)` unconditionally. The CPU smoke test (`test_smoke_eval_cpu.py`) would have errored on the first step. v2.7 §B.2 derives `autocast_enabled = bool(eval_params.amp_enabled) and (device.type == "cuda")`, sets `device_type=device.type`, and falls back to `dtype=torch.float32` when autocast is disabled. CPU smoke runs now silently use fp32 (acceptable — they exist to validate orchestration, not numerics).

**v2.6 — Codex round 6 fixes:**

1. **Wrapper device placement (BLOCKER).** v2.5 §B.0 built the wrapper and called `Driver.restore_from_checkpoint(...)` without ever moving the wrapper to GPU; v2.5 §B.2 then moved the input batch with `t.unsqueeze(0).to(device)`. The first `wrapper(inpt)` would have raised a CPU/GPU-mismatch `RuntimeError`. The trainer pattern (`makani/utils/training/deterministic_trainer.py:132`) is `model_registry.get_model(self.params, multistep=self.multistep).to(self.device)` — `.to(device)` happens before `restore_from_checkpoint`. v2.6 §B.0:
   - `build_wrapper_from_checkpoint(eval_params, ckpt_path, device)` — `device` is now a required positional arg.
   - Body: `wrapper = model_registry.get_model(eval_params, multistep=True).to(device)` *before* `Driver.restore_from_checkpoint`. (The legacy restore path uses `map_location` internally and copies into the model's existing parameters, so post-`.to(device)` is the right ordering.)
   - Final assertion `assert next(wrapper.parameters()).device == torch.device(device)` to catch any future regression.
   - §B.2 explicitly sets `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")` and threads it through.
2. **`sfno_model_handle(eval_params)` was undefined.** v2.5 §B.0 wrote `wrapper = PlasimMultiStepWrapper(eval_params, sfno_model_handle(eval_params))` as if `sfno_model_handle` were a known builder — it is not. The canonical Makani path (`deterministic_trainer.py:132`) is `model_registry.get_model(params, multistep=True)`. With `_install_plasim_patches()` already run, `model_registry.MultiStepWrapper` is rebound to `PlasimMultiStepWrapper`, so `get_model(..., multistep=True)` returns the PlaSim wrapper with `wrapper.preprocessor = PlasimPreprocessor(params)` already wired. v2.6 §B.0 calls `_install_plasim_patches()` then `model_registry.get_model(eval_params, multistep=True)` — no custom builder needed.
3. **Missing `import torch` in §B.0 snippet.** `load_eval_params` references `torch.float32`, `torch.float16`, `torch.bfloat16`. v2.6 adds `import torch` to the imports block alongside `model_registry`, `Driver`, and `_install_plasim_patches`.
4. **Q5 (open question — run-dir vs dataset-stats normalization) closed.** User verified the two `global_means.npy` files are **byte-identical** for this run. v2.6 §J marks Q5 ✅ resolved and `checkpoint_loader.py` adds a SHA256 comparison-with-warning so any future divergence between run-dir and dataset-stats normalization is loud-but-non-fatal.

**v2.5 — Codex round 5 fixes (load-bearing + cleanup):**

1. **`Driver.restore_from_checkpoint` actual signature** (`makani-src/makani/utils/driver.py:347-356`) is `restore_from_checkpoint(checkpoint_path, model, loss=None, optimizer=None, scheduler=None, counters=None, checkpoint_mode='legacy', strict=True)` — a `@staticmethod` taking `checkpoint_path` first, `model` second. v2.4 §B.0 had `restore_from_checkpoint(wrapper, ckpt_path)` (args reversed) and called it as a free function. v2.5 §B.0 imports `from makani.utils.driver import Driver` and calls `Driver.restore_from_checkpoint(ckpt_path, wrapper, checkpoint_mode='legacy')`. The wrapper is passed as `model` because `PlasimMultiStepWrapper.state_dict()` transparently forwards through to `wrapper.model`.
2. **SFNO uses `inp_chans`, not `in_chans`.** Verified at `makani-src/makani/models/networks/sfnonet.py:298-299`: `self.inp_chans = inp_chans; self.out_chans = out_chans`. v2.4 §B.0 asserted `wrapper.model.in_chans == 58` which would have raised `AttributeError` at first run. v2.5 §B.0 asserts `wrapper.model.inp_chans == 58 and wrapper.model.out_chans == 53`.
3. **AMP enabled/dtype derived from `amp_mode`.** Verified `runs/sfno_full/plasim_sim52_full/0/config.json` serialises **only** `amp_mode: 'bf16'` — neither `amp_enabled` nor `amp_dtype` is saved. The trainer derives both at runtime (`makani-src/makani/utils/training/deterministic_trainer.py:84-97`): `amp_mode == 'bf16'` → `amp_enabled = True, amp_dtype = torch.bfloat16`. v2.4 §B.2's autocast block read `eval_params.amp_enabled` and `eval_params.amp_dtype` directly from the loaded params, so autocast would silently be disabled (treated as missing → False) and the eval rollout would run in fp32 — bit-different from training. v2.5 §B.0 mirrors the trainer's derivation block in `load_eval_params` so autocast in §B.2 picks up bf16.
4. **Training SHA recovered from `out.log`** — was previously declared `unknown`. Makani's trainer init logs `git hash: <40-char-sha>` (visible at `runs/sfno_full/plasim_sim52_full/0/out.log:2026-04-27 20:53:17,879`). Full SHA: `106d19d9cad20b20be5e1faf1319b4f0cdb7346b` → `TRAIN_SHA7 = 106d19d`. v2.5 §G.4 removes the `train-unknown` literal and updates the run-tag example to `train-106d19d`. The eval driver now greps `out.log` for the `git hash:` line; falls back to `unknown` only if `out.log` is missing or the line is absent. §G.5 retains the recommendation to write a sidecar `train_code_sha.txt` going forward (more durable than `out.log`).
5. **Stale text cleanup (no semantics change):** `src/sfno_eval/climatology.py` description in §4 layout updated from "per-(sample-of-year, ...)" to "per-(month, day, hour_quarter, ...)"; `compute_climatology.py` output renamed from `climatology_<sample_idx>.nc` to single `climatology_proleptic.nc`; `torch_harmonics` API uncertainty risk row marked ✅ resolved (Q3 already closed in §J).
6. **Forward-pass arithmetic fixed.** v2.4 §G.2 stated "17 016" in the table row and "17 120" in the prose — inconsistent. The correct count is `96 × 56 + 6 × 1454 + 2 × 1458 = 5376 + 8724 + 2916 = 17 016`. v2.5 prose now matches the table; the bogus "8 × 1456" reduction is removed.

**v2.4 — Codex round 4 fixes (load-bearing):**

1. **`valid_autoreg_steps`, not `n_future`, drives the eval-mode dataset horizon.** v2.3 §B.2 set `eval_params.n_future = K - 1` and assumed that propagated through `_plasim_get_dataloader(..., mode="eval")`. Verified at `src/sfno_training/trainer/plasim_trainer.py:103-105`:
   ```python
   n_future = (
       params.get("valid_autoreg_steps") if (mode == "eval") else params.get("n_future", 0)
   )
   ```
   In `mode="eval"`, the constructor pulls `params.valid_autoreg_steps`, not `params.n_future`. The training run has `valid_autoreg_steps = 3` baked into config (used for early-stopping rollouts during training); leaving it unchanged would silently cap the test rollout at 3 steps regardless of the requested K. v2.4 §B.2 sets **both** `eval_params.n_future = K - 1` **and** `eval_params.valid_autoreg_steps = K - 1` before calling `_plasim_get_dataloader`. The pseudocode pins `valid_autoreg_steps` first (it is the active handle in eval mode); `n_future` is set redundantly so future code paths that read it directly remain correct.
2. **Checkpoint loader explicitly enforces `N_in_channels=58`, `N_out_channels=53` from the run-dir `config.json`.** v2.3 §B.2 said `wrapper = PlasimMultiStepWrapper(eval_params, sfno_model_handle(eval_params))` then `restore_from_checkpoint(wrapper, ckpt_path)` — but it did not specify where `eval_params` came from, leaving room for raw-YAML drift. The training run's `runs/sfno_full/plasim_sim52_full/0/config.json` is the **only** authoritative copy of the model-shape parameters and contains:
   ```
   N_in_channels       = 58
   N_out_channels      = 53
   n_state_channels    = 52
   n_diagnostic_channels = 1
   n_forcing_channels  = 6
   ```
   v2.4 §B.0 (new sub-section) and `src/sfno_inference/checkpoint_loader.py` make this load chain explicit:
   1. `eval_params = ParamsBase(); eval_params.update(json.load(open(run_dir/'config.json')))` — start from the frozen training params.
   2. `assert eval_params.N_in_channels == 58 and eval_params.N_out_channels == 53` — guard against silent drift.
   3. Override only the eval-specific fields: `valid_autoreg_steps`, `n_future`, `n_history`, `data_num_shards=1`, `data_shard_id=0`, paths to `test_holdout/`, eval batch size = 1, `amp_enabled` honoured from training, etc.
   4. Then `wrapper = PlasimMultiStepWrapper(eval_params, sfno_model_handle(eval_params)); restore_from_checkpoint(wrapper, ckpt_path)`.
   5. Post-build assertion: `assert wrapper.model.in_chans == 58 and wrapper.model.out_chans == 53`.
   This eliminates the v2.3 risk of someone re-deriving `eval_params` from raw YAML and missing the metadata-time channel-count override that the training-time `parse_dataset_metadata` patch installs.
3. **Persistence baseline for `pr_6h` is undefined; v2.4 excludes it from the persistence comparison (does not synthesise one).** v2.3 §C.1 said "persistence prediction at lead `k` is `truth[s, ...]` (the t=0 state)." `truth[s, ...]` for the dataset is the IC tensor `inp_state` of shape `(1, 52, H, W)` — the diagnostic `pr_6h` is **not** in the IC (`src/sfno_training/data/plasim_forcing_dataset.py:299` — `inp_state` only loads `in_channels = list(range(52))`). The diagnostic appears only on the target side at lead 1 onward. So persistence for channel 52 (`pr_6h`) cannot be defined as "lead-0 truth" because there is no lead-0 truth for that channel. **v2.4 decision:** persistence is computed only for the 52 state channels. The NWP scorecard CSV records `persistence_rmse[c, lead] = NaN` for `c == "pr_6h"` and the report (§F) explicitly notes this as a baseline gap rather than reporting a misleading number. (Alternative considered: read `f['fields_diagnostic'][s, 0, :, :]` at the IC sample as a "lead-0 diagnostic truth" and use that for persistence. **Rejected** because (a) the model never sees this signal at inference time — it is loss-only — so giving persistence access to it would be unfair to the model, and (b) the climate community treats persistence on diagnostic-only channels as undefined; sticking to the convention avoids a paper-review red flag.)

**v2.3 — Codex round 3 fixes (load-bearing):**

1. **Climatology source path corrected.** v2.2 §C.2 said training pool is "98 files in `sim52_astro_64x128/train/`". The model actually trained on **100 files at `/scratch/.../data/makani/sim52_full/train/`**, years 12–111 with no gaps:
   - Years 12–100 symlink to `sim52_astro_64x128/train/`.
   - Years 101–111 symlink to `sim52_astro_64x128/valid/`.
   - Leap-year breakdown (filesystem-verified): **76 × 1455 + 24 × 1459**. (`MOST.0094.h5` is non-leap because its anchor year 99 makes the candidate leap year **100**, which hits the proleptic-gregorian centennial exception: `100 % 100 == 0 AND 100 % 400 ≠ 0`. So 25 candidate leap files − 1 centennial-exception = 24 actual.)
   - Climatology must read this exact directory (`sim52_full/train/`), not `sim52_astro_64x128/train/`.
2. **`PlasimForcingDataset` constructor signature corrected.** v2.2 §B.2 pseudocode used `PlasimForcingDataset(params=..., train=False, ...)`. The real constructor is **keyword-only** with explicit args: `location`, `dt`, `in_channels`, `out_channels`, `n_history`, `n_future`, `diagnostic_dataset_path`, `forcing_dataset_path`, `n_forcing_channels`, `forcing_bias`, `forcing_scale`, `add_zenith`, `data_grid_type`, `model_grid_type`, `bias`, `scale`, `crop_size`, `crop_anchor`, `subsampling_factor`, `return_timestamp`, `return_target`, `relative_timestamp`, `file_suffix`, `enable_s3`, `io_grid`, `io_rank` (`src/sfno_training/data/plasim_forcing_dataset.py:57` and the canonical invocation at `src/sfno_training/trainer/plasim_trainer.py:116`). v2.3 §B.2 instead calls **`_plasim_get_dataloader(eval_params, test_holdout_path, device, mode='eval')`** and uses the returned `(dataloader, dataset, sampler)` — this guarantees identical kwargs to the validation path.
3. **Preprocessor identity fixed.** v2.2 §B.2 created a separate `preprocessor = PlasimPreprocessor(eval_params)` and a separate `wrapper = build_wrapped_sfno(...)`. The wrapper internally creates its own preprocessor at `src/sfno_training/models/stepper.py:33` (`self.preprocessor = PlasimPreprocessor(params)`). Caching forcing into one instance while the wrapper consumes from the other would silently produce wrong outputs. v2.3 §B.2 uses **`preprocessor = wrapper.preprocessor`** — the single instance owned by the wrapper.
4. **NetCDF lead-time schema fixed (off-by-one).** v2.2 §B.4 had `lead_time = n_steps + 1, lead_time coords = np.arange(n_steps+1) * 6`. With `n_future = K - 1` (v2.2 §B.2), the rollout produces **K predictions at lead times {6, 12, ..., 6K} hours** — no lead 0 because lead 0 is the IC, not a prediction. v2.3 §B.4: **`lead_time = K`**, **`lead_time coords = np.arange(1, K+1) * 6` hours**. The IC is captured separately as the `init_state` variable in the same NetCDF (52 channels, no `pr_6h` since the IC has no diagnostic prediction).
5. **Climate rollout length depends on file length.** v2.2 §B.3 said "8 ICs × 1455 steps". With variable file lengths and within-file rollout starting at sample 0, the maximum rollout horizon per file is `(n_samples - 1)`:
   - Non-leap test files (`MOST.0121, 0123, 0124, 0125, 0127, 0128`, n=1455): up to **1454 predictions** each (~363.5 days at 6 h spacing).
   - Leap test files (`MOST.0122, 0126`, n=1459): up to **1458 predictions** each (~364.5 days).
   v2.3 §B.3 uses per-file `K = n_samples - 1`. Total climate-rollout forward passes: 6×1454 + 2×1458 = **11 640** (was 8×1455 = 11 640 in v2.2 by coincidence — same total compute, different per-file horizons).
6. **Stale 1455 mentions cleaned up** in §3 P-1 table footnote, §4 source-layout ASCII tree, §A.3 example, §G.2 SLURM wallclock estimate, §G.2 climatology I/O note, §I risks table.

**v2.2 — Codex round 2 fixes (load-bearing):**

1. **File length is variable: 1455 (non-leap) or 1459 (leap year, +4 samples = +1 day at 6 h spacing).** v2.1 stated "1455 samples per file" everywhere. Filesystem-verified breakdown:
   - **Test split (8 files):** 6 × 1455, 2 × 1459 (`MOST.0122.h5`, `MOST.0126.h5` are leap years).
   - **Training pool (98 files in `train/`):** 75 × 1455, 23 × 1459 (every 4th year). v2.1 referenced "100 training files" and Codex said "76/24"; the actual count is 98 (years 12–100 minus a few packager omissions: `MOST.0094` and several others not present), of which 23 are leap-year. Climatology code must walk the file list and read each file's actual length, not assume 1455.
   - **Pattern:** PlaSim's `plasim_calendar = 'proleptic_gregorian'` includes Feb 29 in leap years, adding 4 × 6 h samples = 24 h to that file.
2. **Climatology indexing changes from `sample_of_year` to `time_of_year_proleptic`.** v2.1 §C.2 locked sample-of-year. With variable lengths that breaks alignment after Feb 29 — sample s=240 in a 1459-file is March 1 12:00 but the same s=240 in a 1455-file is Feb 29 12:00. v2.2 §C.2 rewrites climatology to bin by `(month, day, hour)` (sliced to 6 h granularity → 366 days × 4 hours = 1464 bins per channel). Each bin records mean, std, and `n_contributors` (varies: ~98 for non-leap days, ~23 for leap-day bins).
3. **Eval rollout pseudocode rewritten to mirror `validate_one_epoch` exactly.** v2.1 §B.2 had pseudocode that:
   - Called `cache_unpredicted_features(inp_state, target, forcing, mode='eval')` — wrong signature. Actual signature (`makani-src/makani/models/preprocessor.py:374`): `cache_unpredicted_features(x, y, xz=None, yz=None)`, returning `(x, y)`. The 4 inputs are `(inp_state, tar, inp_forcing, tar_forcing)` from the dataset.
   - Did not call `flatten_history` on the input or per-step targets; `validate_one_epoch` does both (`makani-src/.../deterministic_trainer.py:621, 624`).
   - Did not use `torch.split(tar, 1, dim=1)` to walk lead times; the validation loop does (`deterministic_trainer.py:617`).
   - Used `wrapper(x)` / `pred[:, :52]` for state-feedback; the validation loop uses `inpt = preprocessor.append_history(inpt, pred, idt)` which already encodes the 53→52 slice in PlasimPreprocessor's override (`src/sfno_training/models/preprocessor.py:39`).

   v2.2 §B.2 replaces the pseudocode with the **exact** `validate_one_epoch` body, with `n_future = K - 1` so the loop produces K predictions at lead times {1..K} × 6 h, mirroring v3 of `sfno_tiny_short_training_plan.md` §1.
4. **Run tag now records three SHAs separately.** v2.1 §G.4 mixed two: `eval_sha7` and `train_sha7` (read from `packager_git_sha`). But `packager_git_sha` is the **data-packager** SHA (the code that built the h5 files), not the **training-code** SHA (the code that ran SFNO training). They are different repos / different commits. v2.2 §G.4 splits into:
   - `eval_sha7` — git short SHA of AI-RES at eval submit time.
   - `data_sha7` — `packager_git_sha[:7]` from h5 attrs (= `58413cb` for current dataset).
   - `train_sha7` — separate sidecar file written at training submit time.
5. **`train_code_sha` is currently `unknown` for the existing checkpoint.** Verified `runs/sfno_full/plasim_sim52_full/0/config.json` has zero git/sha/commit keys, and `src/sfno_training/submit_full.slurm` does not capture a SHA. For this evaluation, the run tag will use literal `unknown` for `train_sha7` and the report will note this as a provenance gap. v2.2 §G.5 adds a recommendation to patch `submit_*.slurm` to capture `git rev-parse --short HEAD > $EXP_DIR/train_code_sha.txt` before launching training going forward.
6. **Time-variable semantics confirmed.** `time_plasim` is `float64`, days since per-file Aug-1 anchor (e.g., `0.0, 0.25, ..., 363.5` for a 1455-sample non-leap file; `0.0, ..., 364.5` for 1459-sample leap). `timestamp` is `int64`, seconds since the **same per-file anchor** (NOT seconds since Unix epoch — verified by `timestamp[0] = 0`, `timestamp[1] = 21600`). Use `time_plasim` for climatology binning. v2.2 §A.3 and §C.2 use this.
7. **`makani-src` path corrected** in §B references — was `makani-src/makani/utils/trainer/...`, actual path is `makani-src/makani/utils/training/...` (singular `trainer` → `training`).

**v2.1 — filesystem-verified resolutions (no Codex round needed):**

1. **P-1 resolved.** Test files `MOST.0121.h5..MOST.0128.h5` confirmed present at `/scratch/11114/zhixingliu/SFNO_Climate_Emulator/data/makani/sim52_astro_64x128/test/` (verified 2026-04-29). v2 mistakenly only listed `train/` subdir; the canonical packager layout splits years by `split` attr into `train/` (years 12–100), `validation/` (diagnostic JSONs only — not data), and `test/` (years 121–128). §3 P-1 now records ✅ RESOLVED.
2. **Codex check 1 answered.** All 8 test files have `packager_git_sha = 58413cb11e41`, `rsdt_method = 'astronomical'`, `plasim_calendar = 'proleptic_gregorian'`, `(1455, 52, 64, 128)` — identical to the training pool. Confirmed by direct h5 attr read.
3. **Codex check 2 answered.** Packager **does** write a `split` attribute. Training files: `split = 'train'`; test files: `split = 'test'`. §A.1 sanity check stays as-written.
4. **Codex check 4 answered (option b).** `plasim_time_units` of consecutive files: `0016-08-01`, `0017-08-01`, `0018-08-01`, ..., `0126-08-01`, ..., `0133-08-01`. Each file's anchor restarts on **Aug 1** of a new PlaSim year, with the year tag advancing by exactly 1 per file. This is option (b) from v2 §C.2: each file independently spans one PlaSim year. **Sample-of-year indexing is correct.** v2.1 sets `clim_index_mode = 'sample_of_year'` as the locked default; `time_of_year_proleptic` retained as a fallback flag for future-proofing only.
5. **Time-variable name corrected.** v2 §A.3 says "read the `time` variable"; the actual h5 group has `time_plasim` (PlaSim native time, days since per-file anchor) and `timestamp` (likely seconds since epoch — to verify). v2.1 reads `time_plasim`.
6. **§3 P-1 demoted from BLOCKER to ✅ RESOLVED.** Implementation can begin without packager work.

**v2 — Codex round 1 fixes (all blockers + majors):**

1. **Inference engine swapped from stock `Inferencer` to a custom rollout driver** under `src/sfno_inference/`. v1 §1, §B.1, §B.2 locked Makani's `Inferencer`; that path is **explicitly hard-blocked** at `src/sfno_training/trainer/plasim_trainer.py:93` because stock inference drops the 6 forcing channels. v2 builds the driver on top of `PlasimForcingDataset` + `PlasimPreprocessor` + `MultiStepWrapper`, mirroring `validate_one_epoch` (`makani-src/makani/utils/trainer/deterministic_trainer.py:620`) but extended past `valid_autoreg_steps` for arbitrary horizons.
2. **Test split changed from `MOST.0003..0010` to `MOST.0121..0128`.** The original PlaSim simulation has dedicated held-out test years; using `0003..0010` would be early spin-up files that may have entered the original normalization/statistics pool. v2 §A2 documents the precondition that these files must be packaged into the Makani layout before §A runs (they are not currently on disk under `sim52_astro_64x128/`).
3. **Earth2Studio sparse clone dropped as a runtime dep.** v1 §1 assumed `earth2studio.statistics.{rmse,acc}` would import; verified that `acc` imports `earth2studio.data` which is not in the sparse checkout. RMSE/ACC are now implemented from scratch in `src/sfno_eval/metrics.py` (~50 lines each). v2 §D2 also fixes the call signature: `earth2studio.statistics.rmse(["lat","lon"], weights=...)` takes coordinate dimensions and CoordSystem dicts, not just weights — irrelevant now that we own the implementation.
4. **Calendar facts corrected.** v1 said 1460 6-h steps/year and assumed `MOST.0003-01-01 00:00` ICs. h5 attrs verify: each file is `(1455, 52, 64, 128)` for `fields_state` (1455 samples × 6 h = 363.75 days/file, **not** 365 days; **not** 360 days), `plasim_calendar = proleptic_gregorian`, and `plasim_time_units = 'days since YYYY-MM-DD HH:MM:SS'` per file (e.g., `MOST.0011` is anchored at `0016-08-01`, **not** Jan 1). v2 indexes ICs by **sample-index-within-file**, not by calendar date, and reads each file's `plasim_time_units` + time variable to attach absolute timestamps in the output NetCDFs.
5. **Climatology recomputed from years 12–111.** v1 implicitly relied on `stats/time_means.npy` for climatology. v2 §C2 builds climatology fresh from the 100 training files using sample-of-year indexing (0..1454), independent of `stats/`.
6. **58→53 channel contract is a first-class testable invariant.** v2 adds explicit assertions inside the eval driver and §H.0 dedicates a test (`test_eval_contract.py`) to proving: input is 52 state + 6 forcing concat = 58, output is 53, only `pred[:, :52]` feeds back, `pr_6h` (channel 52) is never recycled, and the forcing tensor at step k+1 comes from h5 truth, not from model output.
7. **Spectra computation switched from 2-D FFT to spherical harmonic transform.** v1 §E.2 said "2-D FFT and radial averaging". On a legendre-gauss grid this is wrong — λ–φ space is non-uniform. v2 §E2 uses `torch_harmonics.RealSHT` (already a Makani dep) for a proper SHT-based KE / variance spectrum. Spectra are also gated to **phase 2** (after RMSE/ACC produce a clean scorecard) — see §0.A.
8. **Climate rollout sample count increased from 3 → 8.** Codex marked "3 ICs is a stability smoke, weak for climate claims." v2 runs **all 8 test years** at 1-year rollout for climate stats (no extra cost — inference budget already allowed for it).
9. **Run-tag now records both eval-code SHA and checkpoint/config provenance.** v1 used eval-time SHA only. v2 §G4: `run_tag = {YYYYMMDD}_{eval_sha7}_ckpt-{ckpt_basename}_train-{train_sha7}` where `train_sha7` is read from `checkpoint config.json` provenance and from the per-file h5 attribute `packager_git_sha`.
10. **Lat-weights confirmed Gaussian quadrature, not cos(lat).** v1 left this as Codex check 5. v2 §B5 + §D1 lock Gauss-Legendre quadrature weights; `torch_harmonics` exposes them, otherwise pre-compute and cache as `stats/lat_weights_legendre_gauss.npy`.
11. **Distribution-shift thresholds reframed as gross sanity, not science.** v1 §A.2 implied a pass/fail gate. v2 §A3 calls these "report-only diagnostics — not a ship gate."
12. **Ensemble eval explicitly out of scope.** Codex confirmed deterministic-only. v2 §0.B.

---

## 0. Phasing & scope discipline

To avoid the v1 trap of stacking too much in one run, v2 splits delivery in two phases:

**Phase 1 — RMSE/ACC scorecard (this plan ships).**
- §A test split, §B custom rollout driver, §C climatology + persistence, §D NWP scoring (RMSE + ACC + bias maps), §F minimal report, §G inference + score SLURMs, §H tests.
- Climate-stats §E is **stubbed only** in phase 1 (zonal-mean bias and time-mean bias maps from the long rollouts, no spectra).

**Phase 2 — Climate-fidelity (separate v3 plan).**
- §E spectra (SHT-based), variance-ratio analysis, drift trajectories, full climate scorecard.
- Triggered only after Phase 1 sanity gate (§D3) passes.

**Out of scope (locked).**
- Ensemble evaluation (no IC perturbation).
- ERA5 / external truth — PlaSim is the reference.
- Recipe-style integration with `external/earth2studio/recipes/eval/` — the recipe stays a *reference template*, not a runtime dep.

---

## 1. Context & current state (unchanged from v1, retained for diffability)

Trained checkpoint:
```
/scratch/11114/zhixingliu/SFNO_Climate_Emulator/runs/sfno_full/plasim_sim52_full/0/
├── config.json                       # frozen Makani params used at training time
├── metadata.json                     # entrypoint = makani.models.model_package:load_time_loop (NOT used; see v1 §B.5 explanation)
├── global_means.npy / global_stds.npy
└── training_checkpoints/
    └── best_ckpt_mp0.tar             # 1.28 GB, val_loss = 0.00210 @ epoch 50
```

Channel order (52 state + 1 diagnostic, **immutable** for the rollout invariant):
```
0   pl
1   tas
2..11   ta1..ta10
12..21  ua1..ua10
22..31  va1..va10
32..41  hus1..hus10
42..51  zg200..zg1000  ← v10.1 contract: pressure-level zg, TOA→surface; zg500 at index 46
52      pr_6h     ← diagnostic; output-only, never feeds back
```

Forcing order (6 channels, supplied by h5 truth at every step):
```
lsm, sg, z0, sst, rsdt, sic
```

---

## 2. Locked decisions (Codex-validated)

| Decision | v2 choice | Source |
|---|---|---|
| Eval style | NWP scorecard (Phase 1) + climate stats (Phase 2) | Interview + Codex confirmation. |
| Inference engine | **Custom rollout driver** under `src/sfno_inference/` | Codex fix 1. Stock `Inferencer` blocked at `plasim_trainer.py:93`. |
| Test years | `MOST.0121..0128` | Codex fix 2. |
| Lat-weights | **Gauss-Legendre quadrature weights** | Codex confirmation. |
| Climatology source | **`/scratch/.../data/makani/sim52_full/train/`** (100 files, years 12–111, 76 non-leap + 24 leap) | Codex round-3 fix 1. |
| Climatology indexing | **`time_of_year_proleptic` — bin by `(month, day, hour_quarter)`** | Codex round-2 fix 2. v2.1's `sample_of_year` breaks at Feb 29 across 1455 vs 1459 files. |
| Sample count per file | **1455 (non-leap) or 1459 (leap)** | Codex round-2 fix 1. |
| Calendar | **proleptic_gregorian**, per-file Aug-1 anchor, leap years add Feb 29 | h5 attrs verified. |
| Climate rollout count | **8 ICs × 1 year** (one per test year) | Codex fix 8. |
| Metrics dependency | **Manual NumPy/Torch implementation in `src/sfno_eval/metrics.py`** | Codex fix 3. |
| Spectra | **SHT-based, Phase 2** | Codex fix 7. |
| Ensemble | Out of scope | Codex confirmation. |
| Run tag | `{date}_eval-{eval_sha7}_data-{data_sha7}_train-{train_sha7}_ckpt-{name}` | Codex round-2 fix 4. Three SHAs separately. |
| Storage | `results/sfno_eval/<run_tag>/` (symlink → `/work2`) | unchanged. |
| Compute | Stampede3 `h100`, single GPU | unchanged. |

---

## 3. Preconditions — must be true before §A runs

### P-1 (✅ RESOLVED 2026-04-29). Test files exist at `sim52_astro_64x128/test/`.

Verified via `ls /scratch/11114/zhixingliu/SFNO_Climate_Emulator/data/makani/sim52_astro_64x128/test/`: all 8 files `MOST.0121.h5..MOST.0128.h5` present. The packager writes years to subdirs by `split` attribute — v2 only listed `train/` and missed this. Per-file h5 attrs confirmed identical processing pipeline to training pool:

| File | split | year | anchor | shape (state) | packager_sha | rsdt_method |
|---|---|---|---|---|---|---|
| `MOST.0011` | `train` | 11 | `0016-08-01` | `(1455,52,64,128)` | `58413cb1…` | `astronomical` |
| `MOST.0012` | `train` | 12 | `0017-08-01` | `(1455,52,64,128)` | `58413cb1…` | `astronomical` |
| `MOST.0014` | `train` | 14 | `0019-08-01` | `(1459,52,64,128)` ¹ | `58413cb1…` | `astronomical` |
| `MOST.0094` | `train` | 94 | `0099-08-01` | `(1455,52,64,128)` ² | `58413cb1…` | `astronomical` |
| `MOST.0121` | `test` | 121 | `0126-08-01` | `(1455,52,64,128)` | `58413cb1…` | `astronomical` |
| `MOST.0122` | `test` | 122 | `0127-08-01` | `(1459,52,64,128)` ¹ | `58413cb1…` | `astronomical` |
| `MOST.0126` | `test` | 126 | `0131-08-01` | `(1459,52,64,128)` ¹ | `58413cb1…` | `astronomical` |
| `MOST.0128` | `test` | 128 | `0133-08-01` | `(1455,52,64,128)` | `58413cb1…` | `astronomical` |

¹ Leap year — file spans Aug 1 of year (Y+5) to Aug 1 of year (Y+6) where (Y+6) is a Gregorian leap year. 1459 = 1455 + 4 extra 6 h samples for Feb 29.
² Centennial exception. Year 100 is divisible by 100 but not 400, so it is **not** a leap year in proleptic_gregorian — file has 1455 samples like a regular year.

**Codex check 1 answered:** identical `packager_git_sha` + `rsdt_method` confirms the test pool is at the same physical-units / variable-coverage state as years 12–100. No re-packaging needed.

§A.1 invocation:
```
python scripts/build_test_split.py \
  --src /scratch/11114/zhixingliu/SFNO_Climate_Emulator/data/makani/sim52_astro_64x128/test/ \
  --dst /scratch/11114/zhixingliu/SFNO_Climate_Emulator/data/makani/sim52_full/test_holdout/ \
  --years 0121,0122,0123,0124,0125,0126,0127,0128
```

### P-2. earth2mip is NOT installed.

`makani.models.model_package.load_time_loop` imports `earth2mip` (`model_package.py:266`). v2 does not use this path — but if any helper code accidentally imports it, jobs will fail. **Mitigation:** the eval driver does *not* import `model_package`; it instantiates the SFNO model directly via `Trainer.restore_from_checkpoint` per the v4 changelog of `sfno_tiny_short_training_plan.md`.

### P-3. `torch_harmonics` is installed.

Required for Gauss-Legendre quadrature weights (§B5) and Phase-2 SHT spectra (§E2). It is a transitive dep of Makani and should already be in the venv. Preflight asserts `import torch_harmonics`. API confirmed: `torch_harmonics.quadrature.legendre_gauss_weights(64, -1.0, 1.0)`.

### P-4 (residual sanity check, not blocking).

The file-index → anchor-year offset is **+5** (verified across MOST.0011, 0012, 0121, 0128). File `MOST.000Y.h5` spans Aug 1 of year (Y+5) → Aug 1 of year (Y+6). A file is **leap** (1459 samples) iff Gregorian year (Y+6) is a leap year:
- `(Y+6) % 4 == 0` AND
- `(Y+6) % 100 != 0` OR `(Y+6) % 400 == 0`.

This rule predicts leap files at `Y ∈ {14, 18, 22, ..., 90, 98, 102, 106, 110}` for the training pool — **24 files**, matching the observed count. The exception is `Y=94` (Y+6 = 100, centennial → non-leap) → 1455 samples, also matching observation.

**Mitigation:** before §C runs the full climatology pass, execute `scripts/trace_calendar_anchors.py` on a small set (`MOST.0014` first leap, `MOST.0094` centennial-exception, `MOST.0122` test leap, `MOST.0128` test non-leap) and verify the per-sample mapping `time_plasim[s] → absolute_dt` puts Feb 29 at exactly the expected sample index. This is a **15-second sanity check**, not a blocker — the math has been re-derived from filesystem evidence.

---

## 4. Source layout (target after v2 ships)

```
src/sfno_inference/                    # NEW — full rollout driver
├── __init__.py
├── rollout_driver.py                  # core: validate_one_epoch-shaped loop, K=arbitrary
├── checkpoint_loader.py               # restore SFNO + PlasimPreprocessor from best_ckpt_mp0.tar
├── nc_writer.py                       # NetCDF dims (init_time, lead_time, channel, lat, lon) + h5 attrs replay
└── README.md                          # explains: this is the v2 inference path required by the hard gate

src/sfno_eval/                         # NEW — scoring & report
├── __init__.py
├── metrics.py                         # rmse_lat_weighted, acc, bias_map (manual, no e2s dep)
├── climatology.py                     # build per-(month, day, hour_quarter, channel, lat, lon) clim from years 12-111 (time-of-year-proleptic indexing)
├── nc_io.py                           # xarray helpers around our NetCDF format
└── (Phase 2 only: spectra.py)

scripts/
├── build_test_split.py                # NEW — symlink MOST.0121..0128 into test_holdout/
├── eval_inference.py                  # NEW — batch driver over (year, IC) pairs
├── compute_climatology.py             # NEW — produces baselines/climatology_proleptic.nc (single file, 366×4 bins)
├── score_nwp.py                       # NEW — NWP scorecard
├── render_eval_report.py              # NEW — Phase 1 report.md
├── submit_eval_inference.slurm        # NEW
├── submit_eval_score.slurm            # NEW
└── submit_eval.sh                     # NEW — chains the SLURM jobs with --dependency

tests/sfno_inference/
├── test_rollout_driver.py
├── test_checkpoint_loader.py
└── test_nc_writer.py

tests/sfno_eval/
├── test_metrics.py
├── test_climatology.py
└── test_eval_contract.py              # 58→53 invariant + forcing-from-truth proof

results/sfno_eval/<run_tag>/           # → /work2/.../results/sfno_eval/<run_tag>/
├── inference/
│   ├── nwp/MOST.0121_ic000.nc..       # 8 yr × 12 IC = 96 files, ~92 MB each, ~9 GB total
│   └── climate/MOST.0121_full.nc..    # 8 files, K=1454 or 1458 (per file_length-1), ~2.36–2.37 GB each, ~19 GB total
├── baselines/
│   └── climatology_proleptic.nc       # (366, 4, 53, 64, 128) — month × hour-quarter bins, ~2.4 GB; sibling n_contributors[366,4]
├── scores/
│   ├── nwp_scorecard.csv
│   ├── bias_maps_<channel>_<lead>.npy
│   └── (Phase 2: spectra/, drift/)
├── plots/
└── report.md
```

---

## A. Build the held-out test split

### A.1 `scripts/build_test_split.py`
- CLI: `python scripts/build_test_split.py --src /scratch/.../sim52_astro_64x128/test/ --dst /scratch/.../sim52_full/test_holdout/ --years 0121,0122,0123,0124,0125,0126,0127,0128`
- Note **`test_holdout/`**, not `test/`. The existing `data/makani/sim52_full/test/` is empty and was carved by `build_subset_dataset.py`; we keep that path as the canonical "production test split for retraining" and put eval data in a sibling `test_holdout/` so eval cannot contaminate retraining splits.
- Behavior: relative symlinks from `test_holdout/MOST.NNNN.h5` → `sim52_astro_64x128/test/MOST.NNNN.h5` (canonical packager test subdir, verified §3 P-1). Idempotent.
- **Sanity check baked in:** for each test file, assert `f.attrs['split'] == 'test'`. Packager attribute confirmed — Codex check 2 answered.

### A.2 Sanity diagnostic (report-only — not a ship gate, per Codex feedback).
For each test-year h5, print:
- Per-channel mean and std → `results/sfno_eval/<run_tag>/diagnostics/test_split_distribution.csv`.
- ±2 σ overlap with training pool means.
- Comment in the report if any channel falls outside training distribution by > 3 σ.

These are flagged for human eyeballing, **not** a gate that blocks the rest of the pipeline.

### A.3 Time anchor extraction.
For each test-year h5: read `attrs['plasim_time_units']`, parse the anchor (`days since YYYY-MM-DD HH:MM:SS`), and read the `time_plasim` dataset (per-sample days since the file's anchor — `float64`, e.g. `0.0, 0.25, 0.5, ..., 363.5` for non-leap and `..., 364.5` for leap files). The sibling `timestamp` dataset is `int64` seconds since the **same per-file anchor** (NOT seconds since Unix epoch — verified). Write `meta/test_year_anchors.json` mapping `MOST.NNNN → {anchor: '0YYY-08-01 00:00:00', n_samples: <1455 or 1459>, dhours: 6, is_leap: bool}`. Used downstream by `nc_writer.py` to attach physical time coords on output NetCDFs.

Confirmed anchors (filesystem-verified §3 P-1):
- `MOST.0121` → `0126-08-01 00:00:00`
- `MOST.0122` → `0127-08-01 00:00:00`  *(extrapolated from +1 yr/file pattern; verify on first read)*
- ...
- `MOST.0128` → `0133-08-01 00:00:00`

### A.4 NWP-mode IC selection — **12 ICs/year, monthly stride, no cross-file rollout**.

For each test file, NWP-mode rollouts launch from 12 ICs spaced approximately monthly within the file. The `K = 56` rollout horizon (= 14 days × 4 samples/day) must fit entirely inside the file, so every IC index `s` must satisfy `s + K < n_samples`. Cross-file rollout is **explicitly out of scope** — the dataset's `_get_indices` would silently return samples from the next file, mixing two physical anchors and breaking the `time_plasim`-based provenance recorded in the output NetCDFs.

**Stride formula** (in `scripts/eval_inference.py`):

```python
def nwp_ic_offsets(n_samples: int, K: int = 56, n_ic: int = 12) -> list[int]:
    """Return n_ic IC sample indices that fit within [0, n_samples - K - 1].

    Spacing is `floor((n_samples - K) / n_ic)` ≈ monthly cadence:
      - non-leap (n=1455, K=56): step = floor(1399 / 12) = 116 → ICs at
        [0, 116, 232, ..., 1276]; max IC + K = 1332 < 1455. ✓
      - leap     (n=1459, K=56): step = floor(1403 / 12) = 116 → ICs at
        [0, 116, 232, ..., 1276]; max IC + K = 1332 < 1459. ✓
    """
    assert n_samples > K + n_ic, f"n_samples={n_samples} too small for K={K}, n_ic={n_ic}"
    step = (n_samples - K) // n_ic
    offsets = [i * step for i in range(n_ic)]
    # Hard guard: every IC + K rollout must terminate inside this file.
    for s in offsets:
        assert s + K < n_samples, (
            f"IC offset {s} + K={K} = {s+K} would cross file boundary at n_samples={n_samples}; "
            "cross-file rollout is not supported"
        )
    return offsets
```

The IC sample indices are the same for leap and non-leap files (since `step` works out identically for n=1455 and n=1459 at K=56, n_ic=12) — keeping the layout uniform across `nwp_scorecard.csv` rows. The corresponding `ic_sample_idx` is recorded as a column so that downstream analyses can reconstruct each IC's calendar date (via `time_plasim[s]` + the file anchor).

For climate-mode rollouts (§B.3), only `s = 0` is used — that's the within-file maximum-horizon rollout. The same `s + K < n_samples` guard applies with `K = n_samples - 1`, which is satisfied by construction.

---

## B. Custom rollout driver — `src/sfno_inference/rollout_driver.py`

### B.0 Eval params + checkpoint contract — `src/sfno_inference/checkpoint_loader.py`

The eval driver must build its `params` object by **loading the run-dir `config.json`** (not the raw training YAML) and asserting the channel-count contract before constructing the wrapper. This keeps the eval-time SFNO geometry bit-identical to training.

```python
import json
import torch
from makani.utils.YParams import ParamsBase  # whatever container the trainer uses
from makani.models import model_registry
from makani.utils.driver import Driver
from sfno_training.trainer.plasim_trainer import _install_plasim_patches

RUN_DIR = "/scratch/11114/zhixingliu/SFNO_Climate_Emulator/runs/sfno_full/plasim_sim52_full/0"

def load_eval_params(run_dir: str = RUN_DIR, *, K: int) -> ParamsBase:
    cfg = json.load(open(f"{run_dir}/config.json"))
    eval_params = ParamsBase()
    for k, v in cfg.items():
        setattr(eval_params, k, v)

    # Hard contract — these are the values the metadata-time patch installed
    # at training; if they ever drift, abort before model build.
    assert eval_params.N_in_channels == 58, (
        f"expected 58 (52 state + 6 forcing); got {eval_params.N_in_channels}"
    )
    assert eval_params.N_out_channels == 53, (
        f"expected 53 (52 state + 1 diagnostic); got {eval_params.N_out_channels}"
    )
    assert eval_params.n_state_channels == 52
    assert eval_params.n_diagnostic_channels == 1
    assert eval_params.n_forcing_channels == 6

    # Eval-only overrides
    eval_params.valid_autoreg_steps = K - 1     # ACTIVE handle in eval mode (plasim_trainer.py:103-105)
    eval_params.n_future = K - 1                # set redundantly for any code that reads it directly
    eval_params.n_history = 0
    eval_params.data_num_shards = 1
    eval_params.data_shard_id = 0
    eval_params.batch_size = 1
    eval_params.global_means_path = f"{run_dir}/global_means.npy"
    eval_params.global_stds_path  = f"{run_dir}/global_stds.npy"
    # forcing_global_means_path / forcing_global_stds_path are inherited from cfg

    # AMP — config.json stores only `amp_mode` ('bf16' for the production run).
    # `amp_enabled` and `amp_dtype` are derived in the trainer at runtime
    # (deterministic_trainer.py:84-97) and are NOT serialised. Eval must
    # mirror that derivation so autocast in B.2 picks the right dtype.
    amp_mode = getattr(eval_params, "amp_mode", "none")
    if amp_mode == "none":
        eval_params.amp_enabled = False
        eval_params.amp_dtype = torch.float32
    elif amp_mode == "fp16":
        eval_params.amp_enabled = True
        eval_params.amp_dtype = torch.float16
    elif amp_mode == "bf16":
        eval_params.amp_enabled = True
        eval_params.amp_dtype = torch.bfloat16
    else:
        raise ValueError(f"Unknown amp_mode: {amp_mode!r}")

    return eval_params

def build_wrapper_from_checkpoint(eval_params, ckpt_path, device):
    """Mirror the trainer's model-build path exactly, then move to device.

    Trainer reference: makani/utils/training/deterministic_trainer.py:132 —
    `self.model = model_registry.get_model(self.params, multistep=self.multistep).to(self.device)`

    `_install_plasim_patches()` rebinds `model_registry.MultiStepWrapper`
    to `PlasimMultiStepWrapper` (and SingleStep likewise) BEFORE
    `get_model` is called, so the returned wrapper is the PlaSim variant
    with `wrapper.preprocessor = PlasimPreprocessor(params)` already wired.
    Patches are idempotent.
    """
    _install_plasim_patches()
    wrapper = model_registry.get_model(eval_params, multistep=True).to(device)

    # Driver.restore_from_checkpoint signature (makani/utils/driver.py:347-356):
    #   Driver.restore_from_checkpoint(checkpoint_path, model, loss=None,
    #       optimizer=None, scheduler=None, counters=None,
    #       checkpoint_mode='legacy', strict=True)
    # Restore AFTER .to(device) so the loaded tensors land on the correct
    # device directly (the legacy path uses `map_location` internally and
    # then copies into the model's existing parameters).
    Driver.restore_from_checkpoint(ckpt_path, wrapper, checkpoint_mode="legacy")
    wrapper.eval()

    # Post-build assertions on the actual SFNO module. SFNO uses `inp_chans`
    # / `out_chans` (verified at makani/models/networks/sfnonet.py:298-299),
    # NOT the more PyTorch-idiomatic `in_chans` / `out_chans`.
    assert wrapper.model.inp_chans == 58
    assert wrapper.model.out_chans == 53
    # Sanity: weights are on the requested device. Compare type+index
    # explicitly because torch.device("cuda") (no index) != torch.device("cuda:0")
    # under `==` even though .to("cuda") resolves to "cuda:<current_device>".
    actual = next(wrapper.parameters()).device
    expected = torch.device(device)
    assert actual.type == expected.type and (
        expected.index is None or actual.index == expected.index
    ), f"wrapper on {actual}, expected {expected}"
    return wrapper
```

The `valid_autoreg_steps` line is the **load-bearing** override per Codex round 4 §1: `_plasim_get_dataloader(..., mode="eval")` reads the dataset's `n_future` from `params.valid_autoreg_steps`, not from `params.n_future`. Setting only `n_future` would silently cap the test rollout at the training-time `valid_autoreg_steps = 3` regardless of K.

### B.1 Reference loop.

The eval rollout is structurally identical to `validate_one_epoch` in `makani-src/makani/utils/training/deterministic_trainer.py:577` (which calls `MultiStepWrapper.forward` repeatedly via `_forward_eval`), but with two extensions:

1. **Arbitrary horizon K**, not capped at `valid_autoreg_steps`.
2. **Per-step output capture** to a NetCDF, instead of only final-step loss.

### B.2 Loop pseudocode — **mirrors `validate_one_epoch` exactly**.

`PlasimForcingDataset.get_sample_at_index` returns a 4-tuple `(inp_state, tar, inp_forcing, tar_forcing)` (`src/sfno_training/data/plasim_forcing_dataset.py:293`). Shapes:
- `inp_state` → `(n_history+1, 52, H, W)`
- `tar` → `(n_future+1, 53, H, W)` (state ‖ diagnostic, z-scored)
- `inp_forcing` → `(n_history+1, 6, H, W)`
- `tar_forcing` → `(n_future+1, 6, H, W)`

For evaluation we set `n_history=0` and `n_future=K-1` so the loop produces **K predictions at lead times {1..K} × 6 h**.

```python
# Inputs: test_holdout_path, ic_global_idx, K (rollout horizon = number of predictions)
# K is per-IC: NWP mode K=56 (= 14 days). Climate mode K = n_samples_in_file - 1
# (= 1454 for non-leap, 1458 for leap test files).

# Build eval_params from the run-dir config.json with the K-dependent overrides
# (sets BOTH valid_autoreg_steps AND n_future to K-1; see §B.0).
eval_params = load_eval_params(run_dir=RUN_DIR, K=K)
assert eval_params.valid_autoreg_steps == K - 1     # ACTIVE handle in mode='eval' (plasim_trainer.py:103-105)
assert eval_params.n_future == K - 1                # redundant guard

# Build the SFNO + PlasimPreprocessor via the patched wrapper class.
# This is the same path used by the trainer; restore_from_checkpoint
# loads the saved weights into wrapper.model and resets wrapper.preprocessor's
# internal buffers. build_wrapper_from_checkpoint asserts inp_chans==58, out_chans==53,
# and moves the wrapper to `device` so model and data tensors collocate.
if torch.cuda.is_available():
    device = torch.device(f"cuda:{torch.cuda.current_device()}")    # explicit index
else:
    device = torch.device("cpu")
wrapper = build_wrapper_from_checkpoint(eval_params, ckpt_path, device)
preprocessor = wrapper.preprocessor                 # USE the wrapper's instance — do not create a new one

# Build the dataset via the patched dataloader factory. This guarantees identical
# constructor kwargs to the validation path (location, in_channels, out_channels,
# n_history, n_future, bias, scale, forcing_bias, forcing_scale, data_grid_type, ...).
# Because eval_params.valid_autoreg_steps == K-1, the dataset will produce
# (n_future+1) = K future targets per sample.
# We discard the dataloader/sampler and index the dataset directly for IC selection.
dataloader, dataset, _ = _plasim_get_dataloader(eval_params, test_holdout_path, device, mode="eval")
assert dataset.n_future == K - 1, (
    f"PlasimForcingDataset.n_future={dataset.n_future}, expected {K-1}; "
    "did valid_autoreg_steps fail to propagate?"
)

# Output normalization stats — load from RUN DIR (see §C.3 / Q5 — byte-identical
# to the dataset-stats copy for this run, but run-dir is the canonical source so
# any future per-checkpoint override stays self-consistent). Move to device and
# reshape to broadcast against (K, 53, H, W) z-scored predictions: (53,) → (1, 53, 1, 1).
out_bias_np  = np.load(eval_params.global_means_path).astype(np.float32)   # shape (53,)
out_scale_np = np.load(eval_params.global_stds_path).astype(np.float32)    # shape (53,)
assert out_bias_np.shape == (53,) and out_scale_np.shape == (53,)
out_bias  = torch.from_numpy(out_bias_np).to(device).reshape(1, 53, 1, 1)
out_scale = torch.from_numpy(out_scale_np).to(device).reshape(1, 53, 1, 1)

inp_state, tar, inp_forcing, tar_forcing = dataset[ic_global_idx]
assert tar.shape[0] == K, f"target T-axis = {tar.shape[0]}, expected K={K}"

# Add batch dim and move to device.
gdata = tuple(t.unsqueeze(0).to(device) for t in (inp_state, tar, inp_forcing, tar_forcing))

# === BEGIN: copy of validate_one_epoch body (deterministic_trainer.py:617-661) ===

inp, tar = preprocessor.cache_unpredicted_features(*gdata)    # caches inp_forcing/tar_forcing into unpredicted_inp_eval/unpredicted_tar_eval; returns (inp_state_z, tar_z)
inp = preprocessor.flatten_history(inp)                       # (1, 52, H, W) since n_history=0

tarlist = torch.split(tar, 1, dim=1)                          # K-tuple of (1, 1, 53, H, W) tensors

predictions = []                                              # list of (1, 53, H, W) z-scored predictions
inpt = inp
for idt, targ in enumerate(tarlist):
    targ = preprocessor.flatten_history(targ)                 # (1, 53, H, W)

    # Autocast: use device.type so CPU smoke tests don't trip on
    # device_type='cuda'. AMP is only enabled when running on CUDA AND
    # eval_params.amp_enabled is True (the trainer-derived bf16 setting);
    # on CPU we silently fall back to fp32, which is fine for orchestration smoke tests.
    autocast_enabled = bool(eval_params.amp_enabled) and (device.type == "cuda")
    with torch.inference_mode(), torch.amp.autocast(
        device_type=device.type,
        enabled=autocast_enabled,
        dtype=eval_params.amp_dtype if autocast_enabled else torch.float32,
    ):
        pred = wrapper(inpt)                                   # _forward_eval → single-step forward, returns (1, 53, H, W)

    predictions.append(pred.detach().clone())

    # advance state for next step. PlasimPreprocessor.append_history (src/sfno_training/models/preprocessor.py:39)
    # asserts pred has 53 channels, slices to first 52 for state-only feedback, and advances the forcing buffer pointer
    # by exactly one slot (forcing for step idt+1 is read from unpredicted_inp_eval, not from pred).
    inpt = preprocessor.append_history(inpt, pred, idt)
# === END: validate_one_epoch body ===

predictions = torch.cat(predictions, dim=0)                   # (K, 53, H, W) z-scored
predictions_phys = predictions * out_scale + out_bias         # de-z-score using run-dir global_means/stds

# Write NetCDF — see B.4.
```

**Key differences from v2.1's pseudocode (now correct):**
- Dataset returns 4 tensors, not 3 — explicit `tar_forcing` for `n_future` future-step forcings.
- `cache_unpredicted_features` takes 4 args, returns 2 (the state pair); forcing is buffered in `unpredicted_inp_eval` / `unpredicted_tar_eval` and consumed inside `wrapper(inpt)` by `append_unpredicted_features`.
- State feedback is via `preprocessor.append_history(inpt, pred, idt)`, which **internally** does the 53→52 slice (PlasimPreprocessor override). Driver code does **not** manually do `pred[:, :52]`.
- `flatten_history` is called on both the input and each lead-time target before scoring/forward — required even at `n_history=0` because the dataset returns shape `(B, 1, C, H, W)` and the model expects `(B, C, H, W)`.

### B.3 The 58→53 contract — **explicit assertions inside the loop**.
The driver asserts on every step:
- `inpt.shape == (1, 52, 64, 128)` after `flatten_history` (no forcing in inpt; forcing is buffered).
- `internal.shape == (1, 58, 64, 128)` *inside* the wrapper after `append_unpredicted_features` (via `register_forward_pre_hook` on `wrapper.model`, per v4 §B.2 of `sfno_tiny_short_training_plan.md`).
- `pred.shape == (1, 53, 64, 128)`.
- **Forcing-from-truth identity** (Codex round-1 fix 6): after `cache_unpredicted_features`, assert `torch.equal(preprocessor.unpredicted_inp_eval[0, ..., 0, :, :], inp_forcing[0, 0, ...])` (step-0 forcing in the buffer matches the dataset's `inp_forcing`). For each subsequent step `idt`, assert `torch.equal(preprocessor.unpredicted_inp_eval[0, ..., idt+1, :, :], tar_forcing[0, idt, ...])` — the forcing at lead `idt+1` came from `tar_forcing[idt]`, **not** from any model output. (Exact buffer-axis layout depends on `cache_unpredicted_features` — verify on first preflight; if this exact slice is wrong, infer the layout from `unpredicted_inp_eval.shape` and update the assertion. The principle is: there exists a slice of the buffer that exactly equals `tar_forcing[idt]` for every step.)
- **53→52 slice is in the preprocessor, not the driver.** PlasimPreprocessor.append_history (`src/sfno_training/models/preprocessor.py:39-56`) asserts `x2.shape == (B, 53, H, W)` then slices to first 52 channels. Driver does **not** do this slice itself. Test `test_eval_contract.py` exercises a 6-step rollout and inspects `inpt.shape == (1, 52, H, W)` after every `append_history` call.
- **`pr_6h` never feeds back.** The 53rd channel (index 52) is sliced off inside `append_history`. Test asserts that mutating the model's `pr_6h` output does not change `inpt` for the next step.

These assertions live inside the rollout loop in **debug mode** (controlled by `eval_params.assert_contract: bool = True`). Production runs can flip to `False` for speed, but the smoke test (§H) always runs with assertions on.

### B.4 NetCDF output schema (`src/sfno_inference/nc_writer.py`).

```
dims:
  init_time   = 1                               # scalar absolute timestamp (from §A3 anchor)
  lead_time   = K                               # K predictions at leads {1..K} × 6 h, NO lead-0
  channel     = 53                              # 52 state + 1 diagnostic (pr_6h)
  channel_ic  = 52                              # IC has no diagnostic
  lat         = 64
  lon         = 128

coords:
  init_time   = parsed datetime64 from h5 attrs
  lead_time   = np.arange(1, K+1) * 6           # in hours: {6, 12, ..., 6K}
  channel     = list of 53 channel names from config
  channel_ic  = list of 52 state channel names (no pr_6h)
  lat         = legendre-gauss latitudes from metadata/data.json
  lon         = equiangular longitudes from metadata/data.json

variables:
  prediction(init_time, lead_time, channel, lat, lon)      # the K × 53 emulator outputs (physical units)
  truth(init_time, lead_time, channel, lat, lon)            # h5 truth at the same lead times (physical units)
  init_state(init_time, channel_ic, lat, lon)               # the IC at lead 0 (52 state channels, physical units)

global_attrs:
  ckpt_path             = checkpoint absolute path
  ckpt_basename         = best_ckpt_mp0
  data_packager_sha     = from h5 attrs['packager_git_sha'] (e.g. '58413cb1…')
  train_code_sha        = from <run_dir>/train_code_sha.txt if present, else 'unknown'
  eval_code_sha         = git short SHA of AI-RES at eval-script run time
  ic_h5_file            = source MOST.NNNN.h5 absolute path
  ic_sample_idx         = int
  rollout_K             = int                     # number of predictions
  dhours                = 6
  plasim_calendar       = 'proleptic_gregorian'
  plasim_year_anchor    = file's plasim_time_units string (e.g., 'days since 0126-08-01 00:00:00')
```

Both `prediction` and `truth` written in **physical units** (de-z-scored using `global_means.npy` / `global_stds.npy` from the run dir, **not** from the dataset stats dir — see §C explanation).

### B.5 Lat-weights: Gauss-Legendre quadrature.
- Use `torch_harmonics` to obtain quadrature weights for the legendre-gauss grid.
  ```python
  from torch_harmonics import quadrature
  cost, w_quad = quadrature.legendre_gauss_weights(64, -1, 1)  # cosθ nodes, weights summing to 2
  lat_weights = w_quad / w_quad.sum()                           # normalize so sum = 1
  ```
- Cache as `stats/lat_weights_legendre_gauss.npy` (one-time, idempotent build).
- Consumed by `metrics.rmse_lat_weighted` and `metrics.acc`.

**API resolved (Codex round 2):** `torch_harmonics.quadrature.legendre_gauss_weights(64, -1.0, 1.0)` is the verified symbol. Numpy fallback `np.polynomial.legendre.leggauss(64)` retained in `metrics.py` as a belt-and-braces alternative.

---

## C. Climatology + persistence — `src/sfno_eval/climatology.py`

### C.1 Persistence baseline — **defined only for the 52 state channels**.

For state channel `c ∈ {0..51}` and IC at sample `s`, persistence prediction at lead `k > 0` is `inp_state[s, 0, c, :, :]` (the IC value, broadcast across all lead times). Computed on the fly inside `score_nwp.py`; no file output needed.

**`pr_6h` (channel 52, the diagnostic) is excluded from persistence.** The IC tensor `inp_state` returned by `PlasimForcingDataset.get_sample_at_index` has shape `(1, 52, H, W)` — `pr_6h` is a target-side-only diagnostic and is not loaded for input (`src/sfno_training/data/plasim_forcing_dataset.py:307-310`). Therefore there is no "lead-0 truth" to persist for that channel. The NWP scorecard CSV records `persistence_rmse[c="pr_6h", lead] = NaN` and the report (§F) explicitly notes:

> Persistence is not defined for diagnostic-only channel `pr_6h` because it has no IC value. Reported persistence comparisons cover the 52 state channels only.

**Why we don't synthesise a `pr_6h` IC from `f['fields_diagnostic'][s, 0, :, :]`:** (a) the model never sees this signal at inference (it is loss-only), so giving persistence access to it would be a strictly unfair comparison; (b) the climate community treats persistence on diagnostic-only channels as undefined — sticking to that convention avoids a paper-review red flag; (c) emulator skill on `pr_6h` is more honestly compared against climatology (§C.2) than against a synthetic persistence baseline.

### C.2 Climatology — built from training pool, indexed by time-of-year proleptic.

**Source:** `/scratch/11114/zhixingliu/SFNO_Climate_Emulator/data/makani/sim52_full/train/` — exactly the directory the model trained on. **100 files** spanning years 12–111, no gaps. Filesystem-verified breakdown: **76 × 1455 + 24 × 1459**. Symlinks: years 12–100 → `sim52_astro_64x128/train/`, years 101–111 → `sim52_astro_64x128/valid/`.

`MOST.0094.h5` is non-leap (1455 samples) — its candidate leap year (anchor year + 1 = 100) hits the proleptic-gregorian centennial exception (`100 % 100 == 0` and `100 % 400 ≠ 0`).

**Why not sample-of-year:** with variable file lengths (1455 vs 1459), sample index s=240 in a 1455-file (Sep 30 00:00) does not land on the same calendar (month, day, hour) as s=240 in a 1459-file (one 6 h slot earlier post-Feb-29). Naive averaging smears the climatology by up to ±0.5 day per leap year. Time-of-year-proleptic indexing handles this exactly.

**Indexing rule:**

Climatology is binned by `(month, day, hour_quarter)` where `hour_quarter ∈ {0, 6, 12, 18}`. Total bins: 366 × 4 = **1464** per channel. Most bins (calendar dates that exist in every year) receive **~100 contributors** (one per training year); the four Feb-29 bins receive only **24 contributors** (one per leap-year file). `n_contributors[366, 4]` is stored alongside `mean`/`std` so downstream consumers (ACC denominator, drift detector) can weight or skip low-N bins.

```python
# For each training file y in sim52_full/train/ (years 12..111):
with h5py.File(f"sim52_full/train/MOST.{y:04d}.h5") as f:
    anchor_str = f.attrs['plasim_time_units']              # 'days since 0YYY-MM-DD HH:MM:SS'
    anchor = parse_anchor_proleptic(anchor_str)            # cftime.DatetimeProlepticGregorian
    t_days = f['time_plasim'][:]                           # (N,) float64, N ∈ {1455, 1459}
    state  = f['fields_state'][:]                          # (N, 52, H, W)
    diag   = f['fields_diagnostic'][:]                     # (N, 1, H, W)
    field  = np.concatenate([state, diag], axis=1)         # (N, 53, H, W)
    for s in range(len(t_days)):
        absolute_dt = anchor + timedelta(days=float(t_days[s]))
        bin_idx = (absolute_dt.month, absolute_dt.day, absolute_dt.hour // 6 * 6)
        # Welford update for clim_sum[bin_idx], clim_sumsq[bin_idx], n_contrib[bin_idx]
```

Output shape: `(366, 4, 53, 64, 128)` for each of `mean`, `std`, plus `n_contributors[366, 4]`. Stored as `baselines/climatology_proleptic.nc` (~2.4 GB each).

**Anchor caveat (P-4, see §3):** the +5 offset between file index and `plasim_time_units` anchor year (verified: `MOST.0011` → `0016-08-01`, `MOST.0121` → `0126-08-01`) means file `MOST.000Y.h5` spans calendar year (Y+5) Aug 1 → (Y+6) Aug 1. A file is leap iff Gregorian year (Y+6) is leap (i.e., (Y+6) % 4 == 0 AND ((Y+6) % 100 != 0 OR (Y+6) % 400 == 0)). Empirically:
- Y=14 → year 20: leap ✓
- Y=94 → year 100: NOT leap (centennial exception) — confirmed `MOST.0094` is 1455 samples.
- Y=2 (if it existed) → year 8: leap.
- Y=98 → year 104: leap ✓.

Within a leap file (1459 samples), the 4 extra samples are the Feb 29 6 h slots in the corresponding Gregorian leap year. Sample 0 is always Aug 1 00:00 of year (Y+5); the leap-day insertion is mid-file ~213 days in.

Until this check passes, climatology computation is gated behind a one-time anchor-trace script (`scripts/trace_calendar_anchors.py`) that produces a CSV `(file, sample_idx, time_plasim_value, computed_absolute_datetime)` for sanity inspection.

**Implementation note (one-pass, low memory):** Welford accumulation in fp32 over (366, 4, 53, 64, 128) = 1.55 G elements × 4 B = **6.2 GB** per accumulator. Two accumulators (sum, sumsq) and one count: **~13 GB**. Fits in CPU RAM on a Stampede3 `skx` node (191 GB). One-pass: walk training file list, read each file's full state/diagnostic into RAM (~1.5 GB per file), update accumulators in place. No GPU needed for climatology build.

### C.3 Normalization stats source (Codex fix 5 nuance).
The training run's `global_means.npy` / `global_stds.npy` (in the **run dir**, not the dataset stats dir) are what the model was actually trained against. Eval must de-z-score with these exact files; using `data/makani/sim52_full/stats/global_means.npy` is *probably* identical but not guaranteed (the run may have applied per-checkpoint overrides). Eval driver loads from the run dir — `runs/sfno_full/plasim_sim52_full/0/global_means.npy`.

---

## D. NWP scoring (Phase 1) — `scripts/score_nwp.py`

### D.1 Lat-weighted RMSE (manual implementation, in `src/sfno_eval/metrics.py`).

```python
def rmse_lat_weighted(pred, truth, lat_weights):
    # pred, truth: (..., lat, lon) in physical units
    # lat_weights: (lat,), sum to 1
    err2 = (pred - truth) ** 2                      # (..., lat, lon)
    err2_lon = err2.mean(dim=-1)                    # (..., lat)
    err2_weighted = (err2_lon * lat_weights).sum(dim=-1)  # (...,)
    return err2_weighted.sqrt()
```

### D.2 ACC (manual implementation).

```python
def acc(pred, truth, clim_mean, lat_weights):
    # all shapes (..., lat, lon), clim_mean broadcast-compatible
    pred_anom  = pred  - clim_mean
    truth_anom = truth - clim_mean
    w = lat_weights.unsqueeze(-1)                   # (lat, 1)
    num   = (pred_anom * truth_anom * w).sum(dim=(-2, -1))
    den_p = ((pred_anom  ** 2) * w).sum(dim=(-2, -1)).sqrt()
    den_t = ((truth_anom ** 2) * w).sum(dim=(-2, -1)).sqrt()
    return num / (den_p * den_t + 1e-12)
```

### D.3 Bias maps (mean error fields).
For each `(channel, lead_time)`: `bias[c, lat, lon] = mean over IC of (pred - truth)`. Saved as `.npy` per `(channel, lead_time)` pair, only for the 5 key channels: `tas, pr_6h, zg500, ua5, ta5` (the v10 contract; v9 inference outputs containing `zg5` are scored against the same set with `zg500` resolved to `zg5` by `scripts/_eval_utils.detect_z500_channel`, per docs/plasim_zg_plev_migration_plan.md §3.10).

### D.4 Lead times scored.
{6 h (k=1), 24 h (k=4), 72 h (k=12), 120 h (k=20), 240 h (k=40), 336 h (k=56)} from each NWP-mode rollout.

### D.5 Output: `scores/nwp_scorecard.csv`.
Columns: `model, channel, lead_hours, ic_year, ic_sample_idx, metric, value`. Tidy long format.
Aggregations (one extra view): `scores/nwp_scorecard_summary.csv` — averaged over IC dimension, columns `(model, channel, lead_hours, metric, mean, std, n_ics)`.

### D.6 **Sanity gate** before declaring scorecard valid.
- Emulator RMSE on `tas` at 6 h **<** persistence RMSE on `tas` at 6 h.
- Emulator ACC on `zg500` at 24 h **>** 0.6 (channel literally `zg500`, not the v9 sigma proxy). Resolution: now that the gate measures literal Z500, the 0.6 threshold can be re-evaluated against PlaSim's simpler atmosphere; left at 0.6 for the first run, revisit after the first emulator scoring on `…_zgplev`.
- Emulator RMSE finite (no NaN/Inf) for all (channel, lead_time) pairs.

If any fail: STOP, escalate to v3 plan with diagnostics. Do not proceed to Phase 2.

For v9-trained checkpoints whose inference outputs carry sigma `zg5` (no `zg500` channel), `scripts/_eval_utils.detect_z500_channel` falls back to `zg5` and the printed gate label reads "Z500 (sigma proxy, v9)" — the threshold meaning is different but the same numeric is retained for continuity (see docs/plasim_zg_plev_migration_plan.md §3.10).

---

## E. Climate scoring stub (Phase 1 — minimal)

Phase 1 produces a *lightweight* climate diagnostic from the 8 × 1-year rollouts:
- Time-mean spatial bias maps for the 5 key channels.
- Zonal-mean of time-mean bias (lat × channel heatmap).
- Variance ratio summary stats (no maps): `mean_over_lat_lon(var_t(emu) / var_t(truth))` per channel.
- Drift: 4-window means at days [1,30], [31,90], [91,180], [181,365].

**Postponed to Phase 2 (v3 plan):**
- KE / temperature variance spectra (SHT-based via `torch_harmonics.RealSHT`, not 2-D FFT).
- Pointwise variance-ratio maps.
- Channel-wise spectral slope diagnostics.

---

## F. Reporting (Phase 1) — `scripts/render_eval_report.py`

`results/sfno_eval/<run_tag>/report.md` includes:
- Header: full run-tag, P-1 status (which test files were used + their `packager_git_sha`), eval-code SHA, training-checkpoint provenance.
- Section 1 — NWP scorecard table (5 key channels at 6 lead times, all 3 baselines).
- Section 2 — ACC line plot per channel (PNG embedded).
- Section 3 — Bias maps (5 channels × 3 lead times = 15 PNGs).
- Section 4 — Climate stub (time-mean bias, zonal-mean heatmap, variance-ratio table, drift trajectories).
- Section 5 — Sanity gate result (PASS / FAIL with which checks failed).
- Section 6 — Phase-2 readiness statement (only emitted on PASS).

---

## G. Compute & SLURM

### G.1 Stampede3 partitions confirmed via `sinfo`.
`h100` partition with GPUs is available (24 nodes, 1031 GB memory).

### G.2 Three jobs (chained).

| Job | Wallclock | GPUs | What it does |
|---|---|---|---|
| `submit_eval_inference.slurm` | 6 h | 1 × H100 | §B for NWP (96 ICs × 56 steps = 5376 fwd) + climate (6 × 1454 + 2 × 1458 = 11 640 fwd). Total ~17 016 forward passes. |
| `submit_eval_score.slurm` | 2 h | 0 (CPU) | §C climatology build + §D NWP scoring + §E climate stub. |
| `submit_eval_report.slurm` | 30 min | 0 (CPU) | §F. |

Inference forward-pass count (matches the per-row arithmetic in the table above): `96 × 56 + 6 × 1454 + 2 × 1458 = 5376 + 8724 + 2916 = 17 016` forward passes. At training step time of ~61 ms (single-step eval should be similar or faster since no optimizer step), that's ~17 minutes compute. With I/O + dataloader overhead, expect 1.5 h. 6 h wallclock has ample margin for retries and assertion overhead.

Climatology build: 100 h5 files (76 × 1455 + 24 × 1459 samples) × Welford accumulation → I/O bound, ~30–60 min on a `skx` CPU node.

### G.3 `scripts/submit_eval.sh`.
```
JOB_INF=$(sbatch --parsable submit_eval_inference.slurm)
JOB_SCO=$(sbatch --parsable --dependency=afterok:$JOB_INF submit_eval_score.slurm)
JOB_REP=$(sbatch --parsable --dependency=afterok:$JOB_SCO submit_eval_report.slurm)
echo "Inference: $JOB_INF, Score: $JOB_SCO, Report: $JOB_REP"
```

### G.4 Run tag.
```
run_tag = ${YYYYMMDD}_eval-${EVAL_SHA7}_data-${DATA_SHA7}_train-${TRAIN_SHA7}_ckpt-${CKPT_BASENAME}
```
- `EVAL_SHA7`: `git rev-parse --short=7 HEAD` of AI-RES at eval job submit time.
- `DATA_SHA7`: `packager_git_sha[:7]` from h5 attrs (the **data packager** code SHA, e.g. `58413cb`). Read from the first test file; assert all 8 test files agree (they do, verified — all `58413cb1…`).
- `TRAIN_SHA7`: read from the training run's `out.log` line `git hash: <40-char-sha>` (logged by Makani at trainer init) and truncated to 7 chars. For the existing checkpoint at `runs/sfno_full/plasim_sim52_full/0/`, the full hash is `106d19d9cad20b20be5e1faf1319b4f0cdb7346b` → `TRAIN_SHA7 = 106d19d`. The eval driver greps `out.log` for this line; if the file is missing or the line is absent (e.g., older runs), it falls back to `unknown` and the report (§F) flags the provenance gap.
- `CKPT_BASENAME`: e.g. `best_ckpt_mp0`.

Example for the current checkpoint, eval run today:
```
20260429_eval-a1b2c3d_data-58413cb_train-106d19d_ckpt-best_ckpt_mp0
```

### G.5 Recommendation: capture training SHA more durably going forward.

The training-time SHA is currently logged once to `out.log` (`git hash: <sha>`). That works for this eval but is fragile — if `out.log` is rotated, truncated, or deleted, provenance is lost. Recommend patching `src/sfno_training/submit_full.slurm` (and `submit_short.slurm`, `submit_tiny.slurm`, `submit_smoke.slurm`) to additionally write `git rev-parse HEAD > $EXP_DIR/train_code_sha.txt` immediately after `EXP_DIR` is created. The eval driver prefers `train_code_sha.txt` if present, else parses `out.log`, else falls back to `unknown`. This is a one-line change per submit script and is recommended in a sibling PR (out of scope for this plan, but flagged here so it doesn't get lost).

---

## H. Tests (`tests/sfno_inference/` + `tests/sfno_eval/`)

| Test | What it asserts |
|---|---|
| `test_eval_contract.py` (**new, load-bearing**) | 6-step rollout from a synthetic 52-channel IC: forward-pre-hook captures shape `(B, 58, 64, 128)`; output shape `(B, 53, 64, 128)`; `pred[:, :52]` feeds back; `forcing[k]` from h5 != `forcing[k+1]` (forcing actually advanced); pr_6h channel does **not** appear in next-step input. |
| `test_rollout_driver.py` | Driver runs end-to-end on a 1-step rollout; output NetCDF has correct dims and physical-unit predictions (de-z-scored). |
| `test_checkpoint_loader.py` | `restore_from_checkpoint` on `best_ckpt_mp0.tar` produces a wrapper whose forward matches `validate_one_epoch`'s output bit-for-bit on a fixed batch. |
| `test_nc_writer.py` | NetCDF round-trip: dims, coords, attrs (including SHAs) preserved. |
| `test_metrics.py` | `rmse_lat_weighted` and `acc` match hand-computed reference at 1e-6 relative on a small synthetic example. |
| `test_climatology.py` | One-pass Welford produces same mean/std as numpy reference on 3 fake h5 files. Both `clim_index_mode` options exercised. |
| `test_eval_params_load.py` (**v2.4 + v2.5 + v2.6 contracts**) | `load_eval_params(run_dir, K=42)` returns `valid_autoreg_steps == 41`, `n_future == 41`, `N_in_channels == 58`, `N_out_channels == 53`, **`amp_enabled == True`**, **`amp_dtype == torch.bfloat16`** (since the run config sets `amp_mode='bf16'`). Mutating only `n_future` (without `valid_autoreg_steps`) and calling `_plasim_get_dataloader(..., mode='eval')` produces a dataset with `dataset.n_future == 3` (the cfg default), proving the round-4 finding. **(v2.5)** Also asserts that after `build_wrapper_from_checkpoint`, `wrapper.model.inp_chans == 58` and `wrapper.model.out_chans == 53` (catches future SFNO attribute renames). **(v2.6)** When invoked with `device='cuda:0'` on a GPU node, asserts `next(wrapper.parameters()).device == torch.device('cuda:0')` (regression guard on the device-placement blocker). On CPU-only smoke runs, the assertion uses `device='cpu'`. |
| `test_persistence_pr6h.py` (**new, v2.4**) | `score_nwp.persistence_rmse` returns NaN for `channel == "pr_6h"` and finite values for the 52 state channels, on a synthetic 6-step rollout. |

CPU smoke test (`test_smoke_eval_cpu.py`) runs the full chain (build_test_split → 1-step rollout → score → report) on a fake 64×128, 1-year, 1-IC dataset to validate orchestration without GPU.

---

## I. Risks & open issues

| Risk | Likelihood | Mitigation |
|---|---|---|
| ~~**P-1 unresolved**: years 121-128 not packaged~~ | ✅ resolved 2026-04-29 | Files at `sim52_astro_64x128/test/`. |
| ~~**Codex check 4 unresolved**: climatology index mode~~ | ✅ resolved 2026-04-29 | option (b) confirmed by per-file `plasim_time_units` Aug-1 anchor pattern. |
| `restore_from_checkpoint` mismatch — distributed init required for 1-rank | Medium | SLURM script sets `MASTER_ADDR=localhost MASTER_PORT=29500 WORLD_SIZE=1 RANK=0`; preflight asserts `DistributedManager.initialize()` succeeds. |
| **(v2.4)** Eval rollout silently capped at 3 steps if `valid_autoreg_steps` not bumped | High if regressed | §B.0 `load_eval_params` always sets both `valid_autoreg_steps = K-1` and `n_future = K-1`; §B.2 asserts `dataset.n_future == K-1` after `_plasim_get_dataloader`; `test_eval_params_load.py` is a regression guard. |
| **(v2.4)** Channel-count drift between training-time YAML and eval-time params | Low | §B.0 `load_eval_params` reads only the run-dir `config.json` (the frozen authoritative copy) and asserts `N_in_channels == 58, N_out_channels == 53`; `build_wrapper_from_checkpoint` re-asserts on the actual model object. |
| **(v2.4)** Persistence baseline misleadingly reported for diagnostic-only `pr_6h` | Low | §C.1 records NaN; §F report explicitly notes the gap; `test_persistence_pr6h.py` enforces. |
| **(v2.6 + v2.7)** Wrapper / data device mismatch (CPU vs GPU) | High if regressed | §B.0 `build_wrapper_from_checkpoint(eval_params, ckpt_path, device)` requires `device`, calls `.to(device)` before `Driver.restore_from_checkpoint`, and asserts the wrapper landed on the right device using a **safe type+index comparison** — `actual.type == expected.type and (expected.index is None or actual.index == expected.index)` — so that `torch.device("cuda")` (no index) is accepted as matching `cuda:0` (v2.7 fix). §B.2 resolves `device` with an explicit `cuda:<current_device>` index so logging and downstream comparisons stay consistent. `test_eval_params_load.py` exercises both branches. |
| Long-rollout numerical blow-up in single precision | Medium | 4-window drift detection (§E); diagnostic flag in report. |
| Forcing channel order drift between training and eval | Low | h5 attrs contain forcing channel list; preflight asserts identity with `forcing_channel_names` from training config. |
| ~~`torch_harmonics` quadrature API name uncertainty~~ | ✅ resolved (v2.2 round 2) — `torch_harmonics.quadrature.legendre_gauss_weights(64, -1.0, 1.0)`. Numpy fallback `np.polynomial.legendre.leggauss(64)` retained as belt-and-braces in `metrics.py`. |
| H5 file lock contention when 100 climatology files are read in one pass | Low | Welford pass is sequential; no parallel readers. |
| Climate stub variance ratio noisy with only ~1454–1458 timesteps per IC | Medium | Phase 2 deferral. |

---

## J. Open questions (numbered for Codex round 6)

1. ~~**(P-1)** Confirm the source NetCDFs for years 121–128 exist and were processed identically to years 12–100.~~ ✅ resolved (v2.1) — files at `sim52_astro_64x128/test/`, identical `packager_git_sha` and `rsdt_method`.
2. ~~**(A.1 sanity)** Does `src/plasim_makani_packager/` write a `split` attribute?~~ ✅ resolved (v2.1) — yes, values `'train'` / `'test'`.
3. ~~**(B.5)** Confirm `torch_harmonics` API for legendre-gauss quadrature weights at `nlat=64`.~~ ✅ resolved (v2.2 Codex round 2 confirmation) — `torch_harmonics.quadrature.legendre_gauss_weights(64, -1.0, 1.0)`.
4. ~~**(C.2)** Climatology indexing — sample-of-year vs time-of-year-proleptic.~~ ✅ resolved (v2.2) — `time_of_year_proleptic` is now the default (was `sample_of_year` in v2.1; reversed because variable file lengths break sample-of-year).
5. ~~**(C.3)** Should eval de-z-score from the run-dir `global_means.npy` or the dataset-stats `global_means.npy`?~~ ✅ resolved (v2.6 / Codex round 6) — verified **byte-identical** for this run. Eval still uses the run-dir copy as the canonical default (`runs/sfno_full/plasim_sim52_full/0/global_means.npy`) so future runs that *do* override stats remain self-consistent. `checkpoint_loader.py` adds a one-time SHA256 comparison against `data/makani/sim52_full/stats/global_means.npy` and warns (does not fail) if they diverge.
6. **(D.6)** Sanity gate threshold for ACC at 24 h — is 0.6 the right floor for `zg500`, or should it be higher given that PlaSim is a much simpler atmosphere than ERA5?
7. **(F)** Phase 1 report format — markdown only, or also a single-PDF artifact for sharing?
8. ~~**(G.4)** Is reading `packager_git_sha` from h5 attrs sufficient, or do we need the **training run's** code SHA?~~ ✅ resolved (v2.2) — three SHAs, `train_sha = 'unknown'` for the existing checkpoint.
9. ~~**(A.3)** What is the `timestamp` h5 dataset?~~ ✅ resolved (v2.2) — `int64` seconds since the same per-file Aug-1 anchor (NOT epoch). Use `time_plasim` (float64 days) for climatology.
10. ~~**(C.2 BLOCKING)** Trace the file-index ↔ anchor-year ↔ leap-year-pattern relationship.~~ ✅ resolved (v2.3) — offset is +5, leap rule is `(Y+6) % 4 == 0 AND ((Y+6) % 100 != 0 OR (Y+6) % 400 == 0)`, predicts 24 leap files in the training pool which matches filesystem. Centennial exception `MOST.0094` (year 100) confirmed non-leap. The `scripts/trace_calendar_anchors.py` script remains in scope as a 15-second sanity check before §C runs.

11. **(NEW, training-subset symlink stability)** `sim52_full/train/MOST.0101..0111.h5` symlink to `sim52_astro_64x128/valid/`. If the dataset is ever re-packaged with a different valid split, those symlinks could break or silently change content. Recommend reading `os.path.realpath()` for each training file at climatology-build time and storing the resolved paths in `meta/climatology_source_files.json` for provenance.

---

## K. Estimated implementation order (post-approval)

| Step | Days | Deliverable |
|---|---|---|
| 0. Resolve **P-1** + Codex checks 1, 4 | 1.0 | Test files packaged; climatology index mode chosen. |
| 1. `src/sfno_inference/` (driver + checkpoint loader + nc writer) + tests | 1.5 | 1-step rollout from real ckpt produces a valid NetCDF with all assertions on. |
| 2. `scripts/eval_inference.py` + SLURM submit | 0.5 | Full 96 + 8 IC inference run completes. |
| 3. `src/sfno_eval/climatology.py` + tests | 1.0 | `climatology_mean.nc` + `climatology_std.nc` produced. |
| 4. `src/sfno_eval/metrics.py` + tests | 0.5 | RMSE/ACC match numpy reference. |
| 5. `scripts/score_nwp.py` + sanity gate | 0.5 | `nwp_scorecard.csv` with PASS/FAIL gate. |
| 6. `scripts/score_climate.py` (Phase 1 stub) | 0.5 | Time-mean bias maps + drift CSV. |
| 7. `scripts/render_eval_report.py` | 0.5 | `report.md` + plots. |
| 8. End-to-end smoke run + writeup | 0.5 | Full chain runs from `submit_eval.sh`. |
| **Total Phase 1** | **~6 days** | |
| Phase 2 (separate v3 plan) | TBD | Spectra, full climate scorecard. |
