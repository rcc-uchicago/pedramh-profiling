---
name: plasim-makani-packager
description: Use when the user wants to convert per-sim-year PlaSim NetCDF (from src/plasim_postprocessor/) and boundary NetCDF (from src/emulator_adaptor/) into the three-dataset Makani-compatible HDF5 layout used to train an SFNO emulator. Covers packager.py, stats.py, metadata.py, validate.py under src/plasim_makani_packager/, the SLURM array dispatch, the 52-state / 1-diagnostic / 6-forcing channel contract, sic clip-provenance, sst land-fill, and per-split timestamp synthesis. Invoke for any task touching src/plasim_makani_packager/*, the packaged HDF5 layout, Makani stats files, the PlasimForcingDataset stub, or the trainer-patch contract owed to src/sfno_training/.
---

# PlaSim → Makani packager — usage and contract

`src/plasim_makani_packager/` packages per-sim-year PlaSim output into a minimally-patched Makani SFNO training dataset. The contract is locked by `docs/plasim_makani_packager_plan.md` (v9) and surfaces through three HDF5 datasets per file:

| Dataset | Shape | Role |
|---|---|---|
| `/fields_state` | `(T, 52, 64, 128)` float32 | Predicted + fed back to next rollout step. |
| `/fields_diagnostic` | `(T, 1, 64, 128)` float32 | Predicted + in loss, **never** fed back (pr_6h). |
| `/forcing` | `(T, 6, 64, 128)` float32 | Prescribed (lsm, sg, z0, sst, rsdt, sic); never predicted. |

Ancillary: `/timestamp` (int64, per-split monotonic seconds, step 21600), `/time_plasim` (float64, raw PlaSim days-since), `/channel_{state,diagnostic,forcing}`, `/lat`, `/lon`. Plan §Phase 1 lists every file-level attr.

## When to use this skill

- Running the packager on a new sim (`sbatch src/plasim_makani_packager/submit.slurm`).
- Computing / inspecting Makani-consumable stats files.
- Generating `metadata/data.json` + the rendered SFNO YAML config.
- Running Phase 4a structural validation or the Phase 4b Makani smoke test.
- Modifying the packager (re-run the unit + smoke tests after every change).
- Debugging sic clip-hard-fail, sst land-fill, or uniform-dT timestamp errors.

## Channel list (locked, plan v9)

```
/fields_state (52)
 0  pl
 1  tas
 2..11   ta1..ta10       (ta1 = TOA = lev[0], ta10 = surface = lev[9])
12..21   ua1..ua10
22..31   va1..va10
32..41   hus1..hus10
42..51   zg1..zg10

/fields_diagnostic (1)
 0  pr_6h

/forcing (6)
 0  lsm     -- static, from MOST
 1  sg      -- static, from MOST
 2  z0      -- varying, from MOST. Land-static, ocean-dynamic (Charnock)
 3  sst     -- varying, from boundary adaptor (NaN-over-land filled with 271.35 K)
 4  rsdt    -- varying, from boundary adaptor (rsdt_method=astronomical)
 5  sic     -- varying, from boundary adaptor (clipped to [0,1])
```

## Environment

The packager needs `xarray + h5py + netCDF4 + numpy`. The Phase 4b smoke additionally needs `torch + makani + physicsnemo`. The project venv at `~/AI-RES/.venv` provides all of these:

```bash
source ~/AI-RES/.venv/bin/activate
export PYTHONPATH=~/AI-RES/src:$PYTHONPATH
```

Makani's `data_helpers.get_timedelta_from_timestamp` rejects `numpy.int64` on Python 3.12. The stub loader in `tests/plasim_makani_packager/stub_forcing_loader.py` monkey-patches this; the production trainer PR (`src/sfno_training/`) must carry the same fix.

## CLI overview

### `packager.py` — per-(sim, year) writer

```
python3 -m plasim_makani_packager.packager
  --sims N [N ...]
  --postproc-root  /path/to/postproc         # contains sim{NN}/MOST.{YYYY}.nc
  --boundary-root  /path/to/boundary_astro   # contains sim{NN}/boundary.{YYYY}.nc
  --output-root    /path/to/out              # writes {split}/MOST.{YYYY}.h5 etc.
  [--train-years 3 100] [--valid-years 101 120] [--test-years 121 128]
  [--sst-land-fill-k 271.35]
  [--task-index N] [--count-tasks] [--overwrite] [--dry-run] [-v]
```

Conditional-required: `--postproc-root` / `--boundary-root` / `--output-root` are needed for any processing run, but **omitted with `--count-tasks`**.

Year 1 and 2 are warmup and always skipped. Year N in a split gets starting timestamp `sum(T_y * 21600 for y prior in same split)`, so concatenation across a split has uniform dT == 21600. This satisfies Makani's cross-file uniform-dT check (`data_loader_multifiles.py:216`).

### `stats.py` — training-split Welford

```
python3 -m plasim_makani_packager.stats
  --output-root /path/to/out
  [--train-years 3 100] [--epsilon 1e-6] [-v]
```

Writes `{output-root}/stats/`:
- `global_means.npy` / `global_stds.npy` — `(1, 53, 1, 1)` float32 (state 52 ‖ diagnostic 1).
- `time_means.npy` — `(1, 53, 64, 128)` float32.
- `forcing_global_means.npy` / `forcing_global_stds.npy` — `(1, 6, 1, 1)` float32.
- `forcing_time_means.npy` — `(1, 6, 64, 128)` float32.
- `sst_land_sentinel_notes.txt` — documents the 271.35 K sentinel's effect on sst stats.

Hard-fails on any channel whose global std (over `T × H × W`) is `< epsilon`. No per-channel exemption: the static forcing channels (lsm, sg) have non-zero spatial std, so they pass on real data. z0 also passes — it carries real spatial + temporal variation (Charnock over ocean).

### `metadata.py` — Makani metadata + YAML render

```
python3 -m plasim_makani_packager.metadata
  --output-root /path/to/out
  [--exp-dir /path/to/runs/sim52_astro_64x128]
  [--config-name plasim_sim52_astro_64x128]
  [--rsdt-method astronomical] [--sst-land-fill-k 271.35]
  [--train-years 3 100] [--valid-years 101 120] [--test-years 121 128]
  [-v]
```

Writes `{output-root}/metadata/data.json` (`coords.channel[:52] == channel_state`, `[52] == channel_diagnostic[0]`) and `{output-root}/config/{config-name}.yaml` (`templates/plasim_64x128.yaml` with `{{OUTPUT_ROOT}}` / `{{EXP_DIR}}` substituted).

### `validate.py` — Phase 4a structural + Phase 4b Makani smoke

```
python3 -m plasim_makani_packager.validate
  --output-root /path/to/out
  --mode {structural,makani_smoke,full}
  [--epsilon 1e-6] [-v]
```

`structural` runs entirely on the packaged HDF5 + .npy stats (no torch / makani). `makani_smoke` invokes `pytest tests/plasim_makani_packager/test_multifile_loader_smoke.py` — requires the full Makani stack. `full` runs both.

## End-to-end recipe (sim52, astronomical rsdt)

```bash
source ~/AI-RES/.venv/bin/activate
export PYTHONPATH=~/AI-RES/src:$PYTHONPATH
POSTPROC=$SCRATCH/AI-RES/data/postproc
BOUNDARY=$SCRATCH/AI-RES/data/boundary_astro
OUT=$SCRATCH/AI-RES/data/makani/sim52_astro_64x128

# 1. Phase 0: boundary adaptor (needs astronomical rsdt)
#    edit src/emulator_adaptor/submit.slurm then: sbatch --array=0-127 ...

# 2. Phase 1: packager SLURM array
N=$(python3 -m plasim_makani_packager.packager --sims 52 --count-tasks)   # 126
# edit src/plasim_makani_packager/submit.slurm (SIMS, POSTPROC_ROOT, BOUNDARY_ROOT, OUTPUT_ROOT) then:
sbatch --array=0-$((N-1)) src/plasim_makani_packager/submit.slurm

# 3. Phase 2-4a (local, fast)
python3 -m plasim_makani_packager.stats    --output-root "$OUT"
python3 -m plasim_makani_packager.metadata --output-root "$OUT" --exp-dir $SCRATCH/AI-RES/runs/sim52_astro_64x128
python3 -m plasim_makani_packager.validate --output-root "$OUT" --mode structural

# 4. Phase 4b Makani smoke (needs torch + makani + physicsnemo)
python3 -m plasim_makani_packager.validate --output-root "$OUT" --mode makani_smoke
```

`scripts/package_sim52_astro.sh` is a user-facing checklist wrapper for steps 3-4.

## Verification — required after any change to the packager

Full test suite (all 31 tests; 7 require makani + torch):

```bash
cd ~/AI-RES && source .venv/bin/activate
python -m pytest tests/plasim_makani_packager/
```

Tests cover:

| File | What it pins |
|---|---|
| `test_channel_flatten.py` | Channel order, sigma ta1=TOA/ta10=surface, forcing order. |
| `test_timestamp.py` | per-split continuous offsets; uniform dT within *and across* files. |
| `test_sic_provenance.py` | adaptor must clip and nothing else (hard-fail), raw quantify report. |
| `test_hdf5_writer.py` | Full process_one round-trip — dim scales, attrs, shapes. |
| `test_stats.py` | `(1, 53, ...) / (1, 6, ...)` stats shapes, epsilon hard-fail. |
| `test_metadata.py` | 53 target names, template substitution. |
| `test_parse_dataset_metadata.py` | Makani parser → `in_channels == out_channels == [0..52]`. |
| `test_multifile_loader_smoke.py` | Phase 4b Makani end-to-end; positive + negative multistep rollout. |

If the Makani-dependent tests fail because `ModuleNotFoundError: No module named 'makani'`, you're on the login node — rerun on a compute node (or in the venv) with the full Makani stack.

## Trainer-patch contract (owed to `src/sfno_training/` follow-up PR)

The packager writes a deliberately non-stock layout: `h5_path=fields_state` (52 channels) with 53 `coords.channel` names. Stock Makani dataloaders crash on channel-count mismatch. The follow-up PR must ship:

1. `PlasimForcingDataset` — returns `(inp_state, tar, inp_forcing, tar_forcing)` with `(n_history+1, C, H, W)` / `(n_future+1, C, H, W)`.
2. `PlasimPreprocessor` — subclass of `Preprocessor2D`; `append_history` auto-slices `pred[:, :52]` to strip diagnostic before feedback. Hard-asserts `x2.shape[1] in {52, 53}`.
3. `PlasimSingleStepWrapper` / `PlasimMultiStepWrapper` — subclasses that install `PlasimPreprocessor`.
4. `PlasimTrainer(Trainer)` — monkey-patches `model_registry.{SingleStepWrapper, MultiStepWrapper}` and `deterministic_trainer.get_dataloader` *before* `super().__init__`. Also overrides `_set_data_shapes` to set `N_in_channels = 58` (with a hard `assert params.n_history == 0`).
5. `train_plasim.py::main` mirroring `makani/train.py::main` for the runtime-injected params (`experiment_dir`, `checkpoint_path`, `resuming`, `amp_mode`, …).

**Hard gate (plan v9):** the follow-up PR must include an integration test that runs `PlasimTrainer(params, world_rank).train(n_iters=1)` to completion with four `isinstance` assertions (dataset / wrapper / preprocessor / `params.N_in_channels == 58`). No GPU-hours before that test passes in CI.

Inference is **out of scope** of the packager plan — stock `Inferencer._inference_indexlist` (`inferencer.py:554-589`) has no slot for our 6 forcing channels when `add_zenith=False`. A separate `src/sfno_inference/` plan is owed.

## What NOT to do

- **Do not** include `pr_6h` in `/fields_state`. Diagnostic leaks into feedback.
- **Do not** manually concat forcing in the trainer — the wrapper's `append_unpredicted_features` already does this.
- **Do not** assert exact sic equality against MOST. The adaptor clips; that's its job. Use `_validate_sic_clipping` which hard-fails on `adaptor.sic != np.clip(MOST.sic, 0, 1)` (tol 1e-6) and quantifies raw MOST-vs-adaptor as report only.
- **Do not** anchor timestamps to PlaSim's actual calendar days-since values (variable-length years break Makani's uniform-dhours check). Use the synthetic per-split continuous scheme.
- **Do not** skip `flatten_history` between `cache_unpredicted_features` and the wrapper call — stock SFNO forward expects 4D tensors.
- **Do not** add silent slicing to `PlasimPreprocessor.append_history`. Hard-assert `x2.shape[1] in {52, 53}`.
- **Do not** flip `n_history > 0` without revising `PlasimTrainer._set_data_shapes`. The v9 hard assert will trip.
- **Do not** reuse a `LossHandler` across different `n_future` values — `multistep_weight` is frozen at construction.
- **Do not** point stock Makani dataloaders at this metadata — the channel-count mismatch will crash normalization. Only `PlasimForcingDataset` can consume this layout.
- **Do not** drive inference through stock `Inferencer` on this dataset until the follow-up PR ships.

## Where to read more

- **Plan (v9)**: `docs/plasim_makani_packager_plan.md` — full rationale, revision history through 9 Codex rounds, trainer-patch contract, verification gate.
- **Source**: `src/plasim_makani_packager/` — 7 modules (packager, stats, validate, metadata, channels, __init__, templates/).
- **Tests**: `tests/plasim_makani_packager/` — 8 test modules + `stub_forcing_loader.py` + `conftest.py` (adds `src/` + test dir to `sys.path`).
- **SLURM template**: `src/plasim_makani_packager/submit.slurm`.
- **Orchestrator**: `scripts/package_sim52_astro.sh`.
- **Dependencies**: `src/plasim_postprocessor/` (produces MOST.*.nc), `src/emulator_adaptor/` (produces boundary.*.nc with `rsdt_method=astronomical`).

## Common failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| `cannot compute timestamp offset for {split}/{year}: prior file missing` | Running packager on non-contiguous years | Package prior years first, or use `--task-index` enumeration that naturally visits them in order |
| `sic NaN-mask parity broken` | Adaptor input (MOST.sic) and adaptor output (boundary.sic) disagree on NaN locations | Re-run adaptor — it should only clip, not introduce or drop NaN |
| `adaptor.sic differs from np.clip(MOST.sic, 0, 1)` | Adaptor altered sic beyond the clip | Check adaptor pipeline for unintended drift |
| `std < 1e-6 on N channel(s)` | Likely a channel is all-zero in every training file (sim never wrote it?) | Inspect the offending variable in the source MOST/boundary file |
| `The time difference between steps is not constant` (Makani) | Trying to load a mixed train+valid dir, or files that weren't packaged by this tool | Use separate `train_data_path` / `valid_data_path` (plan YAML does this); repackage if needed |
| `unsupported type for timedelta seconds component: numpy.int64` | Python 3.12 + stock Makani on our int64 timestamps | Monkey-patch `makani.utils.dataloaders.data_loader_multifiles.get_timedelta_from_timestamp` (see stub) |
| `'YParams' object has no attribute 'target'` | Preprocessor2D wants residual-learning flags | Set `params.target = "tendency"` and `params.normalize_residual = False` before building the preprocessor |
