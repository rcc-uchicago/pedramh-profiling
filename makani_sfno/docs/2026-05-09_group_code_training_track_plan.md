# Plan v5 — Group-code (PanguWeather v2.0 SFNO-v2) training track on Stampede3

Plan file: `/home1/11114/zhixingliu/.claude/plans/fuzzy-bubbling-lerdorf.md`.
Date: 2026-05-09.
Revision: v5 (pre-inference preflight rewritten for wrapper path; sanity gate handled informatively in Phase 1; channel-count typo fixed; explicit `has_diagnostic`, constant-boundary load + spatial z-score, and EMA-state preference in score wrapper; CLI snippets cleaned up).

## Context

Same as v4. Phase 1 = a v10-compatible shim that lets the group's PanguWeather v2.0 code (`nettype=sfno_plasim`) train on our existing v10 zgplev data; produce a checkpoint we can both score offline through our existing AI-RES eval pipeline AND call programmatically as a score function. Phase 2 (full repackage to group's native 13-pressure-level Pangu contract) deferred to a separate plan. Phase 1 is explicitly *not* a strict replication of group SFNO-5410 — eval is "looser comparison".

## Loader & runtime contract audit (verified by file:line)

(All v4 contracts, plus the additions from this revision's review.)

- **HDF5 key layout per training file** — per-level 2D datasets under `/input/`. Sigma vars `f"{var}_{sigma_value}"`; zg `f"zg_{int(level)}.0"`; surface/diag/varying-boundary bare names. Built by `_get_variable_list` (`/work2/.../utils/data_loader_multifiles.py:632-643`); reference: `scripts/infer_sfno5410_blocking_h100_packed.py:113-127`.
- **Filename:** unpadded year + 4-digit index. `12_0000.h5`, `121_0000.h5`. `get_out_path` (line 161).
- **Single `params.data_dir`**; constant-boundary file lookup: `{val_year_start}_0000.h5` (line 740). Use one flat output dir + `train_data_sets`/`validation_data_sets` overrides.
- **Synthetic group calendar.** Source PlaSim timestamps ignored; emit synthetic dates derived from chosen group year.
- **`train_data_sets`/`validation_data_sets` end dates are EXCLUSIVE** (`partition_date_range` line 240). Render emits `end_exclusive_dt = last_valid_init + 6h`.
- **Climatology open is unconditional** (`train.py:459-466`). Real `climatology.nc` required.
- **Train ckpt dir is `checkpoints/`** (`train.py:3613`); long_inference reads `training_checkpoints/` (`long_inference.py:1389-1395`). Bridge with symlink post-train.
- **`long_inference.py` requires:** `--debug` for 1-GPU OR launch via `torchrun --standalone --nproc_per_node=N` (line 1275-1289 reads `os.environ['WORLD_SIZE']`); `--init_datetime` in `"%Y-%m-%d_%H:%M:%S"` format (line 1313-1326); init NC time coord must contain that datetime; `nc_bc_offset=18` shifts boundary reads (line 1268, used in `_get_boundary_data` at line 950+). **Saves only at year boundaries** (line 834, 963) — short rollouts produce nothing. **Phase 1 sidesteps long_inference entirely; uses score-function wrapper instead.** Phase F documents the four corrections.
- **SFNO model constructor** (`networks/modulus_sfno/sfnonet.py:740-756`) reads `variable_list_in`, `variable_list_out`, `constant_boundary_variables`, `lat`, `lon`, `mask_fill`, `mean/std` for surface/upper-air/varying-boundary/diagnostic, and **`params_trainer.has_diagnostic`** (line 756). `in_chans = len(variable_list_in) + len(constant_boundary_variables)`; `out_chans = len(variable_list_out)`.
- **`has_diagnostic` is set in `train.py:3435`** AFTER YParams load: `params['has_diagnostic'] = len(params.diagnostic_variables) > 0`. The score wrapper must do the same before constructing SFNO.
- **Constant-boundary load + spatial z-score** (`data_loader_multifiles.py:740-749`): `_load_constant_boundary_data` reads `{val_year_start}_0000.h5` for `constant_boundary_variables` (`lsm`, `sg`), fills NaN by `mask_fill`, **then z-scores per variable** with mean/std computed from THAT file's spatial extent (not from global stats). The score wrapper must mirror this exactly.
- **EMA-state checkpoint preference** (`long_inference.py:370-380`): if `checkpoint['ema_state']` exists and is non-None, prefer it over `model_state`. The score wrapper does the same.
- **z0 is time-varying in v10**; placed in `varying_boundary_variables`.
- **v10 channel manifest** stored as datasets, not attrs.
- **v10 test split** = years 121-128.
- **Existing AI-RES sanity gate is fail-stop.** `scripts/score_nwp.py:365` returns nonzero on gate failure; `scripts/submit_eval_score.slurm:25` uses `set -euo pipefail` so the chain halts. Phase 1 smoke (2-day, K=8 leads) is unlikely to pass scientific gates. The group eval fork (`submit_eval_group.sh` + `submit_eval_score_group.slurm`) handles this informatively.

## Critical files to add (Phase 1)

(Same as v4 with three additions.)

- `src/sfno_training_group/__init__.py`
- `src/sfno_training_group/env_files/py311_pip_sfno_v2.stampede3.yaml`
- `src/sfno_training_group/env_activate.sh`
- `src/sfno_training_group/tools/_h5_keys.py`
- `src/sfno_training_group/tools/convert_v10_to_group_h5.py`
- `src/sfno_training_group/tools/build_group_stats_netcdf.py`
- `src/sfno_training_group/tools/build_group_climatology.py`
- `src/sfno_training_group/tools/build_init_nc_from_v10.py`
- `src/sfno_training_group/tools/render_yaml.py`
- `src/sfno_training_group/tools/preflight_checks.py` (15 checks total; pre_train 1-10, pre_inference 11-15)
- `src/sfno_training_group/tools/run_train.py`
- `src/sfno_training_group/score_function/__init__.py`
- `src/sfno_training_group/score_function/group_emulator.py`
- `src/sfno_training_group/score_function/_dataset_shim.py`
- `src/sfno_training_group/score_function/_boundary_loader.py`
- `src/sfno_training_group/score_function/run_smoke_rollout.py`
- `src/sfno_training_group/config/plasim_sim52_sigma10_sfno_smoke.yaml`
- `src/sfno_training_group/config/plasim_sim52_sigma10_sfno_full.yaml` (deferred; Phase F)
- `src/sfno_training_group/slurm/submit_train_smoke.slurm`
- `src/sfno_training_group/slurm/submit_inference_smoke.slurm`
- `src/sfno_training_group/slurm/submit_long_inference_full.slurm` (Phase F documentation)
- `src/sfno_training_group/README.md`
- `tests/sfno_training_group/test_v10_to_group_h5.py`
- `tests/sfno_training_group/test_group_stats_netcdf.py`
- `tests/sfno_training_group/test_init_nc_builder.py`
- `tests/sfno_training_group/test_score_function.py`
- `tests/sfno_training_group/test_preflight.py`
- `tests/sfno_training_group/test_render_yaml_end_exclusive.py`
- `scripts/convert_group_inference_to_aires_nc.py`
- `scripts/submit_eval_inference_group.slurm`
- `scripts/submit_eval_score_group.slurm` (NEW in v5: forked from `submit_eval_score.slurm` with informative-mode wrapping of `score_nwp.py`)
- `scripts/submit_eval_group.sh` (NEW in v5: orchestrates the chain with informative mode)
- `docs/2026-05-09_group_code_training_track_plan.md`

No edits to existing Makani path. The two new slurm files (score_group, submit_eval_group.sh) are forks; `submit_eval_score.slurm` and `submit_eval.sh` remain unchanged.

## Phase plan

| # | Phase | Depends on | Effort | Output |
|---|-------|------------|--------|--------|
| A | Build conda env | — | 1 h | env at `/work2/.../envs/group_pangu_sfno_v2/` |
| B.1 | Format converter (per-level keys, flat dir, synthetic calendar manifest) | — | ½ day | data + manifest |
| B.2 | NetCDF stats from converted h5 | B.1 | 3 h | mean/std .nc |
| B.3 | Climatology from converted h5 | B.1 | 2 h | climatology.nc |
| B.4 | Init-NC builder (year 121 IC) | B.1 | 3 h | init_smoke.nc |
| B.5 | render_yaml + preflight (15 checks) | B.1-B.4 | 2 h | green preflight |
| C | Sibling track scaffold + smoke YAML | A | ½ day | YAML + slurm |
| D | Smoke train + ckpt symlink bridge | A, B, C | ½ day wallclock | best_ckpt.tar + bridge |
| G | Score-function wrapper (with EMA pref + has_diagnostic + const-bdry z-score) | D | ½ day | `group_emulator.py` + tests |
| E.1 | Inference smoke via wrapper (year 121 IC, K=8) | G | 1 h | rollout NetCDF |
| E.2 | Output converter shim → 53-channel scorer NetCDF | E.1 | ½ day | converter |
| E.3 | `submit_eval_group.sh` chain end-to-end (informative gate) | E.2 | 2 h | report.md + figures |
| F (deferred) | Long_inference + production-scale | E green | external | full scorecard |

---

## Phase A — Conda env

(Unchanged from v4.) Strip `prefix:`, add `cf_xarray==0.10.0`, `mamba env create -p /work2/.../envs/group_pangu_sfno_v2`, verify torch + harmonics + group SFNO import + NCCL on h100.

---

## Phase B — Data shim

### B.1 — `convert_v10_to_group_h5.py`

Flat output dir; `<year>_<idx:04>.h5` per timestep; 59 keys per file under `/input/` (50 upper-air per-level + 2 surface + 1 diag + 6 forcing); float32 2D `(64, 128)`. Synthetic group calendar — emit year 12 → file `12_0000.h5`, never use source PlaSim timestamps for filename derivation.

Manifest fields per year: `n_timesteps`, `synthetic_start_dt`, `synthetic_last_idx_dt`, `last_train_init_dt`, `train_end_exclusive_dt`, `last_val_init_dt_for_max_lead_K`, `val_end_exclusive_dt_for_max_lead_K`, `z0_temporal_std_mean`. Both `last_*_init_dt` AND `*_end_exclusive_dt` recorded.

z0 audit during conversion. Default placement: `varying_boundary`.

`lfs setstripe -c 1 <DATA_DIR>` once before writing.

CLI:
```
python -m sfno_training_group.tools.convert_v10_to_group_h5 \
  --src $SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_astro_64x128_zgplev \
  --dst $SCRATCH/SFNO_Climate_Emulator/data/group_sfno/sim52_smoke \
  --years 11 12 13 121 \
  --max-forecast-lead-steps 60
# years: 11/12/13 cover val + train; 121 is the test IC source (real test split file).
```

### B.2 — `build_group_stats_netcdf.py`

Recompute stats over converted train h5; xarray Dataset with coords `Z` (10 plev), `Z_2` (10 sigma); 14 variables (5 upper-air + 2 surface + 1 diag + 6 forcing). Audits: zg500 ∈ [5400, 5700] m; `np.exp(stats.pl)` ∈ [80000, 120000] Pa. `attrs["source"] = "recomputed_from_converted_h5"`.

### B.3 — `build_group_climatology.py`

Real `climatology.nc` over converted train years. Schema deduced from `train.py:459-466` (cftime decoder, `time` coord, per-variable arrays).

### B.4 — `build_init_nc_from_v10.py`

Build group-format init NC from one v10 test h5 IC. Schema:
- `time` coord 1 entry, `units = "hours since 0121-01-01 00:00:00"`, `calendar = "proleptic_gregorian"`.
- `sigma` 1-D coord (10 floats, also a variable). `lev` 1-D coord (10 zg pressure values in Pa, also a variable).
- 3D vars `(time, sigma_or_lev, lat, lon)`: ta, ua, va, hus on sigma; zg on lev.
- 2D vars `(time, lat, lon)`: pl, tas.
- No boundary forcing in init NC (read from per-timestep h5 in `boundary_data_dir`).

CLI:
```
python -m sfno_training_group.tools.build_init_nc_from_v10 \
  --src-h5 $SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_astro_64x128_zgplev/test/MOST.0121.h5 \
  --ic-idx 0 \
  --synthetic-init-dt "0121-01-01 00:00:00" \
  --out $EXP_DIR/init_smoke.nc
```

### B.5 — render_yaml + preflight

`render_yaml.py`: substitutes `{{DATA_DIR}}`, `{{EXP_DIR}}`, and **end-exclusive** date strings (last_init + 6h) for `train_data_sets` / `validation_data_sets` from manifest. Test verifies emitted strings.

**Preflight checks (15 total, split into pre_train 1-10 and pre_inference 11-15):**

Pre-train (run before `train.py` invocation):
1. Env imports: torch, h5py, xarray, netCDF4, torch_harmonics, einops, timm, cf_xarray, group SFNO module.
2. YAML parse via group's `utils.YParams` against `--config SFNO`.
3. Calendar manifest exists; required fields per year present.
4. File presence: for each `[start_dt, end_exclusive_dt]` pair, expand via `np.arange((start - origin)/3600, (end_exclusive - origin)/3600, 6)` (matching `partition_date_range`); assert every `(year, idx)` has a corresponding `<year>_<idx:04>.h5`.
5. Key-set audit: open `<val_year_start>_0000.h5`, list `/input/` keys, assert 59-key set matches `_get_variable_list_in/out` derivation.
6. Constant-boundary file `{val_year_start}_0000.h5` exists with `input/lsm`, `input/sg`.
7. GetDataset construction dry run (validate=True, num_inferences=2); finite values.
8. Stats files open; coord lengths and variable list match.
9. Climatology opens with cftime decoder; `time` coord present.
10. z0 audit consistency (if YAML places z0 in constant_boundary, manifest std < 1e-3).

Pre-inference (run before `submit_inference_smoke.slurm` — **rewritten in v5 for the wrapper path**, NOT for long_inference):
11. **Wrapper module imports:** `from sfno_training_group.score_function.group_emulator import GroupEmulator`; assert SFNO_v2 import path resolves.
12. **Checkpoint load (cpu):** `torch.load(<run_dir>/checkpoints/best_ckpt.tar, map_location="cpu", weights_only=False)` succeeds; assert `model_state` key present (and report whether `ema_state` present).
13. **Init NC time coord match:** open `init_nc` with xarray; assert `init_dt` (e.g. `0121-01-01 00:00:00`) is in the time coord (within 1s).
14. **Constant-boundary load + spatial z-score dry run:** load lsm + sg from `<val_year_start>_0000.h5`, apply `mask_fill`, compute spatial mean/std per variable, assert finite + reasonable scale (lsm ∈ [0, 1] before normalization; std > 0 per variable).
15. **Boundary file presence for rollout window:** for `init_year=121, init_idx=0, steps=8`, assert files `121_0000.h5` ... `121_0008.h5` exist (covers IC + 8 boundary reads). Also accounts for `year-rollover` if init_dt is late in year.

Phase split selectable via `--phase {pre_train, pre_inference}` flag.

---

## Phase C — Sibling track scaffold

### Smoke YAML — `config/plasim_sim52_sigma10_sfno_smoke.yaml`

Same as v4. Key knobs:

- `varying_boundary_variables: ['z0', 'sst', 'rsdt', 'sic']` (4 vars).
- `constant_boundary_variables: ['lsm', 'sg']`.
- `train_data_sets`/`validation_data_sets` with placeholders for end-exclusive strings rendered from manifest.
- Defensive long_inference knobs (used only if Phase F invokes long_inference; harmless in smoke):
  ```yaml
  ensemble_inference_hours: 360
  save_forecasts: true
  prediction_duration_days: 2
  long_rollout_years: 1
  forecast_lead_times: [1, 4, 8]
  ```
- Smoke knobs: batch_size 4, max_epochs 1, num_data_workers 4, num_inferences 2, wandb off, fp16, sigma diagnostics auto-disabled.

### Smoke train slurm

(Same as v4.) `set -euo pipefail`; render YAML; pre-train preflight (1-10); torchrun on 4 H100; post-train `ln -sfn checkpoints training_checkpoints` bridge.

### Inference smoke slurm — `submit_inference_smoke.slurm`

(Updated in v5: invokes pre-inference preflight 11-15, not the long_inference-specific dry-run from v4.)

```bash
#!/bin/bash
#SBATCH -J sfno_group_inf_smoke
#SBATCH -p h100 -N 1 --gpus-per-node=1 --cpus-per-task=16 -t 00:30:00
set -euo pipefail
REPO_ROOT="$HOME/projects/SFNO_Climate_Emulator"
source "$REPO_ROOT/src/sfno_training_group/env_activate.sh"

EXP_DIR="${EXP_DIR:?}" RUN_NUM="${RUN_NUM:?}"
RUN_DIR="$EXP_DIR/SFNO/$RUN_NUM"
INIT_NC="${INIT_NC:?}" DATA_DIR="${DATA_DIR:?}"
INIT_DT="${INIT_DT:-0121-01-01 00:00:00}"
STEPS="${STEPS:-8}"

# Pre-inference preflight (checks 11-15)
python -m sfno_training_group.tools.preflight_checks \
  --yaml "$RUN_DIR/rendered.yaml" --config SFNO --phase pre_inference \
  --run-dir "$RUN_DIR" --init-nc "$INIT_NC" --init-dt "$INIT_DT" \
  --data-dir "$DATA_DIR" --steps "$STEPS"

python -m sfno_training_group.score_function.run_smoke_rollout \
  --ckpt "$RUN_DIR/checkpoints/best_ckpt.tar" \
  --yaml "$RUN_DIR/rendered.yaml" --config SFNO \
  --init-nc "$INIT_NC" \
  --boundary-data-dir "$DATA_DIR" \
  --init-dt "$INIT_DT" \
  --steps "$STEPS" \
  --out "$RUN_DIR/inference_smoke/score_fn_rollout.nc"
```

### Long inference full slurm — `submit_long_inference_full.slurm` (Phase F doc only)

Captures the four long_inference corrections:
- WORLD_SIZE: pass `--debug` for 1-GPU OR launch via `torchrun --standalone --nproc_per_node=N`.
- `--init_datetime` required; format `"%Y-%m-%d_%H:%M:%S"` (underscore between date and time).
- Init NC time coord must contain the requested datetime.
- `nc_bc_offset = 18` set internally — preflight must verify boundary indices resolve.
- Output saves only at year boundaries — pick `init_dt` so rollout reaches next Jan 1 within the target window.

Example invocation (Phase F):
```bash
torchrun --standalone --nproc_per_node=1 "$GROUP_PANGU_ROOT/long_inference.py" \
  --config SFNO --yaml_config "$RUN_DIR/rendered.yaml" --run_num "$RUN_NUM" \
  --init_nc_filepaths "$INIT_NC" \
  --init_datetime "0121-01-01_00:00:00" \
  --output_dir "$RUN_DIR/inference_full" \
  --save_basename full_rollout
```

---

## Phase D — Smoke train

(Same as v4.) Convert years 11, 12, 13, 121. Build stats over years 12-13. Build climatology over years 12-13. Render YAML. Preflight 1-10. Submit. Post-train, slurm symlinks `checkpoints/` → `training_checkpoints/`.

Acceptance:
- preflight 1-10 green.
- ≥ 2 optimizer steps with finite loss.
- inverse-transformed sanity values plausible.
- `<run_dir>/checkpoints/best_ckpt.tar` present; symlink resolves.

---

## Phase G — Score-function wrapper (Phase 1 prerequisite)

### G.1 — `_dataset_shim.py`

Dataset-like adapter for SFNO `__init__` (`networks/modulus_sfno/sfnonet.py:740-756`). Provides:

```python
class _DatasetShim:
    def __init__(self, params, mean_ds, std_ds, lat, lon, const_bdry_data):
        # variable_list_in: 50 upper-air per-level + 2 surface + 4 varying-boundary = 56.
        # variable_list_out: 50 upper-air per-level + 2 surface + 1 diagnostic = 53.
        # SFNO line 749: in_chans = len(variable_list_in) + len(constant_boundary_variables) = 56 + 2 = 58.
        # SFNO line 749: out_chans = len(variable_list_out) = 53.
        self.variable_list_in = self._build_var_list_in(params)   # 56 entries
        self.variable_list_out = self._build_var_list_out(params) # 53 entries
        self.constant_boundary_variables = list(params.constant_boundary_variables)  # ['lsm', 'sg']
        self.varying_boundary_variables = list(params.varying_boundary_variables)    # ['z0', 'sst', 'rsdt', 'sic']
        self.upper_air_variables = list(params.upper_air_variables)
        self.surface_variables = list(params.surface_variables)
        self.diagnostic_variables = list(params.diagnostic_variables)
        self.lat = lat; self.lon = lon
        self.mask_fill = params.mask_fill
        # Stats tensors (built from mean_ds/std_ds in load_mean_std style):
        self.surface_mean, self.surface_std = ...
        self.upper_air_mean, self.upper_air_std = ...
        self.varying_boundary_mean, self.varying_boundary_std = ...
        self.diagnostic_mean, self.diagnostic_std = ...
        # Constant boundary already loaded + spatially z-scored:
        self.constant_boundary_data = const_bdry_data
        self.land_mask = ...
        # _ff_std and _delta_std aliases group code looks up:
        self.surface_ff_std = self.surface_std; self.upper_air_ff_std = self.upper_air_std
```

### G.2 — `_boundary_loader.py`

Reads varying-boundary trajectory from converted h5 per `(year, idx)`. Handles year rollover when `init_idx + steps > n_timesteps`.

### G.3 — `group_emulator.py`

```python
class GroupEmulator:
    def __init__(self, ckpt_path: str, yaml_path: str, config_name: str = "SFNO",
                 device: str = "cuda:0", *, prefer_ema: bool = True):
        # 1. Parse YAML via group's utils.YParams.
        params = YParams(yaml_path, config_name)
        # 2. Set has_diagnostic exactly as train.py:3435 does.
        params['has_diagnostic'] = (
            len(params.diagnostic_variables) > 0
            if hasattr(params, 'diagnostic_variables') else False
        )
        # 3. Load mean.nc / std.nc with the load_mean_std-style logic.
        mean_ds, std_ds = self._load_stats(params)
        # 4. Load constant-boundary data + spatial z-score (mirrors
        #    data_loader_multifiles.py:740-749 exactly):
        const_bdry_data, land_mask = self._load_constant_boundary_data(params)
        # 5. Build _DatasetShim.
        shim = _DatasetShim(params, mean_ds, std_ds,
                            torch.tensor(params.lat, dtype=torch.float32),
                            torch.tensor(params.lon, dtype=torch.float32),
                            const_bdry_data)
        # 6. Construct SphericalFourierNeuralOperatorNet_v2(params=params, dataset=shim).
        self.model = SphericalFourierNeuralOperatorNet_v2(params, shim).to(device).eval()
        # 7. Load checkpoint, prefer ema_state (long_inference.py:370-380 pattern).
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        if prefer_ema and ckpt.get("ema_state") is not None:
            state_dict = ckpt["ema_state"]
            self._loaded_state = "ema_state"
        else:
            state_dict = ckpt["model_state"]
            self._loaded_state = "model_state"
        # Strip 'module.' prefix if checkpoint was DDP-wrapped.
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
        self.model.load_state_dict(state_dict)
        self.shim = shim; self.params = params; self.device = device

    def step(self, surface, upper_air, varying_boundary):
        """One forward step.
        surface: (2, 64, 128) physical-space [pl, tas].
        upper_air: (5, 10, 64, 128) physical-space [ta, ua, va, hus, zg].
        varying_boundary: (4, 64, 128) physical-space [z0, sst, rsdt, sic].
        Returns: (out_surface (2, 64, 128), out_upper_air (5, 10, 64, 128),
                  out_diagnostic (1, 64, 128))   physical-space.
        """
        # Apply forward transforms (z-score), call model, apply inverse transforms.

    def rollout(self, init_surface, init_upper_air, boundary_trajectory, steps):
        """Auto-regressive rollout for `steps` 6h timesteps."""

    def save_rollout_netcdf(self, surface_traj, upper_air_traj, diagnostic_traj,
                            init_dt, out_path):
        """Save as NetCDF schema-compatible with the eval converter."""

    @property
    def loaded_state_kind(self) -> str:
        """'ema_state' or 'model_state'. Recorded in saved NetCDF attrs for provenance."""
        return self._loaded_state
```

**Constant-boundary z-score mirror** (`_load_constant_boundary_data` lines 740-749):

```python
def _load_constant_boundary_data(self, params):
    # Read {val_year_start}_0000.h5 for params.constant_boundary_variables.
    file_path = os.path.join(params.data_dir, f"{params.val_year_start}_0000.h5")
    with h5py.File(file_path, "r") as f:
        data = np.stack([f["input"][var][:] for var in params.constant_boundary_variables], axis=0)
    data = torch.from_numpy(data).to(torch.float32)
    # Mask-fill NaNs.
    for i, var in enumerate(params.constant_boundary_variables):
        nans = torch.isnan(data[i])
        if torch.any(nans):
            data[i] = data[i].masked_fill(nans, params.mask_fill[var])
    # Land mask (lsm).
    lsm_idx = params.constant_boundary_variables.index("lsm")
    land_mask = data[lsm_idx].clone()
    # Spatial z-score per variable (NOT global stats from data_train_mean.nc).
    spatial_mean = data.mean(dim=(1, 2), keepdim=True)
    spatial_std = data.std(dim=(1, 2), keepdim=True)
    data = (data - spatial_mean) / spatial_std
    return data, land_mask
```

This is the exact contract the loader expects (line 745-749). It is NOT the same as the global `data_train_mean.nc` stats — constant-boundary normalization is per-snapshot spatial normalization. Mirror precisely.

### G.4 — `run_smoke_rollout.py`

Entry script wrapping `GroupEmulator`. Loads IC from init NC, loads boundary trajectory from converted h5 dir, runs `rollout(steps=K)`, calls `save_rollout_netcdf`.

### G.5 — Tests

`test_score_function.py`:
- Mock checkpoint with `ema_state` (random) AND `model_state` (different random). Default load → `ema_state` (verify by `loaded_state_kind`).
- Mock checkpoint with only `model_state` → load that.
- `_DatasetShim.variable_list_in` length = 56; `variable_list_out` length = 53.
- SFNO `in_chans` reported by model attribute = 58; `out_chans` = 53.
- One step on synthetic IC → output shapes `(2, 64, 128)`, `(5, 10, 64, 128)`, `(1, 64, 128)`; finite.
- Rollout(steps=2) → trajectories with leading dim = 3 (init + 2) for state, 2 for diagnostic.
- Constant-boundary z-score check: lsm normalized to mean ≈ 0, std ≈ 1 over spatial dims.

### G.6 — Acceptance

- Tests pass with mock data.
- `run_smoke_rollout.py` against the Phase D checkpoint produces a finite NetCDF; output attrs include `loaded_state_kind`, `git_sha`, `init_dt`, `steps`.

---

## Phase E — Inference smoke + eval shim

### E.1 — Inference smoke (wrapper-based)

```
INIT_NC=$EXP_DIR/init_smoke.nc \
DATA_DIR=$SCRATCH/SFNO_Climate_Emulator/data/group_sfno/sim52_smoke \
EXP_DIR=$SCRATCH/SFNO_Climate_Emulator/runs/sfno_group_sigma10_smoke \
RUN_NUM=<run num> \
INIT_DT="0121-01-01 00:00:00" STEPS=8 \
  sbatch src/sfno_training_group/slurm/submit_inference_smoke.slurm
```

Output: `$RUN_DIR/inference_smoke/score_fn_rollout.nc`. Acceptance: NetCDF written, finite, plausible magnitudes.

### E.2 — Output converter

`scripts/convert_group_inference_to_aires_nc.py`. Maps wrapper output (per-variable DataArrays + sigma/lev coords) to v10's 53-channel layout (state[0]=pl through state[51]=zg1000, diagnostic[52]=pr_6h). Truth + init_state read from existing v10 test holdout `MOST.0121.h5`. Output schema-compatible with `src/sfno_inference/nc_writer.py`.

### E.3 — `submit_eval_group.sh` chain (informative gate)

**Key change in v5:** the existing scoring slurm halts the chain on sanity-gate failure. For Phase 1 smoke (2-day rollout) the gate is unlikely to pass. Fork:

`scripts/submit_eval_score_group.slurm` — copy of `submit_eval_score.slurm` with:
1. `set -uo pipefail` (drop `-e`) at top.
2. Wrap the `score_nwp.py` invocation in:
   ```bash
   set +e
   python "$REPO_ROOT/scripts/score_nwp.py" --out-root "$OUT_ROOT" ...
   GATE_RC=$?
   set -e
   echo "score_nwp gate exit code: $GATE_RC" | tee "$OUT_ROOT/scores/gate_status.txt"
   if [[ "$GATE_RC" -ne 0 ]]; then
     echo "INFO: sanity gate failed (Phase 1 smoke; informative-only). Continuing chain."
   fi
   exit 0
   ```
3. Always exit 0 from this fork; preserve the gate result in `gate_status.txt` for the report job to read.

`scripts/submit_eval_group.sh` — fork of `submit_eval.sh` that calls `submit_eval_score_group.slurm` instead of `submit_eval_score.slurm`. Steps 3 (report) and 4 (figures) unchanged but read `gate_status.txt` and surface "Phase 1 v10-shim — informative gate result: <PASS/FAIL>" in `report.md`.

Acceptance:
- 53-channel NetCDFs produced.
- `score_nwp.py` runs to completion (success or informative failure recorded).
- `report.md` rendered with explicit gate-status line.
- `figures/` contains plots.
- `gate_status.txt` records the exit code.

---

## Verification (full preflight + acceptance)

1. **Env:** preflight 1 green.
2. **Convert:** flat dir contains `12_0000.h5`, `121_0000.h5` (unpadded year). 59 keys per file. Plausible pl.
3. **Calendar manifest:** synthetic dates, both `last_*_init_dt` and `*_end_exclusive_dt`, z0 audit recorded, year 121 present.
4. **Stats / climatology / init NC:** opens succeed; schema audits pass.
5. **render_yaml:** emits end-exclusive strings; test passes.
6. **Preflight 1-10 green** before train submit.
7. **Train:** ≥ 2 optimizer steps; `checkpoints/best_ckpt.tar` written; symlink bridge created.
8. **Score wrapper unit tests pass** (`test_score_function.py`): channel counts (56/53/58), step shapes, rollout shapes, EMA preference, constant-boundary z-score.
9. **Preflight 11-15 green** before inference smoke.
10. **Inference smoke:** `score_fn_rollout.nc` written, finite, plausible.
11. **Converter:** 53-channel NetCDFs produced.
12. **Score / report / figures:** run to completion; `gate_status.txt` records sanity-gate result; `report.md` surfaces it; figures render.

---

## Risks / unknowns

**R-1 (medium): SFNO constructor drift.** Pin upstream commit; test catches.
**R-2 (medium): bf16 vs fp16.** Phase F deferred wrapper.
**R-3 (medium): Calendar drift.** Synthetic-only emit; preflight 4 verifies.
**R-4 (medium): Boundary trajectory loading.** Preflight 15 verifies file presence for `init_idx + steps` window.
**R-5 (low): cf_xarray pin.** 0.10.0 in env yaml.
**R-6 (low): nc_bc_offset.** Phase F only.
**R-7 (low): Year-boundary save.** Phase F only.
**R-8 (low): z0 placement.** Default `varying_boundary`.
**R-9 (low): Inode pressure.** `lfs setstripe -c 1`.
**R-10 (medium): Looser-comparison framing.** Sanity gate informative-only via fork.
**R-11 (low): EMA-vs-model-state choice.** Default prefer EMA; smoke checkpoint may not have EMA — wrapper falls back to `model_state` and records `loaded_state_kind`.
**R-12 (low): Constant-boundary spatial z-score** is **per-snapshot**, not from global stats — different from how upper-air/surface/varying-boundary are normalized. Wrapper mirrors group code exactly; documented in code + this plan.

---

## Open questions (deferred)

1. Phase 2 (13-plev repackage).
2. bf16 wrapper for Phase F.
3. long_inference full-year rollout (Phase F) with the four corrections.
4. Pinned upstream commit hash captured at first submit.
5. Ensemble inference (Phase F+).

---

## Recommended next steps after plan approval

1. **A** — env build (~1 h).
2. **B.1** — converter + synthetic calendar manifest (½ day).
3. **B.2** — stats (3 h).
4. **B.3** — climatology (2 h).
5. **B.4** — init NC builder (3 h).
6. **B.5** — render_yaml + preflight 15 checks (2 h).
7. **C** — sibling scaffold + smoke YAML (½ day).
8. **D** — smoke train + ckpt bridge (½ day wallclock).
9. **G** — score wrapper + tests (½ day).
10. **E.1** — inference smoke via wrapper (1 h).
11. **E.2** — output converter (½ day).
12. **E.3** — eval chain end-to-end with informative gate (2 h).
13. **F (deferred)** — long_inference + production-scale.

---

## Memory hooks (after Phase D + G green)

- Project: "Group-code training track Phase 1 = v10-compatible shim. Per-level h5 keys (59/file), flat dir, **synthetic** group calendar with **end-exclusive** date strings, real climatology, init-NC builder, ckpt symlink bridge `training_checkpoints/`→`checkpoints/`. Inference smoke uses score-function wrapper, NOT long_inference. Eval chain forks `submit_eval_score.slurm` to handle the sanity gate informatively. Sibling track `src/sfno_training_group/`. Env at `/work2/.../envs/group_pangu_sfno_v2/`."
- Reference: "Per-level h5 key builder canonical reference: `scripts/infer_sfno5410_blocking_h100_packed.py:113-127`."
- Reference: "Group SFNO score-function wrapper at `src/sfno_training_group/score_function/group_emulator.py`. Loads ckpt with EMA preference; sets `params.has_diagnostic` before SFNO construction; loads constant boundary from `{val_year_start}_0000.h5` and applies **spatial** z-score per variable (NOT global stats); exposes `step()` and `rollout()`. Channel counts: `variable_list_in=56`, `variable_list_out=53`, SFNO `in_chans=58` (after constant_boundary concat), `out_chans=53`."
- Reference: "Existing AI-RES sanity gate (`scripts/score_nwp.py:365`) is fail-stop with `set -euo pipefail` in `submit_eval_score.slurm`. The group eval fork (`submit_eval_score_group.slurm`) wraps the score call in `set +e`/`set -e` and writes `gate_status.txt`."
- Reference: "Group SFNO long_inference (Phase F): requires `--debug` (or `torchrun`), `--init_datetime` in `\"%Y-%m-%d_%H:%M:%S\"`, init NC time coord must contain that datetime, `nc_bc_offset=18`, output saved only at year boundaries."
- Reference: "v10 test split = years 121-128. Year 14 is train. Use 121 for inference smoke."
- Project: "Phase 2 (13-plev repackage) deferred."
