# sfno_training_group — group-code (PanguWeather v2.0 SFNO-v2) training track

A v10-compatible shim that lets the group's PanguWeather v2.0 code (`nettype=sfno_plasim`, modulus_sfno backbone) train on our existing v10 zgplev data with no postproc rerun, plus a programmatic Python wrapper for AI-RES score-function calls.

Phase 1 = v10-compatible shim (this directory). Phase 2 (full repackage to group's native 13-pressure-level Pangu contract) is intentionally deferred — see `docs/2026-05-09_group_code_training_track_plan.md` for the rationale.

## Layout

```
src/sfno_training_group/
├── env_files/py311_pip_sfno_v2.stampede3.yaml   # group env adapted for Stampede3 (+ cf_xarray, einops)
├── env_activate.sh                              # module load + conda activate + PYTHONPATH
├── tools/
│   ├── _h5_keys.py                              # canonical per-(var, level) key builder
│   ├── convert_v10_to_group_h5.py               # B.1 — flat dir, per-timestep, synthetic calendar
│   ├── build_group_stats_netcdf.py              # B.2 — recomputed mean/std from converted h5
│   ├── build_group_climatology.py               # B.3 — daily climatology.nc
│   ├── build_init_nc_from_v10.py                # B.4 — IC NetCDF for inference smoke
│   ├── render_yaml.py                           # B.5 — substitutes end-exclusive date strings
│   └── preflight_checks.py                      # B.5 — 15 checks (10 pre_train, 5 pre_inference)
├── score_function/
│   ├── _dataset_shim.py                         # G.1 — minimal dataset-like for SFNO_v2 init
│   ├── _boundary_loader.py                      # G.2 — varying-boundary trajectory reader
│   ├── group_emulator.py                        # G.3 — load ckpt + step()/rollout()
│   └── run_smoke_rollout.py                     # G.4 — Phase E.1 entry script
├── config/
│   └── plasim_sim52_sigma10_sfno_smoke.yaml     # smoke YAML, sed-rendered placeholders
├── slurm/
│   ├── submit_train_smoke.slurm                 # Phase D, single-H100, post-train ckpt symlink
│   └── submit_inference_smoke.slurm             # Phase E.1 standalone (used by submit_eval_inference_group)
└── README.md
```

Phase E.2/E.3 eval pieces live under `scripts/` and `tests/sfno_training_group/`.

## End-to-end smoke recipe

Assumes Phase A env already built at `/work2/11114/zhixingliu/stampede3/SFNO_Climate_Emulator/envs/group_pangu_sfno_v2`.

```bash
# Phase B: convert + stats + climatology + init NC (one-time per data slice)
PYTHONPATH=$PWD/src python -m sfno_training_group.tools.convert_v10_to_group_h5 \
  --src $SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_astro_64x128_zgplev \
  --dst $SCRATCH/SFNO_Climate_Emulator/data/group_sfno/sim52_smoke \
  --years 11 12 13 121 --max-forecast-lead-steps 60

PYTHONPATH=$PWD/src python -m sfno_training_group.tools.build_group_stats_netcdf \
  --data-dir $SCRATCH/SFNO_Climate_Emulator/data/group_sfno/sim52_smoke --train-years 12 13

PYTHONPATH=$PWD/src python -m sfno_training_group.tools.build_group_climatology \
  --data-dir $SCRATCH/SFNO_Climate_Emulator/data/group_sfno/sim52_smoke --train-years 12 13

PYTHONPATH=$PWD/src python -m sfno_training_group.tools.build_init_nc_from_v10 \
  --src-h5 $SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_astro_64x128_zgplev/test/MOST.0121.h5 \
  --ic-idx 0 --synthetic-init-dt "0121-01-01 00:00:00" \
  --out $SCRATCH/SFNO_Climate_Emulator/runs/sfno_group_sigma10_smoke/init_smoke.nc

# Phase D: smoke train (1 epoch, 2 train years, 1 H100, ~5-10 min wallclock)
sbatch src/sfno_training_group/slurm/submit_train_smoke.slurm

# Phase E: inference + converter + score (informative gate)
RUN_NUM=smoke_<YYYYMMDD>_<HHMM> \
  scripts/submit_eval_group.sh
```

## Channel contract (Phase 1 v10 shim)

| Group SFNO   | v10 zgplev mapping                    | count   |
|--------------|---------------------------------------|---------|
| upper_air    | ta, ua, va, hus on 10 sigma levels   | 4×10=40 |
|              | zg on 10 plev (200..1000 hPa)        | 1×10=10 |
| surface      | pl, tas                              | 2       |
| diagnostic   | pr_6h                                | 1       |
| varying bdry | z0, sst, rsdt, sic                   | 4       |
| const bdry   | lsm, sg                              | 2       |
| **shim totals** | `variable_list_in`              | **56**  |
|              | `variable_list_out`                  | **53**  |
| **SFNO**     | `in_chans = 56 + 2`                  | **58**  |
|              | `out_chans`                          | **53**  |

z0 sits in `varying_boundary_variables` (NOT `constant_boundary_variables`) to preserve v10's time-varying-z0 semantics. The convert tool's `_v10_calendar_manifest.json` records `z0_temporal_std_mean` per year for audit.

## Loader-contract pitfalls captured

This track was iteratively reviewed; the following gotchas are encoded into the code/configs so they don't recur:

- **Synthetic calendar** — group's loader derives filenames from chosen group year, not source PlaSim timestamps. Manifest emits synthetic dates and never propagates source-decoded ones.
- **End-exclusive date ranges** — `train_data_sets`/`validation_data_sets` end strings are `last_init + 6h`. `render_yaml.py` enforces this and tests verify.
- **Per-level keys** — `input/<var>_<level>` not `input/<var>` stacked. Reference: `scripts/infer_sfno5410_blocking_h100_packed.py:113-127`.
- **Filename: unpadded year** — `12_0000.h5`, not `0012_0000.h5`.
- **Climatology open is unconditional** — even with `use_sigma_levels=True` (which auto-disables ACC). A real `climatology.nc` is required for any train submit.
- **Train ckpt dir is `checkpoints/`; long_inference reads `training_checkpoints/`** — Phase D slurm symlinks one to the other post-train.
- **`long_inference.py` (Phase F only)** — needs `--debug` for 1-GPU OR `torchrun`; `--init_datetime "%Y-%m-%d_%H:%M:%S"` (underscore between date and time); init NC must contain that datetime; `nc_bc_offset=18`; saves only at year boundaries. Phase 1 sidesteps long_inference; uses the score-function wrapper in `score_function/group_emulator.py` instead.
- **SFNO model constructor** reads `params.has_diagnostic`, `dataset.variable_list_in/out`, `dataset.constant_boundary_variables` — all provided by the shim with the same channel-counting logic as `_get_variable_list`.
- **EMA-state preference** — `GroupEmulator.__init__` prefers `ckpt['ema_state']` over `model_state` if present (matches `long_inference.py:370-380`); strips `module.` DDP prefixes.
- **Constant-boundary normalization** — *spatial* z-score per variable applied at load, not the global mean.nc stats. Mirrors `_load_constant_boundary_data` (data_loader_multifiles.py:740-749).

## Tests

```bash
source $HOME/projects/SFNO_Climate_Emulator/.venv/bin/activate
cd $HOME/projects/SFNO_Climate_Emulator
python -m pytest tests/sfno_training_group/ -v
```

16 tests across converter (7), stats/render (4), score wrapper (5).

## Pinned upstream

Group source is at `/work2/09979/awikner/stampede3/PanguWeather/v2.0/`. Capture commit hash in `docs/2026-05-09_group_code_training_track_plan.md` at first production run. Smoke verified against contracts at file:line in plan v5.
