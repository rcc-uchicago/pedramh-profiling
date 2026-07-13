# `configs/` — Lightning-port configs

> Status: port complete through Phase 6. The config-system decision below is
> settled (Phase 5); this section records the as-landed outcome.

Mirrors `$SNFO_DIR/configs/` (`/project/pedramh/shared/anthonyz/configs/`), which
holds plain YAML files loaded with `yaml.safe_load` (via `common/utils.py::get_yaml`)
plus an archived `configs/old/`. SNFO YAML has three top-level keys — `model:`,
`data:`, `training:` — with cluster suffixes (`*_midway.yaml`, `*_NCAR.yaml`).

## Config system (resolved, Phase 5)

S2S **keeps** its own `v2.0/utils/YParams.py` + sectioned YAML; it did **not**
adopt SNFO's flat `get_yaml` + `model`/`data`/`training` split — a necessary-S2S
divergence. The landed port config is `configs/test_midway.yaml` (section `S2S`),
the default `--yaml_config` for the ported `train.py` / `val.py` / `bench.py`,
loaded via `utils.YParams.YParams` (the smoke harnesses pin `v2.0/config/test.yaml`
the same way).

The S2S hard constraint stands: configs are cluster-specific and fail deep (not
early) — `data_dir`, `checkpoint_path`, and mean/std `.nc` filenames assume a
specific filesystem. On Midway the HDF5 dataset is `/project/pedramh/h5data/h5data`
(no staging step).

No `__init__.py` (configs are data, not a Python package).
