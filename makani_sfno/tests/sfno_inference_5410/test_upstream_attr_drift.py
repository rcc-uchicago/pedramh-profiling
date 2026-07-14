"""Tier 3 of the 5410 yaml regression net (per docs plan v3.1).

Static drift detector. Scans the upstream files we depend on for
``params.<x>`` / ``params['<x>']`` references and compares the union
against an explicit allowlist of attrs that are either:

  - YAML-keys (everything in the SFNO/PLASIM section of the upstream
    yaml, transitively via ``<<: *PLASIM`` / ``<<: *BASE``),
  - main()-injected (set dynamically in ``long_inference.py:main()``
    before any read on the smoke path),
  - Stepper-injected (set in ``long_inference.py:Stepper.__init__``),
  - GetDataset-injected (set in
    ``utils/data_loader_multifiles.py:GetDataset.__init__``),
  - Known-guarded (only accessed inside ``hasattr/getattr/get`` /
    ``in self.params`` checks; failure is silent),
  - Known-unused (accesses live only in commented-out lines, dead
    branches, or alternate ensemble subroutines never reached on the
    smoke path).

When upstream resyncs introduce a new ``params.NEW_ATTR`` access, this
test fails with a diff naming the new attrs. Maintenance is simple:
classify the new attr into one of the bins above and add it to the
matching set, OR — if it is a real new yaml-required key — add it to
``_override_section`` in ``stampede3_yaml_override.py`` and to the
yaml-keys allowlist below.

Pure regex/text scan; no upstream import; runs in <1 s.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_UPSTREAM_REPO = Path(
    "/work2/11114/zhixingliu/stampede3/SFNO_Climate_Emulator/artifacts/derecho_glade/PanguWeather/v2.0"
)
_FILES_TO_SCAN = (
    "long_inference.py",
    "utils/data_loader_multifiles.py",
    "utils/perturbation.py",
)


# Attributes provided by the upstream yaml's BASE / PLASIM / SFNO
# sections (transitively merged in YParams via ``<<: *anchor``).
# Verified by reading
# /work2/.../v2.0/config/SFNO_PLASIM_H5_DERECHO_5410.yaml on
# 2026-05-08.
_YAML_KEYS: frozenset[str] = frozenset({
    # base_config
    "loss", "lr", "weight_decay", "scheduler", "num_data_workers",
    "oc_pct_start", "oc_div_factor", "oc_final_div_factor",
    "log_to_screen", "log_to_wandb", "save_checkpoint",
    "optimizer_type", "plot_animations", "group", "exp_dir",
    "enable_fp8", "fresh_start", "use_transformer_engine",
    "early_stopping", "entity", "project", "name",
    # PLASIM
    "data_dir", "bias_data_dir", "upper_air_variables",
    "surface_variables", "diagnostic_variables", "land_variables",
    "ocean_variables", "mask_output", "constant_boundary_variables",
    "varying_boundary_variables", "train_year_to_year",
    "train_year_start", "train_year_end", "val_year_start",
    "val_year_end", "leap_year", "no_leap_year", "long_validation",
    "long_val_year_start", "long_rollout_years",
    "epochs_per_long_validation", "data_timedelta_hours",
    "surface_mean", "surface_std", "surface_ff_std",
    "upper_air_mean", "upper_air_std", "upper_air_ff_std",
    "boundary_mean", "boundary_std", "diagnostic_mean",
    "diagnostic_std", "climatology_file", "calendar",
    "timedelta_hours", "batch_size", "max_epochs", "has_year_zero",
    "num_levels", "use_sigma_levels", "levels", "sigma_levels",
    "horizontal_resolution", "predict_delta", "epsilon_factor",
    "diagnostic_logs", "diagnostic_acc", "diagnostic_gif",
    "diagnostic_gif_var_dict", "diagnostic_acc_var_dict",
    "diagnostic_spectrum_var_dict", "diagnostic_bias_var_dict",
    "diagnostic_spectra", "forecast_lead_times", "lev",
    "num_inferences", "use_reentrant", "checkpointing", "lat", "lon",
    "checkpoint_save_interval", "max_checkpoints_to_keep",
    "curriculum_learning", "ensemble_validation", "load_exp_dir",
    # SFNO arch (transitively included)
    "nettype", "num_warmup_epochs", "warmup_start_lr", "eta_min",
    "spectral_transform", "filter_type", "operator_type",
    "scale_factor", "embed_dim", "num_layers", "use_mlp", "mlp_ratio",
    "activation_function", "encoder_layers", "pos_embed", "drop_rate",
    "drop_path_rate", "num_blocks", "sparsity_threshold",
    "normalization_layer", "hard_thresholding_fraction",
    "use_complex_kernels", "big_skip", "rank", "factorization",
    "separable", "complex_network", "complex_activation",
    "spectral_layers", "sync_norm",
    # Our overrides (added by stampede3_yaml_override.py)
    "save_forecasts", "ensemble_inference_hours", "save_basenames",
})

# Attributes set dynamically in long_inference.py:main() before any
# read on the smoke path. Verified by reading lines 1245-1370.
_MAIN_INJECTED: frozenset[str] = frozenset({
    "run_iter", "has_diagnostic", "num_ensemble_members",
    "init_nc_filepaths", "nc_bc_offset", "output_dir", "save_basename",
    "ensemble_members_per_pred", "world_size", "batch_size",
    "init_datetime", "final_datetime", "init_nc_timestep_offset",
    "local_rank", "enable_amp", "global_batch_size",
})

# Attributes set in Stepper.__init__ before predict().
_STEPPER_INJECTED: frozenset[str] = frozenset({
    "single_ic_offset", "long_rollout_years",
    "experiment_dir", "checkpoint_dir",
    "best_checkpoint_path", "latest_checkpoint_path",
    "checkpoint_path_globstr", "resuming",
})

# GetDataset.__init__ writes these (data_loader_multifiles.py:401).
_GETDATASET_INJECTED: frozenset[str] = frozenset({
    "forecast_lead_times",  # written if originally falsy
})

# Attributes only ever accessed inside hasattr/getattr/get/in guards.
# Missing keys here are silent; not a yaml requirement on the smoke
# path. Verified by scanning hasattr(params, 'X') and similar guards.
_KNOWN_GUARDED: frozenset[str] = frozenset({
    "boundary_data_dir", "init_datetimes", "mask_fill",
    "prediction_duration_days", "train_data_sets",
    "validation_data_sets", "curriculum_bulk_size",
    "train_date_range", "validation_date_range",
    "save_level_idxs", "save_sigma_level_idxs",
    "early_stop_epoch", "data_grid",
    "img_shape_x", "img_shape_y", "N_in_channels", "N_out_channels",
    "pretrain_encoding",
    "octaves", "period_number", "persistence",  # perlin
    "surface_delta_std", "upper_air_delta_std",  # predict_delta path
})

# Attributes that DO appear in the upstream files but are not on the
# smoke path. Verified by reading the lines they appear on:
#   - in commented-out code,
#   - inside `if self.train` / `if self.validate` branches,
#   - inside `combine_A_ensemble` (only called from a different
#     workflow than long_inference.py main()),
#   - inside `add_gaussian_noise_n_minus_1` / `add_perlin_noise`
#     branches (only reached if `params.perturbation_type` selects
#     them; gated by `epsilon_factor > 0`, which we force to 0).
_KNOWN_NOT_ON_SMOKE_PATH: frozenset[str] = frozenset({
    # Comments only:
    "run_num",          # long_inference.py:1205 (commented)
    "val_start_year",   # long_inference.py:1205 (commented)
    "timdelta_hours",   # long_inference.py:657, 931 typo (commented)
    # Train/validate-only branches (we run inference):
    # (none currently outside _KNOWN_GUARDED)
    # Perturber dead with epsilon_factor=0 (gate at lines 204/558/830):
    "perturbation_type",
})

# Method names / accidentally-matching identifiers — false positives
# from the regex when ``params.foo`` is actually ``foo`` on a method
# call result, not on the YParams object. Excluded from the diff.
_REGEX_FALSE_POSITIVES: frozenset[str] = frozenset({
    "items",  # `for k, v in params_i.params.items()` in main()
    "yaml",   # local 'yaml' module reference, not a params attr
})


# Combine all known sets into the allowlist.
_ALLOWLIST = (
    _YAML_KEYS
    | _MAIN_INJECTED
    | _STEPPER_INJECTED
    | _GETDATASET_INJECTED
    | _KNOWN_GUARDED
    | _KNOWN_NOT_ON_SMOKE_PATH
    | _REGEX_FALSE_POSITIVES
)


# Regex: capture attribute name from `params.X` or `self.params.X`,
# and key name from `params['X']` or `params["X"]`.
_RE_ATTR = re.compile(r"\b(?:self\.)?params\.([A-Za-z_][A-Za-z0-9_]*)")
_RE_ITEM = re.compile(r"\bparams\[\s*['\"]([A-Za-z_][A-Za-z0-9_]*)['\"]\s*\]")


def _strip_comments_and_strings(line: str) -> str:
    """Heuristically remove comments and string literals from a line.

    Not a real Python tokenizer, but good enough for the scan: we
    drop everything after the first unescaped ``#`` outside a string,
    and zero-out ``"..."`` / ``'...'`` regions.

    Drops the entire line if it begins (after whitespace) with ``#``.
    """
    s = line.lstrip()
    if s.startswith("#"):
        return ""
    # Zero out string contents (handles simple cases; nested triple
    # quotes will leak — acceptable for this regex-grade scan).
    line = re.sub(r"'(?:[^'\\]|\\.)*'", "''", line)
    line = re.sub(r'"(?:[^"\\]|\\.)*"', '""', line)
    # Drop trailing line comment.
    hash_idx = line.find("#")
    if hash_idx >= 0:
        line = line[:hash_idx]
    return line


def _scan_file(path: Path) -> set[str]:
    accessed: set[str] = set()
    with open(path) as f:
        for raw in f:
            cleaned = _strip_comments_and_strings(raw)
            if not cleaned:
                continue
            for m in _RE_ATTR.finditer(cleaned):
                accessed.add(m.group(1))
            for m in _RE_ITEM.finditer(cleaned):
                accessed.add(m.group(1))
    return accessed


@pytest.fixture(scope="module")
def upstream_attr_set():
    if not _UPSTREAM_REPO.is_dir():
        pytest.skip(f"upstream repo not present: {_UPSTREAM_REPO}")
    accessed: set[str] = set()
    for rel in _FILES_TO_SCAN:
        p = _UPSTREAM_REPO / rel
        if not p.is_file():
            pytest.skip(f"upstream file missing: {p}")
        accessed |= _scan_file(p)
    return accessed


class TestUpstreamAttrDrift:
    def test_no_unknown_unguarded_attrs(self, upstream_attr_set):
        """Every attr referenced via ``params.<x>`` in scanned upstream
        files must appear in one of the allowlist categories above.

        On failure: classify each NEW attr in the diff into one of:
            yaml-key       → add to ``_YAML_KEYS`` AND to ``_override_section``
                              if it's a real new required override
            main-injected  → add to ``_MAIN_INJECTED``
            guarded        → add to ``_KNOWN_GUARDED`` (verify the access
                              is hasattr/getattr/get-protected)
            unused-on-path → add to ``_KNOWN_NOT_ON_SMOKE_PATH`` (verify
                              the access is in a comment or dead branch)
            false-positive → add to ``_REGEX_FALSE_POSITIVES`` (verify
                              the regex is matching something that is
                              not actually a params-attr access)
        """
        unknown = sorted(upstream_attr_set - _ALLOWLIST)
        assert not unknown, (
            "Upstream attribute drift detected — new params.<x> accesses "
            "found that are not in any allowlist category.\n\n"
            f"Files scanned: {list(_FILES_TO_SCAN)}\n"
            f"New / unclassified attrs: {unknown}\n\n"
            "See module docstring for classification instructions."
        )


class TestAllowlistInternalConsistency:
    """Catches accidental duplicates / typos in the allowlist sets."""

    def test_no_overlap_between_sets(self):
        # YAML keys can overlap with guarded sets (a yaml key may also
        # have a hasattr guard at some access sites); that's fine. We
        # only forbid contradictory categorizations:
        # main-injected vs yaml-keys (same attr can be both — fine).
        # We DO want to flag attrs accidentally in two of the
        # smaller categorical sets.
        smalls = (("stepper", _STEPPER_INJECTED),
                  ("getdataset", _GETDATASET_INJECTED),
                  ("not_on_smoke", _KNOWN_NOT_ON_SMOKE_PATH),
                  ("false_positives", _REGEX_FALSE_POSITIVES))
        for i, (n1, s1) in enumerate(smalls):
            for n2, s2 in smalls[i+1:]:
                inter = s1 & s2
                assert not inter, (
                    f"Allowlist conflict: attrs in both {n1} and {n2}: {inter}"
                )

    def test_no_empty_attr_names(self):
        for s in (_YAML_KEYS, _MAIN_INJECTED, _STEPPER_INJECTED,
                  _GETDATASET_INJECTED, _KNOWN_GUARDED,
                  _KNOWN_NOT_ON_SMOKE_PATH, _REGEX_FALSE_POSITIVES):
            for a in s:
                assert a and a == a.strip(), f"bad attr in allowlist: {a!r}"
