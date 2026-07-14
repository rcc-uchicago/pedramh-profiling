# AI-RES `docs/` index

Implementation plans, audits, and preflight documents in this directory.
Plans are listed chronologically by first-authored date, computed as
`min(first git commit date, filesystem mtime)` so files authored locally
before being bulk-committed reflect their true authoring date.

Existing plan filenames are intentionally **not** date-prefixed: their
cross-reference footprint into `src/`, `scripts/`, `tests/`, and `skills/`
was ~70 files, too large to justify a cosmetic rename. Plans authored
from 2026-05-02 onward use the `YYYY-MM-DD_<name>_plan.md` convention.

## Implementation plans

| Created    | Plan | Notes |
|------------|------|-------|
| 2026-04-17 | [aires_rad_profile_plan.md](aires_rad_profile_plan.md) | `aires_rad` postprocessor profile (radiation + heat fluxes); strict superset of `exp26`; provisional → audited workflow. |
| 2026-04-17 | [plasim_postprocessor_refactor_plan.md](plasim_postprocessor_refactor_plan.md) | PlaSim postprocessor refactor (v1–v4); v5 addendum points at `aires_rad_profile_plan.md`. |
| 2026-04-21 | [plasim_postprocessor_expand_plan.md](plasim_postprocessor_expand_plan.md) | **Superseded 2026-04-21** by `plasim_expansion_and_adaptor_plan.md`. |
| 2026-04-21 | [plasim_expansion_and_adaptor_plan.md](plasim_expansion_and_adaptor_plan.md) | Keeps `pl`; dual sigma + pressure-level `zg` output; separate adaptor module. Supersedes the expand plan. |
| 2026-04-24 | [plasim_makani_packager_plan.md](plasim_makani_packager_plan.md) | v9 packager contract: PlaSim → Makani 3-dataset HDF5; trainer-patch contract. 9 rounds of Codex review. |
| 2026-04-24 | [sfno_training_extraction_plan.md](sfno_training_extraction_plan.md) | **Superseded** by `sfno_training_implementation_plan.md`; v4 is now a pointer file. |
| 2026-04-24 | [sfno_training_implementation_plan.md](sfno_training_implementation_plan.md) | v4: subclass Makani trainer + consume packager HDF5 contract. PR-A (data) + PR-B (trainer). |
| 2026-04-25 | [sfno_full_training_plan.md](sfno_full_training_plan.md) | v1.1 full SFNO emulator training on sim52; training shipped. |
| 2026-04-27 | [sfno_tiny_short_training_plan.md](sfno_tiny_short_training_plan.md) | Tiny + short training gates; rollout diagnostic; preflight checks. |
| 2026-04-30 | [sfno_eval_plan.md](sfno_eval_plan.md) | SFNO emulator evaluation: scoring, rollout, climatology, report rendering. |
| 2026-04-30 | [plasim_zg_plev_migration_plan.md](plasim_zg_plev_migration_plan.md) | v7: sigma → pressure-level `zg` migration; diff-only against packager v9. |
| 2026-05-01 | [dsi_smoke_backup_plan.md](dsi_smoke_backup_plan.md) | v3: DSI smoke training environment bring-up. |
| 2026-05-01 | [dsi_full_training_plan.md](dsi_full_training_plan.md) | v4: DSI full training; depends on `dsi_smoke_backup_plan` phase 1. |
| 2026-05-02 | [2026-05-02_ema_implementation_plan.md](2026-05-02_ema_implementation_plan.md) | EMA (Karras-warmup) for SFNO trainer; legacy save+load only. |
| 2026-05-04 | [2026-05-04_zg1000hpa_migration_plan.md](2026-05-04_zg1000hpa_migration_plan.md) | Drop zg150, add zg1000 (ACE parity); zg500 index 47→46; in-place repackage. |

## Audits, preflights, and checks

| Created    | Doc | Notes |
|------------|-----|-------|
| 2026-04-21 | [weather_emulator_io_postprocessor_check.md](weather_emulator_io_postprocessor_check.md) | I/O postprocessor check. |
| 2026-04-21 | [emulator_adaptor_audit.md](emulator_adaptor_audit.md) | Emulator adaptor audit. |
| 2026-04-21 | [plasim_postprocessor_audit.md](plasim_postprocessor_audit.md) | PlaSim postprocessor audit; references `aires_rad`. |
| 2026-04-21 | [audit_snapshots/](audit_snapshots/) | Manifest snapshots used by the postprocessor audit. |
| 2026-04-22 | [sfno_data_preflight.md](sfno_data_preflight.md) | SFNO training data preflight (v1). |

## Filename convention for new plans

New plans authored from 2026-05-02 onward use the prefix `YYYY-MM-DD_`
based on the calendar date the plan is authored, e.g.
`docs/2026-05-02_<descriptive_name>_plan.md`. Update this index when a
new plan lands.
