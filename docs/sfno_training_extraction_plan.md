# SFNO Training Subproject — Extraction & Refactor Plan

> **v4 (2026-04-24):** SUPERSEDED.
>
> v1–v3 described forking another group's `train.py` (PanguWeather/v2.0) with a 4-group NetCDF input contract (`surface`, `constant_boundary`, `varying_boundary`, `upper_air`). That approach is obsolete — we have since switched to the **PlaSim → Makani three-dataset HDF5 contract** described in `docs/plasim_makani_packager_plan.md` (v9), which subclasses Makani directly instead of forking.
>
> Authoritative documents for the new path:
>
> - **Dataset contract**: `docs/plasim_makani_packager_plan.md` (v9), §"Trainer-patch contract".
> - **Implementation plan**: `docs/sfno_training_implementation_plan.md` (v4 after Codex round 3).
> - **Skill**: `skills/sfno-training/SKILL.md`.
> - **Production code**: `src/sfno_training/`.
> - **Tests**: `tests/sfno_training/`.
>
> Revision history of v1–v3 is preserved in git for archaeology. The body of those revisions is intentionally removed here to prevent any reader from acting on a stale spec.
