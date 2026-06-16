# Round 2 resolution

## Applied

- P1 g0-record-check-too-lax — `_scientific_record_exists` tightened: report.md OR scores/*.csv|json|npy (no longer accepts empty dir). `20260509_gb4_ema` removed from `FAMILYLESS_EVAL_PRUNE_ALLOWLIST`, defaults UNCLASSIFIED_PROTECT; plan §3 G0 row updated.
- P1 squeue-fail-open — `_active_slurm_jobs` returns `None` on squeue failure (was `[]`); `cmd_prune` aborts `--apply` on `None`; plan §5 step 1 reworded to "successfully checked and empty".
- P1 rec_iso-NameError — define `rec_iso = rec_dt.isoformat(timespec="seconds")` before the print.
- P2 reportless-evals-not-in-csv — added `parse_scorecard_csv` fallback that reads `scores/nwp_scorecard_summary.csv` when report.md missing. Verified: +1106 rows for 20260509_y11valid_gb4_k60.
- P2 hparams_json-not-implemented — dropped from plan §4.1 column list; replaced with the actual implementation's column list.
- P2 5410-pr_6h-suppression — §4.2 now notes own-track suppresses 5410 benchmark `pr_6h` row per `render_eval_report.py:39,253` and `project_5410_eval_track`.

## Rejected

(none)

## Contested

(none)
