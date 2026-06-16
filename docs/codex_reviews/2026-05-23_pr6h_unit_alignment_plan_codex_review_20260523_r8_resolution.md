# Round 8 resolution (new loop, post-skill-update; same-day file numbering)

## Applied
- P1 truth-side ×21600 claim unsupported — §1, §3.3, §3.5, §7 reworded to cite empirical ~3,600-4,400× ratio from `docs/2026-05-14_pr_6h_units_mismatch_ticket.md:11-25` with ~5× unexplained gap on top of nominal 21,600 (class-sweep across 4 sites)
- P2 "accumulated/integrated" wording — §1 now uses "6-hour precip proxy" citing `docs/2026-05-06_group_sfno_5410_eval_plan.md:127`; banner reworded with proxy term
- P2 test/smoke commands incomplete — §6 regression and smoke commands now spell out `--run-tag`, `--eval-sha7`, `--data-sha7`, `--train-sha7`, `--ckpt-path` (per `render_eval_report.py:45-51`); smoke command shows how to source from `provenance.txt`
- P2 rationale citation scope — §3.1 + §4.3 banner now cite both `infer_sfno5410_blocking_h100_packed.py:348-349` AND `infer_sfno5410_byo_ic.py:431` with packed-vs-BYO scoping note
