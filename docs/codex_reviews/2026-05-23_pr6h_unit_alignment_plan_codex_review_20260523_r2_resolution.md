# Round 2 resolution

## Applied
- P1 ACC suppression target wrong — §5.2 rewritten: single guard at `render_eval_report.py:174` inside per-channel loop covers both RMSE+ACC; `_render_masked_tas` left untouched
- P2 helper/test shape — §6 tests now reference real rendering helper, drop fake `(channel, model, summary)` signature, regression follows `test_render_eval_report_warmstart.py:53` pattern with `--metadata-json` stub
- P2 override under-specified — §5.4 now uses `PR6H_UNIT_ALIGN="${PR6H_UNIT_ALIGN:-suppress}"` env-var pattern matching existing `TRACK`/`OUT_ROOT`/`CKPT` idiom in `eval_run_report_inline.sh`
- P2 figures paragraph inaccurate — §5.5 corrected: pr_6h is excluded from line plots (`render_eval_figures.py:28`, `:278`); only bias maps overlay, deferred to §10
- P2 skill update too narrow — §8 Phase 6 expanded to full paragraph rewrite of `SKILL.md:224`, with required content listed
