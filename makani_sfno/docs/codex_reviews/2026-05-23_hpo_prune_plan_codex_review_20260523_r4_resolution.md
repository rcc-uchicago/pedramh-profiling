# Round 4 resolution

## Applied

- P2 plan-says-prune--manifest-but-CLI-has-no-such-arg — §5 reworded: `prune [--apply] [--force-active]`, manifest path fixed.
- P2 csv-fallback-no-5410-pr_6h-filter — added the same `(model="5410 benchmark", channel="pr_6h")` skip in `parse_scorecard_csv` for symmetry / future-proofing.

## Applied (self-audit)

- Class sweep on "prose drift" (recurring r1+r3+r4): grepped all `hpo_prune.py <subcmd>` references in plan vs `main()` argparse — all six subcommands and the prune CLI flags now match exactly. No further drift found.

## Rejected

(none)

## Contested

(none)
