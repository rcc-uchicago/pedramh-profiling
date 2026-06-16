# `docs/run_log/` — per-run postmortems and failure analyses

Per-run failure / surprise write-ups. One markdown per investigated run.
Filenames are `YYYY-MM-DD_<short-topic>.md` so a `ls` lists them
chronologically. Anchor each file on the eval `RUN_TAG` or training
`EXP_DIR` so it can be cross-referenced from `provenance.txt`.

## When to add an entry

- An eval result is surprising (regression, anomaly, no-op fix not no-op).
- Two chains collided / one chain overwrote the other.
- A run failed mid-stream (training NaN, eval OOM, checkpoint corruption).
- A long debug session ended with "the bug was X" — write it down so the
  next person doesn't redo the analysis.
- A hypothesis was falsified (negative results are useful — saves the next
  person from re-running the same A/B).

## What NOT to put here

- Forward-looking implementation plans (those go in `docs/YYYY-MM-DD_*.md`
  alongside the existing plan series).
- Skill / instruction updates (those live in `.claude/skills/`).
- Architectural decisions (those belong in the relevant `*_plan.md` plus a
  `memory/project_*.md` if persistent).

## Template

```markdown
# <one-line title> — YYYY-MM-DD

**Run / RUN_TAG / EXP_DIR:** `<exact path or tag>`
**Author / debugger:** <name>
**Date:** YYYY-MM-DD
**Status:** open / mitigated / root-caused
**Severity:** info / regression / blocker

## TL;DR
2–3 sentences. What happened, what we now believe, what we did about it.

## Symptoms
What the eval / training showed. Numbers, with paths.

## What we checked (and ruled out)
Each rejected hypothesis with the evidence that killed it. The point of this
section is that the next investigator does not redo the same checks.

## Root cause
The actual explanation. If still unconfirmed, mark UNCONFIRMED and list
the open A/B that will resolve it.

## Fix / next experiment
Concrete actions taken or proposed. Link to plan files, configs, slurms.

## References
Paths to provenance.txt, eval dirs, training logs, related plans, related
memory entries.
```

## Index

- 2026-05-12 — [v11 EMA tas regression](2026-05-12_v11_ema_tas_regression.md):
  not an EMA bug; trained-recipe artifact. Hypothesis under test:
  no-clip + input noise interaction.
- 2026-05-12 — [v11 / gbhpo40 RUN_TAG collision](2026-05-12_v11_gbhpo40_run_tag_collision.md):
  two eval chains shared `$OUT_ROOT`; led to misattributed scorecards.
  Fixed by `_family-<train_family>` auto-suffix + collision-guard in
  `scripts/submit_eval.sh`.
