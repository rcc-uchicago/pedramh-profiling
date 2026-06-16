# 2026-05-08 — Trim plan for `skills/train-sfno-hpo/SKILL.md`

## Context

`skills/train-sfno-hpo/SKILL.md` is ~280 lines. The bulk is six knob tables
(Fixed-contract, Fixed-by-tier, Quality, Systems, Schedule, CLI), three Tier
A/B/C explanations, a pitfalls list, and a recording convention. Most of it
duplicates content already in the YAMLs, the sibling `sfno-training` skill,
and the Makani argparser.

## Goal

Compress the skill to ~150–180 lines (compressed-playbook style). Move the
full knob inventory and verbose pitfalls to a dated reference doc that the
skill links to.

## Outputs

1. **Rewritten** `skills/train-sfno-hpo/SKILL.md` (~150–180 lines).
2. **New** `docs/2026-05-08_hpo_knob_inventory.md` (full reference; tables +
   verbose pitfalls + full status taxonomy).

## What stays in SKILL.md

- Frontmatter — trimmed from a paragraph to 2–3 sentences.
- "When to use / NOT to use" — kept.
- "Current HPO state" — kept; load-bearing "what's active" section.
- **One Active-only knob table** replacing the six current tables. Rows:
  `batch_size`, `lr` (derived), and the small set most likely to flip to
  Active (`weight_decay`, `optimizer_max_grad_norm`, `valid_autoreg_steps`,
  `multistep_count`). Footer one-liner pointing to the inventory doc for
  Deferred / Ablation-only / Fixed-by-convention rows.
- "Sweep design — three tiers + microbench" — each tier collapsed from
  ~8–10 lines to 2–3 lines (purpose, harness, pass criterion). Tier B
  procedure list kept (it is the operational core).
- Promotion criteria — trimmed to 4–5 lines.
- Recording convention — kept, ~3 lines.
- Pitfalls — only the non-obvious ones (~6 bullets); the rest move to the
  inventory doc.
- "Where to read more" — kept, ~5–6 pointers.

## What moves to `docs/2026-05-08_hpo_knob_inventory.md`

- All six tables in full (Fixed contract, Fixed-by-tier, Quality, Systems,
  Schedule, CLI flags).
- The Status-taxonomy explanation (currently appears twice in SKILL.md).
- Verbose pitfalls (e.g. the `max_epochs` / `scheduler_T_max` re-derivation
  rule — true but rarely-active).

## Redundancies to collapse along the way

- Status taxonomy is explained twice. Keep once in the skill; full version
  in the inventory doc.
- Frontmatter description duplicates the "Current HPO state" prose.
- "Fresh `EXP_DIR` per point" appears in Tier A, Tier B step 4, and
  Pitfalls — consolidate to one place.
- "Microbench ≠ accuracy decision" appears in the feasibility-filter
  section, Tier B intro, promotion criteria, and pitfalls — consolidate.

## Order of operations

1. Draft `docs/2026-05-08_hpo_knob_inventory.md` — move + lightly
   re-organize the six tables and the verbose pitfalls.
2. Rewrite `skills/train-sfno-hpo/SKILL.md` to the compressed form with a
   prominent link to the inventory doc.
3. Verify rewritten SKILL.md is in the 150–180 line band.
4. `grep` for cross-skill links into anchors that will disappear (sibling
   skills already link by file path only — safe).

## Out of scope

- Updating `skills/sfno-training/SKILL.md:144` ("GB=32 default" — stale per
  `project_zgplev_gb_decision`). Flag to user; fix in a separate edit if
  approved.
- Rewriting the SKILL.md description for the eval skills.
- Touching the underlying YAMLs or harnesses.
