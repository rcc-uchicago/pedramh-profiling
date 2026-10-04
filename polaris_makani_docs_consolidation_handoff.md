# HANDOFF — consolidate makani_sfno documentation (fixing doc drift)

Written 2026-10-04. **Scope: documentation only.** No model code, no config, no training runs,
no PBS jobs. This task needs zero compute — don't submit anything. If you find yourself about
to `qsub`, you've left scope.

## 0. Why, in one paragraph

The makani_sfno track has accumulated ~70 files in `makani_sfno/docs/` plus ~12
`polaris_makani_*_handoff.md` files at the repo root plus the makani-tagged entries in
CHANGELOG.md and TODO.md, with no single place to check "does X still hold." This already
caused a real incident: CHANGELOG entries dated 2026-10-02 and 2026-10-03 both cite
`docs/2026-10-02_b_continuation_dryair_handoff.md` §2/§2.3/§3 as the source of the
pre-registered outcome categories used to interpret a measured result — **that file does not
exist anywhere in the repo, on any branch, or in git history.** The underlying job data was
real and independently verifiable; the citation wasn't. Full account: CHANGELOG.md
`2026-10-04 (makani) — Step 2 climate comparison`. That is the seed finding for this task, not
an isolated bug — assume there are more like it until you've checked.

## 1. The verification discipline (this is the actual job, read it before anything else)

The failure mode to avoid is **propagating a citation you haven't opened**. Every existing doc
summarizes some other doc or job log in prose; prose drifts, primary artifacts don't. For every
claim you carry into the merged doc:

1. Find the primary source (job log, CSV row, `qstat -xf` record, or code) — not the nearest
   doc that mentions it.
2. Open it. Confirm it says what the citing doc claims. If it doesn't, or doesn't exist, that
   is itself a finding — record it, don't quietly fix the old prose (CHANGELOG is append-only;
   add a dated correction entry, never edit a past entry's content).
3. Only then write the claim into the merged doc, citing the primary artifact's path, not the
   secondary doc.
4. If you cannot find a primary source for a claim, mark it `❓ UNVERIFIED — no primary source
   found` and keep it visible. An unverifiable claim you flag is more valuable than one you
   silently drop or silently trust.

To check whether a cited file exists anywhere (don't trust `ls` on one branch):
```bash
git log --all --oneline -- '*exact_filename*'          # ever committed, any branch?
python3 -c "import os; [print(os.path.join(d,f)) for d,_,fs in os.walk('.') for f in fs if 'fragment' in f.lower()]"  # on disk now?
```
(`find` is banned on this repo's login node — see CLAUDE.md #2 and the Polaris shell rules;
use `os.walk` or `ls`/`Glob` instead.)

## 2. Inputs to inventory (be exhaustive; don't sample)

- `makani_sfno/docs/*.md` and the CSVs mixed in there (dated 2026-05-02 through 2026-10-03 as
  of this writing — `ls` it fresh, don't assume this list is current).
- Repo-root `polaris_makani_*_handoff.md` (12 as of this writing: `finetune_stability`,
  `g_spatial`, `ace2_ports`, `analysis_ensemble`, `climate_protocol`, `accuracy`,
  `1node_production`, `streaming_driver`, `128node_decision_prompt`, plus `makani_bench_report.md`
  and `ace2_vs_makani_differences.md` which aren't `_handoff`-named but carry the same kind of
  content). `ls *.md` at repo root fresh; this list will have drifted by the time you run it.
- CHANGELOG.md's `(makani)`-tagged entries (every one, not just the recent ones — the oldest
  ones are where "retired claims" live).
- TODO.md's makani P0 section, including the nested handoff pointers inside it.
- `git log --all -- 'makani_sfno/*' 'polaris_makani_*'` for anything decided verbally/in a
  commit message but never written into a doc.
- Check for `makani_sfno/CLAUDE.md` — **does not exist today** (checked 2026-10-04); if you
  create the merged doc, consider whether it should auto-load the way `si/CLAUDE.md` and
  `physicsnemo_ai_rossby/CLAUDE.md` do (see CLAUDE.md "Where to look"). Don't assume this is
  still true by the time you read it — check again.

## 3. Method

1. **Inventory pass first, before writing anything merged.** One table: doc path, date,
   one-line claim/purpose, status guess (plan / result / superseded / unknown). Commit this
   table on its own — it's useful even if the rest of the task stalls, and it's your checkpoint
   if this session runs out of context (see §5).
2. **Verify, per §1, as you go** — not as a separate pass at the end. Keep a running
   "dangling citations / contradictions found" list; this list is a required deliverable, not
   an incidental side effect.
3. **Organize the merged doc by topic, not by date.** Suggested sections (adjust as the
   evidence dictates, don't force facts into a section that doesn't fit):
   - Checkpoint lineages (A / B / C1 / D / F / G / ACE2 ports) — what each is, warm-start
     parentage, current status.
   - Hyperparameters (LR, β₂, grad clip, schedule) — what's confirmed, what's retired-and-why.
   - Depth / `n_future` — the ladder results, what depth buys and costs.
   - Mass conservation / dry-air — trained-with vs post-hoc, the 1-year-vs-5-year trap (fresh
     example: 2026-10-04 entry above).
   - Multi-node scaling — NCCL/libfabric findings, the TCP-vs-CXI correction history.
   - Climate screening methodology — Stage-0 vs the 5-year protocol, the pre-registered ranking
     rules that actually exist (cite the specific handoff §, verified) vs the ones that don't.
   - ACE2 comparison / code audit findings.
   - **Known silent-failure traps** — carry forward the "9 measured silent-failure traps" from
     `polaris_makani_analysis_ensemble_handoff.md` verbatim-verified, don't re-summarize from
     memory of the memory.
   - **Retired / contradicted claims, explicitly "do not resurrect"** — carry forward the "8
     retired claims" from the same handoff, plus the β₂ 0.95→0.999 reversal and any others
     the inventory turns up. This section is as important as "what works."
4. **Tag every entry**: `✅ CONFIRMED WORKING`, `❌ CONFIRMED BROKEN`, `🔁 RETIRED (contradicted by
   later evidence)`, or `❓ UNVERIFIED`. Each gets its evidence citation (job id + exact file
   path you personally opened) and, separately, **WHY** — but only state a mechanism if it's
   actually established; if it's a measured correlation without a confirmed mechanism (e.g. the
   dry-air 1yr-vs-5yr result, or the F/soil-removal result), say so plainly rather than
   upgrading a correlation to an explanation. This project's norm (CHANGELOG 2026-10-04,
   2026-10-03) is to report measurements as measurements, not as verdicts, when no
   pre-registered rule or confirmed mechanism backs the stronger claim.

## 4. Do NOT

- ❌ Delete any existing doc. A superseded doc gets a one-line header added at the top
  (`> Superseded by <new doc> §X — kept for the audit trail`), or gets moved to a
  `makani_sfno/docs/archive/` subfolder. Never silently removed — `makani_sfno/` is a git
  subtree (CLAUDE.md "Repo architecture"); keep edits there minimal and contiguous, and this
  consolidation is exactly the kind of broad edit that warns against sprawling across unrelated
  files in one commit.
- ❌ Edit a past CHANGELOG.md entry's content. Append a dated correction entry instead.
- ❌ Resolve open science questions while writing this (e.g., whether dry-air training "really"
  helps). Record what's measured and what's open; science calls stay jesswan's (CLAUDE.md
  "Division of labor").
- ❌ One giant commit. Inventory table, then per-topic sections, then superseded-doc headers —
  separate commits, same discipline as any other change here (CLAUDE.md "Development
  principles").
- ❌ Push to `main`, or merge your own PR (CLAUDE.md #9) — solo sessions can't self-approve.
- ❌ Assume any specific number, filename, or branch name in this handoff is still current.
  This handoff is itself subject to the same drift it's trying to fix — verify, don't inherit.

## 5. If you run out of context mid-task

This will very likely span a `/clear` or a context compaction. Before that happens (or as soon
as you notice it happened), commit whatever's done — the inventory table alone is useful — and
leave a short status note at the top of the merged doc's own file (not just CHANGELOG) saying
which sections are done, which are started, which are untouched. A future session should be
able to resume from the merged doc itself without re-reading this whole handoff.

## 6. Deliverable

- A new file, suggested path `makani_sfno/KNOWLEDGE.md` (open to a better name/location if you
  find one mid-task — note why you chose it) — the "style model" CLAUDE.md names for this kind
  of living document is `si/bench_midway_notes.md`; match that register, not academic-paper
  register.
- One-line pointers added to CHANGELOG.md's "Where to look" line and to TODO.md's makani P0
  block, so a future session finds it without being told.
- The dangling-citations/contradictions list, reported as its own CHANGELOG entry (this is a
  finding in its own right, same as the 2026-10-04 entry that started this).
- Branch + draft PR (per CLAUDE.md #9); note it in CHANGELOG as open, same as every other PR in
  this repo.

## 7. Definition of done

Inventory table committed. Merged doc committed with every §3 topic populated, every entry
tagged and cited to a verified primary source. Dangling-citations list written up. Superseded
docs marked, not deleted. Pointers added. Draft PR open, noted in CHANGELOG.
