# HANDOFF — consolidate makani_sfno documentation (fixing doc drift)

Written 2026-10-04, revised 2026-10-04 after a 2-seed Opus review found this handoff committing
the exact kind of unverified claim it warns against (see §3, trap count). **Scope: documentation
only.** No model code, no config, no training runs, no PBS jobs.

## 0. Why, in one paragraph

makani_sfno's documentation has drifted enough to cause a real incident: a CHANGELOG entry cited
`docs/2026-10-02_b_continuation_dryair_handoff.md` §2/§2.3/§3 as the source of outcome categories
used to interpret a measured result. That file does not exist in this worktree or branch (checked
2026-10-04). CHANGELOG.md still holds 2 citations to it. The underlying job data was real; the
citation wasn't.

## 1. Step 0: branch and scope — do this before anything else

- **Branch from `feat/makani-b-continuation-dryair` HEAD, not from main.** This handoff,
  `polaris_makani_analysis_ensemble_handoff.md`, roughly 30 of the docs below, and the
  2026-10-03/04 CHANGELOG entries exist **only** on `feat/makani-b-continuation-dryair` — none are
  on `origin/main` (checked: `git show origin/main:polaris_makani_docs_consolidation_handoff.md`
  → `fatal: ... exists on disk, but not in 'origin/main'`). If your worktree defaults to a fresh
  branch off `origin/main`, you will not have this handoff's own prerequisites. Create your working
  branch from this branch's HEAD, name it, and commit the inventory table (§3) to it immediately —
  don't wait until the end to branch.
- **Input scope, named exactly** (counts as of 2026-10-04 on this branch — recount, don't trust
  these numbers by the time you read them):
  - `makani_sfno/docs/*.md` and `*.csv` (91 `.md` + 3 `.csv` at time of writing).
  - Four subdirectories under `makani_sfno/docs/` the first draft of this handoff missed
    entirely: `codex_reviews/`, `hpo_distill/`, `run_log/`, `audit_snapshots/`. **Default: give
    each of these one inventory line (path + count), not claim-by-claim verification** — they're
    in scope to know about, not to fully audit in this pass.
  - Repo-root `polaris_makani_*.md` and related handoff-style docs — get the current list with
    `ls polaris_makani_*.md *makani*.md` at repo root, don't hand-type it (it will be wrong by the
    time you read it; the first draft's hand-typed list of 12 was already off).
  - `makani_port/` (repo root) — `PROGRESS.md`, `HANDOFF_worker.md`, `HANDOFF_monitor.md`,
    `_papercuts.md`, `equivalence_tolerance.md`, and its m0/m5 preregs. Missed entirely in the
    first draft.
  - `MONITOR_makani_streaming_driver.md` (repo root). Also missed.
  - **Every CHANGELOG.md Decisions-log entry about makani, matched by text** (`makani|sfno|ACE2`
    case-insensitive in the entry header), **not** by the `(makani)` tag alone — the tag only
    covers entries from about 2026-09-19 on; makani work starts 2026-09-01 and plenty of earlier
    entries are untagged.
  - **`## Known issues / failed approaches (do NOT re-attempt)`, CHANGELOG.md line 7769** (file is
    7,926 lines total, newest-first in the Decisions log above it). This section is the most
    direct "what doesn't work and why" source in the whole repo and the first draft of this
    handoff never pointed to it.
  - TODO.md's makani P0 block.
  - Docs that exist only on other branches: get the bounded list once with
    `git branch --all | grep -i makani` (roughly 25 branches, not a loop over unknown names), then
    for branches you haven't already checked out, `git ls-tree -r --name-only <branch> --
    makani_sfno/docs polaris_makani_*.md makani_port/` and read hits with `git show
    <branch>:<path>` — never check out or merge another branch. Known example: the canonical
    makani-vs-ACE2 code audit lives only on `feat/makani-port-main`.

## 2. The verification discipline

The failure mode to avoid is **propagating a citation you haven't opened**, and the second-order
failure mode (found by review) is **quoting a count from a doc instead of opening it and
counting** — the first draft of this handoff did exactly that ("9 measured silent-failure traps";
the source has at least 10, plus a separately-described further one). For every claim you carry
into the merged doc:

1. Find the primary artifact: a job's `.o<jobid>`/`.e<jobid>` log, a CSV row, or code — not the
   nearest doc that mentions it. **Job logs are gitignored and scattered**: this worktree's
   `makani_sfno/`, the main checkout's `makani_sfno/`, every other `.claude/worktrees/*/makani_sfno/`,
   and expdirs under `/eagle/projects/lighthouse-uchicago/members/mehta5/`. Build one jobid→path
   index with a single `os.walk` **bounded to those specific roots** (not a scan of all worktrees'
   full trees, not `/eagle` broadly), then look up from it. `qstat -xf` is **not** a reliable
   primary source here — PBS purges old job history, and the Polaris shell rules allow `qstat -x`
   at most once per request, never in a loop over many job ids.
2. Open the artifact. Confirm it says what the citing doc claims, and **if you're quoting a count
   from a doc (N traps, N retired claims), open that doc and count — don't carry the number as
   written**. If a check fails, or the file doesn't exist, that's a finding — record it in the
   dangling-citations list (§3), don't quietly fix the old prose. CHANGELOG.md is append-only:
   add a dated correction entry at the top of the Decisions log, never edit a past entry.
3. When you conclude something "does not exist anywhere," say exactly what you checked (which
   worktree, which branches) — `os.walk('.')` from inside one worktree only proves it's absent
   from *that* worktree, not from the repo.
4. If you cannot find a primary source, mark the claim `❓ UNVERIFIED — no primary source found`
   and keep it visible, rather than dropping it or trusting the secondary doc.

Bulk git/filesystem operations are real risk on this login node (shared, pid-capped at 256,
Lustre-backed) — one Bash call per query, use `timeout`, and read `git log` subject lines only
(`--format=%h %ad %s`) rather than full commit bodies; open a body only when the subject names an
actual decision.

## 3. Method

1. **Inventory pass first, generated mechanically.** One script produces a table (path, date from
   filename, first heading, line count, branch) for every doc in scope; you fill in only a
   `status` guess column (plan / result / superseded / unknown) by hand. Commit this table alone
   as a checkpoint — it's the first thing a resumed session should be able to read. **The
   inventory itself gets no claim verification.**
2. **Verify per §2 as you go**, not as a separate end pass. Keep a running "dangling citations /
   contradictions found" list — required deliverable, not incidental.
3. **Verification is tiered, because this is roughly 180 files and 7,900+ CHANGELOG lines and
   cannot all be verified in one or two contexts — an "every claim" standard will silently become
   sampling without saying so.**
   - **Tier 1 (must verify against the primary artifact):** every number or verdict that goes into
     the Checkpoint-lineages, Hyperparameters, Known-silent-failure-traps, and
     Retired/contradicted-claims sections below.
   - **Tier 2 (secondary citation is acceptable):** everything else — carry it with its source doc
     cited and tag it `❓ secondary-only` rather than silently treating it as verified.
   - **Stop point:** once the inventory plus the Traps and Retired-claims sections are done, commit
     and write the §5 status note even if context remains for more. A correct partial doc beats a
     complete-looking doc that stopped verifying halfway without saying so.
4. **Organize the merged doc by topic**, adjusting the list below as evidence dictates:
   - Checkpoint lineages (A / B / C1 / D / F / G / ACE2 ports) — parentage, current status.
   - Hyperparameters (LR, β₂, grad clip, schedule) — confirmed vs retired-and-why.
   - Depth / `n_future` — the ladder results, what depth buys and costs.
   - Mass conservation / dry-air — trained-with vs post-hoc, the 1-year-vs-5-year trap.
   - Multi-node scaling — NCCL/libfabric findings, the TCP-vs-CXI correction history.
   - Climate screening methodology — Stage-0 vs the 5-year protocol, which ranking rules are
     actually pre-registered (cite the specific handoff §, verified) vs which aren't.
   - ACE2 comparison / code audit findings.
   - **Known silent-failure traps** — pull every trap from `polaris_makani_analysis_ensemble_handoff.md`
     §6 (and its separately-numbered additions, e.g. §0a) by opening the file and counting; don't
     quote a number from this handoff or from memory.
   - **Retired / contradicted claims, explicitly "do not resurrect"** — pull from the same
     handoff's §5 table **and** from CHANGELOG's `## Known issues / failed approaches` section
     (line 7769) and every makani-matched Decisions-log entry, not just the ones tagged `(makani)`.
5. **Tag every entry**: `✅ CONFIRMED WORKING`, `❌ CONFIRMED BROKEN`, `🔁 RETIRED (contradicted by
   later evidence)`, or `❓ UNVERIFIED` (or `❓ secondary-only` per the Tier-2 rule). State a
   mechanism only when it's actually established — if it's a measured correlation without a
   confirmed mechanism, say so plainly rather than upgrading it.

## 4. Do NOT

- ❌ Delete or move any existing doc. A superseded doc gets a one-line header added at the top
  (`> Superseded by <new doc> §X — kept for the audit trail`) — **never moved to an archive
  folder**: moving breaks every existing path citation to it (including ones in append-only
  CHANGELOG entries you can't go back and edit), which manufactures new dangling citations, and
  it's a much larger diff inside a git subtree than a header line.
- ❌ One giant commit — small commits per CLAUDE.md's own "Development principles"; the inventory
  commit, then per-topic-section commits.
- ❌ Push to `main` or merge your own PR — a solo session can't self-approve (CLAUDE.md #9).
- ❌ Resolve open science questions while writing this (does dry-air training "really" help, is
  soil removal "really" the cause) — record what's measured and what's open; those calls are
  jesswan's.
- ❌ **Fix anything you find broken.** If verification turns up a real code/doc disagreement — a
  log contradicting the code, a mis-scored checkpoint, a config that differs from what a doc says
  ran — record it in the dangling-citations list and stop. Do not patch code, rerun scoring, or
  open a code PR; that changes what's computed, outside a documentation-only task and without the
  smoke/equivalence gates CLAUDE.md rules #1/#6 require for any such change.
- ❌ Create an auto-loading `makani_sfno/CLAUDE.md` in this task. If you think the merged doc should
  be (or be pointed to by) one, propose it in the PR description — don't add project-config files
  unilaterally.
- ❌ Assume any specific number, filename, or branch name in this handoff is still current by the
  time you read it. This handoff is itself subject to the drift it's fixing.

## 5. Deliverable and definition of done

- New file: `makani_sfno/KNOWLEDGE.md`. Yes, that's inside a git subtree — accepted, because it's
  a new file (not an edit to subtree-owned content), which is low-risk under CLAUDE.md's
  "keep edits there minimal and contiguous" rule. Match the register of `si/bench_midway_notes.md`
  (CLAUDE.md's named style model for this kind of living doc), not an academic-paper register.
  Keep a status block at the top (done / started / untouched, by section) so a session resumed
  after a context clear needs only this file, not this handoff, to continue.
- Pointers added in CHANGELOG.md's "Status at a glance" makani row and in TODO.md's makani P0
  block. **Do not add a pointer to CLAUDE.md** — that file auto-loads every session; if you think
  it should point here too, say so in the PR description and let the operator decide.
- The dangling-citations/contradictions list, written up as its own CHANGELOG entry.
- Draft PR opened (CLAUDE.md #9), noted in CHANGELOG as open.
- **Done when**: inventory committed; Known-silent-failure-traps and Retired-claims sections
  (Tier 1) fully verified and committed; remaining topic sections populated at least to Tier 2;
  every entry tagged; dangling-citations list written up; no existing doc deleted or moved.
