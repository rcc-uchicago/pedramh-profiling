# Install `codex-review-plan` skill on Derecho

This file packages a Claude Code skill (`codex-review-plan`) and a prompt you can hand to a fresh Claude Code session on Derecho to install it as a **user-level** skill at `~/.claude/skills/codex-review-plan/SKILL.md`.

The skill was edited to be host-portable before export:
- Codex binary is invoked as `${CODEX_BIN:-codex}` (override with absolute path if `codex` isn't on PATH in non-interactive Bash on Derecho).
- `--cd "$PWD"` replaces the hardcoded Stampede3 repo path.
- The unsandboxed rationale was generalized from "Stampede3" to "shared HPC clusters."

The only thing that needs to be true on Derecho before the skill is useful: a working `codex` CLI (logged in, with `~/.codex/config.toml` configured). The skill itself does not depend on the rest of the SFNO_Climate_Emulator repo — it just edits plan docs under whatever project Claude Code is currently in.

---

## Prompt to paste into Claude Code on Derecho

> I want to install a user-level skill called `codex-review-plan` at `~/.claude/skills/codex-review-plan/SKILL.md`. Please:
>
> 1. `mkdir -p ~/.claude/skills/codex-review-plan`
> 2. Write the SKILL.md content below verbatim to `~/.claude/skills/codex-review-plan/SKILL.md` (preserve all formatting, including the YAML frontmatter and triple-backtick code blocks).
> 3. Verify with `ls ~/.claude/skills/codex-review-plan/` and `wc -l ~/.claude/skills/codex-review-plan/SKILL.md` (should be ~188 lines).
> 4. Tell me whether `codex` is on PATH in non-interactive Bash on this host: run `bash -lc 'command -v codex'`. If empty, find it (e.g. `ls $HOME/node20/bin/codex` or `which codex` from an interactive shell) and remind me to set `CODEX_BIN` to that absolute path before invoking the skill.
>
> The SKILL.md content is everything between the `BEGIN SKILL.md` and `END SKILL.md` markers below.
>
> BEGIN SKILL.md
> ```markdown
---
name: codex-review-plan
description: Get a second-opinion review of an AI-RES plan doc from the local Codex CLI (gpt-5.5, unsandboxed), iterating up to 10 rounds. Slash-only trigger `/codex-review-plan <path>`. Each round: fire `codex exec` in the background, wait for the harness wake-up (no polling), categorize Codex's findings against repo evidence and existing memories, apply clear fixes via Edit, surface contested items to the user before applying, save each round's review to `docs/codex_reviews/`, then re-fire Codex with the updated plan + prior review until Codex emits `verdict: APPROVED` or 10 rounds elapse. Use when the user explicitly invokes `/codex-review-plan` with an explicit `docs/YYYY-MM-DD_*_plan.md` path. Do NOT fire on natural-language phrasings like "have Codex look at this" — slash-only.
---

# Codex review plan (iterative)

Drive an automated review-and-fix loop with the local Codex CLI as a second opinion on an AI-RES plan doc. Codex inspects the plan against the live repo, returns a structured review with a verdict tag, this skill applies non-contested fixes and re-invokes Codex, up to 10 rounds.

**Slash-only.** Trigger exactly: `/codex-review-plan <relative-or-absolute-path-to-plan.md>`. Refuse if no path is given. Do NOT fire on phrasings like "have Codex review this" / "send X to Codex" — those stay conversational.

**Async by construction.** Every `codex exec` call uses `run_in_background: true`. Return control to the user immediately with the round number and BashOutput tail-id; pick up when the harness notifies. Never `sleep` or poll.

## When to use this skill

Trigger ONLY on the exact slash invocation:

- `/codex-review-plan docs/2026-05-20_bundled_training_eval_plan.md`
- `/codex-review-plan /abs/path/to/plan.md`

Do NOT trigger on:
- "have Codex review the plan" / "send the plan to Codex" / "what does Codex think" — these are conversational; respond by suggesting the slash command.
- One-off Codex calls for things that aren't plan reviews (e.g. asking Codex to inspect a single file). Use a direct `codex exec` invocation instead — see `reference_codex_cli_bridge` memory.

Refuse if:
- No path argument was passed → tell the user the skill requires an explicit path.
- The path does not exist or is not a `.md` file → surface the error.
- The path is outside `AI-RES/docs/` → confirm with the user before proceeding (skill is scoped to plans, not arbitrary repo files).

## Inputs

| Arg | Required | Notes |
|---|---|---|
| Plan path | yes | First positional argument. Relative to repo root or absolute. Must exist and end in `.md`. |
| `MAX_ROUNDS` (env) | no | Default `10`. Hard cap on iteration; never raise without surfacing to the user. |
| `CODEX_MODEL` (env) | no | Default whatever `~/.codex/config.toml` resolves (`gpt-5.5` as of 2026-05-20). Override with `-m <model>` only on explicit request. |

## Output layout

Reviews land in `docs/codex_reviews/`. Filename: `<plan_basename>_codex_review_<YYYYMMDD>_r<N>.md` where `<plan_basename>` is the plan's stem (e.g. `2026-05-20_bundled_training_eval_plan`) and `N` is the round number within the loop.

**Same-day re-invocation:** before round 1, scan `docs/codex_reviews/` for existing `<basename>_codex_review_<today>_r*.md`; let `N0` = highest existing round + 1, and number this loop's rounds starting at `N0`. This avoids clobbering prior same-day loops.

**Resolution sidecar (loop state).** After each non-terminal round's apply step (and any AskUserQuestion answers), write `docs/codex_reviews/<basename>_codex_review_<YYYYMMDD>_r<N>_resolution.md`. Schema: top-level `# Round N resolution` plus `## Applied` (terse tags), `## Rejected` (tag — one-line rationale), `## Contested` (tag — `user: accepted|declined|deferred`). Lines <120 chars; omit empty sections. This is **loop state for the next round, not a human-readable audit log**. Round N+1's prompt passes it to Codex with instructions to treat Rejected and Contested-`declined`/`deferred` items as out of scope unless the plan has changed to make them materially relevant again. Not written for the terminal round.

`docs/codex_reviews/` is git-tracked. Do NOT commit on the user's behalf; leave that to them.

## Loop workflow

For each round `N` from 1 to `MAX_ROUNDS`:

### 1. Build the prompt

Round 1 prompt (canonical):

```
Review Claude's implementation plan against the repo. Do not modify files; only inspect and report issues.

The plan to review is: <PLAN_PATH>

Please read it in full, then inspect the relevant files it references (configs, slurm scripts, src/ modules, scripts/) to check:

**User-approved intent — recognize and respect.** Before flagging anything, scan the plan for intent markers: memory-slug citations (e.g. `project_*`, `feedback_*`, `reference_*`), a "Decisions" or "Decisions table" section, a "Risks and trade-offs" section, a "What this plan does NOT touch" / "Out of scope" / "Deferred" section, or explicit "from the interview" / "user chose" / "user decided" language. Treat the choices captured there as authoritative — they came from the user, not from Claude's inference. Do not flag them as inconsistent, suboptimal, or missing just because you would have chosen differently. You may still challenge a user-approved choice if you have **concrete grounds**: evidence from the codebase, evidence the plan itself cites, another explicit constraint stated elsewhere in the plan, or a specific reason the choice makes the plan infeasible, unsafe, or scientifically wrong. Pure stylistic preference, or unsupported methodological preference, is not enough — but an evidence-backed methodological or scientific concern is fair game; label it clearly (e.g. "challenges user-approved decision in §X based on …") so the user can weigh it.

1. Correctness — does the plan match the actual code paths, file layouts, config keys, and run conventions in the repo? Flag any mismatches between what the plan asserts and what the code actually does.
2. Completeness — are there steps missing, preflight checks not specified, or edge cases (resume semantics, output-dir collisions, checkpoint naming, EMA vs raw, units conventions like pl/zg/pr_6h) that the plan glosses over?
3. Risk — anything in the plan that could clobber prior production run dirs, silently re-introduce compute the user is trying to save, or violate the own-track v10-only / 5410 unit conventions.
4. Concrete suggestions — for each issue, cite the file:line in the repo that motivates the concern.

Report as a structured markdown review: Strengths / Issues (P0/P1/P2) / Suggested edits. Be specific and terse; do not restate the plan.

End your reply with EXACTLY ONE of these two lines (no other text after it):
verdict: APPROVED
verdict: CHANGES_REQUESTED
```

Round `N>1` prompt appends:

```
This is round <N> of an iterative review.
- Prior round's review: <PRIOR_REVIEW_PATH>
- Prior round's resolution sidecar: <PRIOR_RESOLUTION_PATH>

Read both first, then re-review the (now-edited) plan. Focus on:
- Whether the items marked **Applied** in the resolution sidecar are now correctly resolved in the plan.
- Any new issues introduced by the edits.

Items marked **Rejected**, or **Contested** with user decision `declined` or `deferred`, are out of scope for this review loop. Do NOT re-flag them unless the current plan has changed in a way that makes the original concern materially relevant again (e.g. the section they pertained to has been rewritten and the underlying issue now applies to new content).

Do not re-litigate items you already approved unless the edits affected them.
```

### 2. Fire Codex in the background

Use Bash with `run_in_background: true`:

```bash
"${CODEX_BIN:-codex}" exec \
  --cd "$PWD" \
  --dangerously-bypass-approvals-and-sandbox \
  --color never \
  -o docs/codex_reviews/<basename>_codex_review_<YYYYMMDD>_r<N>.md \
  "<PROMPT>" </dev/null
```

Use `${CODEX_BIN:-codex}` to invoke Codex. If `codex` isn't reliably on PATH in non-interactive Bash on this host, set `CODEX_BIN` to its absolute path (e.g. `$HOME/node20/bin/codex` on Stampede3). `--cd "$PWD"` pins Codex to the project root Claude Code is running in. The sandbox-bypass flag is non-negotiable on shared HPC clusters — see `feedback_codex_unsandboxed_tradeoff`. The `</dev/null` redirect is **required** for background invocations: without it, `codex exec` inherits the parent's stdin and can hang indefinitely waiting on input that will never arrive.

After firing, report to the user: round number, output path, and the BashOutput shell id. Then return control. **Do not poll.**

### 3. On wake-up: read & parse Codex's review

When the background Bash completes (harness notifies), read the output file. Parse:
- Verdict line: scan the last 5 non-blank lines for `verdict: APPROVED` or `verdict: CHANGES_REQUESTED`. If neither found, treat as `CHANGES_REQUESTED` and note the missing tag for the user.
- Findings, by P0 / P1 / P2 headers.

### 4. Categorize each finding

For every Issues item, judge it against repo evidence and existing memories (`MEMORY.md` is loaded into context). Place each in one bucket:

| Bucket | Criterion | Action |
|---|---|---|
| **Apply** | Clearly correct: cited code/line matches Codex's claim and no existing memory contradicts. | Edit the plan to address. |
| **Reject (Codex wrong)** | Contradicted by repo evidence or an existing memory. Common case: Codex flags a number/decision the user has already validated and saved. | Do NOT edit. Note in round summary with rationale (e.g. "contradicts `project_bundled_eval_tail_timing`"). |
| **Contested** | (a) ambiguous — Codex's claim plausible but not verifiable without user input on intent; or (b) a scope/compute/risk decision the user should make per `feedback_respect_compute_scope`, `feedback_protect_prior_runs`, etc.; or (c) a non-surgical edit that materially changes the plan's **scope, experiment choices, output paths, or scientific assumptions** (see §5 for what is NOT contested under this clause). | Pause loop. Use AskUserQuestion to surface contested items before applying any of them. |

### 5. Apply non-contested fixes

For each Apply-bucket item: Edit the plan doc directly. Prefer surgical edits — fix the flagged inaccuracy in place. Group edits into one Edit call per replacement when possible.

**Class-sweep before patching.** Before editing for a finding, identify its **class** — the kind of issue (e.g. "v10/v11 unit confusion", "missing preflight check", "wrong default for a config key", "incorrect units on `pr_6h`", "checkpoint-path mismatch", "unguarded `set -u` variable"). Execute the sweep as **enumerate → batch-fix → re-grep**: build the full hit list across the plan and relevant source files (configs, slurm scripts, `src/`, `scripts/`) *before* editing, apply all clearly related fixes in one pass, then re-run the same search to confirm zero residual hits. **Never apply a class-sweep finding as a one-off edit to only the cited line** — Codex typically cites one example of a recurring problem; partial fixes leave the bug elsewhere and Codex will re-raise it next round. If the sweep surfaces related instances that materially expand scope (per §4 (c) criteria), route the whole cluster — not just the cited finding — through the contested path.

**Coverage/pattern classes — prefer discovery over enumeration.** When the class is a hardcoded list (file-extension allowlist, glob pattern, channel-name whitelist, recognized config keys), don't just add the missing case and move on. If Codex flags one or two missing entries (e.g. `.yml` after fixing `.yaml`), switch the check itself from a static list to **content- or path-driven discovery** — iterate the directory, resolve actual referenced files, match by content type. One more `if` clause invites Codex to find the next missing case next round.

Non-surgical edits (replacing a whole section, restructuring, adding a new subsection) are fine — **apply them automatically without pausing** — as long as the change is one of:
- Clarifying wording or fixing internal inconsistency
- Reorganizing existing content without altering its substance
- Adding validation steps, preflight checks, or tests
- An obviously correct fix (e.g. correcting a path, config key, units, or default that the cited code disproves)

A non-surgical edit becomes **contested** (route to §4 (c) and pause via AskUserQuestion) only if it materially changes the plan's **scope** (e.g. adding/removing experiments, expanding budget), **experiment choices** (e.g. swapping models, datasets, hyperparameters the user fixed), **output paths** (e.g. moving a run dir), or **scientific assumptions** (e.g. changing a units convention or a memory-backed decision). Wording, structure, and additive safety checks do NOT trigger the pause.

### 6. Check verdict & decide loop continuation

- If verdict is `APPROVED` and the Apply bucket is empty for this round → loop ends, report success.
- If verdict is `APPROVED` but you applied fixes anyway (rare: Codex approved with nitpicks) → loop ends, report success and list what you cleaned up.
- If verdict is `CHANGES_REQUESTED` and `N < MAX_ROUNDS` → proceed to round `N+1`.
- If verdict is `CHANGES_REQUESTED` and `N == MAX_ROUNDS` → loop ends, report to user that Codex did not approve after 10 rounds, summarize the still-outstanding issues from the final review, and list any contested items still pending user decision.

### 7. Inter-round meta-pattern check

Before firing round N+1, look across the prior rounds' resolution sidecars and the current round's findings. If the current round's findings include the **same class** as an earlier round (e.g. round 1 fixed one v10/v11 unit confusion and round 2 finds another; round 1 added one preflight check and round 2 finds another missing; round 1 fixed one unguarded variable and round 2 finds another), pause and run a **class-sweep self-audit** on the plan for that class before firing N+1: re-grep the plan and the relevant source files for any further instances, apply additional fixes per §5's class-sweep rule, and note them under Applied in round N's resolution sidecar with a `(self-audit)` suffix on the tag. This catches the case where Codex surfaces the same root issue one instance at a time across rounds.

Skip this check if no class repeats across rounds, or if the loop is terminating (APPROVED or `N == MAX_ROUNDS`).

### 8. Per-round summary to user

After each round (before re-firing or finalizing), post a concise turn message containing: `Round N/MAX — verdict: APPROVED|CHANGES_REQUESTED`, the review file path, and three counted sections — **Applied** (one-line summary + plan line), **Rejected** (finding — one-line rationale, e.g. "contradicts memory X"), **Contested — awaiting your decision** (finding — why I'm asking). Omit empty sections.

Then either:
- If contested items: AskUserQuestion, wait for answer, apply per-answer, **write the resolution sidecar** for round N, then fire round N+1.
- If no contested items and verdict is CHANGES_REQUESTED: **write the resolution sidecar** for round N, then fire round N+1 immediately.
- If verdict is APPROVED or N == MAX_ROUNDS: print the final summary and stop. (No sidecar for the terminal round — nothing will consume it.)

## Final summary (loop done)

On loop end, print: `Codex review loop complete — APPROVED at round N` (or `not approved after 10 rounds`), the list of round review paths, the final plan path with edit count, and — if not approved — outstanding P0/P1 items from the final review.

## Cost & guardrails

Codex calls hit the user's OpenAI account (not Anthropic). Each round on a ~600-line plan used ~276k tokens in the 2026-05-20 baseline; budget roughly `200k–400k tokens × rounds`. Do NOT silently exceed `MAX_ROUNDS=10`. If the user asks for more rounds mid-loop, confirm explicitly before continuing past 10.

If a `codex exec` call exits non-zero or the output file is empty: do NOT auto-retry. Stop the loop, surface the failure (stderr tail + exit code), and ask the user how to proceed. **One exception — apparent hang:** if the process is still alive and the `-o` file has stopped growing for several minutes, the cause is usually a missing `</dev/null` redirect (Codex waiting on inherited stdin). Verify the redirect is in the command, kill the run, and re-fire once before treating it as a model failure.

## Anti-patterns

- Applying every Codex finding without judgment. Codex will sometimes contradict memories or repo facts (see the walltime case in the 2026-05-20 baseline review where `project_bundled_eval_tail_timing` was authoritative).
- Skipping the AskUserQuestion step on contested items to "save a turn."
- Committing review files or plan edits on the user's behalf.
- Lifting `MAX_ROUNDS` silently. Ten is a budget cap, not a default-to-extend.

## Related

- `reference_codex_cli_bridge` — base `codex exec` invocation pattern.
- `feedback_codex_unsandboxed_tradeoff` — why we use `--dangerously-bypass-approvals-and-sandbox` on shared HPC clusters.
- `feedback_plan_to_docs` — plans live in `docs/YYYY-MM-DD_*_plan.md`.
- `feedback_interview_first` — clarify before non-trivial setup.
- `feedback_respect_compute_scope`, `feedback_protect_prior_runs` — common sources of "contested" findings.
> ```
> END SKILL.md
