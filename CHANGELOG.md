# CHANGELOG — pedramh-profiling living document

This is the **living document**: the shared memory across sessions. It records
what's done, what's in progress, what's blocked, measured results, and — most
importantly — **failed approaches so they aren't re-attempted**. Update it before
you stop working. Newest entries at the top of each section.

See **CLAUDE.md** for how to work here and **DESIGN.md** for what/why.

Format for entries: `YYYY-MM-DD — <what happened> — <result/measurement> — <what it means / next>`.

---

## Status at a glance

| Track | State |
|---|---|
| Repo published (s2s / s2s-lightning / si) | ✅ done |
| SNFO → SI rename (repo-wide) | ✅ done |
| Polaris (PBS) bring-up | ⬜ not started — handoff written (`polaris_handoff_prompt.md`) |
| Correctness baselines captured (DESIGN.md §4) | ⬜ not started — **blocks all optimization** |
| Test harness (tier-1 equivalence/unit + `--fast` smoke) | ⬜ not started |
| Optimization ladder (DESIGN.md §5) | ⬜ not started |

## Next actions (pick from the top)

1. **Polaris bring-up** — probe → 1-GPU → 4-GPU smoke for each model via PBS;
   write `polaris_pbs_notes.md`. Follow `polaris_handoff_prompt.md`.
2. **Capture correctness baselines** (DESIGN.md §4) for each model — nothing can be
   optimized safely until these exist.
3. **Stand up the test harness** — CRPS/KL numerical checks, normalize↔inverse
   round-trip, tiny-model forward/backward, `--fast` smoke.
4. Then start the optimization ladder (torch.compile first), one gated commit each.

## In progress

- _(none yet — claim your task here, e.g. "IN PROGRESS: Polaris probe (@you)")_

## Decisions / changes log

- **2026-07-13** — Added `DESIGN.md`, `CLAUDE.md`, and this `CHANGELOG.md` — the
  project's design spec, working guide (Fable 5; small commits + tests pass;
  explicit "things NOT to do"), and living document — patterned on
  `smsharma/clax` and the MARSHAL/decrypto Midway playbooks. Establishes the
  **numerical-equivalence-vs-baseline** gate as the correctness oracle. Next: begin
  Polaris bring-up + baseline capture.
- **2026-07-13** — Published the repo and completed the repo-wide **SNFO → SI**
  rename (SI is the correct name; SNFO was a mislabel). NGC key scrubbed to
  `$NGC_API_KEY`. `main` is branch-protected (PR + 1 review).

## Failed approaches (do NOT re-attempt)

- _(none recorded yet — when something doesn't work, record it here with the reason,
  e.g. "Tried enabling torch.compile default mode on the port — graph break at the
  VAE reparam sampling; needs `dynamic=False` + a compile-region boundary. — <date>")_

## Benchmark results

_(record per-cluster bench deltas here as they land — model, cluster, config,
samples/s, peak mem, and the equivalence result for any optimization. Compare only
within a cluster, never A100 vs H100 NVL.)_
