# makani port — PROGRESS (newest first)

Format: `## YYYY-MM-DD HH:MMZ — <one-line state>` then **Progress / Surprises / Decisions / Next**,
bullets only, with job ids, tokens and shas. A fresh session reads the top 3 entries.

## 2026-10-01 — handoff written; nothing ported yet

**Progress**
- Handoffs: `HANDOFF_worker.md` (milestones M0–M8), `HANDOFF_monitor.md`; this file; `_papercuts.md`
  seeded with the traps of the 2026-09-29/30 sessions.
- Measured: our makani = 0.2.0 @ `c9704308` (2026-04-23); upstream `main` = `a0aa4c4f` (2026-09-30),
  184 commits / 209 files ahead, same `__version__`.

**Surprises**
- Upstream `main` needs `zarr>=3`; our venv pins zarr 2.18.7 (`polaris_setup_sfno_venv.sh:82`).
- `4c40a0e0` "Guard spectral weight splitting on spatial_distributed" may explain our `w=4`
  problem (7669001: h2w4 trains but its loss is ~26 % high) — unread.

**Decisions** (operator, 2026-10-01): port to latest `main`; keep the same checkpoints; small commits,
green tag per milestone, revert to the previous green commit on breakage; one worker + one monitor.

**Next**
- Worker: create `feat/makani-port-main` + worktree `.claude/worktrees/makani-port` from
  `feat/makani-f-finetune` (record the base sha here); start M0 (`api_delta.md`, golden prereg).
