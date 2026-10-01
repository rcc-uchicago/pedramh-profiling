# makani port — PROGRESS (newest first)

Format: `## YYYY-MM-DD HH:MMZ — <one-line state>` then **Progress / Surprises / Decisions / Next**,
bullets only, with job ids, tokens and shas. A fresh session reads the top 3 entries.

## 2026-10-01 — M0(a) api_delta written; 3 STOP items found; golden prereg + PBS next

**Progress**
- Branch `feat/makani-port-main` + worktree `.claude/worktrees/makani-port` cut from
  `feat/makani-f-finetune` @ **`ed3d75c65f45`** (monitor confirmed the same sha).
- Bare full clone `$MEMBER_ROOT/external/makani-main.git`; target pinned to full sha
  `a0aa4c4fe5c40207d4fcc3da61d4c656a4ea0346` (= `main` head today).
- Old venv baseline matches the monitor's: `direct_url.json` `c7f1fba5…`, `RECORD` `380b1a77…`.
- `makani_port/api_delta.md`: all 30 import sites resolve at both shas; 9 behavioural BREAKs, 5
  predicted NUMERICS sources (N1–N5), 3 STOP items.

**Surprises**
- `4c40a0e0` and `4af60539` are **not** our `w=4` problem (a spectral-loss guard; tests only). The
  M7 lead is `18c4582`: `DistributedInstanceNorm2d` normalised in **bf16** at our pin, fp32 on main.
- main **rejects our grid declaration** (`equiangular` + cell-centred lat 89.5…−89.5) in
  `parse_dataset_metadata` — every job dies at config load. Science-owned (api_delta §3.2).
- Legacy checkpoints pickle `ruamel` `ScalarFloat`/`Anchor` (optimizer/scheduler state); main's
  `weights_only=True` loader rejects the whole file, model-only loads included (§3.1).
- The trainer now `torch.compile`s every non-SHT loss term by default — loss values leave bitwise.
- `8cf889ec` "pname checksum" is a save-time gather-plan crc; it stores nothing — no load issue.

**Decisions** — none taken; the three STOPs are proposed with defaults in api_delta §3.

**Next**
- M0(b) `m0_golden_prereg.md`, M0(c) `polaris_makani_port_golden.pbs` (old venv only, run twice).
- Operator: answer api_delta §3.1 (ruamel allowlist) and §3.2 (grid declaration) before M2/M3.

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
