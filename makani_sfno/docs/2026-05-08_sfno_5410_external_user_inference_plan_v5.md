# Plan v5: SFNO-5410 external-user inference path on Stampede3

> **Status: REVISION OF v4 to align with eval-track explicit-K
> implementation now in progress.** Date: 2026-05-08. Author:
> zhixingliu (via Claude). Audience: Codex reviewer + a group member
> (the eventual end user).
>
> Supersedes plan-v4 (`..._v4.md`). v4 was approved by Codex on
> 2026-05-08 (4th-round review), but the pipeline shape changed
> mid-stream because a parallel session is implementing the eval-track
> explicit-K-horizon plan v3.1
> (`docs/2026-05-08_sfno_5410_explicit_K_horizon_plan.md`). That work
> already lays down 6 of the 8 upstream hunks plus a `prediction_duration_days`
> override pattern, so the user-track no longer carries that weight.
>
> ## v4 → v5 changes
>
> | v4 design | v5 (post-eval-track-rebase) |
> |---|---|
> | User-track owns all 8 upstream hunks (4 alloc + 2 continuation + 2 perturbation gates) | **User-track owns only 2 hunks** (perturbation gates at lines 558, 830). The 6 partial-horizon hunks (4 alloc + 2 continuation) are owned by the eval-track explicit-K plan and are **already applied** as of 2026-05-08 16:00 |
> | Patch ledger: NEW file `docs/2026-05-08_panguweather_local_patches.md` | **Append a new section to existing `docs/2026-05-04_makani_local_patches.md`** (LP-004). The standalone v4 panguweather ledger is obsolete and will be deleted as part of v5 implementation |
> | Test: NEW `tests/sfno_inference_5410/test_upstream_patch_present.py` checks all 8 markers | **Test already exists** (created by eval-track), checks 4 alloc + 2 cont. v5 **extends it** to also check exactly 2 perturbation-gate markers, and updates the docstring to describe the full 8-hunk surface |
> | `prediction_duration_days` only set in the user yaml builder | **`prediction_duration_days` is now also set by `stampede3_yaml_override.py`** (eval-track, K=60). User-track yaml builder still sets it for arbitrary `horizon_days`. The two paths are independent — same yaml key, different drivers |
> | `_override_section_user` writes `ensemble_inference_hours = min(8784, horizon_days*24)` | Unchanged. Note that eval-track's `_override_section` writes `ensemble_inference_hours = (K+1)*6` for the K=60 case (= 366h) — the two formulas converge at small horizons |
> | Marker comment text: `AI-RES-Stampede3 local patch (2026-05-08, hunk N)` | **Match eval-track convention**: `# AI-RES local patch (2026-05-08): <reason>`. No "Stampede3" prefix; no "hunk N" suffix; the strict-count grep is the sole consistency check |
> | Sequence step 1: apply 8-hunk patch | **Sequence step 1: apply 2-hunk perturbation-gate patch** (Hunks 2 and 6 from v4 are the only remaining ones). All other steps unchanged |

---

## 1. Context (unchanged from v4)

A group member on G-819272 will run SFNO-5410 inference on Stampede3
with **her own initial-condition NetCDF**. Use case: arbitrary IC,
arbitrary horizon in days, optional perturbation ensemble. sim52
boundary mode only (BYO deferred per user decision). Permission
scope: G-819272 group-readable.

## 2. Upstream invariants — current state (post eval-track patch)

### 2.1 Yearly-chunk save model — 6 hunks already applied, 2 remaining

Verified 2026-05-08 16:00 against the live tree at
`/work2/.../v2.0/long_inference.py`:

```
$ grep -c 'min(next_year_jan1, self.params.final_datetime)' long_inference.py
4   # ← 4 allocator hunks applied (eval-track LP-003)
$ grep -c 'current_datetime < self.params.final_datetime' long_inference.py
2   # ← 2 continuation hunks applied (eval-track LP-003)
$ grep -c 'epsilon_factor > 0\. and self.params.init_datetime.year == 1' long_inference.py
2   # ← 2 perturbation-gate sites still pristine (user-track LP-004)
$ grep -c 'current_year < self.params.final_datetime.year' long_inference.py
0   # ← no stragglers from the pre-patch year-only continuation
```

So the partial-horizon plumbing (allocators bound by `final_datetime`,
continuation compares datetimes not years) is already production-ready.

**Remaining for user-track**: drop the `init_datetime.year == 1` gate
on the two perturbation sites. Pristine code:

```python
# long_inference.py:558 (async path) and ~line 830 (sync path)
if self.params.epsilon_factor > 0. and self.params.init_datetime.year == 1:
    print('Perturbing ICs...')
    input_surface, input_upper_air = self.perturber(input_surface, input_upper_air)
```

After v5 patch:

```python
# AI-RES local patch (2026-05-08): drop year-1 gate so perturbation fires for any IC year.
if self.params.epsilon_factor > 0.:
    print('Perturbing ICs...')
    input_surface, input_upper_air = self.perturber(input_surface, input_upper_air)
```

NB: the sync-path line number may have drifted slightly relative to
v4's "830" anchor because the eval-track allocator hunk above it
added 2 lines. Confirm by grep before edit; the surrounding
`Perturber` block is the stable anchor.

**Why this is safe for the eval-track** (already validated in
LP-003's "skipped hunks" rationale): eval-track pins
`epsilon_factor=0.0`, so dropping the year-1 condition is a
no-op there. The two paths are orthogonal.

### 2.2 BCS dataloader: yaml key `prediction_duration_days` (unchanged from v4)

Same insight as v4 §2.2. Eval-track plan also relies on this; the
key is now set in two yaml builders:

| Builder | Key value |
|---|---|
| `src/sfno_inference_5410/stampede3_yaml_override.py` (eval-track) | `(K+1) * 6 / 24` (= 15.25 for K=60) |
| `src/sfno_inference_5410/user_inference.py` (user-track, NEW) | `horizon_days` (user-supplied) |

No upstream patch needed.

### 2.3 `_get_inference_duration` is dead (unchanged from v4)

Still zero callers. Still no patch.

### 2.4 ensemble_inference_hours (unchanged from v4)

Same analysis as v4 §2.4. User-track caps at `min(8784, horizon_days*24)`.
Eval-track sets `(K+1)*6` (= 366 for K=60). Both fall inside the
"safe upstream sizing" envelope.

### 2.5 Boundary year decoupling (unchanged from v4 §2.5)

### 2.6 Environment (unchanged from v4 §2.6)

## 3. Design

### 3.1 Upstream patch: 2 hunks in `long_inference.py` (perturbation gates only)

The 6 partial-horizon hunks are already applied as part of LP-003
(eval-track). User-track adds:

**Hunk 2 (v4 numbering retained for clarity)** — line 558 (async path,
perturbation gate):
```diff
-                    if self.params.epsilon_factor > 0. and self.params.init_datetime.year == 1:
+                    # AI-RES local patch (2026-05-08): drop year-1 gate so perturbation fires for any IC year.
+                    if self.params.epsilon_factor > 0.:
                         print('Perturbing ICs...')
                         input_surface, input_upper_air = self.perturber(input_surface, input_upper_air)
```

**Hunk 6** — same change at the sync-path mirror site (line ~830 ±2,
confirm by grep). Same diff, no other context change.

**Marker convention**: matches LP-003 exactly — `# AI-RES local patch
(2026-05-08): <reason>`. The strict-count grep below is the sole
consistency check; no "hunk N" suffix is required because the count
of perturbation-gate markers (described below) uniquely identifies
the user-track entries.

**Patch tracking**: Append `## LP-004 — Perturbation gate (user-track,
arbitrary IC year)` to `docs/2026-05-04_makani_local_patches.md`
following the same section template as LP-003 (Why / Diff /
Verification / Backup / Re-application). Strict-count verification:

```bash
GATE_PRE=$(grep -c 'epsilon_factor > 0\. and self.params.init_datetime.year == 1' \
    /work2/.../v2.0/long_inference.py)
GATE_POST=$(grep -c '# AI-RES local patch (2026-05-08): drop year-1 gate' \
    /work2/.../v2.0/long_inference.py)
[[ "$GATE_PRE" -eq 0 && "$GATE_POST" -eq 2 ]] || \
    echo "perturbation-gate patch incomplete: pre=$GATE_PRE (expect 0) post=$GATE_POST (expect 2)"
```

**Drop**: delete `docs/2026-05-08_panguweather_local_patches.md`
(v4 artifact). The makani ledger
(`docs/2026-05-04_makani_local_patches.md`) is now the single source
of truth for upstream patches. No tombstone in this repo — `git log`
is the audit trail.

### 3.2 Module: `src/sfno_inference_5410/user_inference.py` (rewrite, unchanged from v4 §3.2)

Surface and override body identical to v4 §3.2. The `prediction_duration_days`
and `ensemble_inference_hours = min(8784, horizon_days * 24)` lines are
retained verbatim. Add a header note that the eval-track yaml override
also sets `prediction_duration_days` (under a different key derivation)
so the two paths are independent yaml builders, not call-graph
neighbors.

### 3.3 Module: `src/sfno_inference_5410/preflight.py` (MODIFY/EXTEND)

This module already exists from the eval-track explicit-K work; it
exports `assert_upstream_patched` (4 allocator + 2 continuation
strict counts) plus K-related helpers, and is depended on by
`scripts/eval_inference_5410.py` and
`tests/sfno_inference_5410/test_upstream_patch_present.py`. **Do
not remove or rename any existing symbol.** v5 adds to it.

v5 additions on top of v4 §3.3:

1. **Keep `assert_upstream_patched` unchanged** — it is the eval-track
   contract for the 6 LP-003 markers, and both
   `scripts/eval_inference_5410.py` and
   `tests/sfno_inference_5410/test_upstream_patch_present.py` already
   import it. **Add `assert_perturbation_gate_patched` beside it and
   export both.**

   ```python
   def assert_perturbation_gate_patched(upstream_long_inference_path: Path) -> None:
       """Verify the year-1 perturbation gate is dropped (LP-004, Hunks 2 + 6).

       Strict counts:
         - 0 occurrences of pristine 'epsilon_factor > 0. and self.params.init_datetime.year == 1'
         - 2 occurrences of '# AI-RES local patch (2026-05-08): drop year-1 gate'
       """
   ```

   Called only when `epsilon_factor > 0` (deterministic runs don't
   need this gate dropped). The user-track CLI invokes both
   `assert_upstream_patched` (always — covers LP-003) and
   `assert_perturbation_gate_patched` (only when
   `epsilon_factor > 0` — covers LP-004).

2. **6-hour alignment check** — unchanged from v4 §3.3 (Codex
   round-4 required edit). Still pre-scaffold.

3. **Generated-yaml `prediction_duration_days` check** — unchanged
   from v4 §3.3.

### 3.4 Module: `scripts/run_sfno_5410_inference.py` (unchanged from v4 §3.4)

### 3.5 CLI flags (unchanged from v4 §3.5)

### 3.6 SLURM template (unchanged from v4 §3.6)

### 3.7 Tests

| File | Status | Scope |
|---|---|---|
| `tests/sfno_inference_5410/test_upstream_patch_present.py` | **EXTEND existing** | Add a third test `test_perturbation_gate_strict_counts` checking 0 pre-patch + 2 post-patch perturbation markers. Update module docstring to describe all 8 hunks (6 + 2). |
| `tests/sfno_inference_5410/test_user_inference.py` | **NEW** | Tier-1 yaml regression (incl. `prediction_duration_days == horizon_days`, `ensemble_inference_hours == min(8784, horizon_days*24)`, perturbation_type only when `epsilon_factor>0`) |
| `tests/sfno_inference_5410/test_preflight.py` | **NEW** | Each preflight helper, including `assert_perturbation_gate_patched` (positive + negative cases via `tmp_path`-staged fake source) |

### 3.8 User guide (unchanged from v4 §3.8)

### 3.9 Permissions (unchanged from v4 §3.9)

## 4. File list

| File | Status | Purpose |
|---|---|---|
| `/work2/.../v2.0/long_inference.py` | **PATCH** (2 hunks) | drop year-1 perturbation gate (Hunks 2 + 6 of original v4 numbering) |
| `docs/2026-05-04_makani_local_patches.md` | **APPEND `## LP-004` section** | track the 2 user-track hunks alongside LP-003 (the 6 eval-track hunks) |
| `docs/2026-05-08_panguweather_local_patches.md` | **DELETE** | obsolete v4 artifact; makani ledger is the single source of truth |
| `src/sfno_inference_5410/user_inference.py` | **REWRITE** | yaml builder (v1 stale) |
| `src/sfno_inference_5410/preflight.py` | **MODIFY/EXTEND** | already exists from eval-track explicit-K work (exports `assert_upstream_patched` + K helpers used by `scripts/eval_inference_5410.py` and `test_upstream_patch_present.py`). v5 adds `assert_perturbation_gate_patched` and the 6-hour alignment check; keeps every existing symbol. |
| `scripts/run_sfno_5410_inference.py` | **NEW** | user-facing CLI |
| `scripts/submit_sfno_5410_user_inference.slurm` | **NEW** | SLURM template |
| `tests/sfno_inference_5410/test_upstream_patch_present.py` | **EXTEND** | already exists (LP-003); add perturbation-gate strict-count test |
| `tests/sfno_inference_5410/test_user_inference.py` | **NEW** | yaml regression |
| `tests/sfno_inference_5410/test_preflight.py` | **NEW** | preflight units |
| `docs/2026-05-08_sfno_5410_external_user_guide.md` | **NEW** | the deliverable |

**Not modified:**
- `src/sfno_inference_5410/stampede3_yaml_override.py` — already updated by eval-track (K threading + `prediction_duration_days`); user-track does not touch it
- `scripts/{eval_inference_5410.py, build_5410_yaml_override.py}` — already updated by eval-track
- The eval SLURMs — already updated by eval-track
- `scripts/preflight.py` (training preflight; distinct from 5410)
- Upstream `PanguWeather/v2.0/` files OTHER than `long_inference.py`

## 5. Verification plan

Order: eval-track work must be on `main` (or merged into the working
branch) **before** v5 implementation. Confirm by checking that the 6
LP-003 markers grep correctly (§2.1 commands).

1. **Static patch presence** —
   `pytest tests/sfno_inference_5410/test_upstream_patch_present.py -q`.
   Expects: 4 alloc + 2 cont (LP-003) + 0 pre-patch perturbation + 2
   post-patch perturbation markers.
2. **Unit tests** — `pytest tests/sfno_inference_5410/ -q`:
   - existing eval Tier 1, 2, 3 + LP-003 patch test (pass — eval-track work)
   - new test_user_inference, test_preflight (login-node, <30s)
3. **Dry-run** — `scripts/run_sfno_5410_inference.py --dry-run` for both
   deterministic (`horizon_days=30`) and ensemble (`horizon_days=365,
   num_members=4`); confirm yaml + scaffold + manifest.
4. **Smoke deterministic on H100, 1-year** — IC = sim52 `121_0000.nc`,
   `--horizon-days 365`. Expected: `*_y0121.nc` with 1460 timesteps;
   bit-exact match (modulo logging) to existing eval-track smoke
   **with `epsilon_factor=0`** (so the only thing the v5 patch
   affects — perturbation — is unreached).
5. **Smoke deterministic on H100, sub-year** — IC = sim52 `121_0000.nc`,
   `--horizon-days 30`. Expected: `*_y0121.nc` with 120 timesteps; no
   IndexError. (This is also the eval-track K=60 smoke shape modulo
   horizon length — confirms LP-003 + LP-004 cohabit cleanly.)
6. **Smoke deterministic on H100, partial-final-year** — IC dated
   Dec 15 0125 (build via clipping a sim52 NC), `--horizon-days 30`.
   Expected: 2 NCs (`*_y0125.nc` with 68 timesteps, `*_y0126.nc` with
   52 timesteps); confirms LP-003 (already merged) handles
   year-rollover with partial final chunk. v5 doesn't change this
   behavior; the smoke is a regression check.
7. **Smoke ensemble on H100** — IC dated 0125-06-15, `--horizon-days 365,
   --num-members 4, --epsilon-factor 1e-3, --perturbation-type gaussian_noise`.
   Expected:
   - 4 NCs `*_member000_y0125.nc` … `*_member003_y0125.nc`
   - filenames in user's calendar (confirms LP-004 dropped year-1 gate)
   - members differ from each other
8. **Permission verification** — apply chmod, log before/after stat.
9. **Doc walkthrough** — read user guide cold, follow each step.

## 6. Future work (out of scope for this iteration)

(Unchanged from v4 §6.)

## 7. Risks (revised for v5)

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| 1 | Eval-track LP-003 reverted before user-track LP-004 lands | high | `test_upstream_patch_present.py` would catch (4 alloc / 2 cont counts go to 0); coordinate sequencing in §10 |
| 2 | LP-004 line numbers shifted from v4's "558, 830" anchors due to LP-003 edits | low | Use `grep` for the pristine `init_datetime.year == 1` anchor instead of line numbers; sync-path now ≈ line 832 |
| 3 | Future upstream resync overwrites both LP-003 and LP-004 | high if resync | Both ledgered in `docs/2026-05-04_makani_local_patches.md` with strict-count verification; resync runbook applies them in order |
| 4 | Smoke 4 (1-year deterministic) accidentally exercises the perturbation gate | low | `epsilon_factor=0` in that smoke; the new gate code is unreachable |
| 5 | `prediction_duration_days` interacts oddly with the IC dataset | low | Eval-track Tier-2 already covers IC dataset construction with `prediction_duration_days` set |
| 6 | Multi-year leap+non-leap span template mismatch | medium | Per-step preflight (unchanged from v4) |
| 7 | Output dir collision | low | Pre-scaffold freshness (unchanged) |
| 8 | $SCRATCH purge | low | Document |
| 9 | Permission widening exposes more than needed | low | Path-specific |

## 8. Codex review answers carried over from v4

(All five v4 round-4 Codex answers still apply — they were about
patch hoisting safety, `prediction_duration_days`, capped
`ensemble_inference_hours`, partial-final-year time coords, and the
8-hunk patch test. v5 narrows the user-track patch scope to 2 hunks
but does not change any of those answers.)

## 9. Diff vs v4 — explicit list of plan-content changes

To make Codex review minimal:

1. **§ Header table** — added.
2. **§2.1** — replaced "8 hunks total" claim with a current-state grep
   table; user-track now owns 2.
3. **§2.2** — added note about two yaml builders writing the same key.
4. **§3.1** — collapsed from 8 hunks to 2; removed Hunks 1, 3-5, 7-8;
   updated marker convention.
5. **§3.3** — kept `assert_upstream_patched` unchanged (still
   imported by `scripts/eval_inference_5410.py` and
   `test_upstream_patch_present.py`); added
   `assert_perturbation_gate_patched` beside it. Module marked
   MODIFY/EXTEND, not NEW (eval-track already created it).
6. **§3.7** — `test_upstream_patch_present.py` is now EXTEND, not NEW.
7. **§4 file list** — updated PATCH count (2), added DELETE for
   panguweather ledger, added EXTEND for the test.
8. **§5 verification** — added prereq "eval-track on `main` first";
   smoke 4 explicitly noted as `epsilon_factor=0`.
9. **§7 risks** — replaced v4's risk #1 ("8-hunk patch breaks on
   resync") with finer-grained risks #1-#3 about the LP-003 / LP-004
   ordering.
10. **§10 sequence** — see below.

## 10. Sequence of work

Prereq: confirm eval-track LP-003 is applied to upstream
`long_inference.py` (4 allocator + 2 continuation markers, 0
pre-patch year-only continuation markers). If it has been reverted
or if a resync has dropped it, halt and re-coordinate.

1. Apply 2-hunk perturbation-gate patch (Hunks 2 + 6 from v4
   numbering, now LP-004) to `/work2/.../v2.0/long_inference.py`.
2. Append `## LP-004 — Perturbation gate (user-track, arbitrary IC
   year)` section to `docs/2026-05-04_makani_local_patches.md`.
3. Delete `docs/2026-05-08_panguweather_local_patches.md`.
4. Extend `tests/sfno_inference_5410/test_upstream_patch_present.py`
   with the perturbation-gate strict-count test; run it.
5. Rewrite `src/sfno_inference_5410/user_inference.py` per §3.2.
6. Implement `src/sfno_inference_5410/preflight.py` per §3.3
   (including `assert_perturbation_gate_patched`).
7. Implement `tests/sfno_inference_5410/test_user_inference.py` and
   `test_preflight.py`; run all 5410 tests.
8. Implement `scripts/run_sfno_5410_inference.py` per §3.4-§3.5.
9. Verify §5.3 dry-run.
10. Implement SLURM template §3.6.
11. Apply permissions per §3.9; log.
12. Run §5.4-§5.7 smokes (1-year, sub-year, partial-final-year, ensemble).
13. Write user guide §3.8 cold-walkthrough.
14. Hand to user.

## 11. Open questions for Codex round-5 review

1. Is the v5 split — eval-track owns LP-003 (6 hunks, K-driven),
   user-track owns LP-004 (2 hunks, perturbation-gate) — preferable
   to a consolidated single-patch design? v5 picks split because
   LP-003 is already merged and they're orthogonal; revisit if
   Codex sees a reason to consolidate.
2. Is appending LP-004 to the existing makani ledger preferable to
   keeping `docs/2026-05-08_panguweather_local_patches.md` as a
   separate source of truth? v5 picks consolidation; the alternative
   adds a search surface for future-you.
3. Should `assert_perturbation_gate_patched` live in
   `src/sfno_inference_5410/preflight.py` (user-track) or be moved
   into a shared helper alongside `assert_upstream_patched` (which
   eval-track placed there)? v5 puts both in
   `preflight.py` to keep the patch-presence checks colocated;
   the only constraint is that both helpers stay grep-targetable
   from the strict-count test.
