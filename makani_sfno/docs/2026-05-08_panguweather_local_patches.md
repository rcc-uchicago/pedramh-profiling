# PanguWeather/v2.0 local patches for SFNO-5410 user inference

> **Status: REVIEW-ONLY. Patch NOT yet applied to upstream as of
> 2026-05-08.** This doc presents the proposed diff for review;
> apply only after user approves.
>
> Date: 2026-05-08. Author: zhixingliu (via Claude). Driven by
> approved plan v4 (`docs/2026-05-08_sfno_5410_external_user_inference_plan_v4.md`).

## Target file

```
/work2/11114/zhixingliu/stampede3/SFNO_Climate_Emulator/artifacts/derecho_glade/PanguWeather/v2.0/long_inference.py
```

(Same upstream tree shared by the eval-track and the new
user-inference path.)

## Why patch

Two upstream behaviors block the user-inference use case:

1. **Yearly-chunk save model**: rollouts only save NetCDFs at Jan-1
   year boundaries. Sub-year horizons (e.g. 30 days) and
   partial-final-year horizons (e.g. Dec 15 + 30 days = Jan 14)
   produce empty or truncated output. See plan v4 §2.1 for line-by-
   line evidence.

2. **Year-1 perturbation gate**: `Perturber.add_perturbations()` is
   only applied when `init_datetime.year == 1`. ICs in any other
   year produce silently-deterministic rollouts despite
   `epsilon_factor > 0`.

The eval-track (96-IC, full sim52 year, deterministic) is unaffected
by both: its rollouts are exactly Jan 1 → Jan 1 (so the
yearly-chunk save fires once at the right moment) and it always
runs with `epsilon_factor=0`. So the patches are **semantically
equivalent for the eval-track** and **enable the new user-track**.

## Patch surface — 8 hunks

| # | Line(s) | Path | Purpose |
|---|---|---|---|
| 1 | 554-555 | async, initial allocator | Bound first chunk endpoint by `final_datetime` |
| 2 | 558 | async, perturbation gate | Drop `init_datetime.year == 1` |
| 3 | 720-722 | async, continuation | Continue while `current_datetime < final_datetime` (was: `current_year < final.year`) + hoist `current_datetime = next_output_datetime` |
| 4 | 724-725 | async, year-rollover allocator | Same `min(next_year_jan1, final_datetime)` as Hunk 1 |
| 5 | 834-835 | sync, initial allocator | Mirror of Hunk 1 |
| 6 | 830 | sync, perturbation gate | Mirror of Hunk 2 |
| 7 | 998-1000 | sync, continuation | Mirror of Hunk 3 |
| 8 | 1002-1003 | sync, year-rollover allocator | Mirror of Hunk 4 |

All 8 hunks are necessary; the four-pair symmetry (async vs sync,
initial vs rollover) is required because upstream duplicates the
same logic across two execution paths.

## Unified diff

```diff
--- a/long_inference.py
+++ b/long_inference.py
@@ -550,11 +550,15 @@
                     # Create arrays for first year of output data
                     current_datetime = self.params.init_datetime
                     current_year = self.params.init_datetime.year
                     next_year_offset_hours = current_datetime.hour % self.params.timedelta_hours
                     time_step_in_year = 0
-                    next_output_datetime = self.dataset.datetime_class(current_year+1, 1, 1, hour=next_year_offset_hours,
-                                                                    has_year_zero = self.params.has_year_zero)
+                    # AI-RES-Stampede3 local patch (2026-05-08, hunk 1): bound chunk endpoint by final_datetime
+                    # so partial-final-year rollouts save correctly. See docs/2026-05-08_panguweather_local_patches.md.
+                    next_year_jan1 = self.dataset.datetime_class(current_year+1, 1, 1, hour=next_year_offset_hours,
+                                                                  has_year_zero = self.params.has_year_zero)
+                    next_output_datetime = min(next_year_jan1, self.params.final_datetime)
                     
                     # Perturb initial conditions if using perturbations
-                    if self.params.epsilon_factor > 0. and self.params.init_datetime.year == 1:
+                    # AI-RES-Stampede3 local patch (2026-05-08, hunk 2): drop year-1 gate so perturbation fires for any IC year.
+                    if self.params.epsilon_factor > 0.:
                         print('Perturbing ICs...')
                         input_surface, input_upper_air = self.perturber(input_surface, input_upper_air)
@@ -716,16 +720,18 @@
                                 await save_queue.put((deepcopy(ensemble_datasets), particle_idxs.numpy(), ensemble_start, ensemble_end))
                                 await asyncio.sleep(0)
                                         
                             
                             current_year += 1
-                            # If this was not the final year
-                            if current_year < self.params.final_datetime.year:
-                                # Get the number of time steps in the next year
-                                current_datetime = next_output_datetime
+                            # AI-RES-Stampede3 local patch (2026-05-08, hunk 3): continuation by datetime, not year.
+                            # Hoists current_datetime assignment out of the if-body (semantically identical;
+                            # current_datetime isn't read after the rollout loop exits).
+                            current_datetime = next_output_datetime
+                            if current_datetime < self.params.final_datetime:
+                                # Get the number of time steps in the next chunk (may be partial final year)
                                 next_year_offset_hours = current_datetime.hour % self.params.timedelta_hours
-                                next_output_datetime = self.dataset.datetime_class(current_year+1, 1, 1, hour=next_year_offset_hours,
-                                                                                has_year_zero = self.params.has_year_zero)
+                                # AI-RES-Stampede3 local patch (2026-05-08, hunk 4): same final_datetime cap as hunk 1.
+                                next_year_jan1 = self.dataset.datetime_class(current_year+1, 1, 1, hour=next_year_offset_hours,
+                                                                              has_year_zero = self.params.has_year_zero)
+                                next_output_datetime = min(next_year_jan1, self.params.final_datetime)
                                 output_inference_steps = int((next_output_datetime - current_datetime).total_seconds() // 3600 / self.params.timedelta_hours)
                                 
                                 # Create new numpy arrays for the next year of data
@@ -826,15 +832,18 @@
                     current_datetime = self.params.init_datetime
                     current_year = self.params.init_datetime.year
                     next_year_offset_hours = current_datetime.hour % self.params.timedelta_hours
                     time_step_in_year = 0
                     
                     # Perturb initial conditions if using perturbations
-                    if self.params.epsilon_factor > 0. and self.params.init_datetime.year == 1:
+                    # AI-RES-Stampede3 local patch (2026-05-08, hunk 6): drop year-1 gate (mirror of hunk 2).
+                    if self.params.epsilon_factor > 0.:
                         print('Perturbing ICs...')
                         input_surface, input_upper_air = self.perturber(input_surface, input_upper_air)
                         
-                    next_output_datetime = self.dataset.datetime_class(current_year+1, 1, 1, hour=next_year_offset_hours,
-                                                                    has_year_zero = self.params.has_year_zero)
+                    # AI-RES-Stampede3 local patch (2026-05-08, hunk 5): bound chunk endpoint by final_datetime (mirror of hunk 1).
+                    next_year_jan1 = self.dataset.datetime_class(current_year+1, 1, 1, hour=next_year_offset_hours,
+                                                                  has_year_zero = self.params.has_year_zero)
+                    next_output_datetime = min(next_year_jan1, self.params.final_datetime)
                     output_inference_steps = int((next_output_datetime - current_datetime).total_seconds() // 3600 / self.params.timedelta_hours)
                     output_surface = np.zeros((input_surface.shape[0], output_inference_steps,
                                                     input_surface.shape[1], input_surface.shape[2], input_surface.shape[3]),
@@ -994,16 +1003,18 @@
                                 self.save_prediction(deepcopy(ensemble_datasets), particle_idxs.numpy(), ensemble_start, ensemble_end)
                                 save_time += time.time() - save_start
                                         
                             
                             current_year += 1
-                            # If this was not the final year
-                            if current_year < self.params.final_datetime.year:
-                                # Get the number of time steps in the next year
-                                current_datetime = next_output_datetime
+                            # AI-RES-Stampede3 local patch (2026-05-08, hunk 7): continuation by datetime (mirror of hunk 3).
+                            current_datetime = next_output_datetime
+                            if current_datetime < self.params.final_datetime:
+                                # Get the number of time steps in the next chunk (may be partial final year)
                                 next_year_offset_hours = current_datetime.hour % self.params.timedelta_hours
-                                next_output_datetime = self.dataset.datetime_class(current_year+1, 1, 1, hour=next_year_offset_hours,
-                                                                                has_year_zero = self.params.has_year_zero)
+                                # AI-RES-Stampede3 local patch (2026-05-08, hunk 8): same final_datetime cap (mirror of hunk 4).
+                                next_year_jan1 = self.dataset.datetime_class(current_year+1, 1, 1, hour=next_year_offset_hours,
+                                                                              has_year_zero = self.params.has_year_zero)
+                                next_output_datetime = min(next_year_jan1, self.params.final_datetime)
                                 output_inference_steps = int((next_output_datetime - current_datetime).total_seconds() // 3600 / self.params.timedelta_hours)
                                 
                                 # Create new numpy arrays for the next year of data
```

## Hoisting safety analysis (Hunks 3, 7)

The continuation rewrite moves `current_datetime = next_output_datetime`
from line 722 (inside the `if current_year < final.year:` body) to
before the new condition, so the new check `current_datetime <
final_datetime` evaluates the updated value.

**Why this is safe** (Codex round-4 confirmed):

- `current_datetime` is a local variable. After the rollout loop
  exits, it is not read anywhere downstream. Setting it
  unconditionally one extra time at loop-exit has no side effect.
- The original code's first action inside the if-body was the
  assignment we hoisted; all subsequent reads of `current_datetime`
  inside the body see the same value either way.
- The 24h-model branch at lines 743-754 (and the sync mirror at
  1021-1032) reads `current_datetime` indirectly via the
  surrounding rollout state, not via the assignment we hoisted.

## Verification

After applying:

```bash
# 1. Static patch presence (login-node, <1s)
pytest tests/sfno_inference_5410/test_upstream_patch_present.py -v

# 2. Eval-track regression (login-node, <30s)
pytest tests/sfno_inference_5410/ -q

# 3. Sub-year smoke on H100 (per plan v4 §5.4)
sbatch scripts/submit_sfno_5410_user_inference.slurm  # configured for 30-day rollout
```

Expected:
- Test 1 confirms all 8 patch markers present.
- Test 2 confirms eval-track Tier 1, 2, 3 still pass.
- Smoke 3 produces exactly 120 timesteps in `*_y0121.nc`.

## Resync procedure

If a future upstream resync overwrites this file:

1. `test_upstream_patch_present.py` will fail loud, listing missing markers.
2. Re-apply the diff above (the 8 hunks each carry a `AI-RES-Stampede3 local patch (2026-05-08, hunk N)` comment for grep targeting).
3. Re-run verification §1-3.
4. Update this doc's date if the resync introduces context shifts that require diff regeneration.

## Application

To apply the patch:

```bash
# From repo root
cd /home1/11114/zhixingliu/projects/SFNO_Climate_Emulator
# Show what would change
git -C /work2/11114/zhixingliu/stampede3/SFNO_Climate_Emulator/artifacts/derecho_glade/PanguWeather/v2.0 diff long_inference.py  # confirms clean state
# Apply hunks via Edit tool (8 separate edits) or `patch` command if exported as .patch file
```

The Claude wrapper applies the 8 hunks in order via the Edit tool,
each preserving the exact context lines shown in the diff above.

## Files this affects (downstream)

- Eval-track (`scripts/eval_inference_5410.py`,
  `scripts/submit_eval_inference_5410*.slurm`): semantically
  unchanged. The eval rollouts are full Jan-1 → Jan-1 years with
  `epsilon_factor=0`, both of which behave identically pre- and
  post-patch.
- User-track (`scripts/run_sfno_5410_inference.py`, new):
  enabled by these patches.

## Out of scope

- Other upstream files (`utils/perturbation.py`, `utils/data_loader_multifiles.py`, `networks/sfnonet.py`, etc.) are unmodified.
- Eval-track wrapper code (`src/sfno_inference_5410/stampede3_yaml_override.py`, `scripts/eval_inference_5410.py`) is unmodified.
