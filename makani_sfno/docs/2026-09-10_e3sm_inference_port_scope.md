# E3SM inference port — why it is needed, and exactly what it takes

Scoped 2026-09-10 by reading the code and the packed data, not by estimate.
Every claim below carries its evidence; anything unverified is marked.

Companion: `2026-09-10_lagged_ensemble_design.md` (what the port is *for*).

> **Headline: the port is much smaller than
> `polaris_makani_analysis_ensemble_handoff.md` implies, and it is smaller in a
> specific way.** That document scopes "four localized changes plus a dataset
> swap". The dataset swap is a **no-op** — the dataset layer is already generic
> and already runs on E3SM. What that document does *not* name (a calendar/
> anchor gap) is the only item requiring a decision.

---

## PART 1 — WHY WE HAVE TO CREATE A PORT AT ALL

Four independent reasons. The first is the load-bearing one.

### 1.1 Stock makani's inference loop has no slot for exogenous covariates

Our recurrence is `x_{t+1} = f(x_t ⊕ u_t)` — 7 exogenous forcing channels that
must be re-supplied from the dataset at **every** rollout step. Two of them
(`sst`, `solin`) vary in time, so carrying stale values means running the model
on inputs it was never trained to see.

Stock's rollout cannot supply them. **Verified 2026-09-10 by reading
`makani/utils/inference/inferencer.py`:**

- `:660` — `self.preprocessor.cache_unpredicted_features(None, None, inpz, tarz)`.
  The first two arguments are **hardcoded `None`**; only `inpz`/`tarz`
  (zenith angle) are passed.
- `:649-653` — the per-step token unpacks as `(tar, tarz_raw, ttar)` or
  `(tar, ttar)`. There are exactly two optional extras — **zenith and
  timestamp** — and no slot for a general covariate tensor.
- Contrast the *training* path, `deterministic_trainer.py:686`:
  `inp, tar = self.preprocessor.cache_unpredicted_features(*gdata)` — the full
  data tuple is forwarded, which for our fork carries `inp_forcing` /
  `tar_forcing`.

Our `PlasimForcingDataset` returns **four** tensors (`inp_state`, `tar`,
`inp_forcing`, `tar_forcing`). Stock's inference unpacking cannot consume them.

⚠ It would not crash. It would roll forward on whatever the preprocessor last
cached and produce confident, physically-wrong output. That is why the fork
hard-fails instead: `plasim_trainer.py:131`, `assert mode != "inference"`.

### 1.2 Input and output channel counts differ

`N_in_channels=107`, `N_out_channels=101`. Stock's `MultifilesDataset` assumes
`in_channels == out_channels` and a single `/fields` HDF5 key (recorded at
`plasim_forcing_dataset.py:37-39`, which is why the fork subclasses it). The
101st output is a **diagnostic head** (`PRECT`) that is supervised but must be
**dropped before feedback** — 101 predicted, 100 fed back. Stock has no notion
of an output channel that is not part of the state.

### 1.3 The fork's own inference module is pinned to the *other* dataset

`src/sfno_inference/` exists and handles 1.1 and 1.2 correctly, but
`load_eval_params` asserts the PLaSim contract with **literal integers**
(`checkpoint_loader.py:74-82`: 58 / 53 / 52 / 1 / 6). On an E3SM config it
aborts before the model is built. This is the bulk of the mechanical work in
Part 2.

### 1.4 Scheduler and cluster

`scripts/submit_eval.sh` is SLURM on Stampede3. We are PBS on Polaris. Per
CLAUDE.md #7 this means adding a **sibling** script, never editing the SLURM
one in place.

### What is NOT a reason

- ❌ *"The inference gate blocks us."* It does not. `eval_inference.py:264`
  calls `_plasim_get_dataloader(..., mode="eval")`, so the
  `assert mode != "inference"` at `plasim_trainer.py:131` never fires. That
  gate blocks **stock makani's `Inferencer`**, not the fork's own path.
- ❌ *"We need a new dataset class."* We do not — see §2.0.

---

## PART 2 — WHAT NEEDS TO BE DONE

### 2.0 Verified as already working — do NOT redo these

| # | claim | evidence |
|---|---|---|
| 1 | **`PlasimForcingDataset` is already generic** | `plasim_trainer.py:163` passes `n_forcing_channels=params.get("n_forcing_channels", 6)`; E3SM's config sets **7**. This is the class E3SM *training* already runs on. The `52`/`6` in its docstrings are stale prose, not code |
| 2 | **`nc_writer` is shape-driven** | `:92-93` derives `n_chan` / `n_chan_ic` from tensor shapes. `channel_ic = channel_names[:n_chan_ic]` takes the first 100 of 101 — correct, because `PRECT` is last |
| 3 | **Lat orientation guard passes** | pack `metadata/data.json` lat = `[89.5, 88.5, …]`, strictly descending; `nc_writer.py:111` requires exactly that |
| 4 | **Channel-name cross-check passes** | pack `channel_state` (100) ‖ `channel_diagnostic` (`['PRECT']`) = 101, matching the run config's `channel_names` position for position, so `_resolve_and_check_channel_names` is satisfied |
| 5 | **`rollout_one_ic`'s body is channel-generic** | `rollout_driver.py:163-164` reads `n_state` / `n_out` from `eval_params`; 52/53 are defaults, not hardcodes |
| 6 | **Pack layout keys are identical** | `fields_state` / `fields_diagnostic` / `forcing`, same as PLaSim |
| 7 | **A test split exists** | `test/2048.h5`, `test/2049.h5`, 1460 samples each |

### 2.1 The mechanical changes

| # | file:line | change | size |
|---|---|---|---|
| **A** | `checkpoint_loader.py:74-82` | Five asserts on literal `58 / 53 / 52 / 1 / 6`. Replace with values read from config, **keeping the internal-consistency check** — `n_state + n_forcing == N_in` and `n_state + n_diag == N_out`. That is the part with real value; the literals are not | ~10 lines |
| **B** | `checkpoint_loader.py:237` | `assert wrapper.model.out_chans == 53` → `eval_params.N_out_channels` | 1 line |
| **C** | `rollout_driver.py:83` + `eval_inference.py:251` | `_load_run_norm_stats(..., n_out: int = 53)`; the caller omits `n_out`, so a 101-channel stats file is checked against 53. Fails loudly (shape check at `:95`), but pass the real value | 1 line |
| **D** | `rollout_driver.py:281-302` | 🐛 **Real bug — see §2.2** | ~10 lines |
| **E** | anchor / calendar | **Decision required — see §2.3** | ? |
| **F** | `scripts/submit_eval.sh` | Polaris PBS sibling. Copy the env-bootstrap block **verbatim** from `polaris/polaris_makani_env_probe.pbs` | 1 script |

A-D total roughly 25 lines and are mechanical.

⚠ **Constraint on every one of them:** `src/sfno_inference/` is inside a
`git subtree` **and** is shared with the Stampede3 `eval-sfno-own` skill. Each
fix must **generalize**, never special-case E3SM (CLAUDE.md #5). Concretely:
derive from config, do not branch on dataset name.

### 2.2 Change D — the silent-wrong-physics bug

`_extract_truth_sic` guards on the forcing-stats **length** (`< 6`) and then
reads **positional index 5**.

E3SM's forcing vector, from the run's `config.json`:

```
[0] lsm   [1] topo   [2] glacier   [3] natveg   [4] sst   [5] solin   [6] ice
```

Length 7 passes the `< 6` guard, index 5 is **`solin`**, and sea ice is at
index **6**. The function therefore returns solar insolation under a confident
`truth_sic` label, which `nc_writer` writes with
`attrs["units"] = "fraction"` and a description about ice masking.

**Fix:** look the channel up **by name** against `channel_forcing`, and return
`None` (with a warning) when the name is absent. Name lookup also fixes the
PLaSim path, where positional index 5 happens to be correct today only by
coincidence of ordering — which is what makes this a generalizing fix rather
than a special case.

### 2.3 Change E — the anchor gap, and the calendar under it

**Symptom.** `_resolve_ic_provenance` (`rollout_driver.py:347`) reads
`f.attrs["plasim_time_units"]`. The E3SM converter **never writes it** —
verified, zero occurrences in `polaris/convert_e3sm_to_makani_alldata.py`. So
`anchor = ""` and `nc_writer._parse_anchor_to_datetime64` raises
`ValueError: unparseable anchor: ''`.

✅ That failure is **loud**, so there is no silent-wrong risk here.

**The real issue underneath.** E3SM as packed uses a **noleap (365-day)
calendar**. Measured, not assumed: `.pack_logs/test.log` reports `T=1460` for
**2048, a leap year** — 365 × 4, not 366 × 4. And `time_plasim` is a
**split-cumulative** day count from the split's first year (the `offset`
accumulates across years at `convert_e3sm_to_makani_alldata.py:487,491`), not a
within-year one.

⇒ Anchoring at `{first_year}-01-01` and adding those days onto a proleptic-
Gregorian `numpy.datetime64` drifts **one day per leap year crossed**. In the
two-year test split, `2049.h5`'s dates would already be off by one.

**Two ways out:**

| option | cost | consequence |
|---|---|---|
| **Label leads by step index / hours-since-IC; drop absolute dates** | trivial | The natural coordinate for a lagged ensemble anyway — members are defined by lead offset, not calendar date. Downstream scoring that keys on lead time is unaffected |
| **Carry the calendar properly** | converter + writer + downstream | Write `plasim_time_units` and a `calendar: noleap` attribute into the pack; use `cftime` in the writer. Correct dated output, but touches anything expecting `datetime64` |

⚠ This has **zero effect on the model or on any metric**. It is purely what
label gets written on the output files.

**Recommendation: take step-index labelling** unless calendar-dated NetCDF is
required by a downstream consumer.

### 2.4 New work for the lagged ensemble (not part of the port)

→ `2026-09-10_lagged_ensemble_design.md` §5. Summarised: a stagger-`d` offset
generator, member alignment by absolute target index, and a weighted
combination step.

---

## 3. Order, and what gates what

1. Per-lead metrics ✅ (commit `652e9505`, `PERLEAD_METRICS_OK`, job 7602739)
2. Re-score base + C1 at `K=20` 🔵 (jobs 7603089 / 7603090) → yields `sigma(k)`
3. Read the shape of `err(k)` → decides whether the ensemble direction is sound
4. Port changes A-D + F (needed for *any* inference, lagged or not)
5. Decision E
6. Lagged-ensemble driver

Steps 4 and 5 do not depend on step 3 and can proceed in parallel with it. Step
6 does depend on step 3.
