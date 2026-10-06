# makani B: dry-air findings and the ACE2 corrector gap (2026-10-05)

A discussion handoff. It covers what the B-lineage 5-year runs actually show, two
corrections to readings that were circulating, and why our "ACE2 dry-air fix" is not
ACE2's conservation scheme. Evidence lives in CHANGELOG.md (2026-10-02/03/04/05
`(makani)` entries) and the run directories cited below; this file only links them.

> **⚠ CORRECTION (2026-10-06): the DRYAIR numbers below are in Pa, not hPa.**
> `readout.log`'s `drift PS (phys, lead L − lead 1)` is in Pa (`lead1=98565`). B22's
> −6.8 hPa (−680 Pa) and B24's +16 hPa (+1600 Pa) were converted; the DRYAIR24
> "−9.5 to −25.6 hPa (med ≈ −21)" and DRYAIR01 "−78 → −502 → −1496 hPa" were copied
> raw. Correct values: **DRYAIR24 −0.10 to −0.26 hPa (med ≈ −0.21)**, ~30× smaller than
> B22 and ~75× smaller than B24, consistent with its 1-yr screen (−0.145 hPa, dry drift
> +0.000, `climate_screen_7711659`); DRYAIR01 −0.8 → −5 → −15 hPa before truncating.
> So TL;DR 2 is reversed and TL;DR 3 / §2 fall away: ΔTMQ ≈ −21 Pa / g ≈ −2 kg/m², not
> −214. Confirmation from the member NetCDFs: job 7719173
> (`polaris/polaris_dryair_budget_readout.pbs`, `DRYAIR_BUDGET_OK`). §1d/§3 still hold:
> survival is not attributable to dry-air, and the port is still 1 of ACE2's 3 steps.
>
> **Measured (7719173, 2026-10-06):** the formula `PS − g·TMQ` is correct. DRYAIR24 has dDRY = 0.00
> and dPS −0.14 to −0.24 hPa = g·ΔTMQ (TMQ ~27 → ~25 kg/m², global mean never negative). B22/B24's
> ±6–17 hPa is almost entirely **dry-mass** drift (dDRY ≈ dPS). The soil-fix screen (7719183) drifts
> −9 to −18 hPa in 1 yr, also all dry. Detail: CHANGELOG 2026-10-06.

**TL;DR**
1. DRYAIR24 survives 5 years (8/8), but so do plain B22/B24. Survival is not
   attributable to dry-air training.
2. DRYAIR24's PS drift (median ≈ −21 hPa) is *larger* than B22's (−6.8). Neither is
   physically good: real global-mean PS varies by well under 1 hPa.
3. With the dry-air fix on, PS can only drift if column water drifts. −21 hPa implies
   global-mean TMQ fell by ≈ 214 kg/m², which would make it strongly negative. That is
   **inferred, not measured**: check it first (§4).
4. We ported **one of ACE2's three corrector steps**. ACE2 forces water positive first
   and closes the moisture budget afterwards. Without those two steps, the dry-air step
   turns water errors into PS drift (§3).

---

## 1. Progress: the B lineage and what has run

### 1a. Checkpoint family tree

```
A   prod1n_b32_sgdr                single-step production (best = epoch 243)
└─► B   nf4_prod_b16_r1            A fine-tuned on the 4-step-rollout objective
    ├── B_e01  best_ckpt_mp0.tar           B after 1 epoch (lowest validation loss)
    ├── B_e21…B_e24  ckpt_mp0_v0…v3        "B22", "B24" = B after 22 / 24 epochs
    ├─► fs_anneal_dryair_nf4_b16_rb01      job 7709268, done. Restarted from B_e01
    │     ├── DRYAIR01 = its epoch 1         CONSERVE_DRY_AIR=1 in training
    │     └── DRYAIR24 = its epoch 24 (= its best_ckpt)
    └─► fs_anneal_soilfix_nf4_b16_rb01     job 7715005, RUNNING on capacity
          (same recipe, SOILFIX=1 instead of dry-air; started 2026-10-05 17:40)
```

- **Epoch numbers count from the start of each run.** "01" in B01 and in DRYAIR01 are
  different checkpoints; DRYAIR01 is a descendant of B01.
- **Shared fine-tune recipe** (`makani_sfno/polaris/submit_finetune_stability_arm.sh`):
  global batch 16, 2 nodes, LR 4e-4, cosine annealing to 1e-6, `SCHED_TMAX=22`,
  1-epoch warmup with `LR_START=0.01`.
- **Epoch 1 runs at a constant 4e-6** because warmup steps once per epoch (CHANGELOG
  ~L1099). DRYAIR01 is therefore barely moved from B_e01. An earlier verbal claim that
  it was trained "at a high LR" was wrong.

### 1b. The 5-year protocol (an IC ensemble)

`makani_sfno/polaris/polaris_climate_run.pbs`, jesswan's protocol:

- 8 members per checkpoint. Member `i` starts at 2044 frame `1092 + 16·i` (Oct 1, 5, …,
  29, every 4 days) and runs `7667 − 16·i` steps, so all members end on 2049-12-31 18:00.
- Scoring starts 2045-01-01 (Oct–Dec spin-up discarded).
- Deterministic: no noise, no seeds. Members differ only in start date. All arms use
  the same 8 dates, so member `i` is comparable across arms.
- Forcing is read from data by valid time (`climate_driver.read_forcing_chunk`), so all
  members share the same forcing. Spread measures internal variability only.
- **Results are used as replicates, not as a forecast ensemble.** We report counts and
  ranges; no ensemble mean, CRPS or spread–skill ratio has been computed.
- The 1-year screens (`polaris_climate_screen.pbs`) use **one** start each, so they are
  not ensembles. The "n=6" in the Step-2 screen means 6 *checkpoints*, not members.

### 1c. 5-year results (all start from B_e01 or are B_e01)

| arm | job | survival | PS drift @ ~5 yr | first >3σ |
|---|---|---|---|---|
| B_e01 plain | 7649597 | 1/8 (survivor −269 hPa); failures 2.5–5.0 yr | — | — |
| B_e01 + dry-air at inference only | 7707597 | 0/8, collapses 2.28–2.48 yr | diverged | 0.79–1.15 yr |
| **B22** plain | 7707597 | **8/8** | −6.1 to −8.8 (med ≈ −6.8) | 1.06–1.43 yr |
| **B24** plain | 7707597 | **8/8** | +13.0 to +17.5 (med ≈ +16) | 0.95–1.37 yr |
| DRYAIR01 | 7711720 | 0/8, collapses 3.88–4.14 yr | diverged | 0.97–1.73 yr |
| **DRYAIR24** | 7711720 | **8/8** | −9.5 to −25.6 (med ≈ −21) | 0.88–1.47 yr |

Verified from `$MEMBER_ROOT/runs/makani_eval/climate_protocol_7711720/`: both DRYAIR arms
ran with `dry_air_fix=1` **at inference too**, and their `ckpt_epoch` values are 1 and
24 (`member_*.log`). DRYAIR2400's PS trajectory reaches −20 hPa by about 1 yr and then
wanders between −10 and −26 hPa (bounded). DRYAIR01 goes −78 → −502 → −1496 hPa
before truncating.

### 1d. What the results support

- **Supported:** the extra 4-step training after B_e01 is what makes B stable to 5
  years. B22/B24 (no dry-air) and DRYAIR24 (dry-air) all reach 8/8.
- **Supported:** training with dry-air beats applying it only at inference (2.4 yr
  collapse → 3.9–4.1 yr at 1 epoch → 5-yr survival at 24 epochs).
- **Not supported:** that dry-air *caused* the stability, or reduced mass drift.
- **Weak:** the DRYAIR01 vs inference-only gain. Both are ≈ the B_e01 weights plus the
  same inference fix, and chaotic rollouts move failure times under tiny weight changes.
- **The 1-yr screen was misleading:** DRYAIR at 1 yr drifted 60–1500× less than plain B
  (−0.04 to −0.49 hPa), then lost the advantage by 5 yr.
- **Approximate control:** B22/B24 already act as the "no-constraint control"
  (B_e01 + more plain training). The difference is the LR schedule (the dry-air run
  restarted warmup and cosine). **Not verified** whether B's own schedule matches the
  fine-tune recipe.

---

## 2. Is −21 hPa a good sign?

**No.** Global-mean PS is about 985 hPa, and in E3SM it moves well under 1 hPa per year
(water-vapour cycle only). −21 hPa is about 2% of atmospheric mass and 20–40× natural
variability. The only positive is that it is bounded rather than runaway. B22's
−6.8 hPa is also too large.

**It should not be possible with the fix on.** `mass_fix.py` (CHANGELOG ~L1002) shifts PS
each step so that the global mean of **PS − g·TMQ** equals the step input's. With dry
mass pinned, ΔPS = g·ΔTMQ:

> −21 hPa = −2100 Pa ÷ 9.81 ≈ **−214 kg/m²** of global-mean TMQ, against a real global
> mean of about 25 kg/m².

If this holds, DRYAIR24 "conserves dry air" by letting water vapour go unphysical, and
the PS drift is a symptom. A precedent exists: the negativity probe 7650442 found TMQ
cells at −89 to −289 kg/m² (CHANGELOG ~L1079). **Unmeasured:** `readout.log` prints
only PS, Z3, T and TREFHT. The member NetCDFs should contain TMQ.

---

## 3. ACE2 mechanism differences: the corrector

ACE2's atmosphere corrector (`ACE2_retrain/ace_exp/fme/fme/core/corrector/atmosphere.py`
~L182–216; our ACE2 config `ACE2_retrain/config_polaris.yaml:178–190`) runs three steps
**in order, every step**:

| step | ACE2 | our makani `conserve_dry_air` |
|---|---|---|
| 1. force water positive: `specific_total_water_0…7`, `Q2m` | ✅ **first**. The code comment: *"do this step before imposing other conservation correctors, since otherwise it could end up creating violations of those constraints"* | ❌ none. TMQ is unclamped. An opt-in, default-off `NonNegativeConstraint` (PRECT, SOILWATER_10CM, TMQ) exists (CHANGELOG ~L603) but was not used |
| 2. conserve dry air: global mean of PS − water fixed to the input's | ✅ | ✅ `mass_fix.py` |
| 3. `moisture_budget_correction: advection_and_precipitation`: global water changes only by E − P | ✅ | ❌ none |

**Why it matters.** Step 2 only pins the *difference* between PS and water. In ACE2,
steps 1 and 3 anchor water, so PS stays anchored with it. In ours water is free, so any
water drift is converted one-for-one into PS drift. DRYAIR24 therefore tested *ACE2's
dry-air step without its water guards*, not ACE2's conservation scheme.

**Smaller differences**

| | ACE2 | ours |
|---|---|---|
| water term | 8 per-level total-water fields (vapour + condensate) on hybrid `ak/bk` coordinates | single `TMQ` column (vapour only) |
| precision | float32 or float64 (`_dry_air_precision`) | float32 (after the bf16 rounding bug, fix `6c29bf18`) |
| applied | training + inference, every step | training (`CONSERVE_DRY_AIR=1`) and/or inference (`--dry-air-fix`) |

**Porting cost.** Positivity is cheap. The moisture budget may not port:
`ace2_vs_makani_differences.md` ~L169–172 already flags that our 101-channel E3SM
contract is not obviously budget-closable. Whether makani has both precipitation and
evaporation terms to close it is unverified, and it is a jesswan science call.

Broader makani-vs-ACE2 comparison: `ace2_vs_makani_differences.md` (corrector rows ~L71–74)
and `makani_sfno/docs/2026-09-10_ace2_comparison_the_corrector.md`.

---

## 4. Open questions and next steps (for discussion)

1. **Measure TMQ under DRYAIR24.** This tests §2 directly. The existing
   `makani_sfno/polaris/polaris_negativity_probe.pbs` (`DRY_AIR_FIX=on`, 1-yr rollout,
   per-lead TMQ min / %-negative / mean, debug queue) looks sufficient, since DRYAIR24
   reaches −20 hPa within the first year. Alternatively, read TMQ from
   `climate_protocol_7711720/member_DRYAIR24*.nc` in a debug job (no venv Python on login
   nodes). Debug 1-node runs are pre-authorized.
2. **If TMQ is negative:** a faithful port means adding positivity *before* the dry-air
   step (`NonNegativeConstraint` on TMQ), then deciding on a moisture budget. This needs
   a §4 equivalence gate with the flag off, like `DRYAIR_OFF_EQUIV_OK`.
3. **Soil-fix (7715005)** finishes about 05:40 on 2026-10-06. Its screen and 5-yr steps
   are in TODO.md P0. The soil-fix corrector may share the same "no positivity/budget
   guard" gap, so check before reading its result.
4. **Combined dry-air + soil-fix arm:** hold until (a) the soil-fix result and (b) the
   TMQ check. The two flags are independent in the submit script (L83–84), but nobody has
   checked that the correctors compose correctly on the same step.
5. **Clean control:** B22/B24 are approximate. Before spending a node on a matched
   no-constraint run from B_e01, verify B's LR schedule against the fine-tune recipe.
6. **Capacity is max_run 1 per project.** 7715005 holds it, so new long runs either
   chain with `DEPEND=7715005` or go to preemptable. Check with the operator before
   submitting either.

## 5. Sources

- CHANGELOG.md (this branch): 2026-10-05 soil-fix entries (~L145–200); 2026-10-04 Step 2
  (~L302–360); 2026-10-02 Step 1 (~L520–568); dry-air fix definition and bf16 bug
  (~L999–1016); LR warmup (~L1099); negativity probe (~L1059–1079).
- Run dirs: `$MEMBER_ROOT/runs/makani_eval/climate_protocol_{7649597,7707597,7711720}/`
  (`readout.log`, `member_*.log`), `climate_screen_7711659/`.
- Code: `makani_sfno/polaris/polaris_climate_run.pbs`, `submit_finetune_stability_arm.sh`,
  `makani_sfno/src/sfno_inference/climate_driver.py`, `sfno_training/models/mass_fix.py`.
