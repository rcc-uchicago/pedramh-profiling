# HANDOFF — SFNO apples-to-apples: retrain PanguWeather, train ai-rossby, compare

Read this whole file first. Work in
`/eagle/projects/lighthouse-uchicago/members/mehta5/pedramh-profiling`, branch
`fix/tsoi-fill-270` (or a fresh branch off it).

**Goal:** train the **same SFNO architecture** on the **same E3SM data** with the
**same corrected normalization**, once through the **PanguWeather** harness and once
through the **ai-rossby (PhysicsNeMo)** harness, so the two are comparable —
and leave results + checkpoints **readable and writable by the whole group**.

---

## 1. What "apples-to-apples" actually requires

This is the part to get right; everything else is mechanics. The comparison is only
meaningful if these are **identical**, and it is only honest if the ones that
**cannot** be identical are stated in the write-up.

### Must match — verify each, don't assume

| axis | value | how to verify |
|---|---|---|
| architecture | `SphericalFourierNeuralOperatorNet_v2`, `embed_dim 512`, `num_layers 12`, `num_blocks 16`, `spectral_layers 3`, `scale_factor 1`, `instance_norm` | **parameter count must be identical** — see §4 |
| variable set | the 108-field contract, **same order** | `ai_rossby_variable_contract.py --check-artifacts` |
| normalization | the corrected stats (`$PANGU_AUX/data_2015-2050_{mean,std_corr}.nc` and the zarr built from them) | `check_normalization.py` → `NORMALIZATION_OK` |
| data years | train 2015–2044, val 2045–2048 | both configs |
| precision | bf16 | both configs |
| batch / world size | `batch_size 1`, 4 ranks (global batch 4) | both configs |
| seed | same value, set explicitly | Pangu `--global_seed`; ai-rossby `cfg.seed` |
| epochs | same target | `MAX_EPOCHS` |
| hardware | 4× A100-40GB, single node | same queue, ideally same node type |

### Cannot match — disclose these, do not paper over them

1. **The data path differs by construction.** PanguWeather reads the **E3SM HDF5**
   archive directly; ai-rossby reads the **converted Zarr** stores. Same numbers,
   different I/O and different sample-ordering machinery. This is a *harness*
   comparison, not a kernel comparison — say so.
2. **Sample order / sampler semantics differ.** PanguWeather's validation IC logic
   uses a `linspace` endpoint; ai-rossby's sampler is arange-like (hence the
   1-sample 2049 tail store). See handoff-v2 §7.7.
3. **`checkpointing: 3`.** In SFNO this is *graduated* (`>= 3` forces strictly more
   recompute than `>= 1`). Both sides are SFNO now, so set it the **same** on both —
   but if you ever compare against a PanguPlasim run, note it is a flat boolean there
   and the same number means something different.
4. **Loader workers.** jesswan's reference run used `num_data_workers: 1`; ai-rossby
   defaults to 8. A prior sweep measured ~9% wall throughput and 10× step-time
   jitter between 1 and 8. **Pin both to the same value** and record it.

> **The failure mode to avoid:** reporting "harness A is X% faster than harness B"
> when the two differ in four uncontrolled ways. An earlier comparison in this repo
> claimed 1.51× and was really 1.33× measured on mismatched windows, with three more
> confounds underneath. CHANGELOG 2026-08-05 has the post-mortem.

---

## 2. State — what is DONE and verified (do not redo)

| item | status |
|---|---|
| Corrected normalization | ✅ regenerated over all 51,100 files. `SST` **8.4407 / 12.0659**, `TSOI_10CM` **271.1259 / 16.3902**. `check_normalization.py` → `NORMALIZATION_OK`, **23/23 channels**, `TSOI` spread **0.9968** (was 0.122) |
| Pre-fix stats preserved | ✅ `$PANGU_AUX/pre_fix/` — the values the 85-epoch checkpoint was trained under |
| ai-rossby norm zarr | ✅ rebuilt, **bitwise-identical** to the `.nc` across all 26 vars/levels |
| E3SM → Zarr conversion | ✅ **35 stores**: train 2015–2044 + val 2045–2048 + 2049 tail. 43,800 train samples |
| ai-rossby training smoke | ✅ `PANGU_PLASIM_RUN_OK` (job 7341412) — but that was **PanguPlasim**, not SFNO |
| Profiling harness (ai-rossby) | ✅ `profile_train.py` + NVTX + CSV + nsys; `parse_nsys.py`-compatible |
| §4 equivalence machinery | ✅ `equivalence.py` + `compare_baselines.py` + PBS gate; `baselines/ai_rossby_pangu_plasim/` |
| `torch.compile` | ⛔ measured **1.40×** but **FAILS the §4 gate** under both procedures — **do not enable** |

---

## 3. Step 0 — access (do this FIRST; it is currently broken)

**The requirement is read *and* write for the group. Today it is read-only.**
Everything is `drwxr-sr-x` / `-rw-r--r--` with `umask 0022`, so other
`lighthouse-uchicago` members can read results but cannot write, and cannot fix or
extend a run. The setgid bit *is* set (group ownership inherits) — only the mode is wrong.

```bash
# 1. new files group-writable, for every future job
#    add to polaris_env.sh (it currently sets no umask):
umask 0002

# 2. fix what already exists
chmod -R g+rwX $MEMBER_ROOT/runs $MEMBER_ROOT/bench \
               $AI_ROSSBY_DATA $PANGU_AUX \
               <repo>/baselines
find $MEMBER_ROOT/runs $AI_ROSSBY_DATA -type d -exec chmod g+s {} +   # keep group inheritance

# 3. verify from another account before declaring done
#    (group-write is the claim; `stat` showing drwxrwsr-x / -rw-rw-r-- is the evidence)
```

**Decide where results live.** `$MEMBER_ROOT` is a *personal* dir
(`members/mehta5`). For group read/write, prefer a shared area — the project root
`/eagle/projects/lighthouse-uchicago/` is `drwxrws---` (group-writable), so this
works and does not exist yet:

```bash
mkdir -p /eagle/projects/lighthouse-uchicago/shared/sfno_comparison/{runs,bench,baselines}
chmod -R 2775 /eagle/projects/lighthouse-uchicago/shared
```

Then point both harnesses' run dirs there. **This is a group-affecting decision —
confirm with the owner before creating a top-level directory.**

---

## 4. Step 1 — build the ai-rossby SFNO parity config (the real work)

ai-rossby currently trains `PanguPlasimLegacy`. It must train **SFNO** instead.

⚠ **`conf/model/sfno_e3sm.yaml` is the WRONG starting point.** It is the
*speed-benchmark* variant (`jsw_256`): only 3 surface vars, no land, 1 diagnostic,
`embed_dim 256`, **and permuted group order** (`TOPO` first; `Z3`/`RELHUM` swapped).
Against our stores that is a silent failure — `ClimateZarrDataset` stacks by
store-attrs order, so a permutation is correctly-shaped and raises nothing.

Create `conf/model/sfno_e3sm_parity.yaml` = **`sfno_e3sm.yaml`'s architecture block**
+ **`pangu_plasim_e3sm.yaml`'s variable block**, with two changes:

* `embed_dim: 512` — jesswan's production value (the measured 1.18 B model), not 256.
* `surface_variables` = the **folded 8**
  `[TREFHT, U10, RHREFHT, PS, PSL, TMQ, SOILWATER_10CM, TSOI_10CM]`, and **omit
  `land_variables` entirely**. Reason: `SfnoPlasim.__init__` accepts **no**
  `land_variables`/`ocean_variables` (unlike `PanguPlasimLegacy`, which slices
  `[surface|land|ocean]`). The store's `surface_variables` attr is already that
  folded 8, and `train.py::_surface_channel_names` defaults `land_variables` to `[]`,
  so this lines up exactly — but only if you fold, and only in this order.

**Gate it before training anything:**

```bash
python3.12 ai_rossby_variable_contract.py --check-artifacts \
  --model-config   physicsnemo_ai_rossby/examples/weather/ai_rossby/conf/model/sfno_e3sm_parity.yaml \
  --dataset-config physicsnemo_ai_rossby/examples/weather/ai_rossby/conf/dataset/e3sm_pangu_parity.yaml \
  --converter      physicsnemo_ai_rossby/tools/data/e3sm/pangu_h5_to_zarr.py \
  --store          $AI_ROSSBY_DATA/e3sm/train/2015.zarr      # expect VARIABLE_PARITY_OK
```

*(The checker's `surface_variables` comparison is already the folded form —
`STORE_SURFACE` — so the folded config is what it expects.)*

### Also fix while you are there

`conf/model/sfno_e3sm.yaml` carries a **stale comment** claiming land variables "are
NOT written by the per-year converter". They are — folded into the store's
`surface_variables` (verified on `train/2020.zarr`). Correct it so nobody re-derives
the wrong conclusion.

---

## 5. Step 2 — the architecture gate (cheap, decisive, do not skip)

ai-rossby's `SfnoPlasim` wraps
`physicsnemo.experimental.models.modulus_sfno.SphericalFourierNeuralOperatorNet` and
its docstring claims it is *"faithful to PanguWeather v2.0
`networks/modulus_sfno/sfnonet.py`"*. **That is a claim, not a proof**, and the whole
comparison rests on it.

**Test: at identical config the two must report identical parameter counts.**
PanguWeather's E3SM SFNO is a measured **1,182,108,160** params
(`polaris_bench_report.md`). Build ai-rossby's parity config and print
`sum(p.numel() for p in model.parameters())` — it must match exactly.

* **Match** → the architectures are the same; differences are harness/data-path.
* **Mismatch** → you are comparing two different models. Stop and characterise the
  difference before running anything long. Report it either way.

Stronger version if you want it: a §4-style fixed-weights check across the two
implementations (`equivalence.py`'s `MODE=fixed` records loss + `grad_norm` with no
optimizer step, so nothing compounds).

---

## 6. Step 3 — run both

**Queue: use `-q capacity`, NOT `preemptable`.** `capacity` takes 1–4 nodes for
**≤168 h** at `Priority=150`; `preemptable` only runs on nodes `prod` isn't using and
**did not start a single one of 9 jobs in 11.5 h** on 2026-08-05. Caveat:
`capacity`'s `max_run 1 / max_queued 2` are **per PROJECT**, so check the slot is free
and coordinate:

```bash
qstat -a | awk 'NR>5 && $3 ~ /capacity/ {print $2, $10}'      # who holds it
```

At the measured **449.6 ms/step** (ai-rossby PanguPlasim; re-measure for SFNO — it is
~19× the parameters) a 100-epoch run is ~150 h and **fits one 168 h job**. SFNO will
be slower: budget from a fresh bench, not from this number.

* **PanguWeather retrain** — `PanguWeather/v2.0/HPC_scripts/polaris_train_e3sm_sfno_alldata_full.pbs`
  already points at the corrected `$PANGU_AUX/*.nc`. Confirm it invokes **`train.py`**
  (not `train_optimized.py`) and set the same seed/epochs as ai-rossby.
* **ai-rossby** — `physicsnemo_ai_rossby/polaris/polaris_pangu_plasim.pbs` with
  `model=sfno_e3sm_parity`. Rename the script if "pangu_plasim" becomes misleading.

**Before the long run, re-measure both** with the existing harnesses (a 20+80-step
bench each) so the throughput comparison rests on matched CSV rows rather than on
epoch-time arithmetic. `profile_train.py` (ai-rossby) and `PANGU_BENCH=1` (Pangu)
write the **same 19 canonical columns**; that is the comparison surface.

---

## 7. Traps confirmed the hard way

1. **Never trust an exit code or a truncated log.** Key on the PASS token. A killed
   pytest exited 0 with a 6-byte log twice in one session.
2. **Login nodes are not a test environment** — the same suite returned rc=0, rc=1 and
   rc=130 within an hour. Use `polaris/polaris_recipe_tests.pbs`.
3. **Never resubmit a stuck job without diagnosing.** `queue_tags` + large
   `eligible_time` = the queue has no nodes; walltime and priority are irrelevant.
   Anchor the grep — `wfp_eligible_time_exp` is a substring trap. → notes §1b
4. **Channel order is a silent failure** — the whole reason the preflight exists.
5. **`torch.compile` fails the §4 gate** (grad_norm 5.3% at *identical weights*,
   output extremes off 0.4–0.6 σ). Do not enable it for these runs.
6. **`physicsnemo_ai_rossby/` is a git subtree**, including
   `examples/weather/ai_rossby/` — keep edits minimal and contiguous. → notes §6b
7. **Check `time` length, not file existence.** `val/2046.zarr` existed but held **1**
   timestep (the old tail store) where production needs 1460.

---

## 8. Open decisions for the owner

* **Where results live** — personal `$MEMBER_ROOT` vs a new group-writable
  `/eagle/projects/lighthouse-uchicago/shared/`. Creating the latter affects everyone.
* **`embed_dim 512` vs `256`.** 512 reproduces jesswan's 1.18 B production model and
  is the like-for-like choice; 256 is ~4× cheaper and still a valid comparison **if
  both sides use it**. Pick one and use it on both.
* **jesswan's sign-off on the fills** (`TSOI_10CM` stats regenerated at the unchanged
  270 fill; `SST` 270 → −1.8) is still required before any resulting model's numbers
  are reported (DESIGN §1).
* **Epoch budget.** jesswan's reference run reached epoch **85 of 100** in 210 h of
  compute. Decide whether to match 85, run the full 100, or fix a smaller K for both.
