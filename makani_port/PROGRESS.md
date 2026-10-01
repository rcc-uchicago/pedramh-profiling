# makani port — PROGRESS (newest first)

Format: `## YYYY-MM-DD HH:MMZ — <one-line state>` then **Progress / Surprises / Decisions / Next**,
bullets only, with job ids, tokens and shas. A fresh session reads the top 3 entries.

## 2026-10-01 ~18:15Z — c8 + c10 landed on the operator's word; slot-3 series complete; slot-3 gate next

**Progress** (`feat/makani-port-main-dev`)
- `afe780ab` c8: `compat.parse_dataset_metadata_scoped` — rebinds `parse_dataset_metada.verify_grid_type` around our
  own call only, waives exactly `equiangular` + our 180 cell-centred rows (N→S, 1e-3°), restores in `finally`, prints
  `GRID_VERIFY_OPTOUT equiangular_cellcentred n_lat=180 source=…` once per process; stock call on the pin. Wired in
  `train_plasim.py` and `scripts/preflight.py` (the only two production call sites). 4 tests in `test_compat_port.py`.
- `4f0f51d8` c10: `ScalarFloat` + `Anchor` safe globals at `compat` import, only when main's
  `checkpoint_helpers.load_checkpoint` exists. 1 test (anchored ScalarFloat round-trips through makani's loader; any
  other class still refused). Login checks: `py_compile` only.

**Decisions**
- Operator (~18:00Z, direct, after the ACE2 side track): "**ignore the check c8 for now and do the weighting in the
  score**" = `grid_declaration.md` option 1 now; band-area weights in scoring (`sfno_eval/metrics.py`), training and
  validation keep makani's naive weights. "For now": revisit with jesswan before any from-scratch campaign.
- §3.1 option (a) was already ruled at 05:05Z; c10 implements it unchanged from the worker's draft.

**Next**
- Fast-forward the port worktree to the dev head (0 dirty) → `qstat` → `qsub polaris/polaris_makani_port_slot3.pbs`
  from its `makani_sfno/` → `PORT_M2M3 gate=green`.

## 2026-10-01 ~17:40Z — tolerance ruling applied; slot-3 job written; only c8/c10 stand between it and the qsub

**Progress** (`feat/makani-port-main-dev`, pushed)
- `072880dd` `equivalence_tolerance.md` + `grid_declaration.md` from `c260333f` (copied, byte-identical; this branch is
  now their canonical home). `07849ea8` erratum line: P0.2 complex reference = `complex128`.
- `e3f8fd3d` Part 0 amended to ruling §2b/§3.6 (P0.1 → `n2_null.json`; P0.2 fp64 + `n_complex ≥ 1` + total; new P0.6
  `trace_pertensor_ref.json` + `N1_TRAIN_PRETEST`).
- `bdf7e698` `port_golden_npy_diff.py` compares bytes and is NaN-aware; a mask mismatch = ERROR, rc 1.
- `c1607adc` harness knobs in `port_golden_train.py` (off = unchanged trace) + `port_part0_compare.py` (4/4 stdlib).
- `3f27c216` M3: `port_golden_infer.py --load-only` + `port_ckpt_load_compare.py` (3/3; the 10 golden entries carry all
  5 fields).
- `d608c340` `polaris_makani_port_slot3.pbs` + `polaris_makani_port_suites.pbs` (`-rA`, every touched test). `bash -n`
  only.

**Surprises**
- Ruling §2b's `vector_norm(g.double())` drops the imaginary part of a complex grad, which is G4's own target. The
  monitor filed `ruling_erratum.md`.

**Decisions** — none new.

**Next**
- Operator: c8 + c10. Then commit them (c10 makes the new-venv M3 loads possible) → fast-forward the port worktree to
  the dev head (0 dirty) → `_papercuts.md` → `qstat -u rmehta1987` (7698872 holds one debug slot; one may queue behind
  it) → `qsub polaris/polaris_makani_port_slot3.pbs` from the port worktree's `makani_sfno/` → `PORT_M2M3 gate=green`.

## 2026-10-01 ~17:00Z — operator opens threshold equivalence; tolerance panel 7697688 running; preregs frozen until its ruling

**Decisions** (operator ~16:30Z, relayed by the monitor)
- Port equivalence gates may use a threshold instead of bitwise. Proposed table: branch
  `docs/makani-port-tolerance-table` @ `db6822b9` (`equivalence_tolerance.md`, `grid_declaration.md`). Do not merge;
  the monitor updates it with the ruling.
- Panel = debug job **7697688** (read-only) → `runs/makani_port/review/tolerance_panel/7697688/ruling.md`,
  `PORT_TOL_DONE`. **No `m4_infer_prereg.md` and no M5 tolerance parts** until it lands; the table then goes into the
  preregs verbatim, committed before the jobs.
- Unchanged: c8/c10 still await the operator, and a tolerance does not replace c8. Old-venv-vs-golden regression gates
  in slot 3 stay bitwise unless the ruling says otherwise.
- m5 prereg Part 0 (`215b4ec4`) is already committed. Its only numeric bound, P0.2 `max_rel ≤ 1e-5`, is a
  `_foreach_norm` correctness check, not an equivalence gate. If the ruling covers it, the ruling wins by amendment.

**Next**
- 7697688 holds a debug slot. Check `qstat -u rmehta1987` before the slot-3 qsub.

## 2026-10-01 ~16:50Z — c9 (G4 dataset) staged ahead of c8; api_delta §3.4; c8/c10 still with the operator

**Progress**
- `59d225ff` c9: `PlasimForcingDataset` keeps the pin's HDF5 read path on main (`compat.MAKANI_HAS_BACKENDS`), with a
  geometry-only stand-in backend. Lands ahead of c8 (monitor OK; independent of c8's code).
- `580a4763` api_delta §3.4: `_lNN` levels are terrain-following. No lNN→hPa alias as pressure; the FCN3.1 alias is for
  grouping only.

**Surprises** — none.

**Decisions**
- **c9 is NOT the grid opt-out.** Its stand-in carries a `GridSpec` (a `NamedTuple`, `backends/base.py:124`); main
  verifies only in `_describe_grid(from_file=True)` (`base.py:419-420`), so the dataset path skips verification by
  design (G4, as on the pin). The config-load raise at `parse_dataset_metada.py:54` is untouched by c9 and is **c8's**
  scope. Until c8 lands, every new-venv job dies at config load on our `data.json`.

**Next** — unchanged: operator on c8/c10 → c8 → c10 → P0.1–P0.4 harness → slot-3 PBS → qsub.

## 2026-10-01 ~16:30Z — slot-3 series 6/10 code commits in on `feat/makani-port-main-dev`; c8 (grid rebind) BLOCKED by the auto-mode classifier

**Progress**
- Port branch: docs commit `de36065e` (monitor's 12:44Z/15:14Z entries, HANDOFF §3a, `golden/infer_ext.json`), pushed.
- Dev worktree `.claude/worktrees/makani-port-dev`, branch `feat/makani-port-main-dev` from `de36065e`. Commits, in §3a order:
  `4bb277e5` R3/R4/G8 docs → `3160e7bf` c1 LG fixture latitudes (test-only) → `5b4d45a7` c2 test param builders →
  `fe074c28` c3 timedelta shim on `backends.base` → `0e3e762e` c4 fin×fout→matmul helper (+ `checkpoint_loader.py:297`
  3-line hunk, preflight, `test_compat_port.py`) → `602eaf3e` c5 `--odirect_config` → `c7c50e88` c6 `dataloader_state=`
  → `a8dd9ff5` c7 LossHandler `compile=False` → `215b4ec4` m5 prereg Part 0. Login checks only (`py_compile`).
- N2-PRE threshold `1e-3`: golden trace min grad norm 6.396964e-03 (`0x1.a33b4p-8`, step 1), max 0.566, 0/20 clipped.

**Surprises**
- The classifier refused c8 (the scoped `verify_grid_type` rebind, ruled in api_delta §3.2) as "Security Test Removal".
  Not worked around. Nothing was written; the tree is clean at `215b4ec4`. c10 (the ruamel safe-globals allowlist) may
  get the same refusal.
- **HB-0 is undefined on this pack.** The `_lNN` levels are terrain-following hybrid levels
  (`convert_e3sm_to_makani_alldata.py:30-45`), and `data.json` has no hyam/hybm. Panel fact 11 ("needs the pack's plev
  list") assumed isobaric levels. Parked in m5 prereg P0.5 until the coefficients or a ruling exist.

**Decisions** — none new. c9–c10 not started (they follow c8 in §3a order).

**Next**
- Operator: allow c8 (and c10) explicitly, or re-rule §3.2/§3.1. Then c8 → c9 G4 dataset (draft in `m2_drafts_8436bfdf/`)
  → c10 → the P0.1–P0.4 harness → the slot-3 PBS (draft suites list + `test_compat_port`, `test_train_plasim`,
  `test_ema_integration`, packager parse tests) → fast-forward the port worktree → qsub.
- HB-0: hyam/hybm/P0 from the E3SM archive, or jesswan's ruling on the approximation.

## 2026-10-01 15:14Z — R2 GREEN (7693069 PORT_GOLDEN_EXT_OK tolerance=bitwise); slot 3 (M2+M3) next, no worker running

**Progress**
- 7693069 @ `2b1c9030` (dirty 2 = docs only; old venv, makani `c9704308`), 12:51–12:53Z: `GOLDEN_INFER` ×5
  `control_bitwise=True`, `PORT_GOLDEN_INFER_OK n=5`, `GOLDEN_MATCH ext_vs_ref`, `PORT_GOLDEN_EXT_OK tolerance=bitwise
  ref=…/golden_ext/7681949`. All 10 goldens are now deterministic across jobs, so M4 gates all 10.
- The port worktree is unfrozen. The debug queue is empty.

**Next**
- §3a slot 3: the M2 commit series in the dev worktree `feat/makani-port-main-dev`, starting from
  `$MEMBER_ROOT/runs/makani_port/m2_drafts_8436bfdf/`. The test-only fixture fix comes first. Also R3/R4 docs and the
  m5 prereg Part 0.
- Copy `golden_ext/7681949/infer_ext.json` to `makani_port/golden/infer_ext.json` (M4 reads it).

## 2026-10-01 12:44Z — order panel 7682039 PORT_ORDER_DONE; its order is now the task layout (written by the monitor)

Written by `port_makani_monitor`: the order landed at 05:16Z, after both the worker and the monitor sessions had gone
idle, so nobody relayed it and R2 was never queued. **Execution order = `HANDOFF_worker.md` §3a**. Full text, including
the contested facts and the decision branches: `$MEMBER_ROOT/runs/makani_port/review/order_panel/7682039/order.md`.

**Progress**
- 7681949 @ `2b1c9030` (done 05:01:46Z; log `makani_sfno/makani_port_gext.o7681949`): `NPY_DIFF_SUMMARY n1 bitwise=5/5`,
  `N1_PRETEST bitwise=5/5` (N1 inert, so the M4 prereg stays bitwise), `PHYSICSNEMO_PARITY_OK files=724
  tree_sha256=630587a5…`, `PORT_GOLDEN_EXT_WRITTEN out=$MEMBER_ROOT/runs/makani_port/golden_ext/7681949`. This is R1.
- Order panel 7682039 (2 Opus 5.5 critics, risk-first and value-first, plus a Fable 5.1 moderator; read-only):
  `PORT_ORDER_DONE`, all three rc=0.
- The worker's **uncommitted M2 drafts** (`train_plasim.py`, `checkpoint_loader.py`, `preflight.py`, `compat.py`,
  `plasim_trainer.py`, `plasim_forcing_dataset.py`, `polaris_makani_port_suites.pbs`) were copied out of its job tmp to
  `$MEMBER_ROOT/runs/makani_port/m2_drafts_8436bfdf/`. Unreviewed and untested. Start the slot-3 series from them.

**Surprises** (the panel, checked against the files)
- The G1 "M2 discovery job" isn't needed. The fixture latitudes are `linspace(-80, 80, H)` (`test_hdf5_writer.py:45`),
  stamped `legendre-gauss`, so every `parse_dataset_metadata` call raises. The failure list is known in advance. Fix it
  with a test-only commit that writes true Legendre-Gauss nodes; the §3.2 waiver does not cover it.
- The review's clipped probe "at max_grad_norm 1.0" proves nothing: the golden trace is `clipped=false` on every step
  (max 0.567). N2-PRE uses a threshold **below the trace minimum** (step 0 ≈ 6.4e-3; read all 20 from `trace_train_r1.json`).
- M6 needs no code, because `polaris_env.sh:101-102` already honours `POLARIS_SFNO_VENV`. Do not add `SFNO_VENV` as a
  second override name.

**Decisions**
- Operator (~05:40Z, before the panel): "**jesswan approves all**": NonNeg, hydrostatic incl. the moist variant, the
  grid-weight fix, `weight_decay_mode`, FCN3.1. *Verbal, via operator; written re-confirmation pending*
  (`$MEMBER_ROOT/runs/makani_port/jesswan_approvals.md`). Approved = usable in NEW opt-in arms, default OFF, and the
  off-path stays bitwise.
- Operator: follow the panel's order. Moderator rulings: M3 folds into the M2 gate job. R2 is never folded with an M2-sha
  job. S-NN wiring lands after M4 green and before the M5 job (its off-path check rides in slot 5). B1 stays post-M8.
  M7 depends only on the M2 gate, and is the fallback job whenever a gate STOPs. Fsurg is off the pending-golden list
  (no checkpoint exists).
- Dev work moves to a second worktree, `feat/makani-port-main-dev`. The port worktree only runs gates and is
  fast-forwarded between jobs.

- **R2 submitted 12:50Z as debug job `7693069`** (operator: "submit the job") at HEAD `2b1c9030`, from `makani_sfno/`,
  `GOLDEN_REF=…/golden_ext/7681949`. Its provenance line will show `dirty_tracked_files=2`: this entry and HANDOFF §3a,
  both docs the job never executes. **The port worktree is frozen until 7693069 ends.** Key on
  `PORT_GOLDEN_EXT_OK tolerance=bitwise` in `makani_sfno/makani_port_gext.o7693069`.

**Next** (the moderator's first 3 actions; action 1 = 7693069, done)
1. Check `pids.current`, then one `qstat -u rmehta1987` (debug slot is empty as of 12:44Z). From `makani_sfno/` at
   `2b1c9030`: `qsub -v GOLDEN_REF=/eagle/projects/lighthouse-uchicago/members/mehta5/runs/makani_port/golden_ext/7681949
   polaris/polaris_makani_port_golden_ext.pbs` → `PORT_GOLDEN_EXT_OK tolerance=bitwise`. The port worktree is frozen
   until it ends.
2. Create the dev worktree `feat/makani-port-main-dev` from `2b1c9030` (Lustre-safe git, `_papercuts.md:20`). Commit
   docs: R3 (api_delta #8 a–d, `enable_odirect`, G3 drop), R4 (TODO.md post-M8 entries, incl. the B1 passthrough fact),
   G8 note.
3. Slot-3 commit series, in §3a's order, test-only fixture fix first. Alongside it, `m5_train_prereg.md` Part 0
   (N2-PRE, CRPS ref, `_foreach_norm` check) and the HB-0 numpy script. Login-node checks only.

## 2026-10-01 05:30Z — operator rulings recorded; golden extension write job next

**Progress**
- `m0_golden_prereg.md` Amendment 1 (E1 golden extension ×5, E2 N1 pre-test, E3 physicsnemo
  parity) + `polaris/polaris_makani_port_golden_ext.pbs`, `scripts/port_golden_npy_diff.py`,
  `scripts/port_physicsnemo_parity.py`, `golden_checkpoints_ext.json` — committed before the qsub.

**Decisions** (operator, relayed by the monitor, 2026-10-01)
- ~05:05Z: carry out **every** recommendation of review 7681379, in its order, each through its own gate.
- ~05:05Z: **§3.1 = option (a)** — add exactly `ruamel.yaml.scalarfloat.ScalarFloat` and
  `ruamel.yaml.anchor.Anchor` to `torch.serialization` safe globals in our compat layer; **never**
  `MAKANI_ALLOW_UNSAFE_CHECKPOINT_LOAD`.
- ~05:10Z: `NonNegativeConstraint` (PRECT, SOILWATER_10CM, TMQ) becomes an **opt-in** right after M4
  (new venv only, feature-detected, wired in our model-build path, default OFF, names from config,
  `NONNEG_CONSTRAINT on|off channels=…` per job; flag-off must stay bitwise vs golden). Evidence for
  jesswan = a pre-registered inference-only climate screen (A e243, nf4p_r1; on vs off; eps/mode/leak
  sensitivity). **No training, no default-on, no production config change without jesswan's written
  sign-off** — "awaiting jesswan".
- Post-M8 items (review §3) become TODO.md entries now; implemented only after M8.

## 2026-10-01 05:10Z — M1 GREEN (7680816 VENV_MAIN_OK, tag makani-port/m1-green @ 6a30605a); review 7681379 read

**Progress**
- Job 7680816 at `6a30605a`, dirty 0: `SFNO_VENV_MAIN_BUILT`, `VENV_MAIN_OK … symbols=29/29
  old_venv_unchanged=1`, `PORT_M1 gate=green`. Old venv dist-info still `c7f1fba5…`/`380b1a77…`.
- `sfno-venv-main` resolved versions (reproducible rebuild = these): torch 2.8.0 (base conda),
  makani 0.2.0 @ `a0aa4c4f`, torch_harmonics @ `2edb24ed` (reports **`0.9.2a`**; old venv reports
  `0.9.2.dev75+g2edb24ed` — **same commit**, the build had no git metadata; CUDA-ext build parity
  unverified), physicsnemo 2.2.0a0 **non-editable** from tree `1674e93e`, **zarr 3.4.0**, numpy 2.2.6
  (base), h5py 3.16.0 (overlay), wandb 0.22.1 (base), DALI 2.2.0, xarray 2025.9.1 (base), netCDF4 1.7.4.
- `GRID_WEIGHT_DELTA` (A e243, single-step z-space l2, 64 valid samples, old venv): channel-mean
  loss_rel **−7.41e-4** (cell-centred vs makani naive); worst channels RELHUM_l00 **+2.45 %**,
  RELHUM_l01 +1.79 %, RELHUM_l02 +1.67 %; polar (|lat| ≥ 85°) share of the error 0.30 % (naive) vs
  0.45 % (cell); weights max_abs_row 7.6e-5 at ±89.5°, L1 6.31e-3. Measurement only — for jesswan.
- Operator review job 7681379 (read-only) `PORT_REVIEW_DONE`; `runs/makani_port/review/7681379/review.md`.

**Surprises**
- pip "dependency conflicts" ×3 in `build.log`, verdicts: (i) **harmless** — base-conda packages we
  never import (sglang/vllm/verl: transformers, xgrammar, numpy<2, setuptools; boto3: botocore;
  datasets/gcsfs: fsspec ≤ 2025.9 — fsspec 2026.6.0 is also the old venv's); (ii) **harmless in
  practice** — nvidia-physicsnemo 2.2.0a0's declared `torch>=2.10`, torchvision, timm, tensordict,
  gitpython, importlib-metadata, termcolor, urllib3 minimums: the old venv runs the same 2.2.0a0 on torch
  2.8 in every green job; (iii) **none** names zarr, numcodecs, DALI or makani.
- Review: a `try/except` around `parse_dataset_metadata` would leave params half-filled (it raises at
  `parse_dataset_metada.py:54` before channel names / dataset dict) → §3.2 is a scoped rebind instead.
- Review: golden covers 5 of the 10 checkpoints named in the handoff; N1 (compile removed from the
  dhconv contraction) is untested and could make bitwise M4 impossible by construction.

**Decisions** — none new. §3.1 still unruled (M3 only).

**Next** (review's "Before M8" list, in order)
- One old-venv debug job: M0 prereg amendment + golden write for the 5 missing ckpts; N1 pre-test
  (`TORCH_COMPILE_DISABLE=1` vs golden); sha256 of both venvs' physicsnemo installs.
- M2 code (api_delta §2 + scoped grid rebind); M5 prereg additions (clipped probe, CRPS probe, per-lead h5).

## 2026-10-01 04:05Z — M0 GREEN (7679553 PORT_GOLDEN_OK, tag makani-port/m0-green @ 81269603); M1 next

**Progress**
- Write 7676860 `PORT_GOLDEN_WRITTEN` (sha `347c6431`, 11:54 wall); gate 7679553 `PORT_GOLDEN_OK
  tolerance=bitwise` vs 7676860 (sha `812696030bae`; differs from `347c6431` only in this file) —
  `GOLDEN_MATCH infer_vs_ref / val_vs_ref / train_vs_ref`, `F_FINETUNE_TESTS_OK 6/6`, old venv
  `c9704308` with the monitor's dist-info sha256s.
- Tag `makani-port/m0-green` → `812696030bae`, pushed. Manifests copied to `makani_port/golden/`
  (`infer.json`, `trace_val_r1.json`, `trace_train_r1.json` from the write run = the reference).
- Golden numbers: A val-only loss `0x1.b3af2p-7` (0.013296); 20-step trace clipped=0, grad norm max
  0.567 (monitor); post-trace val 0.0240.

**Surprises**
- The warm-start trace perturbs A hard (fresh AdamW at lr 2e-3, no warmup: loss 0.0102 → 0.138).
  Good for amplifying drift; it is an **equivalence probe only**, never a training-quality number.

**Decisions**
- Operator ruling on api_delta §3.2, **relayed by the monitor** (2026-10-01): keep `equiangular`;
  dataset-scoped opt-out of `verify_grid_type` (backend gets the file's own lat/lon + a narrow catch
  around `parse_dataset_metadata` for our `data.json`), one `GRID_VERIFY_OPTOUT equiangular_cellcentred`
  line per job; outputs bitwise; measure the weighting error for jesswan (`GRID_WEIGHT_DELTA`).
- §3.1 (ruamel allowlist) **not yet ruled** — gates M3; not implemented.

**Next**
- M1: `polaris_setup_sfno_venv_main.sh` → `sfno-venv-main` at `a0aa4c4fe5c4…`, built in one debug job
  that also prints `VENV_MAIN_OK` and the `GRID_WEIGHT_DELTA` measurement (debug = one queued job).

## 2026-10-01 01:25Z — M0 write run 7676860 queued (sha 347c6431); gate run follows it

**Progress**
- `m0_golden_prereg.md` committed at 01:21:13Z in `347c64319ad2` — before the qsub.
- `polaris/polaris_makani_port_golden.pbs` + `scripts/port_golden_{infer,train,compare}.py` +
  `makani_port/golden_checkpoints.json` (5 ckpts, stored epochs read from the files).
- Job **7676860** (debug, write mode) submitted from `makani_sfno/`. Worktree frozen until it ends.

**Surprises**
- C1 e24 is `ckpt_mp0_v23.tar`; its `best_ckpt_mp0.tar` is e18. Fsurg has no checkpoint
  (7671977 `rc=124` at 2880 s). N6 (`18c4582` on h1w1) measured inert for every golden ckpt.

**Decisions** (with the monitor): golden trace = warm start from A e243 + grad norm + `clipped`.

**Next**
- 7676860 → expect `PORT_GOLDEN_WRITTEN`; then `qsub -v GOLDEN_REF=$MEMBER_ROOT/runs/makani_port/golden/7676860 polaris/polaris_makani_port_golden.pbs`
  → `PORT_GOLDEN_OK` = M0 gate → tag `makani-port/m0-green`, copy the JSONs to `makani_port/golden/`.
- Operator: api_delta §3.1 / §3.2 answers still needed before M2/M3.

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
