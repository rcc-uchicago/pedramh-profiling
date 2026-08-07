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
| Polaris (PBS) bring-up | ✅ **all 4 runnable models GREEN on 4×A100**, and Pangu is now proven **reproducible by a second user** (7253591, loss identical to the installer's run); **SI too** (7253603). Their deps were private to rmehta1987 until today's shared top-ups (PanguWeather-SFNO, SI, Makani-SFNO, PhysicsNeMo) + probe + all 3 data converters proven on real data. S2S/port scripts delivered but blocked on an ERA5 Globus stage. See `polaris_pbs_notes.md`. |
| **Profiling (PanguWeather SFNO on A100)** | ✅ **DONE — see `polaris_bench_report.md`.** Harness ported (PanguWeather had **zero** instrumentation), loader sweep + nsys captured. **VERDICT: GPU-bound** (loader idle **0.7%**) and **elementwise-bound** (61% of GPU time pointwise vs 15% GEMM) ⇒ `torch.compile` (§5 rung 1) is the right first lever, now on evidence. Model is **1.18 B params**, not ~79M. SI/makani/physicsnemo **not yet profiled**. |
| §4.0 prerequisites — **`s2s/v2.0`** | 🟡 **seed knob DONE + GPU-verified** (`--seed`/`$S2S_SEED`/YAML + `--deterministic`, `s2s/v2.0/utils/seeding.py`; 10 tests `SEEDING_OK` on CPU **and on an A100**, job 7253738 rc=0); tiny config + VAE noise-fix still **block baseline capture** |
| §4.0 prerequisites — **`PanguWeather`** (the focus; a separate fork, nothing propagates) | ✅ **ALL THREE MET.** seed knob ✅ **already existed — do NOT port `seeding.py` here** (`--global_seed`→`seed_torch`, seeds numpy+torch+CUDA, forces `cudnn.deterministic`; stronger than s2s's legacy path). VAE noise hook ✅ **built** (`utils/vae_noise.py`, 16 tests `VAE_NOISE_OK`) but **inert on `sfno_plasim`** (no VAE). `tiny_baseline.yaml` ✅ **written AND run** — job 7255583: **7,166,656 params** (165× smaller than the real 1.18 B), 0.023 s/step, **1.00 GB**. ⇒ **baseline capture is no longer blocked on building anything** |
| **E3SM data prep (PhysicsNeMo zarr)** | 🟡 **7 defects found, 5 fixed, 4 open**; verified `SEQZARR_VERIFIED` on a 24-year random fixture (job 7257786). **The full ~1 TB conversion is NOT cleared to run** — 4 open defects + 5 decisions. `polaris_data_prep_handoff_prompt.md`. makani's converter **unaudited**; Pangu's stats prep audited (clean, metadata-only). |
| Correctness baselines captured (DESIGN.md §4) | 🟡 **machinery BUILT and first baselines captured** (`baselines/ai_rossby_pangu_plasim/{eager,compiled_default}.json`, job 7353187) — `equivalence.py` + `compare_baselines.py` + a PBS gate. **First verdict: `torch.compile` FAILS** (4.02e-01 > 1e-2), so rung 1 is measured (1.40×) but **not adopted**. Open question for the owner: §4.1 gates on a 20-step bf16 *training* trajectory, which compounds — step 0 agrees to 8.3e-4. Other models still have no baseline |
| Test harness (tier-1 equivalence/unit + `--fast`) | 🟡 3 test files now exist + self-run (`SEEDING_OK`, `BENCH_INSTR_OK`, `VAE_NOISE_OK`); no `conftest.py`/`--fast` yet |
| Optimization ladder (DESIGN.md §5) | 🟡 **rung 1 MEASURED on ai-rossby, not adopted**: `torch.compile` (default mode) = **1.401×** (step_med 449.6→320.8 ms, peak mem 24.98→21.07 GB; jobs 7352022 vs 7352948). Still **not enabled anywhere** — §4 equivalence gate does not exist yet. Measuring is unblocked; adopting is not. PanguWeather/SI/s2s rungs untouched |

### Smoke status matrix (probe → 1-GPU → 4-GPU)

| Model | Midway | Polaris |
|---|---|---|
| Toolchain probe | — | ✅ `PROBE_OK` (job **7253681**, the fixed probe run as a second user: `sys.path` free of `~/.local`, and it imports the REAL `modules.train_module` rather than an empty namespace package — the hollow check that hid `cf_xarray`. 4×A100-40GB; makani/physicsnemo need the §6 venv, non-blocking) |
| S2S (`torchrun`) | ✅ runs (Midway scripts GREEN) | ⛔ blocked on ERA5 stage (scripts ready) |
| S2S-Lightning | ⚠️ standalone smoke config-path fixed 2026-07-13 — **needs a Midway run to reconfirm** | ⛔ blocked on ERA5 stage (scripts ready) |
| SI | ✅ runs (Midway scripts GREEN) | ✅ **4-GPU GREEN** (7252700: step_med 0.400 s, peak 30.98 GB) **and reproducible by a SECOND USER** — job **7253603** (`PYTHONNOUSERSITE=1`): step_med 0.399, peak 30.69 GB, rc=0 |
| PanguWeather SFNO | — | ✅ **4-GPU GREEN** (7252271) **and reproducible by a SECOND USER** — job **7253591** (`PYTHONNOUSERSITE=1`) rc=0 with loss **0.3411, identical** to the as-installer run |
| Makani SFNO | — | ✅ **4-GPU GREEN** (job **7253465**, current script: train loss 2.61 / val 2.38 + ckpt; first green 7252769 pre-rework; pack `CONVERT_OK` 7252728) — runs from the isolated SFNO venv |
| PhysicsNeMo SFNO | — | ✅ **4-GPU GREEN** (job 7252933, rc=0: loss 0.889, val err 0.541, ckpt saved; 1-GPU 7252816 also green; zarr `CONVERT_OK`) |

## Next actions (pick from the top)

0. **FOCUS (2026-07-15): the work is on `PanguWeather/`, not `s2s/v2.0`.** They are
   95%-identical forks (DESIGN §2c) — but **copies, not shared imports**, so nothing
   propagates between them. Both consequences the earlier handoff listed here are now
   **resolved, and one of them was wrong**:
   - PanguWeather's missing NVTX/`S2S_BENCH` instrumentation: ✅ **ported** (2026-07-15,
     range names identical per CLAUDE.md #10), and the legacy path is proven unchanged
     (job 7255505 reproduces the green 0.3411 bit-identically).
   - "the `--seed` knob lives in `s2s/v2.0/utils/seeding.py` only — port it deliberately":
     ❌ **do NOT.** PanguWeather already has `--global_seed` → `seed_torch()`, which is
     *more* complete than s2s's legacy path. See the Known-issues entry.


1. **Capture the §4.1 baseline** on TINY — **nothing blocks it any more.** All three §4.0
   prerequisites are met on PanguWeather: the seed knob already existed, the VAE hook is
   built, and `tiny_baseline.yaml` is written *and run* (job 7255583: **7,166,656 params**,
   165× smaller than the real 1.18 B; 0.023 s/step; **1.00 GB**), so a K=20 baseline is
   ~0.5 s of compute.
   Procedure: world size 1, fixed seed (`--global_seed`), K=20 steps, per-step loss
   trajectory + output summary stats → `baselines/pangu_sfno/` as JSON/CSV (§4.2 — text
   only, never tensors).
2. **Then, and only then, rung 1 of the §5 ladder: `torch.compile`.** The profile now says
   this is the right lever on evidence — **61% of GPU time is elementwise over ~1506
   launches/step vs 15% GEMM** (`polaris_bench_report.md` §4.2), i.e. fusion-starved.
   `TORCH_COMPILE_MODE` is now genuinely wired and deliberately unset — it was **not**
   plumbed in PanguWeather despite an earlier claim in this file (see the correction entry
   below). Expect longer warmup (raise `S2S_BENCH_WARMUP` to 40); profile eager, bench
   compiled.
3. **Profile the other three models** (`polaris_bench_report.md` covers only PanguWeather
   SFNO). SI is cheapest — it already has `SI_BENCH_*`/`SI_NVTX` and a green Polaris bench
   (7252700/7253603). makani/physicsnemo have no comparable harness.
4. **Fix the loader's missing `worker_init_fn`** — it would make `num_data_workers` an
   output-neutral knob and unlock a measured **+9% wall throughput with 10× less jitter**
   (`1 → 8`). Today the worker count changes the noise realization, so the win cannot pass
   the §4 bitwise gate. Ship it with a test pinning sample→noise independence from worker count.
5. **Stand up the test harness proper** — three self-running test files now exist
   (`SEEDING_OK`, `BENCH_INSTR_OK`, `VAE_NOISE_OK`) but there is no `conftest.py` or
   `--fast`. Add CRPS/KL numerical checks and the normalize↔inverse round-trip.
6. **Unblock `s2s`/the port** — still the ERA5 Globus stage.

## In progress

- **ai-rossby PanguPlasim bring-up — ✅ ALL THREE HANDOFF-v2 STEPS GREEN (2026-08-05).**
  Branch `fix/tsoi-fill-270`. The smoke-scale pipeline is end-to-end proven: corrected
  statistics → rebuilt norm zarr → a 4×A100 training run that converges.
  1. **Normalization regen** — job **7340945**, exit 0, **34:54** over all 51,100 h5 files
     (~2.15 TB). `MOMENTS_OK` → `NORM_NC_OK` → `NORMALIZATION_OK` → `E3SM_NORM_REGEN_OK`.
  2. **Norm zarr rebuilt** — bitwise-identical to the source `.nc` across all 26 vars/levels.
  3. **Training smoke** — job **7341412**, exit 0, **9:36**. `PREFLIGHT_OK`
     (`VARIABLE_PARITY_OK 21/21` ×3 stores) + `PANGU_PLASIM_RUN_OK`.
  4. **Production conversion** — ✅ **DONE 2026-08-06**: 35 stores (train 2015–2044,
     val 2045–2048 + 2049 tail), 43,800 training samples, verified on disk. Took
     3 h through `debug`; `preemptable` never scheduled it (see the 2026-08-06 entry).
     No inode problem materialised — `lfs quota` shows files quota/limit **0 = no cap**.
  **Next: the long training run — and use `-q capacity`, NOT `preemptable`.**
  `capacity` takes **1–4 nodes for ≤168 h** at `Priority=150` and is actively
  scheduling (12 running); the project's slot is currently free. At the *measured*
  449.6 ms/step (not the stale 537) the 100-epoch run is **~150 h — it fits in ONE
  job**, versus 3× 72 h `preemptable` links that cost jesswan ~80 h of inter-link
  queue wait plus a lost partial epoch per kill. Caveat: `capacity`'s
  max_run 1 / max_queued 2 are **per PROJECT**, so coordinate before taking the slot.

- **Pipelines runbook delivered (`polaris_pipelines_plan.md` + operator guide
  `polaris_pipeline_runbook.md`); the §0 smoke sequence is DONE (all four green, see the
  decisions log). Remaining wait: jesswan's zarr transfer** (announced 2026-07-16, not yet on
  disk). The moment it lands:
  `cd physicsnemo_sfno && qsub -v STORE=<transferred-path> polaris/polaris_verify_store.pbs`
  (PASS = `SEQZARR_VERIFIED` per store + `STORE_VERIFY_OK`), then train against it via
  `qsub -v SEQZARR_ALLYEARS_DATA=<dir> polaris/polaris_sfno_allyears.pbs` — if it verifies at
  the current generation, our own ~11 h / ~1.43 TB conversion is unnecessary.
- **Data-prep PR open for review — https://github.com/rcc-uchicago/pedramh-profiling/pull/11**
  (branch `polaris-data-prep`). ⚠️ **Stacked on `polaris-profiling` (#10), itself stacked on
  `polaris-pbs-bringup`** — merge in that order. Carries the 7 converter defects (5 fixed, 4
  open) and **5 decisions** needing jesswan/us: `polaris_data_prep_decisions.md`.
  **The analysis is 1/3 done** — makani's 367-line converter is unaudited (§8).
- **Profiling PR open for review — https://github.com/rcc-uchicago/pedramh-profiling/pull/10**
  (branch `polaris-profiling`). ⚠️ **Stacked on `polaris-pbs-bringup`**, which is still
  unmerged, so PR #10's diff includes those commits until it lands. **Merge the bring-up PR
  first.** A solo session cannot self-approve — maintainer review/merge needed.
- **Polaris bring-up PR open for review** — branch `polaris-pbs-bringup` pushed; open at
  https://github.com/rcc-uchicago/pedramh-profiling/compare/main...polaris-pbs-bringup
  (a solo session cannot self-approve — maintainer review/merge needed).
- **Layout change: the SFNO codebases are now `git subtree`s of this repo** (not
  separate checkouts as the handoff assumed, not submodules). Imported **unsquashed for
  full provenance**: upstream commits are real ancestors of HEAD (jesswan-uc 8,
  feynmanliu214 38, ktangsali 203). Cost: 313 → 4,769 files, .git 3.9 MB → 306 MB.
  **Was 3; `physicsnemo_ai_rossby/` made it 4 on 2026-08-04** (`0777be0f` ← `87002adb`) —
  and its `examples/weather/ai_rossby/` recipe is subtree-owned too, despite reading as
  fork-owned. See `polaris_pbs_notes.md` §6b's warning box before editing anything there.
  Bidirectional merging (`subtree pull` from them, `subtree split` + PR to them) and the
  rule-#8 exception for imported third-party junk are documented in
  `polaris_pbs_notes.md` §6b. Note: pushing this repo now needs
  `git -c pack.threads=1 push` — the ALCF login node's process cap kills multi-threaded
  pack (`unable to create thread` / `git-pack-objects died`); the same cap forced the
  physicsnemo `subtree add` onto a compute node.
- **Deferred, ready:** **ERA5 Globus stage** → unblocks the S2S + S2S-Lightning smokes
  (scripts already preflight `ERA5_NOT_STAGED`).
- **makani / physicsnemo — torch_harmonics conflict RESOLVED via an isolated venv.**
  makani 0.2.0 needs the *public* `precompute_latitudes`, absent from every torch-2.8-safe
  release (0.7.4/0.8.0); 0.9.1 ships wheels only (no sdist) and its `attention/_C.so`
  ABI-breaks torch 2.8. Resolution (per user): `polaris_setup_sfno_venv.sh` builds an
  isolated `--system-site-packages` venv with **torch_harmonics 0.9.x from GitHub source**,
  so the base conda keeps 0.7.4 and the GREEN Pangu/SI smokes need no re-validation.
  **Trap:** a `--system-site-packages` venv re-enables the USER site, which `site.py` puts
  *before* the venv — the base's `--user` 0.7.4 shadowed the venv and makani still failed;
  fixed with `PYTHONNOUSERSITE=1` in the venv + both SFNO PBS scripts.
  Two more launch traps (both encoded in the scripts): `torchrun` resolves to the BASE
  conda launcher (whose shebang pins the base python) because the venv inherits torch and
  has no torchrun — use `python -m torch.distributed.run`; and makani's `--batch_size` is
  GLOBAL, so the **rank count must divide it** (`global_batch_size % data_parallel_size == 0`
  — `--batch_size=1` on 4 ranks fails). Plus an **upstream makani bug** (pin
  `c970430`): `self.logger` is assigned only when `log_to_screen` is truthy (rank-0 only)
  yet `deterministic_trainer.py` calls it unconditionally → every non-zero rank died;
  patched in our `plasim_trainer.py` wrapper, not in makani.
  **RESULT: Makani pack GREEN (`CONVERT_OK`, 7252728) and Makani SFNO 4-GPU smoke GREEN
  (7252769: train loss 2.19, val 2.05, checkpoint written, rc=0), **re-confirmed on the
  post-rework script by 7253465** (train 2.61 / val 2.38, 7.10 s of real training).**
  **PhysicsNeMo is ALSO GREEN** (1-GPU, job 7252816, rc=0: loss 1.082, val err 0.776,
  checkpoint saved; zarr store `CONVERT_OK` with max|zarr-h5|=0 + all-finite). Its four
  traps: hydra's PATH-form defaults make `model=tiny_sfno` impossible (added
  `conf/config_e3sm_sfno.yaml` + `--config-name`); mlflow is NOT optional and mlflow 3.x
  refuses its own file store (`MLFLOW_ALLOW_FILE_STORE=true`); `datapipe.parallel=false`
  is a broken fallback (DALI rejects `prefetch_queue_depth`); `validation.num_steps` must
  be >=2 (matplotlib squeezes the axes array) and `dataset.dataset_filename` must be
  repointed. **PhysicsNeMo 4-GPU is GREEN too** (job 7252933, rc=0: 4 ranks, loss 0.889,
  val err 0.541) — so all four runnable models are green on 4 GPUs.

## Decisions / changes log

- **2026-08-06** — **SFNO apples-to-apples: ARCHITECTURE GATE PASSED (job 7364984) — the
  two implementations are BITWISE identical, not merely same-sized.** Parity config,
  gate, per-epoch telemetry and group-write all in.
  > **Gate result (4×A100, ai-rossby venv, fp32, rc=0, `SFNO_PARITY_OK`):**
  > `in_chans=105 / out_chans=101` as predicted; **both models 1,182,108,160 params**,
  > matching the independently measured reference (job 7255410) exactly; and at
  > **identical weights on identical input the two agree to 0.000e+00 relative error**
  > on `surface`, `upper_air`, `diagnostic`, `loss` (5.26061580e-02) and `grad_norm`
  > (2.16819166e-01). Handoff §5 asked only for equal parameter counts — this is the
  > stronger fixed-weights form, so any remaining difference between the two runs is
  > **harness and data-path, not model.**
  Implements `polaris_sfno_comparison_handoff.md`. Owner decisions taken this session:
  results stay in `$MEMBER_ROOT` with the modes fixed (no new group-level dir),
  **`embed_dim 512`** (jesswan's production value), and the two long runs go to
  **different queues** — ai-rossby on `capacity`, PanguWeather on `preemptable` — which
  sidesteps `capacity`'s one-running-job-per-PROJECT cap (`max_run=[p:PBS_GENERIC=1]`,
  verified with `qstat -Qf`) instead of serialising them.
  - **The architecture claim is now PROVEN, not asserted** (`compare_sfno_parity.py`).
    `SfnoPlasim`'s docstring claimed fidelity to PanguWeather's
    `SphericalFourierNeuralOperatorNet_v2`; the vendored `modulus_sfno/` is in fact a
    near-verbatim copy — **5 of 7 files byte-identical**, the other two differing only by
    `@torch.compiler.disable` decorators (inert in eager) and `return_latent` plumbing
    (adds no module ⇒ no parameter). The gate keeps an **exact-line** allowlist, not
    regexes: a regex broad enough to cover the `return_latent` refactor (`^\s*else:$`)
    is broad enough to hide a real change. Negative control confirms an added
    `nn.Linear` still fails it. Config block: **26/26 architecture keys identical.**
  - **Two traps found that would each have silently broken the comparison:**
    1. **The effective learning rate is 3.0517578125e-05, NOT the `lr: 1e-4` written in
       the SFNO block.** That line is dead — `get_optimizer` (train.py:706-709) checks
       `hasattr(params,'loglr')` FIRST, and `loglr: -13` is inherited from the base
       config, so the peak LR is `2**-13 * global_batch_size/16` at 4 ranks × batch 1.
       Taking the file at face value would have run ai-rossby at **3.3×** the reference.
       It is also world-size-dependent, so it must be recomputed if ranks change.
    2. **The handoff's §4 claim that the checker already expects a folded config was
       wrong.** `--check-artifacts` compared `model.surface_variables` against the
       *unfolded* 6 and required `land_variables`/`ocean_variables` keys — which
       `SfnoPlasim.__init__` cannot accept (`build_model` forwards every config key as a
       kwarg). Checker now handles both shapes explicitly and reports which it checked;
       12 tests, incl. negative cases proving the folded branch is not a way to skip the
       land channels. **19/19 PASS folded, 21/21 PASS split (unchanged).**
  - **`torch.compile` stays OFF** for both runs — measured 1.40× but FAILS the §4 gate.
  - **Per-epoch telemetry added to BOTH PRODUCTION trainers** (`epoch_telemetry.py`,
    duplicated in each tree because they share code by copy; a drift test pins the
    copies identical). Opt-in (`{PANGU,AI_ROSSBY}_EPOCH_TELEMETRY=1`), on by default in
    the two run scripts. **Why not the bench harness:** it structurally cannot answer
    "what do epochs 1-6 cost" — `profile_train.py` builds **no EMA** and scopes its
    scheduler to the bench window, and `PANGU_BENCH` truncates the run and exits. Two
    CUDA events/step + one all-reduce/epoch, **no per-step sync** (a sync would remove
    CPU/GPU overlap and change the number being measured). 21 shared columns.
    - **This immediately exposed a real harness difference:** PanguWeather **skips the
      `update_ema` sweep entirely** until `ema_warmup_epochs` (6), so its step cost jumps
      at that epoch; ai-rossby calls `ema.update` from epoch 1 and warms only the *decay
      value*, paying the sweep throughout. On 1.18 B params that is ~4.7 GB of
      elementwise traffic per step. Recorded honestly in `ema_active` rather than
      smoothed over. Window boundaries are pinned by test so both sides span the same
      work (Pangu's per-iteration RMSE/wandb block is deliberately *outside* it — it has
      no ai-rossby counterpart).
  - **Group write fixed** (`umask 0002` in `polaris_env.sh`, before its `mkdir`s;
    `chmod -R g+rwX` + setgid over runs/bench/ai_rossby_data/pangu_polaris_data/
    baselines). The setgid bit was already correct — only the mode was wrong.
  - **Dead knob found:** `cfg.max_checkpoints_to_keep` is declared in ai-rossby's
    `conf/config.yaml` and **never read**, and physicsnemo's `save_checkpoint` does not
    prune ⇒ checkpoints accumulate unbounded at ~14 GB each (4.7 GB weights + 9.4 GB
    AdamW moments). `CKPT_INTERVAL` (default 5 ⇒ ~280 GB/100 epochs) is the only lever
    today. Eagle shows no quota cap, so this is a tidiness issue, not a blocker.
  - **New files:** `compare_sfno_parity.py`, `epoch_telemetry_test.py`,
    `ai_rossby_variable_contract_test.py`, `{PanguWeather/v2.0/utils,
    physicsnemo_ai_rossby/examples/weather/ai_rossby}/epoch_telemetry.py`,
    `conf/model/sfno_e3sm_parity.yaml`, `conf/training/sfno_e3sm_parity.yaml`,
    `polaris/polaris_sfno_{parity_gate,e3sm}.pbs`, `polaris/polaris_bench_sfno_e3sm.pbs`.
    Also corrected a **false comment** in `conf/model/sfno_e3sm.yaml` claiming the land
    variables are not written by the per-year converter — they are, folded into the
    store's `surface_variables` (verified on `train/2020.zarr`).
  - **ai-rossby SFNO training smoke GREEN — job 7364985, `SFNO_E3SM_SMOKE_OK`.** First
    time the vendored SFNO has ever been executed in the ai-rossby venv.
    `PREFLIGHT_OK` (`VARIABLE_PARITY_OK 19/19` against every one of the 35 stores),
    `steps_per_epoch=10950` — exactly the Pangu arithmetic (43,800/4) — finite loss
    **2.947**, val_loss **2.995**, and the first telemetry row:
    `epoch=1 n=20 step_med=751.1ms gpu_busy=94.9% peak=25.89GB ema=1`. Peak leaves
    ~14 GB of headroom on a 40 GB A100. `gpu_busy 94.9%` at `num_workers=1` puts loader
    idle at ~5%. (`step_mean` 1133 ms vs `step_med` 751 ms with std 1658 is first-step
    autotune in a 20-step window; it washes out over 10,950 steps — read the median.)
    Confirmed end-to-end that `umask 0002` works: the CSV landed `-rw-rw-r--`.
  - **`validation=off` does NOT turn validation off** — found by the smoke appearing to
    hang for 5 minutes after training finished. `conf/validation/off.yaml` disables only
    the ROLLOUT; a single-step `val_loss` pass still runs over every sample whenever
    `dataset.val_zarr_path` is set (train.py:832-842). Measured: **322.6 s per epoch**
    over the 4 val years (5,840 samples) — 13× the 20-step smoke it was validating.
    `dataset.val_zarr_path=null` is what actually skips it; the smoke path now does
    (override with `SMOKE_VALIDATE=1`). Two consequences: budget ~5.4 min/epoch of
    validation into the ai-rossby production run, and note this is a **disclosed
    non-matching axis** — PanguWeather's production validation is 129 ICs × 60-step
    rollouts, entirely different work. The telemetry row deliberately covers TRAINING
    only (`epoch_end` fires before validation), so the step comparison stays clean.
  - **MATCHED BENCH ROWS — the honest harness number is 1.166×, ai-rossby faster.**
    Jobs **7364997** (ai-rossby) and **7364998** (PanguWeather, `CONFIG_NAME=
    E3SM_SFNO_H5_POLARIS_ALLDATA`), same node type, 20+80 steps each. Every
    controlled axis identical in the CSVs: `n_gpus 4`, `batch_per_gpu 1`, bf16,
    `ddp_find_unused false`, **`n_loaders 1`**, `n_steps_counted 80`, and — proven by
    the gate — the same 1,182,108,160-param model.

    | | ai-rossby | PanguWeather | ratio |
    |---|---|---|---|
    | `step_med` | **0.71996 s** | **0.83975 s** | **1.166×** |
    | `samples_per_s` | 5.556 | 4.763 | **1.166×** |
    | `compute_med` | 0.71994 | 0.83753 | 1.163× |
    | `step_p90` | 0.72346 | 1.11494 | 1.541× |
    | `step_std` | 0.00814 | 0.19743 | 24× |
    | `peak_mem_gb_max_rank` | 21.40 GiB | 28.76 **"GB"** = 26.79 GiB | **1.25×** |
    | loader idle | 0.63% (`data_idle_frac`) | 9.05% (`loader_wait_frac`) | — |
    | `cpu_prep_med` | 1.5e-05 s | 2.2e-03 s | 149× |

    `step_med` and `samples_per_s` agree to three digits (internal consistency).

    > **⚠ UNIT MISMATCH in `peak_mem_gb_max_rank` — the two harnesses do not
    > report the same unit.** PanguWeather divides by **1e9** (decimal GB,
    > `train.py:1344`); ai-rossby's `profile_train.py:347` and
    > `epoch_telemetry.py:224` divide by **1024³** (GiB). Pangu's numbers are
    > therefore inflated ~7.4% relative to everything else in this document.
    > Pangu's `28.76 "GB"` is **26.79 GiB**, its headroom is **12.70 GiB** (not
    > 10.7), and the memory ratio is **1.25×** (not 1.34×). Third cross-harness
    > measurement inconsistency found today, after the missing EMA and the node
    > variance — every one of them made the two sides look more different than
    > they are. **Convert before comparing anything against a Pangu row.**

    > ### ⚠ RETRACTED SAME DAY — 1.166× IS NOT CONFIRMED. **Do not quote it.**
    > The two rows were measured in **separate jobs on different nodes** —
    > ai-rossby on `x3001c0s13b0n0`, PanguWeather on `x3003c0s37b0n0`. The sweep
    > (job 7365119, node `x3109c0s19b1n0`) then re-measured the **identical**
    > `ckpt3/sg0/w1/bs1` config at **0.6517 s** against the parity bench's
    > **0.7200 s** — **1.105×, from the node alone.**
    >
    > Node-to-node spread (10.5%) is therefore the same order as the effect
    > attributed to the harness (16.6%), and with n=2 nodes the *direction* of the
    > bias is unknown, so the true gap could be larger, smaller, or absent.
    >
    > This is the same failure class as the 1.51×→1.33× post-mortem below —
    > caught here only because the sweep happened to re-run the baseline config.
    > **Fix: bench both harnesses in ONE job on ONE node, alternating A/B/A/B/A/B
    > for repeats.** Until that runs, the only defensible statements are the
    > architecture gate (exact) and the within-job sweep ratios (same node,
    > back-to-back).
    >
    > **Lesson for every future comparison here: a cross-JOB ratio on Polaris is
    > not a measurement.** One node, one job, interleaved repeats, or it does not
    > count.
  - **Two things worth chasing, both now measurable:** (a) `compute_med` alone differs
    by 1.163×, i.e. the gap is NOT mostly loader — with an identical model, that is
    harness overhead inside the step; (b) PanguWeather is slower **while not yet paying
    for EMA** (`_ema_active()` is False until epoch 6) whereas ai-rossby sweeps from
    epoch 1. Expect the gap to WIDEN at Pangu's epoch 6 — which is exactly what the
    per-epoch telemetry was built to catch.
  - **Epoch budget arithmetic** (from the measured rows, for the still-open decision):
    ai-rossby **2.19 h/epoch** training + 0.09 h validation ⇒ 100 epochs ≈ **228 h**,
    which does NOT fit one 168 h `capacity` job. PanguWeather **2.55 h/epoch** training
    + production validation (129 ICs × 60-step rollouts, est. 0.2–0.7 h) ⇒ 100 epochs
    ≈ **275–325 h**, i.e. 4–5 uninterrupted 72 h `preemptable` links at best.
  - **BEST-CONFIG SWEEP for ai-rossby SFNO — job 7365119, 12 configs, ONE node
    (`x3109c0s19b1n0`), back-to-back.** These ratios are same-node/same-job and are
    therefore trustworthy in a way the cross-job parity ratio is not.

    | config | cls | step_med | samp/s | vs ckpt3 | peak GB |
    |---|---|---|---|---|---|
    | `ckpt3` (current) | N | 0.6517 | 6.138 | 1.000× | 21.40 |
    | `ckpt2` | N | 0.5116 | 7.819 | **1.274×** | 33.96 |
    | `ckpt1` | N | 0.4986 | 8.023 | **1.307×** | 36.06 |
    | `ckpt0` | N | 0.4987 | 8.020 | 1.307× | 36.11 |
    | `ckpt1`+static_graph | N | — | — | **CUDA OOM** | — |
    | `ckpt0`+static_graph | N | — | — | **CUDA OOM** | — |
    | `ckpt3`+workers=8 | N | 0.6579 | 6.080 | **0.991×** | 21.40 |
    | `ckpt1`+workers=8 | N | 0.5013 | 7.980 | 0.997× | 36.06 |
    | any `batch_size` 2 or 4 | C | — | — | **CUDA OOM** | — |

    - **`checkpointing` is the only lever that pays.** The expensive branch is `>=3`
      (checkpoint every block); dropping to 2 recovers 1.274× of the 1.307× available.
      **`ckpt0` and `ckpt1` are indistinguishable** (0.4986 vs 0.4987 s, 36.06 vs
      36.11 GB), so the `>=1` encoder/decoder branch costs and saves nothing — 1 is
      the sensible floor and there is no reason to run 0.
    - **`num_workers` is worthless here, and slightly negative** (0.991×, 0.997×) —
      exactly as the measured `data_idle_frac` of 0.0068 predicted. **The old "+9%
      throughput at 8 workers" does NOT transfer**: that was PanguPlasim at 449 ms/step;
      at ~500-650 ms with a 1.18 B model the loader is already fully hidden and extra
      workers only add startup cost. Do not re-try this knob on SFNO.
    - **`ddp_static_graph` OOMs** at ckpt1/ckpt0 — it retains extra state. The
      `sfno_plasim.yaml` comment calling it "safe to enable here" is not true at low
      checkpointing. Untested at ckpt2/ckpt3, where there is headroom.
    - **`batch_size` > 1 is unavailable** at any fast setting; even bs=2 at ckpt1 OOMs.
  - **ZeRO-1 SWEEP — job 7366778, real `train.py` (EMA on), 8 configs, one node.
    ZeRO is REQUIRED, not optional: without it the entire speedup is unreachable.**
    Predictions were written into the script before it ran; **6 of 8 exact**.

    | config | step | vs base | peak GB | predicted |
    |---|---|---|---|---|
    | `ckpt3` no-ZeRO | 711.5 ms | 1.000× | 25.91 | ~25.8 ✅ |
    | `ckpt2` no-ZeRO | — | — | **OOM** | "marginal ~38.4" ❌ |
    | `ckpt1` no-ZeRO | — | — | **OOM** | OOM ✅ |
    | `ckpt3` + ZeRO | 724.1 ms | 0.983× | **19.46** | fits ✅ |
    | `ckpt2` + ZeRO | 574.1 ms | **1.239×** | 32.02 | ~31.8 ✅ |
    | `ckpt1` + ZeRO | 561.0 ms | **1.268×** | 34.12 | ~33.9 ✅ |
    | `ckpt0` + ZeRO | 559.6 ms | 1.271× | 34.17 | ~33.9 ✅ |
    | `ckpt1` + ZeRO, bs=2 | — | — | OOM | "may now fit" ❌ |

    - **ZeRO-1 saves 6.45 GB** (25.91 → 19.46 at ckpt3; predicted 6.6) and costs
      **1.7%** in step time — the all-gather plus losing the fused AdamW kernel.
    - **Without ZeRO, ckpt2 and ckpt1 both OOM**, so on plain DDP this model is
      pinned at ckpt3 and none of the 1.24-1.27× is reachable. That makes ZeRO
      the enabling change, not a memory nicety.
    - **The EMA accounting is now empirically anchored**: `ckpt3` real training
      measured **25.91 GB** vs the bench's 21.40 GB and the earlier smoke's
      25.89 GB — a 0.02 GB reproduction. Bench peaks understate production by
      the 4.4 GB EMA copy, as derived.
    - **Bench step times also understate production by ~9%**: same `ckpt3`
      config, 651.7 ms in the bench vs 711.5 ms in the real trainer — the EMA
      sweep over 1.18 B params plus DDP metric reduction and logging.
    - **The one wrong prediction is the useful one.** `ckpt2`/no-ZeRO was called
      "marginal ~38.4 GB" and OOM'd ⇒ **these estimates run ~1-2 GB optimistic.**
      Do not trust a margin under ~2 GB.
    - **Not fragmentation.** The OOM reports 38.29 GiB allocated with only
      57.76 MiB reserved-but-unallocated, so `expandable_segments` would not have
      helped — these are genuine capacity limits. (The `PYTORCH_CUDA_ALLOC_CONF`
      vs torch-2.10 `PYTORCH_ALLOC_CONF` naming question is therefore hygiene,
      not a live defect.)
  - **RECOMMENDED ai-rossby production config: `model.checkpointing: 2` +
    `use_zero_optimizer: true`** — 1.239× at 32.02 GB (7.47 GB headroom), over
    `ckpt1`'s 1.268× at 34.12 GB (5.37 GB). `ckpt1` buys 2.3% for 2.1 GB less
    headroom, and the estimates just proved ~1-2 GB optimistic; a 170 h run that
    dies at hour 100 costs far more than 2.3%. `ckpt0` is strictly worse than
    `ckpt1` (same speed, more memory) — as the first sweep also found.
    At 1.75 h/epoch + 0.09 validation, 100 epochs ≈ **184 h**, down from ~225 h.
    **Still not adopted**: needs the §4 equivalence gate, and a validation-inclusive
    smoke (a 2-epoch/20-step run suffices — `peak_mem_gb` is run-to-date, so
    epoch 2's row already includes epoch 1's validation).
  - **⚠ THE BENCH SWEEP'S MEMORY NUMBERS UNDERSTATE PRODUCTION BY ~4.4 GB — and it flips the
    recommendation.** `profile_train.py` builds **no EMA**; `train.py` does, and
    `ModelEMA` holds a full fp32 shadow copy = 1,182,108,160 × 4 B = **4.40 GB**.
    Confirmed against the smoke: train.py peak 25.89 GB vs bench peak 21.40 GB, a
    **4.49 GB** delta. Adding it back, against 39.49 GiB usable:
    | config | bench peak | + EMA | verdict (before validation) |
    |---|---|---|---|
    | ckpt3 | 21.40 | ~25.8 | fine — matches the smoke's 25.89 |
    | ckpt2 | 33.96 | ~38.4 | **~1 GB margin — marginal** |
    | ckpt1 | 36.06 | ~40.5 | **would OOM** |
    So the fastest configs are **not actually reachable in production**. `ckpt2` is the
    only candidate and it is borderline before validation is even counted. **Nothing is
    adopted**: this needs a real short training run (EMA + validation) at ckpt2, then
    the §4 equivalence gate vs ckpt3, before it goes anywhere near a production run.
  - **Worth a cheap check:** torch 2.10 documents the allocator knob as
    `PYTORCH_ALLOC_CONF`, while `polaris_env.sh` exports `PYTORCH_CUDA_ALLOC_CONF`. If
    the old name is no longer honored in the ai-rossby venv, `expandable_segments:True`
    is silently inactive there — which would change every OOM verdict above.

  - **PRODUCTION RUNS (CANCELLED, pending the best-config decision) — 100 epochs each, to CONVERGENCE (owner's call: the goal
    is trained models, not only parity/profiling).** Both pre-chained with
    `polaris_submit_chain.sh`, so neither needs daily human resubmission; every link
    re-runs its script from the top and resumes from the run's checkpoint.
    - **ai-rossby** → `capacity`, **2 × 168 h**: **7365020** (Q) → 7365021 (H).
      ~228 h needed, so link 2 finishes it. This holds the project's ONLY capacity
      slot (`max_run=1` per project) — coordinate before queueing anything else there.
    - **PanguWeather** → `preemptable`, **6 × 72 h**: **7365022** (Q) → 7365023-27 (H).
      ~275–325 h needed; 6 links = 432 h of margin, deliberately over-provisioned
      because a link that starts after training completed just resumes a finished run
      and exits in minutes. `preemptable` is scheduling again (26 running at submit
      time), unlike 2026-08-05 when it never started a single job.
    - Both at `seed=0`, `num_data_workers/num_workers=1`, bf16, batch 1 × 4 ranks,
      `checkpointing: 3`, telemetry on for every epoch. Watch progress with the
      per-epoch CSVs under `$MEMBER_ROOT/bench/epoch_telemetry_*.csv`, not the exit
      code — a preempted link exits non-zero and that is normal.
  - **Group read+write DONE and verified**: `ai_rossby_data` stores are `drwxrwsr-x`
    with `-rw-rw-r--` chunks, likewise `runs/`, `bench/`, `pangu_polaris_data/`,
    `baselines/`. The `chmod -R` over the 35 zarr stores took ~50 min (millions of
    chunk files) — run it in the background, not inline.
  - **Still open:** jesswan's sign-off on the fills (DESIGN §1) before any resulting
    model's numbers are reported.

- **2026-08-06** — **PRODUCTION CONVERSION COMPLETE (35 stores) — after routing it
  through `debug`, because `preemptable` never scheduled it.**
  - **Final dataset**, verified on disk (not from logs): `train/` **30 stores,
    2015–2044**, contiguous, 1460 timesteps each = **43,800 training samples**;
    `val/` **2045–2048** full years + **2049** as the 1-sample tail store. Exactly
    the handoff §5 Step 4 target. Production training is no longer blocked on data.
  - **⚠ `val/2046.zarr` existed but was WRONG — and "it's on disk" would have hidden
    it.** It was the 1-sample tail store from the first smoke (time len **1**,
    22 MB) whereas production needs 2046 as a full val year and 2049 as the tail.
    Rebuilt to 1460. Skipping it would have given production a validation year of
    one sample — surfacing much later as a metrics oddity, not as a conversion bug.
    **Check `time` length, never mere existence.**
  - **The scheduling lesson, and a wrong diagnosis of mine.** I blamed queue times
    on requested walltime and resubmitted three times on that theory (18 h → 3 h →
    2 h). Wrong: the jobs had been **eligible 11 h 28 m** with
    `comment = Insufficient amount of resource: queue_tags`. `preemptable` only
    runs on nodes prod isn't using and had none (backlog grew 106 → **134 queued**
    while we waited). Walltime was never the constraint.

    | queue | evidence, 2026-08-05/06 |
    |---|---|
    | `debug` | **9/9 jobs started**, median wait **19 s**, worst 27 min |
    | `preemptable` | 18 h job: never started in 12 h. 8× 2 h jobs: never started in 11.5 h |

    **Every piece of real work that completed came through `debug`**; everything
    sent to `preemptable` after 16:08 did nothing. Check `eligible_time` (with an
    anchored grep — `wfp_eligible_time_exp` is a substring trap) before theorising
    about priority.
  - **How it was done:** `debug` allows one job at a time, so a driver submitted 5
    sequential chunks of ≤7 stores (6.2 min/store measured ⇒ ~43 min against a
    55 min wall). All five green on `CONVERT_ALL_OK`; **3 h wall total.** The driver
    stops on the first failed chunk rather than leaving a silent hole. This is only
    expressible because of the `${VAL_YEARS-}` fix (`b5f80060`) — with `:-`, every
    train-only chunk would have silently converted `val/2045` and raced.

- **2026-08-05** — **DESIGN §4 equivalence gate BUILT, first baselines captured — and
  `torch.compile` FAILS it, so the 1.40× is NOT adopted.** Commit `402d336e`,
  job **7353187**.
  - **What now exists.** `baselines/ai_rossby_pangu_plasim/{eager,compiled_default}.json`
    — the repo's first §4 artifacts — plus `equivalence.py` (deterministic K=20
    capture), `polaris/compare_baselines.py` (§4.1's metric, stdlib-only), and
    `polaris/polaris_equivalence_compile.pbs` (both captures in ONE job, so the
    only difference is the change under test). §4.1 was the item blocking every
    optimization; it is now runnable in ~3 min.
  - **The verdict, stated plainly: FAIL.**

    ```
    max rel err 4.019e-01 > 1e-2   at forward_output_stats.surface.mean
    loss trajectory alone          1.113e-01  (step 11, diagnostic)
    surface.std    0.44623 -> 0.43339   (2.9%)
    upper_air.max  1.44501 -> 1.28017  (11.4%)
    ```

    The `std`/`max` drifts are on well-scaled quantities, so this is **not** just
    the near-zero-mean artifact the headline number might suggest.
  - **The divergence COMPOUNDS — it does not start large:**

    | step | 0 | 2 | 3 | 10 | 15 |
    |---|---:|---:|---:|---:|---:|
    | rel err | **8.30e-04** ✓ | 3.37e-03 | 8.29e-03 | 1.82e-02 ✗ | 1.30e-02 ✗ |

    Step 0 — identical weights, one forward+backward — agrees to **8.3e-4**, well
    inside tolerance. So the per-step computation matches; what fails is the
    20-step trajectory, because each optimizer step feeds slightly different bf16
    weights forward. **That is expected of any change perturbing bit-level
    arithmetic, correct ones included.**
  - **Recorded as a failure anyway, and the tolerance stays 1e-2.**
  - **FOLLOW-UP (job 7353316, `MODE=fixed`): compile fails the GENEROUS test too.**
    Added `AI_ROSSBY_EQUIV_MODE=fixed` — K forward+backward passes with **no
    optimizer step**, so weights are identical every iteration and step *i* cannot
    inherit from step *i−1*. Backward still runs (63% of step time, where the
    fusion is) and `grad_norm` is recorded. This separates the two hypotheses, and
    **both turn out to be real**:

    | step | 0 | 5 | 10 | 19 |
    |---|---:|---:|---:|---:|
    | train mode | 8.3e-4 | 8.0e-3 | **1.8e-2** ↗ | 1.6e-3 |
    | **fixed mode** | 8.3e-4 | 1.2e-3 | 8.9e-4 | 1.8e-3 |

    Compounding was genuine (fixed-mode loss error is flat at ~1e-3). **But a real
    per-step difference remains, which the compounding had masked:**
    `grad_norm` **5.33e-02** at identical weights, `surface.std` 3.02e-02,
    `surface.min` 1.34e-01 (**0.606 of the field's own σ**), `upper_air.min`
    6.53e-02 (0.400 σ). The aggregate loss agrees to ~0.1% while the output
    *extremes* differ by 0.4–0.6 σ — an averaged scalar hiding pointwise
    disagreement, which is exactly why §4.1 asks for output stats *alongside* the
    losses.
  - **Verdict strengthened: NOT adopted.** The test built to give compile its best
    chance still fails on quantities unrelated to trajectory amplification.
  - Consequently the procedural question (should §4.1 gate fusion changes on fixed
    weights rather than a training trajectory?) is **still worth settling on its
    own merits, but it is not what blocks this lever** — and it was never something
    to settle by adjusting a number until it passed (rules #1/#11).
  - **Also open:** this is a **world-size-1** capture. `torch.compile` is applied
    beneath the DDP wrap, so §4.1 additionally requires a 4-GPU baseline before
    adoption could even be considered.
  - The comparator's failure modes were unit-tested **before** it gated anything:
    identical and 0.5%-drift pass; 5% drift, seed mismatch, output-shape drift and
    3% stat drift are all correctly rejected.

- **2026-08-05** — **DESIGN §5 rung 1 MEASURED on ai-rossby: `torch.compile` = 1.40×.
  Getting there exposed a silent correctness bug that would have inflated it.**
  Commit `6972a940`.
  - **The bug, first.** `train_step` introspects the model's forward signature to
    decide whether to pass `train=True` + targets. `torch.compile` returns an
    `OptimizedModule` whose forward is dynamo's wrapper, so introspection answered
    **`eager=True` vs `compiled=False`** (measured). Compiling therefore dropped
    `train=True`, flipping `if self.checkpointing > 0 and train:` false —
    activation checkpointing silently OFF and a different forward path. A speedup
    measured that way times a *different computation* (rule #1), **and flatters
    itself**, since disabled checkpointing is faster. Fixed with a bounded
    `_unwrap_model` peeling `_orig_mod` then `.module` (`_orig_mod` first —
    `OptimizedModule` forwards unknown attrs, so `hasattr(m,"module")` misses it).
  - **The measurement** (jobs **7352022** eager vs **7352948** compiled; 4×A100,
    bf16, `default` mode, 40 warmup + 20 steps, `train/2015.zarr`):

    | | eager | compiled | ratio |
    |---|---:|---:|---:|
    | `step_med` | 449.6 ms | **320.8 ms** | **1.401×** |
    | `samples_per_s` | 8.90 | 12.47 | 1.401× |
    | `samples_per_s_wall` | 8.80 | 12.29 | 1.397× |
    | `peak_mem_gb_max_rank` | 24.98 | 21.07 | 1.186× |
    | `n_params` | 60,708,112 | 60,708,112 | identical ✓ |

  - **The profile predicted this.** nsys put `backward` at **63%** of the step
    (288.2 ms vs `forward_loss` 37.0 ms — a 7.8× ratio where ~2× is healthy) with
    `elementwise_kernel` holding the top three kernel slots: fusion-starved, which
    is exactly what compile fixes. The memory drop is the same effect from another
    angle — fused kernels stop materialising intermediates. **`default` mode, NO
    cuda graphs**; `reduce-overhead` may add more given ~62k launches/step.
  - **NOT ENABLED.** DESIGN §4 requires equivalence vs a captured baseline and §5
    leaves rung 1 unset until it exists — it still doesn't. Shipped as a
    measurement + correctness fix, not a config change. **The gate is now cheap:**
    the harness already runs a real `train_step` and merely discards the losses;
    recording them would both capture the repo's first `baselines/<model>/` entry
    and gate this lever. That is the next job.
  - **Login nodes are not a test environment.** The same suite returned `rc=0`,
    `rc=1`, and `rc=130`-at-scipy-import within one hour, then passed cleanly in
    isolation minutes later — pure contention. Added
    `polaris/polaris_recipe_tests.pbs` (compute node, gates on pytest's summary
    LINE not the exit code, since a killed run can exit misleadingly):
    **`RECIPE_TESTS_OK 13 passed`** (job 7353083, 46 s). The login node had been
    displaying 6 dots for a 13-test suite — truncated output that I twice read as
    a result. **A truncated capture with `rc=0` is not evidence, and one failure
    on a shared login node is not evidence of a code defect.**

- **2026-08-05** — **ai-rossby profiling harness built and GREEN; `n_params` overturns
  the standing explanation for the SFNO-vs-PanguPlasim speed gap.**
  Commits `b5f80060` (converter fix) → `84dbacf6` (harness).
  - **The measurement** (job **7352022**, 4×A100, bf16, 5+20 steps on `train/2015.zarr`):
    `step_med` **449.6 ms**, p90 449.9, std 0.27 ms, `samples_per_s` 8.90,
    `peak_mem` **24.98 GB**, `data_idle_frac` 0.0107, **`n_params` 60,708,112**.
  - **`n_params` was never logged before, and it inverts the story.** ai-rossby is
    **60.7 M params vs SFNO's measured 1,182,108,160 — 19.5× SMALLER** — yet only
    ~1.6× faster per step. So the gap was never SFNO's spectral transforms (the
    profile puts cuFFT/SHT at **3.3%**). A 19.5× smaller model gaining only 1.6×
    means **ai-rossby is badly under-utilizing the GPU.** That is now the first
    thing to profile, and it is on evidence rather than assumption.
  - **The old step time was ~19% pessimistic.** `LaunchLogger`'s 537 ms is an epoch
    *mean* including first-step warmup; the synced median is **449.6 ms**. Every
    wall-clock estimate derived from 537 ms is stale.
  - **Memory was never the constraint we assumed.** 24.98 GB of 40 GB at
    `batch_size=1` — ~15 GB headroom, and `checkpointing`/`embed_dim` were never needed.
  - **Shape.** `profile_train.py` is a *wrapper*, not a fork: it imports
    `build_model`/`build_datapipe`/`build_loss`/`make_optimizer`/`make_scheduler`/
    `train_step` from the real trainer, so it cannot drift (rule #4). `train.py` is
    unchanged; `train_loop.py` gains gated NVTX only. CSV's first 19 columns are
    s2s's, in order, with 4 appended (`samples_per_s_wall`, `data_idle_frac`,
    `config_sha16`, `n_params`).
  - **Verification, in full:** existing `test_train_loop.py`/`test_multistep_train_step.py`
    6 pass **before** the edit, 6 after with NVTX unset, 6 with `AI_ROSSBY_NVTX=1`;
    the new drift guard is **negative-tested** (renaming `backward`→`backwards` fails
    it with the exact diff, revert byte-identical); PanguWeather's own guard still
    `BENCH_INSTR_OK (10 tests)`.
  - **Two bugs caught during implementation, both by reading the real call sites:**
    (a) `make_optimizer`/`make_scheduler` need `_flatten_{optimizer,scheduler}_cfg`
    first — passing raw nested config silently builds the WRONG optimizer;
    (b) `multistep_train_step`'s rollout gets ONE `forward_loss` range, not K, or
    `parse_nsys.py`'s per-range median becomes meaningless.
  - **A 4-reviewer adversarial pass rewrote the plan first**, and killed a claim I had
    repeated several times: the **"1.51× faster" figure was apples-to-oranges**
    (SFNO's `Time taken for epoch` includes validation + checkpoint; ai-rossby's was
    train-only). Like-for-like it is **1.33×**, and even that carries four
    uncontrolled confounds — chiefly that **`checkpointing: 3` is NOT a matched
    setting**: graduated in SFNO (`>= 3` does strictly more recompute), a flat
    boolean in ai-rossby (`> 0`, so 1/2/3 are byte-identical). Also uncontrolled:
    `num_data_workers` 1 vs 8, and her step time drifting 702→832 ms (+18.5%).
  - **Converter landmine fixed** (`b5f80060`): `${VAL_YEARS:-2045}` defaults on
    *empty* as well as unset, so splitting the conversion across parallel jobs gave
    every train-only shard `val/2045` — concurrent racing writers on one store with
    `--overwrite`. `qsub -v VAR=" "` does not dodge it (PBS strips it to empty;
    verified with `od`). Fixed `:-`→`-` plus a tail-year guard, four cases unit-tested.

- **2026-08-05** — **Handoff-v2 Steps 1-3 executed: corrected normalization regenerated,
  norm zarr rebuilt, and the ai-rossby PanguPlasim training smoke is GREEN on 4×A100.**
  The whole point of the exercise, measured: **`TSOI_10CM` normalizes to spread 0.9968,
  was 0.122.**
  - **Step 1 — regeneration** (job **7340945**, exit 0, **34:54**). Full 51,100-file /
    ~2.15 TB moments pass. Measured vs the handoff's predicted values:
    `SST` **8.4407 / 12.0659** (predicted 8.44 / 12.06), `TSOI_10CM` **271.1259 / 16.3902**
    (predicted 271.09 / 16.39). The gate `check_normalization.py` passes **all 23 contract
    channels** at mean ≈0 / spread ≈1. The two fill warnings are the documented false
    positives (`PFTDATA_MASK` is degenerate — valid values are only ever 1; `SST`'s −1.8
    fill sits exactly *on* the data minimum, so the `valid_lo <= fill` test trips on a
    float boundary).
  - **Step 2 — norm zarr rebuilt.** Old store moved to `…zarr.prefix` (rollback path).
    Verified beyond the handoff's 2-field spot-check: **all 26 vars × all levels are
    bitwise-identical to the source `.nc`** (worst abs diff 0.000e+00). Old→new:
    `SST` 109.963/123.908 → 8.441/12.066; `TSOI_10CM` 105.229/133.802 → 271.126/16.390.
  - **Step 3 — training smoke** (job **7341412**, exit 0, **9:36**, 4 ranks, bf16,
    365 steps/epoch, 537 ms/iter). `PREFLIGHT_OK` + `PANGU_PLASIM_RUN_OK`. Loss finite and
    decreasing within the epoch (8.122e-01 → 6.949e-01 → 6.870e-01), epoch loss 8.410e-01,
    **val_loss 6.708e-01**. **All four "never executed" unknowns from handoff §5 are now
    resolved:** the model constructs (the `sol_in` patch holds on a real config), the
    tensors line up, **it fits in 40 GB with no lever pulled** (`checkpointing: 3` and
    `embed_dim: 240` were NOT needed), and the windowing divides.
  - **The window_size question was answerable statically** — no run needed. 18 levels → 9
    after the level patch, **+1 for the surface row = 10**, which the vertical window of 2
    divides. (`pangu_plasim_e3sm.yaml:65`'s comment reaches the right conclusion by wrong
    arithmetic — it stops at 9.) Horizontally the downsampled 45×90 is *not*
    window-divisible, but `Transformer3DBlock` applies `get_pad3d`/`crop3d`, so it pads.
  - **Cross-check against jesswan's shipped `.nc` — this is the real validation, and it is
    stronger than her regeneration would have been.** The defect touches 2 channels; the
    other 24 carry 0% NaN and so have no fill dependence, meaning her Jul 8 file is an
    *independent* reference for them. **18 of 26 variables reproduce her numbers** (most to
    ~1e-8). `U`/`V`/`RELHUM` appear changed only under *relative* error on near-zero means —
    absolute diffs are ≤0.15 on values up to 83, inside §3's ±1%. Note this tests our
    arithmetic against a differently-computed source, whereas a second regeneration by her
    would share our fill assumptions rather than test them.
  - **⚠ NEW, latent — cloud std floors differ.** Shipped `_std_corr` floored **16**
    near-zero levels to 1.0; ours floors **9**. Where hers underflowed to exactly 0 (and so
    got floored), our float64 shifted accumulation leaves catastrophic-cancellation
    residue: `CLDICE` ×4 at **2.03e-12** and `CLDLIQ` ×3 at 7e-28…6.6e-24 instead of 1.0.
    **Inert today** — `CLDICE`/`CLDLIQ`/`CLOUD` are commented out of
    `E3SM_SFNO_H5_POLARIS.yaml:52` and absent from the 108-field contract — but **anyone
    re-enabling those channels would divide by ~1e-12.** Deliberately NOT "fixed": the
    floor is a science call, and rule/trap #8 forbids raising `STD_ZERO` globally because
    `PRECT`'s real std is ~8.3e-8. Flagged for jesswan alongside the fills.
  - **Scheduling: Step 1 was rerouted to `debug`.** The handoff's job 7337234 sat queued
    ~5 h behind **106 preemptable jobs**. Measured the cost first (0.62 s/file × 51,100 ÷ 32
    workers ≈ 16 min), so it fits debug's 1 h cap — it ran in 34:54. Submitted under
    `TAG=dbg` so its outputs (`data_dbg_*.nc`) could not collide with the queued job's,
    which stayed queued as a fallback; results were promoted to the canonical names
    afterward and 7337234 was cancelled (state F, never ran). **Reusable trick:** the `TAG`
    env var makes a racing submission safe.
  - **⚠ The handoff's backup step had never been performed.** `$PANGU_AUX/pre_fix/` did not
    exist, so job 7337234 would have destroyed the only local copy of the pre-fix statistics
    (the ones §2's analysis and the 85-epoch checkpoint rest on). Backed up before anything
    else. Anyone re-running this: do that step first, literally first.
  - **jesswan's own regeneration still has not started** (checked 2026-08-05 ~03:00 UTC):
    every stats file in `$E3SM_ROOT` is still dated Jul 8 17:01, and her newest PBS job
    (`7322826`, `physicsnemo_e3sm_sfno_peryear`, ended Aug 4 17:19 **exit 1**) is a training
    job. Caveat: ALCF homes are 0700, so this is "no evidence", not proof.
  - **Stale-doc sweep (handoff §8).** `e3sm_h5_to_seqzarr.py` said TSOI land mean **268 K**
    in two places; measured is **272 K** — corrected (comment-only, no behavior change, so
    the makani coupling in rule #5 is unaffected). The other two §8 items —
    `verify_pangu_store.py`'s "six checks" and the CHANGELOG's `train.py:636,739` anchors /
    "NOTHING SUBMITTED YET" line — were **already fixed** in `3cf6c6c4`.

- **2026-08-04** — **SST-fill investigation (jesswan): the degC/Kelvin bug is confirmed, my
  "it doesn't affect training" framing was WRONG, and an audit shows the defect class is
  structural — it exists only where stats are precomputed externally.**
  - **Correction to my own earlier claim.** I argued normalization is affine-invertible so a
    first-layer weight absorbs the bad fill. **That holds in fp32 only.** AMP was on
    (`enable_amp` defaults `not args.no_amp`; no launcher passes `--no_amp`), and under bf16
    the ocean SST range collapses to **75 distinct values** (fp16: ~575). Quantization is
    irreversible — no downstream weight recovers it. jesswan is right: SST is a *feature*, so
    training and inference are both affected. Measured: ocean signal **0.093σ**, ~**0.47 °C
    per bf16 level**.
  - **⚠ R2 CONFIRMED INDEPENDENTLY, and it reframes the cost.** `SST`, `ICE` and `sol_in` are
    **bit-identical across all 35 years** at the same day-of-year (`max|diff| = 0` for 2015 vs
    2020/2035/2049), while prognostic `TREFHT` varies normally. So this archive has **no
    interannual SST variability and no SSP245 warming trend** — measured global-mean ocean SST
    trend is *exactly* 0.000 °C/decade. The quantization therefore costs no ENSO/trend signal
    (there is none); it costs **spatial + seasonal** structure — SST fronts, upwelling, the ice
    edge — and because the field repeats exactly, that blurring is a *systematic* forcing bias
    seen 35 times, not averaging noise.
  - **Audit (jesswan's ask) — physicsnemo_sfno and makani are CLEAN, and immune by
    construction**, via two different mechanisms:
    | path | SST fill | stats | exposed? |
    |---|---|---|---|
    | PanguWeather | 270 (degC field) | precomputed `.nc`/`.npz` | **YES** |
    | **ai-rossby (ours)** | 270 (parity) | precomputed zarr **+ `amp: bf16`** | **YES — worst** |
    | physicsnemo_sfno | −1.8 ✅ | **writes none**, online BatchNorm | no |
    | makani | −1.8 ✅ | **in-stream from PACKED data** | no |
    Both already document SST as degC and explicitly rejected the npz because it "was computed
    under a land-fill convention of ~270 for SST"
    (`e3sm_h5_to_seqzarr.py:242`, `convert_e3sm_to_makani.py:56-57`). **Nothing to change there.**
    ⇒ **The bug class exists only where an externally-precomputed stats file is paired with a
    separately-configured fill.** That is exactly PanguWeather and our ai-rossby path.
  - **Corrected constants are exactly derivable — no raw recompute needed.** Because the land
    mask is static AND SST repeats bit-identically (R2), the 35-year stats equal any year's,
    and the shipped 270-fill pair inverts analytically to ocean-only moments (land fraction
    0.37352, ocean mean 14.546 °C, ocean std 11.514 — cross-checked against a direct 12-sample
    measurement: 14.598 / 11.488). Re-filling gives:
    | SST fill | mean | std | ocean signal |
    |---:|---:|---:|---:|
    | 270 (current) | 109.9630 | 123.9083 | 0.093σ |
    | **0.0** | **9.1130** | **11.5140** | **1.000σ** |
    | **−1.8** | **8.4407** | **12.0659** | **0.954σ** |
    Either is ~10× the current signal and ~36× the bf16 levels. **−1.8 additionally unifies
    all three pipelines**; 0.0 is marginally better numerically. jesswan regenerates the
    PanguWeather `.nc`; these are the cross-check.
  - **Deferred by jesswan's call: "fix SST first".** `TSOI_10CM`'s mismatch stands —
    shipped stats encode a **0-fill** while the config fills **270**, measured over 12
    samples (0.195σ vs 1.606σ if matched = **8.2× under-weighted**). Unlike SST this channel
    is **scored** (a `land_variable`, i.e. prognostic), so it hits a loss term, not an input.
    Fix in the same regeneration pass when SST is settled.

- **2026-08-04** — **ai-rossby PanguPlasim bring-up: Step-0 variable-parity gate PASSED
  (10/10), then steps 1-6 implemented. Nothing submitted to the scheduler yet.**
  Commits `b28b2e7c` (gate) → `b10d5e5c` (venv + code + configs) → `2924b90b` (norm store +
  conversion tooling).
  - **The gate.** `ai_rossby_panguweather_variable_parity.md` proves the run trains the
    **identical 108-field variable set** as jesswan's PanguWeather E3SM run — group for
    group, name for name, **order for order**, level for level, fill for fill. The contract
    lives once in `ai_rossby_variable_contract.py::PLANNED` and is machine-checked two ways:
    `--check-ground-truth` (vs jesswan's YAML) and `--check-artifacts` (vs our configs,
    converter and store attrs). **26/26 pass.** stdlib-only so it runs on a login node
    before any venv exists — but needs `python3.12`, since the bare `python3` there is 3.6.
  - **Ground truth identified from run artifacts, not assumed:** the trainer logged
    `E3SM_SFNO_H5_STAMPEDE_jsw.yaml` at startup in 7 of the 10 `e3sm_train_*.e` logs, from
    jesswan's Stampede3 `$WORK` (`jwan4`), with a descending loss. It turned out not to
    matter: **all four** E3SM configs (`_STAMPEDE_jsw`, `_DERECHO_jsw`, `_POLARIS`,
    `_POLARIS_ALLDATA`) carry a byte-equivalent variable contract.
  - **⚠ Trap that would have been a FALSE PASS:** PanguWeather carries two level lists.
    `levels: [5, 10, 20, …]` are nominal hPa *labels*; `sigma_levels: [4.714998…, …]` are
    the values actually embedded in the H5 keys, and `use_sigma_levels: True` makes the
    latter what reaches the loader. The check compares `sigma_levels` — verified identical
    to ai-rossby's 18 levels, `max|diff| = 0.0`.
  - **ai-rossby's shipped E3SM defaults were NOT the contract** — 3 surface vars vs 6 (+2
    land), 1 diagnostic vs 3, and **two groups in a different order**. Order is the silent
    one: `ClimateZarrDataset` stacks tensors in **store-attrs** order while fills and loss
    build from the **model-config** lists (`dataset.py:533` vs `train.py:657` fill / `:735`,`:770` loss, via `_surface_channel_names()` at `:92`), so a
    permutation is correctly-shaped and `torch.cat` raises nothing. Hence the preflight.
  - **Code edits (vendored tree).** `sol_in` added to `_solar_names` in *both*
    `pangu_plasim_legacy.py` and `pangu_plasim.py` (E3SM's solar field is neither `rsdt`
    nor `toa_incident_solar_radiation`, so the ctor raised `ValueError`). One
    `_surface_channel_names()` helper in `train.py` replaces four restatements of "the
    surface tensor is `[surface|land|ocean]`" — the two sites the handoff named, **plus two
    it did not**: the `ArchesWeatherLoss` branch (off our `loss=mae` path, but a landmine
    for a later loss switch) and `channel_equal_weight`'s `n_surf`, which would
    under-weight the surface term against the channels it scores.
  - **Fills — the user's Kelvin confirmation, and where it does not hold.** `TSOI_10CM =
    270 K` is right (soil temp is Kelvin; land mean 268 K ⇒ effectively a mean-fill).
    **`SST` is degC in this archive** (measured −1.80…32.92), so the same 270 is ~8× its
    maximum — R4, inherited from a mechanical rename of ERA5/PlaSim's
    `mask_fill['sst']=270.` where the field really was Kelvin. **Reproduced deliberately**
    (parity is the point, and the shipped stats were computed under the same 270-fill, so
    fill and stats agree — unlike `TSOI_10CM`, R3). `SST = -1.8` + masked stats remains the
    fast-follow. **Both still need jesswan's sign-off before any numbers are reported.**
  - **Normalization store BUILT + verified** (login node, seconds):
    `$AI_ROSSBY_DATA/e3sm/norm/normalization_2015-2050.zarr`, 26 vars / 18 levels, from
    `$PANGU_AUX/data_2015-2050_{mean,std_corr}.nc` — the *same constants jesswan's run
    used*, converted not recomputed. Round-trip vs source over all 23 contract vars:
    `max|diff| = 0`. The `Z`→`Z_2` fix the handoff anticipated was **already applied**
    upstream by `polaris_prepare_e3sm_stats.py`; the tool's rename branch is a no-op here.
    No zero/tiny std among the 23 (that is what `_std_corr` corrects).
  - **Handoff corrections worth keeping:** (a) `uv` *does* ship with the ALCF conda module,
    but only after **`conda activate base`** — `module load conda` alone puts neither
    `python` nor `uv` on PATH. (a2) **The extras list was incomplete: `--extra
    datapipes-extras` is REQUIRED.** The fork promoted `xarray`/`zarr`/`netCDF4` to core
    deps but **not `dask`**, which `pangu_h5_to_zarr.py` imports to allocate the Zarr
    template. Caught by running the converter on 8 timesteps locally; otherwise it would
    have died `ModuleNotFoundError` on the compute node **after queueing**. The venv script
    now installs and verifies it. (b) The dataset config is a **new** file,
    `conf/dataset/e3sm_pangu_parity.yaml`, **not** an overwrite of `conf/dataset/e3sm.yaml`
    — that one is the SFNO speed-bench config for a different channel set. Launch with
    `dataset=e3sm_pangu_parity`.
  - **Venv:** `polaris_setup_ai_rossby_venv.sh` → `AI_ROSSBY_VENV_OK`. torch 2.10.0+cu129,
    zarr 3.2.1, torch_harmonics 0.9.1, wandb 0.27.0; physicsnemo editable from **this**
    checkout (13 GB, `${MEMBER_ROOT}/conda-envs/ai-rossby-venv`, via
    `UV_PROJECT_ENVIRONMENT` so it stays out of the git tree). `AI_ROSSBY_VENV` +
    `AI_ROSSBY_DATA` added to `polaris_env.sh` — **no shared fallback for the venv**,
    deliberately: it is an editable install, and the PBS scripts hard-fail
    `AI_ROSSBY_WRONG_CHECKOUT` instead.
  - **Preflight is real, and proven so.** `polaris_pangu_plasim.pbs` refuses to launch
    unless every produced store's attrs AND the model/dataset configs match the contract
    (`PREFLIGHT_OK`). Negative-tested by pointing `--store` at the normalization store:
    5 store checks FAIL loudly rather than no-op'ing. Also wired
    `dataset.nan_fill_strict` (the transform supported `strict` but `train.py` never
    passed it) — on for the smoke, so a NaN reaching the model raises at its origin
    instead of surfacing later as an unattributable NaN loss.
  - **Subtree:** `physicsnemo_ai_rossby/` imported **unsquashed** from jesswan's local copy
    (`ai-rossby`, HEAD `87002adb`, a real ancestor of HEAD). 4,037 files; `.git` 310 → 315 MB
    only, because it shares objects with the existing `physicsnemo_sfno/` subtree (both fork
    awikner/physicsnemo). `subtree add` ran fine on the **login** node with
    `-c pack.threads=1` — the compute-node workaround the physicsnemo import needed was not
    required here (local path, no network fetch).

- **2026-08-04** — **Pivot: training moves to the ai-rossby recipe (PanguPlasim), for exact
  PanguWeather parity — and the SFNO 103-var conversion `7324098` was cancelled + its 463 GB
  partial store deleted.** The SFNO `unified_recipe` v2 packed store can't feed ai-rossby (which
  needs a per-variable **zarr v3** store from its own `tools/data/e3sm/pangu_h5_to_zarr.py`), so
  finishing it bought nothing for the chosen path. Decisions (with the user): model
  **PanguPlasimLegacy** on the exact PanguWeather 108-field groups (SST/ICE prescribed);
  **normalization parity-first** (reuse shipped stats + reference fills `SST=270, TSOI=270`),
  masked-stats recompute + `SST=-1.8` as a fast-follow; ai-rossby vendored as a git subtree
  `physicsnemo_ai_rossby/` on `fix/tsoi-fill-270` with its OWN uv venv (zarr≥3, torch≥2.10);
  cluster Polaris. **Full plan + a variable-parity assertion gate (Step 0) → handoff
  `polaris_ai_rossby_pangu_handoff_prompt.md`** (adversarial + cold reviewed; caught the silent
  channel-order trap, the `sol_in` solar-name blocker, and the OOM launch flags). The TSOI 0→270
  code change (`aa43824a`) stands — consistent with ai-rossby's parity fills, still pending
  jesswan sign-off.

- **2026-08-03** — **TSOI_10CM ocean fill changed `0.0 → 270 K`, and a 103-var E3SM SeqZarr
  regeneration submitted** — branch `fix/tsoi-fill-270`, commit `aa43824a`. rmehta1987's
  decision after establishing that `0.0` is 0 K over the ~61% ocean (out-of-distribution — the
  ~268 K coastline cliff dominates BatchNorm's σ and starves the channel ~26× of gradient, R3),
  while **270** is mid-distribution (land mean 268 K) and matches PanguWeather's own
  `mask_fill['TSOI_10CM']=270.`. Because PhysicsNeMo normalizes online (BatchNorm), fill and
  stats can't disagree — the old "keep 0.0" reason (npz stats at 0-fill) doesn't apply here.
  - **⚠ SCIENCE CHANGE — needs jesswan's sign-off** before any resulting model's numbers are
    reported (DESIGN §1). Made explicitly + recorded (store `nan_fill` attr + here), not silently.
    `verify_seqzarr.py` imports `NAN_FILL`, so it checks the 270 placement automatically.
  - **Jobs:** verify `7324097` (debug, N=120 random draw + exhaustive bitwise incl. TSOI=270;
    PASS=`SEQZARR_VERIFIED`) → full `7324098` (preemptable, **HELD** on
    `-W depend=afterok:7324097`; runs only if verify exits 0). **Output →
    `/eagle/.../members/mehta5/e3sm_seqzarr_allyears_tsoi270`** (`e3sm_train.zarr` 2015–2046 +
    `e3sm_val.zarr` 2047–2049, 103+5 ch, ~1.43 TB, ~11 h). **Quota OK: 16.17 / 50 TB used** (the
    old 15 TB / 1.86 TB-free note below is stale — quota was raised).
  - **⚠ Operational:** the PBS jobs read the converter from the **working tree at runtime**, so
    keep the checkout on `fix/tsoi-fill-270` (TSOI=270) until `7324098` finishes; the produced
    store's `nan_fill` attr is the post-hoc proof it used 270. This regen is the field-set that
    **matches PanguWeather** (clouds excluded + all 7: PS/TMQ/RHREFHT/FSNT/FSNTOA/SOILWATER/TSOI).

- **2026-07-16** — **The runbook survived a 3-agent gauntlet (2 adversarial + 1 cold, Fable 5),
  which refuted four of its claims and surfaced one operational blocker; all fixed, plus the
  no-babysitting mechanics the owner asked for.**
  - **🔴 QUOTA (operator-sim agent): the eagle project quota is 13.14 of 15 TB used — ~1.86 TB
    free, SHARED project-wide** — while the full plan needs ~3.1 TB and the incoming zarr
    transfer itself needs ~1.43 TB of the same pool. Runbook now leads with a storage-budget
    section (one PhysicsNeMo dataset copy ever; `myquota` ≥1.5 TB free before any TB-scale job;
    freeing/raising quota is an owner decision that precedes the first big write).
  - **Walltime + chains: the "resume automatically" claim was wrong for walltime expiry**
    (`-r y` requeues only PREEMPTED jobs; a walltime kill just stops). Fixed three ways: the 3
    production launchers (Pangu full, makani alldata full, PhysicsNeMo allyears) go 24 h → **72 h**
    (queue max, verified `qstat -Qf preemptable`: max walltime 72:00:00, 20 queued/10 running
    per user); new **`polaris_submit_chain.sh`** pre-submits N `-W depend=afterany` links so a
    multi-day run needs zero monitoring (links resume from checkpoint; post-completion links
    exit in minutes). **Mechanics proven**: jobs 7259371/72 — link 2 held on link 1's afterany,
    clean chain qdel, zero node-hours. Also `polaris_verify_store.pbs` gains `-r y` (read-only
    idempotent ⇒ a preempted EXHAUSTIVE verify requeues instead of being silently deleted).
  - **Refuted by the fact-check agent, fixed in the runbook:** the "~27 min (measured)"
    validation figure (it is an extrapolation from 112.6 s at 9 ICs; now stated as such, budget
    20–30 min); "PNGs refreshed each epoch" (they ACCUMULATE — ~103 new files/epoch, ~51,500
    over 500 epochs; `model_package_*` is the opposite: ONE dir overwritten every save);
    the Route B success rule (a resumed conversion legitimately prints a skip line INSTEAD of
    `CONVERT_OK` — keying on it would fail a correct run; key on `ZARR_ALLYEARS_COMPLETE`);
    the headroom arithmetic (13.9 GiB is vs the RESERVED peak 25.6, not the 25.0 used).
  - **Misleading-as-written, fixed:** `PHYSICSNEMO_WRONG_CHECKOUT` protects only the
    PhysicsNeMo jobs — **makani has no checkout guard** and silently falls back to the shared
    venv (= the builder's working tree) if the per-user venv build is skipped; the venv build
    command must run from YOUR OWN checkout (it editable-links whatever tree it runs from);
    jesswan-hardcoded paths became `<you>` placeholders; "fails within a second" softened
    (module loads come first, and two-levels-deep dirs pass the polaris_env.sh check).
  - **Cold-agent findings adopted:** train/val splits are NOT aligned across pipelines
    (PhysicsNeMo trains through 2046 — years Pangu/makani validate on; only **2047–2049 is
    unseen by all three**) and epoch budgets differ ~30× — recorded as runbook §7.6: no
    cross-model skill comparison without a protocol from jesswan; per-run "finished" log lines
    added (Pangu `DONE ---- rank 0`, PhysicsNeMo `Finished training!`, makani = a later link
    exits immediately); makani's pack has NO value-level verifier (stated in its section with
    the same honesty as PhysicsNeMo's); debug re-runs serialize (~2–3 h for all four); the
    stale 162-ch/3.5-GB narrative in the Pangu full launcher's header got a correction banner
    (18.9 GB measured, 108 ch, 72 h, chain pointer).
  - **W&B online logging: PROVEN live** (previous entry's smoke 7259364 synced run `gywtrgsr`
    to wandb.ai through the ALCF proxy mid-training; `ALLDATA_SMOKE_OK` unchanged). The Pangu
    smoke now carries the full launcher's online block — the block itself had never executed
    anywhere before this.

- **2026-07-16** — **🟢 The §0 smoke sequence ran with the owner's go-ahead: ALL FOUR GREEN at
  the current contracts, same day.** Every prior green predated the change it would validate;
  that debt is now paid. Read each verdict from the log token, not rc:
  - **Pangu validation smoke (7259271): `PANGU_VAL_SMOKE_OK`** — the headline. The unproven
    `utils/metrics.py` CPU-climatology fix EXECUTED for the first time (all 4 ranks reported
    `clim_*=cpu`): validation peak **25.048 GiB**, train peak 25.127, device 39.495 ⇒
    **13.86 GiB headroom**, `valid_loss=0.6989` finite, full production leads [1,12,20,40,60].
    **The §4a adversarial arithmetic (38.8–40.8 GiB ⇒ requeue-forever) is REFUTED with the fix
    in place** — Pangu production training is cleared on memory. Planning number: validation
    112.6 s at 9 ICs ⇒ ~27 min/epoch at production's 129 ICs.
  - **Pangu 108-ch training smoke (7259296): `ALLDATA_SMOKE_OK`** — 105-in/101-out, peak
    26.98 GB, `step_med` 0.643 s (n=12; the trustworthy number remains 0.602 s at n=80).
  - **PhysicsNeMo allyears smoke (7259303): `ALLYEARS_SMOKE_OK` + `PHYSICSNEMO_CSV_OK`** —
    seam stores bitwise-verified, 103+5 schema pinned, and the §6 CSV tee is now proven
    END-TO-END: 7 rows exactly as designed (5 minibatch, decreasing loss 1.518→1.330; 1 epoch
    row lr=1e-3 + GB/s; 1 validation row 1.883), git_sha stamped.
  - **makani 100/1/7 smoke (7259321): `ALLDATA_SMOKE_OK`** — fresh pack `CONVERT_ALLDATA_OK`,
    contract + names verified against the converter, real training (6.79 s) + checkpoint, and
    the benign `N_in_channels=107 ... expected 58` watchdog warning appeared exactly as
    designed. The stale 154-ch pack was MOVED (not deleted) to
    `${MEMBER_ROOT}/data/e3sm_makani_alldata_smoke_154ch_stale` — delete when convenient.
  - Also: `polaris_pipeline_runbook.md` added — the operator-facing guide (plain scientific
    language, full paths) for running the PhysicsNeMo pipeline off the transferred dataset
    (Route A: verify in place) or a rebuild (Route B). §0/§5 of the plan and the runbook's §2
    table updated with these results.

- **2026-07-16** — **The three-pipeline runbook is written (`polaris_pipelines_plan.md`) and the
  handoff's §6 tee + two gap-closing scripts are implemented. Nothing was submitted** (per the
  handoff: design, don't launch). Method: two Fable-5 agents swept PanguWeather and makani code;
  PhysicsNeMo read directly; every command in the plan carries its PASS token and what the gate
  cannot catch.
  - **§6 PhysicsNeMo CSV tee: DONE.** New `examples/weather/unified_recipe/bench_csv.py` +
    5 delimited vendor-divergence blocks in the vendored `train.py` (subtree-pull conflict
    surface is those blocks; a tripwire test greps for them). Frozen 10-column schema
    (`timestamp,epoch,step,loss,lr,gb_per_s,valid_error,n_gpus,git_sha,run_name`), rank-0 only,
    env-gated (`PHYSICSNEMO_BENCH_CSV`, unset ⇒ byte-identical no-op). **No per-step GPU sync
    added**: the loss is `.clone()`d (CUDA-graph static-buffer aliasing — N unclone'd refs would
    all read the last step) and buffered; conversion happens on a 100-row flush, the cadence at
    which LaunchLogger already syncs by string-formatting the tensor. Proven:
    `BENCH_CSV_OK (23 tests)` (login node, stdlib-only test). End-to-end proof wired but
    pending a job: the allyears smoke now also requires `PHYSICSNEMO_CSV_OK` (a real minibatch
    row) and the allyears launcher writes `${RUN_DIR}/metrics.csv` by default.
  - **`physicsnemo_sfno/polaris/polaris_verify_store.pbs` (new): verify an EXISTING SeqZarr
    store at any path** — built for the store jesswan is transferring in (not present yet,
    checked today). Three layers: attrs/generation vs the converter's current lists; **exact
    chunk-key set per array** (deliberately not `nchunks_initialized`, which counts `.partial`s
    by prefix); stride-sampled bitwise round-trip vs the h5 archive (auto-stride ≈480 samples,
    nudged off divisors of 1460 so it never re-samples the same calendar dates; `EXHAUSTIVE=1`
    escape for the full read). Attrs+chunk layer tested against the on-disk old-generation
    store — correctly refused it (`STORE_INCOMPLETE` + `STORE_WRONG_GENERATION`). The PBS has
    never been submitted. Note recorded while testing: **the stores at
    `${MEMBER_ROOT}/e3sm_seqzarr/` are pre-exclusion 157+5 smoke-scale relics** (CLDICE in
    attrs, no sentinel) — only the tracked 162-era launchers may use them.
  - **`makani_sfno/polaris/polaris_sfno_alldata_full.pbs` (new, NEVER RUN):**
    `e3sm_alldata_full.yaml` existed with **no launcher** (the "2 pbs" were pack-full +
    train-smoke — a gap the handoff missed). Mirrors the green locked launcher + the alldata
    smoke's derived contract gates. Its header carries the blocker in bold: **a 100/1/7
    checkpoint cannot currently be evaluated** (`sfno_inference/checkpoint_loader.py:74-82`
    hard-asserts 52/1/6; stock makani inference is hard-gated off) — decide the eval path
    before burning 100 epochs.
  - **Handoff corrections found by verification** (details in the plan §5): Pangu
    `inference.py` cannot run E3SM-SFNO at all (nettype gate + a checkpoint layout train.py
    never writes) — `ensemble_inference.py` with h5-derived ICs is the only path runnable from
    what is staged; the bias-`.npy`/`long_validation` caveat binds **train.py**, not the
    inference scripts; train.py's `--amp_dtype` help text says fp16 but the code default is
    bf16 (`train.py:265`) while both inference scripts default fp16 — a train/infer precision
    asymmetry to decide before comparing numbers; Pangu's full/smoke PBS headers still carry
    stale 162-ch/3.5-GB comments (flagged, deliberately not edited this session).
  - **Deliberately NOT done:** no Pangu inference PBS wrapper (the h5-IC ensemble path has
    never been exercised anywhere — the plan gives the exact first interactive command instead
    of encoding guesses into a script); no edit to `polaris_env.sh`'s `_pick` (the allyears
    smoke header records that as a deliberate stay-away; new scripts use their own env names).

- **2026-07-16** — **🔴 The E3SM archive replays ONE year of ocean forcing 35 times, and two of
  three pipelines mis-normalize a channel. Measured, adversarially verified, and written up in
  `polaris_e3sm_variable_reference.md` (per-variable reference + risk register R1–R12) and
  `data_for_training.md` (which risks actually affect *training*).** Method that produced these:
  4 agents — 3 adversarial (told to refute) + **1 cold, given no conclusions at all**. The cold
  one found the single worst issue; nobody was looking for it. Two of my own interpretations were
  corrected by the adversaries. Do this again.
  - **`SST`/`ICE`/`sol_in` are BITWISE identical across all 35 years** at the same index-in-year.
    1,224 md5 comparisons (12 indices × 34 years × 3 vars) + 480 random: **0 mismatches**; distinct
    inodes (not hardlinks); valid cells compared by value. Control: atmospheric fields **never**
    matched (1,632 comparisons; `TREFHT` differs by up to 30.6 K). Global SST mean is
    `14.574015 °C` in 2015, 2020, 2030, 2040 **and** 2049. Cause: `boundary_data/*_masked.nc` hold
    exactly **1460 steps = one year**; `netcdf-to-h5_e3sm.py` re-slices from `chunk_id=0` every
    year. The frozen 2015 in-file timestamp is the SAME bug, not a second one.
    **Intent is unresolved — only jesswan can say.** `CTL_SST0051` reads as a deliberate fixed-SST
    control; `SSP245AMIP` reads as a transient scenario that should warm. Both readings fit the
    name. Within-year seasonality is intact and strong (σ: SST 11.5, `sol_in` 403) — only the
    *interannual* axis is dead. **Not a training blocker** (it is a valid prescribed boundary);
    it contaminates metrics if a model *forecasts* those fields, which PhysicsNeMo does.
  - **PhysicsNeMo's normalizer silently erases precipitation.** `BatchNorm2d(momentum=None,
    affine=False)` on **raw physical units**, `eps=1e-5` → amplitude `σ/√(σ²+eps)`. `PRECT`'s σ is
    **7.7e-8 m/s** (m/s!), ~40,000× below `√eps` → amplitude **2.5e-5**. The loss is a *global* L2
    over all channels, so gradient share scales as amplitude²: PRECT's is **~6e-10**. The model
    converges and forecasts **climatological-mean rain**. Zero skill, no error, and the BatchNorm
    state is exported into the inference package. **NOT a data defect** — in mm/day σ is 6.72 →
    amplitude 1.0000. Fix belongs in the training path; **does not gate conversion**.
  - **PanguWeather's `TSOI_10CM` is normalized with stats that don't match its fill.** Config fills
    **270**; `compute_normalization_e3sm.py` never sets that key so its stats encode a **0** fill
    (npz 105.229/133.802 vs predicted 0-fill 105.266/133.857; 270-fill would be 271.13/16.43).
    A *predicted* channel ends up ~**26×** under-weighted. **Inherited from jesswan's own
    `_DERECHO_jsw.yaml:66` / `_STAMPEDE_jsw.yaml:66` — live in the group's existing runs**, not
    introduced here. The 270 fill itself is *good* (0.02σ from the valid mean).
  - **The `SST` 270 fill is inherited from a Kelvin ancestor.** `compute_normalization{,_plasim}.py`
    both carry `mask_fill['sst'] = 270.` / `['ts'] = 270.` — ERA5/PlaSim names for **Kelvin** fields.
    The E3SM copy renamed them mechanically; `SST` is **degC** (metadata says so). Fingerprint that
    it was mechanical: the same edit produced `mask_fill['TREFHT'] = 270.`, and TREFHT has **zero
    NaN** — dead code. 270-filling `SST_masked.nc` reproduces the shipped npz **exactly** (1e-7).
  - **Units metadata lies on 4 fields.** `SST`'s `long_name` is literally *"potential temperature"*;
    `RHREFHT` says units `1` but is percent; `PCT_*` say `unitless` but run 0–100. On this archive a
    variable name is not evidence **and neither is the attribute**. Measure.
  - Also corrected: the store is **2.15 TB**, not the "~1 TB" the older docs repeat; checkpoints are
    **18.9 GB**, not ~3.5 GB (`runs/pangu_sfno_full/.../checkpoints/` was already 177 GB at ~9
    epochs); the E3SM SFNO model is **1.18 B params**, not ~79 M.

- **2026-07-16** — **Cloud variables excluded from ALL three models (science owner).** All three
  pipelines now agree on the same **108 of 162** channels. PanguWeather already excluded them;
  makani's ALLDATA converter and PhysicsNeMo's `e3sm_h5_to_seqzarr.py` (`EXCLUDED_VARS`) now do too.
  PhysicsNeMo: **162 → 108, predicted 157 → 103**, store **2.15 TB → ~1.43 TB**. Verified by running
  the converter on the real archive: `excluded 54 channels`, `max|zarr-h5| = 0.0`, `CONVERT_OK`, then
  `SEQZARR_VERIFIED (EXHAUSTIVE: 6 × 108 = 648 channel-samples, bitwise)`, and the store's own attrs
  inspected independently (103+5+54 = 162, zero clouds survive, `TSOI_10CM`/`SOILWATER_10CM` intact).
  - **The duplication bug this exposed is the real lesson.** Nothing derived the channel counts:
    `157`/`5` was restated across ~13 sites, and makani's converter had `N_STATE, N_TARGET,
    N_FORCING = 154, 155, 7` as literals the asserts did **not** check — it would have written a
    correct 100-channel pack advertising 154 in its metadata. Both are now **derived**. The
    trainer reads counts from the store's attrs behind a `STORE_WRONG_GENERATION` gate; a stale
    162-channel store can no longer be silently adopted.
  - The biggest miss was **not** a `157` literal: `verify_seqzarr.py:107-135` structurally assumed
    the store partitions all 162 h5 keys, so it would have hard-failed **every** new store —
    including the tracked `polaris_verify_data_prep.pbs`, which rebuilds its fixture each run.

- **2026-07-16** — **Scope widened (owner): training is now Phase 1, not out of bounds.** DESIGN.md
  rewritten (375 → 314 lines) around *implement-first, then-profile*; three overlapping model tables
  merged into one six-codebase table; CLAUDE.md's "Why we're here" updated to match (it said "NOT
  retraining" and is auto-loaded, so it outranked DESIGN). **The division of labor is the line that
  matters and did not change: bring-up and training are ours; the science — variable sets, fill
  values, channel roles, physics — is jesswan's.** The rewrite also found ~10 stale claims in the
  old DESIGN (it said the §4 prerequisites "NONE exist yet"; all three exist).

- **2026-07-16** — **PanguWeather's bench knobs renamed `S2S_*` → `PANGU_*`.** PanguWeather is its
  own project — a fork of `s2s/v2.0` by **copy, not import** — and nothing outside it ever read
  these (verified: the only cross-project bench consumers are `s2s-lightning/common/bench_callback.py`,
  which genuinely imports `s2s/v2.0`, and `si/parse_nsys.py`). The `S2S_` prefix was decoration the
  copy carried along, and it kept inviting the conflation (it led me to cite CLAUDE.md #5 — which
  governs `s2s/v2.0/` — about a PanguWeather-only file).
  - **`train.py` now errors `LEGACY_BENCH_ENV` if an `S2S_BENCH*`/`S2S_NVTX` knob is set.** Required,
    not politeness: `BENCH = os.environ.get(...) == "1"` means *unset ⇒ silently no benchmarking*, so
    a stale script or doc (e.g. `polaris_bench_report.md`'s reproduction command) would have produced
    a run that measured nothing and exited 0. Verified the guard fires on `S2S_BENCH=1`/`S2S_NVTX=1`
    and passes on `PANGU_BENCH=1`/nothing-set.
  - **NVTX range names and CSV columns are UNCHANGED** (CLAUDE.md #10) — only the env knobs moved, so
    every prior bench row stays comparable. `s2s/` and `s2s-lightning/` keep `S2S_*`; historical
    CHANGELOG entries below are left as written (they record what was true then — the guard is what
    catches anyone following them).

- **2026-07-15** — **🔴 E3SM data prep: the PhysicsNeMo converter had 7 defects, and the smoke
  could not see the worst 3. Full analysis + the 5 open decisions:
  `polaris_data_prep_handoff_prompt.md`.** Prompted by "how do we confirm the conversion is
  correct before the ~1 TB run" — the answer was that we couldn't.
  - **The smoke was green because it was the one configuration where the bugs were invisible.**
    It converts 64+16 samples from **2015 only**: the archive's frozen in-file year is
    *correct* for 2015; `--max-samples` defaulted to **64**, which is what the smoke passes
    explicitly; and the zero-fill-on-interrupt trap needs an interrupt. Guardrail #4 exactly.
  - **Fixed (5):** `--max-samples` default 64→None (the "full" run wrote **64 of 51,100** and
    printed CONVERT_OK); the time axis now takes the **year from the filename** (measured: the
    archive stamps 2015 into EVERY file — `2049_1459.h5` → `2015-12-31 18:00:00`); `longitude`
    0..359 → **0.5..359.5** cell centres (verified vs `boundary_data/TOPO.nc`; `train.py:447`
    reads it into the inference model package, so it georegistered every product half a degree
    west); a `conversion_complete` sentinel (zarr pre-allocates with `fill_value: 0.0`, so a
    preempted run leaves a **right-shaped store of silent zero slabs** and the trainer gate only
    checked `shape[0] >= 1000`); and the **four `means_/stds_` arrays DELETED** — dead
    (nothing reads them: the datapipe asks only for `time/predicted/unpredicted`, `train.py`
    normalizes with `BatchNorm2d(momentum=None, affine=False)`, `model_packages.py:85-86` saves
    *that* batchnorm's stats) **and wrong** (npz SST assumes a 270 fill; the store fills −1.8).
  - **Dead-and-wrong metadata is worse than none** — it fooled **two independent auditors**
    (me and a cold Fable 5 agent) into "the model can't see SST". It can; the arrays are never
    read. Deleted, not corrected. The converter has **no npz dependency** now: layout + fill.
  - **New `--random-sample N --seed S`** + `polaris/polaris_verify_data_prep.pbs`: a small store
    spanning EVERY year, verified **exhaustively**. Green: **`SEQZARR_VERIFIED (EXHAUSTIVE:
    40 samples × 162 channels = 6480 channel-samples, bitwise)`**, job **7257786**, 24 distinct
    years. The full script now needs `-v CONFIRM_FULL=1` (the `--max-samples` fix *armed* it).
  - **STILL OPEN (4, documented not fixed):** a non-contiguous `--years` store passes every gate
    with **8766 h seams**; `nchunks_initialized` is defeatable (zarr 2.18.7 counts `.partial`
    files by prefix — measured 6/6 while a sample was a zero slab); a preemption during the
    *val* conversion **destroys the completed 9 h train store**; `--random-sample` ignores
    `--start-sample` but records it.
  - **5 DECISIONS for jesswan/us** (§0 of the handoff). The sharpest: **PhysicsNeMo forecasts
    SST and sea ice; Pangu and makani prescribe them — and this is an AMIP run, where they are
    prescribed by definition.** `UNPREDICTED` is *our* list, so that one is ours to fix.
  - **The npz "Kelvin" open issue is RESOLVED and its recorded diagnosis was WRONG.** SST is
    **°C** (measured `[−1.80, 32.21]`). The npz mean of 109.963 is arithmetically a **270
    land-fill of °C data**: `0.374×270 + 0.626×14.70 = 110.06`. The npz is not broken — it
    *encodes* Pangu's `SST: 270.` fill. Do not "fix" it to Kelvin.
  - **Analysis is 1/3 done:** PhysicsNeMo's converter audited, Pangu's stats prep audited (64
    lines, metadata-only, clean). **makani's 367-line converter is unaudited** — it flips
    latitude, truncates to 10 of 18 levels, renames channels, and its stats *are* live. §8 of
    the handoff scopes it.

- **2026-07-15** — **🔴 CORRECTION + fork-drift audit: `TORCH_COMPILE_MODE` was NOT plumbed in
  PanguWeather, and this file said it was.** Prompted by the question "have the
  `bench_report.md` optimizations — especially the ViT ones — already been done in
  PanguWeather?". Checked instead of assumed (DESIGN §2c: the forks share code by **copy**, so
  "nothing tells you the other copy drifted"). Full table: `polaris_bench_report.md` §6b.
  - **The error:** the harness port brought `S2S_BENCH`/NVTX across but **not** the compile
    knob. PanguWeather had only a commented-out `torch.compile(self.model, mode='default')`
    (`train.py:639`) and no env read — exactly as DESIGN §2c's table already said
    (`TORCH_COMPILE_MODE`: s2s **2**, PanguWeather **0**). I should have read my own table.
    The commented-out `export TORCH_COMPILE_MODE=…` in both new bench scripts was therefore a
    **live trap**: uncomment it → no compile, no error → "torch.compile doesn't help this
    model". **Now genuinely wired** in `get_model()` (gated; unset ⇒ legacy path) + a test
    that fails if the knob is ever disconnected again.
  - **The drift is BIDIRECTIONAL** — each fork has something the other lacks:
    **`static_graph=True`** is in s2s and **missing in PanguWeather**;
    **`gradient_as_bucket_view=True`** is in PanguWeather and **missing in s2s**.
  - **PanguWeather is AHEAD on several**: bf16 is the YAML default (`amp_dtype: bfloat16`)
    rather than an env knob defaulting to fp16; `--async_save` reaches more files; there are
    more `os.path.isfile` checkpoint guards. The per-iteration `empty_cache()` removal and the
    per-param `.item()` removal landed in both, independently.
  - **`static_graph=True` is a candidate, NOT a known win — do not just copy it.**
    `bench_report.md` §4 changed bf16 and `find_unused_parameters=False`+`static_graph=True`
    **together** (+5.3% for the pair), so `static_graph`'s isolated contribution was never
    measured, and PanguWeather already has the expensive half. Worse, s2s needs a
    **dead-module freeze** (`layer_perturbation2`/`layer_purturbation_e2`, `train.py:437-444`)
    to make `static_graph` legal; PanguWeather has no such freeze, so copying it across could
    fail at runtime on `pangu_plasim`.
  - **The ViT/Swin optimizations: there are none to port.** `bench_report.md` §3's findings
    (LayerNorm-backward 2nd-largest, layout conversions, `roll`, matmul only 6th) are
    **profiler observations explicitly deferred to `torch.compile`**, and rung 2
    (FlexAttention) is unstarted — in *either* fork. A diff of the two `networks/pangu.py`
    shows the **only** perf-relevant divergence is s2s's NVTX ranges: SDPA is already in
    `EarthAttention3D` in both, and both have the same 2 `torch.roll`s, 13 `LayerNorm`s and
    identically commented-out block checkpointing. **And the ViT does not run on Polaris** —
    the green path is `sfno_plasim` → `networks/modulus_sfno/`, which never touches
    `pangu.py`. The ViT (and its VAE, and the new `vae_noise` hook) belong to `pangu_plasim`,
    blocked on PLASIM data.
  - **Worth noting:** the H100 **ViT** and the A100 **SFNO** — different architectures —
    profile the *same way*: elementwise-dominated, matmul secondary. Two independent
    measurements, one conclusion: `torch.compile` is rung 1.

- **2026-07-15** — **Profiling phase: PanguWeather SFNO profiled on 4×A100. Full report:
  `polaris_bench_report.md`.** Branch `polaris-profiling` (stacked on the still-unmerged
  `polaris-pbs-bringup`). Headlines:
  - **Instrumentation had to be built first** — PanguWeather carried 0 NVTX ranges and no
    `S2S_BENCH` (DESIGN §2c). Ported from `s2s/v2.0` with range names + CSV columns
    byte-identical (CLAUDE.md #10), gated so unset knobs ⇒ legacy path byte-for-byte.
    **Proven, not asserted:** job **7255505** (no `S2S_BENCH`) reproduced the GREEN
    reference **7253591** exactly — train loss **0.3411**, valid_loss
    **0.7049359679222107**, bit-identical. Adapted where the fork differs: the scaler can be
    `None` (bf16), EMA is real hot-path work s2s lacks (own `ema` range), and `amp_dtype` is
    recorded from the dtype actually used (PanguWeather takes precision from the YAML, so
    reading `$S2S_AMP_DTYPE` would have mislabelled every row).
  - **VERDICT 1 — GPU-bound.** Loader idle is **0.7%** at the shipped `num_data_workers: 1`
    (job 7255410). The §5 kernel ladder is **not** premature.
  - **VERDICT 2 — elementwise-bound, not matmul-bound.** **61%** of GPU time is
    pointwise/elementwise over ~1506 launches/step; GEMM is only **15%**; NCCL 10.5%;
    cuFFT/SHT just 3.3% (job 7255503). Memory-bandwidth bound and fusion-starved ⇒
    `torch.compile` is the right first lever **on evidence**, not assumption.
  - **The model is 1,182,108,160 params** — 1.18 B, **not** the "~79M" DESIGN/CLAUDE.md
    assume (that figure is the Pangu/Swin model, not the E3SM SFNO). 26.98 GB peak of 40 GB.
  - **`cpu_prep_frac` is NOT the data-loader idle fraction** — a trap worth remembering. It
    times `_prepare_inputs_batch` on an **already-fetched** batch (0.3–0.6% of the step even
    with the loader deliberately starved). The blocking fetch happens in `__next__`,
    *between* steps, inside no step window. Worse, it was **fatal**: the elapsed-vs-sum
    self-check fires on an input-bound run and **refuses the row**, i.e. the harness aborts
    exactly when the loader is the finding. Now measured (`loader_wait_med`/`loader_wait_frac`,
    appended after s2s's 19 columns) and folded into the check, which makes it *tighter*.
    **Falsified before believed:** `workers=0` moved it 0.7% → 14.8% (21×) while
    `cpu_prep_frac` stayed flat — a metric that cannot move proves nothing.
  - **`elapsed` was sampled AFTER `cudaProfilerStop()`** (inherited from s2s), folding the
    profiler's buffer flush into the measured wall time. Job **7255503** read `elapsed=51.8s`
    vs `sum=25.7s` and threw away a good bench row — on **every** nsys run. The timers were
    fine; the clock was stopped in the wrong place. Fixed; the re-run **7255557** records
    cleanly at rc=0.
  - **`samples_per_s` is a STEP RATE, not throughput** — it excludes the loader gap. At
    `workers=0` it reads 6.50 while the truth is 5.53; quoting it would have ranked the
    **slower** config first. Convert: `wall = samples_per_s × (1 − loader_wait_frac)`.
  - **`num_data_workers` is NOT output-neutral here** — `data_loader_multifiles.py:1031/1102`
    draws per-sample gaussian noise **inside the workers** (`epsilon_factor: 0.1`) and there
    is **no `worker_init_fn`**, so the worker count changes the noise realization and moves
    the loss. `1 → 8` is **+9% wall throughput and 10× less jitter** (step_p90 0.826→0.603) —
    recorded as a **finding, not a recommendation**. Clean fix: a seeded `worker_init_fn`.
  - **Nothing was optimized.** `TORCH_COMPILE_MODE` is wired and left unset; the §4 gate is
    not executable until a baseline is captured.

- **2026-07-15** — **§4.0 on PanguWeather: the seed prerequisite was already satisfied.**
  The handoff implied `--seed` needed porting from `s2s/v2.0/utils/seeding.py`. It does
  **not**: `train.py:3825` has `--global_seed` (default 0) → `seed_torch()` (`:3742`, called
  at `:3785`), which seeds `PYTHONHASHSEED`/numpy/torch/CUDA and sets `cudnn.benchmark=False`
  + `cudnn.deterministic=True`. That is **stronger than s2s's legacy path** — the numpy gap
  that made s2s's baselines irreproducible does not exist here, which is *why* 0.3411 is
  bit-reproducible. **Porting `seeding.py` would create two competing seed mechanisms — don't.**
  Remaining gaps: Python's `random` unseeded, `torch.use_deterministic_algorithms(True)` never
  set. Side note: `cudnn.benchmark=False`/`deterministic=True` are therefore **always on** —
  a performance fact hiding inside a reproducibility mechanism.

- **2026-07-14** — **DESIGN §4.0 seed knob: DONE.** `s2s/v2.0/train.py` gains `--seed` and
  `--deterministic`; the logic lives in the new shared `s2s/v2.0/utils/seeding.py` (imported
  by S2S *and* the port, CLAUDE.md #5 — additive, nothing existing changed).
  Precedence: `--seed` > `$S2S_SEED` > the YAML's `seed:` > **legacy**.
  - **Opt-in by design.** No seed => the historical path is preserved *byte-for-byte*
    (`torch.manual_seed(world_rank)`, `cudnn.benchmark=True`), which is what lets this ship
    without re-validating the existing greens. A test pins that property.
  - **What was actually broken:** `torch.manual_seed(world_rank)` seeded torch only.
    **numpy was never seeded** — and `train.py:1251` draws the validation sample from it
    (`np.random.randint`) — so two runs of the "same" config diverged. `random` was unseeded
    too, and `cudnn.benchmark=True` picks kernels by timing. A "reproducible baseline" on
    that footing was not reproducible.
  - **Rank offset:** `seed + world_rank`, preserving the legacy intent (distinct streams per
    rank, so the loader's per-sample noise — `data_loader_multifiles.py:474-481`, drawn in
    the workers — doesn't correlate across ranks). At rank 0 the applied seed IS the seed,
    so a §4.1 world-size-1 baseline is comparable with the port's `seed_everything(s)`. A
    multi-rank baseline is NOT comparable across launchers — documented in
    `seeding.equivalent_to_seed_everything`.
  - **Tests:** `s2s/v2.0/test/seeding_test.py` — 10 assertions, **`SEEDING_OK`**, runs with
    no ERA5/GPU/cluster (deliberate: the S2S+port data smokes are blocked on the ERA5 stage,
    so the mechanism is proven without them). Covers same-seed reproduction, different-seed
    divergence, the numpy gap, byte-identical legacy, precedence (incl. **seed 0**, the
    classic falsy bug), loud failure on a bad seed, rank offsets, and model-level identical
    init+forward+backward. `polaris_seeding_test.pbs` runs the CUDA half on a real GPU and
    **fails rc=4 if CUDA was not visible** — a skipped test must never read as a pass.
    **GPU-verified: job 7253738, rc=0, `SEEDING_OK (10 tests)` +
    `CUDA was visible -> the CUDA RNG assertion really ran`.** So CUDA RNG reproducibility is
    demonstrated on the device a baseline would actually be captured on, not just on CPU.
  - **Still blocking baseline capture:** `tiny_baseline.yaml` and the VAE noise-fix hook.
    Also note `--deterministic` needs `CUBLAS_WORKSPACE_CONFIG=:4096:8` exported *before*
    python starts (the PBS script does it); `enable_determinism()` warns rather than
    pretending when it is missing.


- **2026-07-14** — **🔴 The "GREEN" smokes were green for ONE PERSON. Fixed.** A cold
  5-agent audit of the *fixed* tree (the second gauntlet) surfaced that Pangu/SI depended on
  `pip install --user` packages living in `PYTHONUSERBASE=/home/rmehta1987/.local/...`.
  **ALCF home dirs are mode `0700`**, so those packages are readable by their owner alone.
  Every Pangu/SI "GREEN" was therefore unreproducible by the rest of the project — the exact
  opposite of this deliverable's purpose — and `polaris_running_the_smokes.md` told jesswan
  "they use software already installed on Polaris", which was false.
  - **Proof, not inference:** job **7253539** re-ran Pangu with `PYTHONNOUSERSITE=1` (which
    reproduces a second member's view of the filesystem) and died on
    `ModuleNotFoundError: No module named 'tensorly'`. Impersonating the other user is the
    only way to catch this class of bug; a normal re-run by the installer always passes.
  - **Fix:** `polaris_setup_base_topups.sh` installs netCDF4 / zarr / torch_harmonics 0.7.4 /
    tensorly / tltorch / cftime / numcodecs into the **shared, world-readable**
    `$POLARIS_TOPUPS` on eagle; Pangu/SI/S2S/probe prepend it to `PYTHONPATH`.
  - **Two traps inside the fix**, both now guarded:
    (1) `pip install --target` can't see the base conda, so it re-resolved the world and
    silently pulled **torch 2.13 + CUDA 13 + numpy 2.5.1** (4.1 GB) — which, being on
    PYTHONPATH, would have **shadowed the base's torch 2.8/cu12.9** and moved every smoke
    onto an untested toolchain. `--no-deps` + a hard fail if `torch|numpy|nvidia|triton`
    land in the target; now **64 MB**, and it asserts torch/numpy still come from base.
    (2) `$POLARIS_TOPUPS` must NEVER go on PYTHONPATH in an SFNO job — its
    torch_harmonics 0.7.4 would shadow the venv's 0.9.x and re-break makani
    (`PYTHONNOUSERSITE` does **not** block PYTHONPATH). Both SFNO scripts now assert
    torch_harmonics resolves inside their venv (`ERROR TORCH_HARMONICS_SHADOWED`).
  - **Proven fixed, not assumed:** Pangu **7253591** (`PYTHONNOUSERSITE=1`) rc=0 with loss
    **0.3411 — bit-identical** to the installer's 7253401, and SI **7253603** step_med 0.399 /
    peak 30.69 GB (vs 0.400 / 30.98: noise). Identical rather than merely similar matters: it
    shows the shared top-ups serve the *same code* the greens ran on. Version pins in
    `$POLARIS_TOPUPS` match the old `~/.local` ones exactly.
  - **Regression-proofed:** `polaris_require_topups()` (in `polaris_env.sh`, called by **all 8**
    base-conda jobs — the SFNO pair is deliberately exempt) fails the run with `ERROR TOPUPS_MISSING` or `ERROR PRIVATE_DEPS_ON_PATH`
    if a dep ever resolves from a private home again. Both branches tested: unsetting
    PYTHONPATH reproduces the original bug and the guard catches it. **Note the asymmetry the
    guard exists for — this bug is invisible to the one person who could fix it**, because
    their own runs pass.
  - **A reasoning error worth remembering.** When deriving the top-ups list I used the rule
    *"missing for the installer too => off the smoke path"*. That is only valid for code that
    has **actually run green**. The S2S/port smokes have never run on Polaris (blocked on
    ERA5), so for them "missing for everyone" means **broken for everyone** — and I dropped
    `cf_xarray` on that basis. It is a bare import at `s2s-lightning/modules/train_module.py:52`
    reached from both port entrypoints. The port would have died at import right after a
    multi-TB Globus stage. Caught by the third cold gauntlet; `cf_xarray` is now in the
    top-ups and all 5 entrypoint chains import as a second user.
  - **The probe's port check was hollow.** It imported `common, data, modules`, which have no
    `__init__.py` — namespace packages, so the import succeeds without executing any of the
    smoke's code. That is how a bare missing import survived a green `PROBE_OK` while the
    docs claimed the port's env was "proven by the probe". The probe now imports
    `modules.train_module`, and `polaris_require_topups` covers **8/8** base-conda jobs (it
    was 5/8; the probe and both port jobs were missed, contradicting this very entry's
    earlier claim of "every base-conda job").
  - **Lesson (generalise):** never `pip install --user` a dependency the project must share,
    and never accept "it's green" from the environment that installed it. The probe
    (7251974) had the same blind spot — it certified "all models import" while importing
    from a private home; it now imports through the shared dir and warns if `~/.local` is on
    `sys.path`.

- **2026-07-14** — **Makani's re-run was a silent no-op (`rc=0`, zero steps).** With a
  hardcoded `--run_num 0`, `train_plasim` auto-resumed from a checkpoint that already
  satisfied the smoke's `max_epochs=1`: job **7253454** printed `Total training time is
  0.00 sec` and exited **0**. `RUN_NUM` now defaults to `${PBS_JOBID%%.*}`, plus a gate that
  forces `rc=4`/`ERROR NO_CHECKPOINT` when a run exits 0 without writing its checkpoint.
  Revalidated by **7253465** (train 2.61 / val 2.38, 7.10 s of real training).
  **`rc=0` is not a PASS criterion for a resumable trainer** — key on the work token.
  Related: the smokes have **no seed knob** (DESIGN §4.0), so their losses move run-to-run
  (7252769: 2.19/2.05 vs 7253465: 2.61/2.38 on identical code) — they are **not** an
  equivalence baseline.

- **2026-07-14** — **Audit fixes (docs + scripts).** `CONVERT_OK` re-attributed **7252736 →
  7252728** (7252736 packed nothing and failed rc=1; verified from the log). The
  `disassemble_input` note corrected: **fixed** in `train_module.py` (`1fef2473`) but still
  **open** in `bias.py:226`, `ae_module.py:68`, `combined_module.py:185/287`. All 3
  converters now honour the advertised `$E3SM_ROOT` (only makani did). All 10 `*.pbs` now
  source `polaris_env.sh` (5 didn't, so the notes' "every script pins the caches" was false).
  `polaris_logs/.gitkeep` committed — the dir is gitignored, so the probe's `#PBS -o
  polaris_logs/` had nowhere to deliver in a fresh clone. CLAUDE.md's "Polaris/PBS = single
  `python`" corrected (6 of 10 use torchrun). Cleanup doc's `.npy` "loaded by" column fixed
  (`pangu_lite.py` only mentions the masks in a comment) — the conclusion (never blanket-
  ignore `*.npy`) was right, the evidence wasn't.


- **2026-07-14** — **Polaris (PBS) bring-up.** Confirmed cluster facts (`-A
  lighthouse-uchicago`, 4×A100-40GB sm80, `debug` queue, `filesystems=home:eagle`,
  `/local/scratch`); env = base ALCF conda (`module load conda`, torch 2.8/cu12.9) +
  `pip install --user` netCDF4/zarr/**torch_harmonics 0.7.4** (0.9.1 ABI-breaks torch 2.8).
  **Probe GREEN** (job 7251974). **PanguWeather-SFNO 4-GPU smoke GREEN** (job 7252271):
  climatology CDF-5→NETCDF4 auto-prep + 1 bounded epoch, train loss 0.3411, DDP
  validation, rc=0. Two traps recorded in `polaris_pbs_notes.md`: (1) Pangu `--debug`
  hardcodes `world_size=1` → OOMs under `torchrun -n4` (bound with `--epochs 1`
  instead); (2) Lustre needs `HDF5_USE_FILE_LOCKING=FALSE`. Authored all
  `polaris_*.pbs` (S2S/port/SI/Pangu/makani/physicsnemo) + 3 data converters +
  repointed configs. **S2S/port blocked on an ERA5 Globus stage** (not on Polaris).
  **SI, Makani and PhysicsNeMo also went GREEN** (7252700 / 7252769 / 7252816) — the
  latter two from an isolated SFNO venv (see the In-progress entry). Caches/TMPDIR
  pinned to eagle (persistent), not node-local scratch (per user). A 5-agent cold
  adversarial audit independently re-confirmed every GREEN claim against the raw logs and
  surfaced the fixes applied in `3c0b4e5`. Full detail: `polaris_pbs_notes.md`.
- **2026-07-13** — Model policy set to **main = Opus 4.7 (xhigh effort), subagents =
  Fable 5**. Trimmed CLAUDE.md to stay <200 lines while adding: filled the real
  Midway env paths, a per-model smoke table (what to run + PASS signal), the
  launcher-shape + env-bootstrap rules, the `test.yaml` trap (rule #12), and a
  "where to look" doc map. Ran two cold Fable-5 agents to source the additions.
- **2026-07-13** — PR #4 (`polaris-pbs-handoff`) merged to `main` (`4c283f2`);
  `polaris_handoff_prompt.md` is on `main`.
- **2026-07-13** — Cold adversarial review of the docs (three Fable-5 agents); applied
  the findings (SI `bench.py --config <path>` command, DESIGN §2 launch table,
  `data_prep` NVTX name, a concrete §4 + its §4.0 prerequisites, baseline
  `.pt`-vs-`.gitignore` fix, interactive-allocation preface, `pytest --fast` hedge).
  **Also fixed a real regression:** the port smokes hardcoded a cwd-relative
  `v2.0/config/test.yaml` (pre-monorepo) → now resolved relative to `__file__`.
- **2026-07-13** — Added `DESIGN.md`, `CLAUDE.md`, `CHANGELOG.md` (design spec,
  working guide, living doc) patterned on `smsharma/clax` + the MARSHAL/decrypto
  playbooks. Establishes the **numerical-equivalence-vs-baseline** gate as the oracle.
- **2026-07-13** — Published the repo; repo-wide **SNFO → SI** rename (SI is correct;
  SNFO a mislabel). NGC key scrubbed to `$NGC_API_KEY`. `main` branch-protected.

## Known issues / failed approaches (do NOT re-attempt)

Each is attributed to its source doc — verify there before acting.

- **(PanguWeather) Don't assume a `bench_report.md` optimization reached this fork — the drift
  is bidirectional.** `static_graph=True` is in s2s only; `gradient_as_bucket_view=True` is in
  PanguWeather only; bf16/`--async_save`/checkpoint-guards are *ahead* in PanguWeather. Full
  table before acting: `polaris_bench_report.md` §6b.
- **(PanguWeather) Do NOT copy `static_graph=True` across without the dead-module freeze** —
  s2s freezes `layer_perturbation2`/`layer_purturbation_e2` (`train.py:437-444`) to make it
  legal, and PanguWeather has no such freeze. Its isolated gain was also never measured
  (`bench_report.md` §4 changed it together with bf16). — `polaris_bench_report.md` §6b.
- **(PanguWeather) There are no ViT/Swin optimizations to port — they were never implemented
  in either fork**, and the ViT doesn't run on Polaris anyway (`sfno_plasim` uses
  `networks/modulus_sfno/`, never `networks/pangu.py`). `bench_report.md` §3's LayerNorm/
  layout/`roll` findings are observations deferred to `torch.compile`; SDPA is already in both.
  Don't go hunting for a missing port that doesn't exist. — `polaris_bench_report.md` §6c.
- **(polaris_env.sh) `-v SEQZARR_DATA=…` (and the other `_pick` vars) CANNOT be overridden** —
  `_pick` never reads its first argument, and `polaris_env.sh:155` exports unconditionally. Job
  7257791 was submitted with `-v SEQZARR_DATA=…_fresh` to force the PhysicsNeMo smoke to rebuild
  its store with a changed converter; it silently used the OLD cached store and passed. A gate
  that cannot be pointed at fresh data is not a gate.
  — `polaris_data_prep_handoff_prompt.md` §4.
- **(E3SM archive) The .h5 construction is NOT in this repo** — it lives in
  `/eagle/.../jesswan/PanguWeather/data_utils/` (`netcdf_to_h5*.py` ×3, `get_stats.py`, adapted
  "from FourCastNet repo"). Read it before deciding the fill questions: it is the ground truth
  for the 270 sea-surface-temperature fill, the 19 constant cloud channels, and the frozen
  `time` year. Three `netcdf_to_h5` variants exist and nothing records which built the archive.
  — `polaris_data_prep_handoff_prompt.md` §8c-bis.
- **(E3SM data prep) `CONVERT_OK` is NOT a verification** — it checks 1 channel of 1 sample
  (0.01%) and is blind to the NaN fill by construction (its probe channel is chosen as one
  with no fill). Require `SEQZARR_VERIFIED` from `polaris/verify_seqzarr.py`.
  — `polaris_data_prep_handoff_prompt.md` §4.
- **(E3SM data prep) The PhysicsNeMo smoke store CANNOT validate the full conversion** — it is
  64+16 samples of **2015 only**, and all three worst defects were invisible at exactly that
  scale. Use `polaris/polaris_verify_data_prep.pbs` (`--random-sample`, spans every year).
  — `polaris_data_prep_handoff_prompt.md` §5.
- **(E3SM archive) `input/time` is FROZEN AT 2015 in every file** — `2049_1459.h5` carries
  `2015-12-31 18:00:00`. Month/day/hour track the index; only the year is wrong. Never build a
  time axis from the in-file label; take the year from the filename. Upstream defect.
  — `polaris_data_prep_handoff_prompt.md` §1.
- **(E3SM stats) The npz SST mean of ~110 is NOT "Kelvin data"** — that inference (previously
  recorded here as an open issue) is refuted. SST is °C; 110 is arithmetically a 270 land-fill:
  `0.374×270 + 0.626×14.70 = 110.06`. The npz encodes Pangu's `SST: 270.` fill and is
  self-consistent. Do not "fix" it. — `polaris_data_prep_handoff_prompt.md` §3.
- **(zarr) `nchunks_initialized` is not a completeness check** in zarr 2.18.7 — it counts chunk
  keys by *prefix* regex, so a `.partial` left by a kill mid-write counts as written (measured
  6/6 while a sample was an all-zero slab). Compare the exact expected key set.
  — `polaris_data_prep_handoff_prompt.md` §4.
- **(PanguWeather) Do NOT port `s2s/v2.0/utils/seeding.py` into PanguWeather** — it already
  has `--global_seed` → `seed_torch()`, which is more complete than s2s's legacy path. Two
  seed mechanisms racing to set the same global RNG is a regression, not a port.
  — `polaris_bench_report.md` §6.
- **(PanguWeather) `cpu_prep_frac` is NOT the data-loader idle fraction** — it times
  `_prepare_inputs_batch` on an already-fetched batch and stays at ~0.4% even with the loader
  deliberately starved to a 14.8% stall. Use `loader_wait_frac`. — `polaris_bench_report.md` §3.
- **(PanguWeather) `samples_per_s` is a step rate, not wall throughput** — it excludes the
  between-step loader fetch. Convert with `× (1 − loader_wait_frac)` before comparing two
  configurations, or you will rank the slower one first. — `polaris_bench_report.md` §3.
- **(PanguWeather) Don't read the NVTX sub-ranges as GPU time** — they are pushed/popped on
  the CPU thread and measure *enqueue*; they sum to 55% of the step, and the rest is the
  terminal `cuda.synchronize()` draining the GPU. `backward = 280 ms` is CPU launch work, not
  47% of GPU time. Attribute GPU time from the kernel table. — `polaris_bench_report.md` §4.1.
- **(PanguWeather) Bumping `num_data_workers` changes the LOSS, not just the speed** — the
  loader draws per-sample gaussian noise inside the workers (`epsilon_factor: 0.1`) with no
  `worker_init_fn`. The +9% from `1 → 8` is real but cannot be validated by the §4 bitwise
  gate. Fix `worker_init_fn` first. — `polaris_bench_report.md` §3.
- **The E3SM SFNO is 1.18 B params, not "~79M"** — the 79M figure (DESIGN, CLAUDE.md #12)
  describes the Pangu/Swin model. Don't carry 79M-era resource intuition onto the SFNO path.
  — `polaris_bench_report.md` §1.

- **(Polaris) `torch_harmonics` version box** — makani 0.2.0 imports the *public*
  `torch_harmonics.quadrature.precompute_latitudes`, which does NOT exist in 0.7.4 or
  0.8.0 (private `_precompute_latitudes`). 0.9.1 has it but ships **wheels only** (no
  sdist on PyPI — `--no-binary :all:` cannot build it) and its prebuilt
  `attention/_C.so` fails on torch 2.8 with `undefined symbol:
  _ZNK3c1010TensorImpl15incref_pyobjectEv`, so `import torch_harmonics` dies outright.
  Don't re-try pinning a PyPI version — install from the GitHub source (compiles `_C`
  against the local torch) **and re-verify the green Pangu-SFNO smoke**, or isolate the
  SFNO frameworks in their own venv. — `polaris_pbs_notes.md` §6.
- **(Polaris) Pangu `--debug` is single-GPU ONLY** — it hardcodes `world_size=1`, so
  under `torchrun --nproc_per_node=4` all 4 ranks init as rank-0-on-GPU-0 and OOM the
  40 GB A100. Bound a smoke with `--epochs 1` instead. — `polaris_pbs_notes.md` §5.
- **(Polaris) SI `calendar: 'noleap'` crashes the loader** — noleap is an *idealized*
  cftime calendar that forces `has_year_zero=True`, clashing with `has_year_zero: False`
  at `si/data/amip_new.py:667` (`cannot compute the time difference between dates with
  year zero conventions`). Use `'standard'` (correct for a non-leap-year smoke); a full
  run crossing a leap year needs a loader fix. — `polaris_pbs_notes.md` §5.
- **Port standalone smokes had a stale cwd-relative config path** (`v2.0/config/test.yaml`,
  pre-monorepo) → `FileNotFoundError` before any GPU work. Fixed 2026-07-13 (resolve
  relative to `__file__`). If a port smoke can't find the config, check this first.
- **"Missing kernel tables" are NOT a profiler/ptrace limit** — an unconditional
  `restore_checkpoint()` crashed on `FileNotFoundError` before any GPU work (a
  byte-identical CUDA-API fingerprint). Fixed with the `os.path.isfile` guard. If a
  profile has no kernel table, **read the `.err` first.** — `bench_report.md` §II.7.
- **S2S batch ≥3/card (bf16) is a trap** — throughput collapses near allocator
  saturation and 4/card OOMs; the known-good ceiling on ~94 GB cards is **2/card**.
  — `bench_report.md` §II.4.
- **`num_data_workers=0` fakes a GPU-idle "bottleneck"** (large idle %; SI's first
  4-GPU bench failed its sanity check on HDF5 reads). Known-good: 8 workers +
  `--cpus-per-task=8`. — `bench_report.md` §II.7 / `si/bench_midway_notes.md`.
- **Inference: always pass `--async_save`** — synchronous NetCDF saving throttles
  rank 0 well below the other ranks. — `bench_report.md` §II.7.
- **Don't remove SI's fp32 island around the spherical-harmonic transform** — bf16
  breaks `torch_harmonics` (`view_as_complex` rejects bf16); it's wrapped in
  `autocast(enabled=False)` on purpose, cost ≈ 0. — `si/bench_midway_notes.md` §3–4.
- **SI + `torch.compile max-autotune` / nsys-on-compiled-SI** — reported to
  crash/segfault (CUDA-graph capture; tracing the compiled DDP backward). Use
  `default` compile mode; profile eager, bench compiled. — `si/bench_optim_sweep.sh`
  header (verify before relying on it).
- **DSI handoff-latency investigation** — several hypotheses already investigated;
  the current lead is a **driver/CUDA mismatch** (not interconnect). Read
  `bench_report.md` §I.5/§I.7 and don't re-run the ruled-out ones.

## Open questions (answer + record here)

- **Baseline node class.** The SI optimization-sweep CSVs (`si/bench_optim_*.csv`)
  ran on the **test partition H100**, a different node class from the pedramh-gpu
  H100-NVL numbers — re-measure a pedramh-gpu baseline to compare like-for-like.
- ~~**A100 (Polaris) memory**~~ — **RESOLVED** by probe 7251974: `nvidia-smi` on-node
  reports **40960 MiB/GPU** (4× A100-SXM4-40GB, driver 570.124.06). See
  `polaris_pbs_notes.md` §1.
- **SI compile gain** — reported ~+62% (`default` mode) but a `*_postfix` re-run is
  lower; quote it as a range until re-measured on pedramh-gpu.

## Benchmark results

**Read the existing evidence before capturing baselines or claiming a speedup**
(compare only within a cluster, never A100 vs H100 NVL):
- `s2s/v2.0/bench_report.md` — S2S H100-NVL baselines + the step-time / VAE-encoder split.
- `si/bench_midway_notes.md` — SI bench + decisions log (refutes the "H200" label).
- `si/bench_optim_*.csv` + `si/bench_optim_sweep.sh` header — the 2026-05 one-lever-
  at-a-time SI optimization sweep (test-partition H100 — a different node class).
- `s2s-lightning/LIGHTNING_PORT.md` — the port's DDP/AMP/bench wiring + per-phase
  smoke-id table. **The port-vs-v2.0 nsys caveat** is in the header of
  `s2s-lightning/midway_bench_nsys_port.sh`: the port's per-step NVTX window opens at
  `on_train_batch_start` (after H2D), so its `step_med` excludes the transfer —
  compare throughput via `samples_per_s_wall`, never `step_med`.
- `s2s/v2.0/HPC_scripts/bench_methodology.md` — what every `bench_results.csv` column
  means and why timing is `cuda.synchronize`-bracketed.

**Hardware identity (do not reintroduce refuted labels):** `pedramh-gpu` is
**H100 NVL (~93 GB)**, NOT "H200" (a commit message said so; refuted in
`si/bench_midway_notes.md`) and NOT "80 GB H100". NVLink is within socket-pairs only
(GPU0↔1, GPU2↔3); the host is PCIe Gen4. The Midway H200 *test* partition is a
separate node class (full-mesh NVLink, PCIe Gen5).

**How to capture a baseline (BLOCKED on §4.0):** procedure = DESIGN.md §4.1, storage
= §4.2 (JSON/CSV summary in git, tensors on cluster storage), metric definitions =
`bench_methodology.md`. Record each capture as a dated row here.

_(record new per-cluster bench deltas below — model, cluster, config, samples/s,
peak mem, and the equivalence result for any optimization.)_
