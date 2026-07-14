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
| Polaris (PBS) bring-up | ⬜ not started — handoff on branch `polaris-pbs-handoff` (PR pending) |
| §4.0 prerequisites (seed knob, tiny config, VAE noise-fix) | ⬜ not started — **blocks baseline capture** |
| Correctness baselines captured (DESIGN.md §4) | ⬜ not started — **blocks all optimization** |
| Test harness (tier-1 equivalence/unit + `--fast`) | ⬜ not started |
| Optimization ladder (DESIGN.md §5) | ⬜ not started |

### Smoke status matrix (probe → 1-GPU → 4-GPU)

| Model | Midway | Polaris |
|---|---|---|
| S2S (`torchrun`) | ✅ runs (Midway scripts GREEN) | ⬜ |
| S2S-Lightning | ⚠️ standalone smoke config-path fixed 2026-07-13 — **needs a Midway run to reconfirm** | ⬜ |
| SI | ✅ runs (Midway scripts GREEN) | ⬜ |

## Next actions (pick from the top)

1. **Polaris bring-up** — probe → 1-GPU → 4-GPU smoke for each model via PBS;
   write `polaris_pbs_notes.md`. Follow `polaris_handoff_prompt.md` (currently on
   branch `polaris-pbs-handoff` — merge or check it out first).
2. **Build the §4.0 prerequisites** — a `--seed` knob in `s2s/v2.0/train.py`, a
   `tiny_baseline.yaml`, and a VAE noise-fixing hook. Nothing can be optimized
   safely until the equivalence gate is executable.
3. **Capture correctness baselines** (DESIGN.md §4) for each model.
4. **Stand up the test harness** — CRPS/KL numerical checks, normalize↔inverse
   round-trip, tiny-model forward/backward, a `conftest`-registered `--fast`.
5. Then start the optimization ladder (torch.compile first — enable the existing
   `TORCH_COMPILE_MODE` plumbing), one gated commit each.

## In progress

- _(none yet — claim your task here, e.g. "IN PROGRESS: Polaris probe (@you)")_

## Decisions / changes log

- **2026-07-13** — Ran a cold adversarial review of the new docs (three Fable-5
  agents) and applied the findings. Fixes: corrected the SI `bench.py` command
  (`--config <path>`, no `--yaml_config`); corrected the DESIGN §2 launch table
  (port & SI use `srun`/`ntasks==devices` on Midway, plain `python` on Polaris);
  fixed the S2S NVTX range name (`data_prep`, not `preprocess`); made §4 concrete
  (metric, tolerances, determinism flags, VAE noise-fixing) and split out the §4.0
  prerequisites that must be built first; fixed the baseline `.pt`-vs-`.gitignore`
  contradiction (commit JSON/CSV summaries only); reconciled "never run directly"
  with the interactive-allocation preface; hedged the `pytest --fast` gate; added a
  one-time-setup section; named the existing `TORCH_COMPILE_MODE`/`S2S_AMP_DTYPE`
  knobs in §5; declared CLAUDE.md the cluster-facts SSOT. **Also fixed a real code
  regression:** the port smokes (`smoke_train_module.py`, `smoke_datamodule.py`)
  hardcoded a cwd-relative `v2.0/config/test.yaml` from the pre-monorepo layout —
  now resolved relative to `__file__` (needs a Midway run to reconfirm end-to-end).
- **2026-07-13** — Added `DESIGN.md`, `CLAUDE.md`, and this `CHANGELOG.md` — the
  design spec, working guide (Fable 5; small commits + tests pass; explicit "things
  NOT to do"), and living document — patterned on `smsharma/clax` and the
  MARSHAL/decrypto Midway playbooks. Establishes the
  **numerical-equivalence-vs-baseline** gate as the correctness oracle.
- **2026-07-13** — Published the repo and completed the repo-wide **SNFO → SI**
  rename (SI is the correct name; SNFO was a mislabel). NGC key scrubbed to
  `$NGC_API_KEY`. `main` is branch-protected (PR + 1 review).

## Known issues / failed approaches (do NOT re-attempt)

- **Port standalone smokes had a stale cwd-relative config path** (`os.path.abspath("v2.0/config/test.yaml")`)
  from the pre-monorepo layout → `FileNotFoundError` before any GPU work. Fixed
  2026-07-13 to resolve relative to `__file__`. If a port smoke fails to find the
  config again, check this first.
- _(record other dead-ends here with the reason, e.g. "Tried torch.compile default
  mode on the port — graph break at the VAE reparam sampling; needs `dynamic=False`
  + a compile-region boundary. — <date>")_

## Benchmark results

Existing measured evidence (Midway) already lives in the repo — read these before
capturing new baselines or claiming a speedup:
- `s2s/v2.0/bench_report.md` — S2S H100 baselines.
- `si/bench_midway_notes.md` — SI A100/H100-NVL bench report + decisions log
  (note: it refutes the "H200" label for `pedramh-gpu`).
- `s2s-lightning/LIGHTNING_PORT.md` — the port's DDP/AMP/bench wiring + its
  nsys-vs-v2.0 comparison caveats.

_(record new per-cluster bench deltas below as they land — model, cluster, config,
samples/s, peak mem, and the equivalence result for any optimization. Compare only
within a cluster, never A100 vs H100 NVL.)_
