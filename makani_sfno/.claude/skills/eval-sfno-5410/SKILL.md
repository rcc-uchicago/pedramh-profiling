---
name: eval-sfno-5410
description: Evaluate the group SFNO-5410 (blocking) emulator on Stampede3 via the H100 + packed Derecho env path. Production rolls 96 ICs at K=60 forecast leads (15 days, 360 h) using `scripts/infer_sfno5410_blocking_h100_packed.py` with the exact blocking source tree, epoch-48 checkpoint, and packed Derecho env. The SLURM wrapper is `scripts/submit_eval_inference_5410_packed.slurm`. Auto-launch policy: smoke on plain phrasing; production requires an explicit "production" or "full" keyword. Group conventions (`pl=ln(p_s)`, `zg=gpm`, `pr_6h=rate×6h`) must NOT be unit-converted on top of pipeline outputs. Use when the user asks to evaluate the group SFNO-5410 emulator, run the 5410 smoke, or kick off the production 96-IC inference. For the AI-RES own emulator, defer to the sibling skill `eval-sfno-own`.
---

# Evaluate the group SFNO-5410 emulator

End-to-end on a smoke run: submit the smoke SLURM with `LIMIT_ICS=1`, return the job ID, exit. End-to-end on a production run: submit the same SLURM (no `LIMIT_ICS`) for the full 96-IC sweep, return the job ID and `RUN_ROOT`, exit.

The reliable Stampede3 runtime is **H100 GPU + packed Derecho env + exact blocking source tree + epoch-48 checkpoint**. CPU/.venv-based diagnostics from the project virtualenv do not reproduce Derecho and are not used here. Background and the full diagnostic trail are in `docs/2026-05-09_sfno_5410_investigation_history.md`.

**You SUBMIT and RETURN.** Inference is multi-hour; do not block, poll, or sleep. After printing job IDs and the relevant `run_root`, your job is done. The user re-invokes you to check status.

## Architecture

`scripts/infer_sfno5410_blocking_h100_packed.py` is a single Python process driven by the packed Derecho env. It builds upstream's `Stepper` ONCE (loading the 106M-param SFNO model + epoch-48 checkpoint + mean/std once), then loops over the 96 ICs (8 years × 12 ICs/year) calling `Stepper.reconfigure_for_ic` for each. Boundary template years are `51/52` (prescribed-boundary), not target years `121..128`. `nc_bc_offset = 0` matches upstream validation.

Key paths (defaults baked into the script — override only if you know what you're doing):

| Component | Path |
|---|---|
| Packed Derecho env Python | `/work2/11114/zhixingliu/stampede3/SFNO_Climate_Emulator/artifacts/derecho_runtime/aires_env_20260509/unpacked/bin/python` |
| Blocking source tree | `/work2/11114/zhixingliu/stampede3/SFNO_Climate_Emulator/artifacts/derecho_blocking/source_trees/forecast_modules/PanguPlasim/` |
| Epoch-48 checkpoint | `/work2/11114/zhixingliu/stampede3/SFNO_Climate_Emulator/artifacts/derecho_blocking/sfno5410_blocking_epoch48_20260509/checkpoints/ckpt_epoch_48.tar` |
| H5 sigma data | `/scratch/11114/zhixingliu/SFNO_Climate_Emulator/data/derecho_glade/sim52/h5/sigma_data` |

Production rolls each IC for **K=60 forecast leads** (15 days, 360 h). Tqdm ticks **61** raw 6-hour steps per IC for **60** scored leads (the 61st forecast is computed and discarded by upstream's `< shape[1]` save guard) — expected, not a bug.

## When to use this skill

Trigger when the user asks any of:

- "evaluate 5410" / "evaluate the group emulator" / "evaluate the group 5410 emulator"
- "run 5410 smoke" / "run the 5410 smoke test"
- "run the production 5410 eval" / "full 5410 eval" / "kick off the 96-IC 5410 run" → **production** keyword path
- "is the 5410 eval done?" / "5410 eval status"

Do NOT trigger for: anything mentioning v9 sigma, v10 zgplev, or the own emulator without "5410" — those belong to `eval-sfno-own`.

## Auto-launch policy

| User wording | Path | Confirmation needed? |
|---|---|---|
| "evaluate 5410", "run 5410", "5410 smoke" | smoke (`LIMIT_ICS=1`) | no — auto-launch |
| "production 5410", "full 5410", "96-IC 5410" | production (96 ICs) | no — explicit keyword IS the gate |
| Ambiguous ("can you run the 5410 eval?") | ask once, then proceed per answer | yes — clarify smoke vs full |

The explicit "production" / "full" keyword is itself the human checkpoint; do not add a second confirmation gate on top. Always print the job ID and `RUN_ROOT` after submit so the user can interrupt if it picked the wrong path.

## Smoke launch (auto on plain phrasing)

```bash
cd $HOME/projects/SFNO_Climate_Emulator
LIMIT_ICS=1 RUN_ROOT="$WORK2/SFNO_Climate_Emulator/results/sfno_eval_5410/$(date +%Y%m%d)_smoke_$RANDOM" \
    sbatch scripts/submit_eval_inference_5410_packed.slurm
```

Expected: 1 IC, fast turn-around. Surface the job ID and `RUN_ROOT` to the user.

## Production launch (on explicit "production" / "full" keyword)

```bash
cd $HOME/projects/SFNO_Climate_Emulator
RUN_ROOT="$WORK2/SFNO_Climate_Emulator/results/sfno_eval_5410/$(date +%Y%m%d)_blocking_96ic_h100_packed" \
    sbatch scripts/submit_eval_inference_5410_packed.slurm
```

Use a **fresh `RUN_ROOT`** for each production submission. Never reuse a previous run root — the script writes into `$RUN_ROOT/inference/upstream_raw/` and refuses to overwrite without `FORCE=1`.

**Capacity caveat — sbatch may reject this submission.** TACC enforces a per-user pending-job limit on Stampede3. If the own-track 4-job chain is already pending, the 5410 submit can be rejected at submission time with no jobid assigned. Wait for an own-track job to clear and resubmit. Surface this caveat to the user before submitting if `squeue -u $USER | wc -l` already shows several pending entries.

After the inference job completes, scoring/report/figures need to be chained manually for now — the legacy `scripts/submit_eval_5410.sh` driver is wired to the deprecated `eval_inference_5410.py` orchestrator and **must be rewired** before it can drive the packed-env path end-to-end. Until then, return only the inference job ID and `RUN_ROOT` to the user; surface the rewiring as follow-up work if they ask for the full chain.

## Group conventions (do NOT unit-convert)

The 5410 pipeline emits values under the group's conventions; downstream code must NOT re-convert these:

- `pl = ln(p_s)` — log-surface-pressure, dimensionless. Already log-space.
- `zg` — geopotential metres (gpm), not geopotential (m²/s²). Do not multiply by g or divide by 9.81.
- `pr_6h` — rate × 6h, i.e. accumulated mass per 6-hour interval (not flux × seconds). Do not multiply by 6×3600.

If you see a downstream script trying to apply g, 9.81, or 21600 to one of these — stop and ask the user before changing anything. The own-track has different conventions; conflating them silently corrupts results.

## Output tree (5410 production, after inference completes)

```
<run_root>/
└── inference/
    ├── raw_manifest.csv              # per-IC RMSEs at canonical leads
    ├── inference_metadata.json       # run provenance
    └── upstream_raw/                 # ⇐ inference output, 96 NetCDFs
        └── Y<Y>_s<ssss>_member000_y<YYYY>.nc
```

Aggregate ~125 GB across 96 files. Surface `<run_root>` and the inference output dir to the user.

## Regenerate figures only (no rerun of inference/scoring)

Applies to a 5410 `RUN_ROOT` that *has already been scored* manually (i.e. `scores/nwp_scorecard_summary.csv` and `scores/bias_maps_*.npy` exist — these are produced by `scripts/score_nwp.py`, **not** by the inference SLURM). When the figure-rendering script changes and you want to back-fill new styling onto such a run, rerun the script directly:

```bash
OUT=/path/to/results/sfno_eval_5410/<run_tag>

source $HOME/projects/SFNO_Climate_Emulator/.venv/bin/activate
cd $HOME/projects/SFNO_Climate_Emulator
python scripts/render_eval_figures.py \
    --out-root "$OUT" \
    --track 5410        # required: keeps pr_6h in native kg m^-2 (6h accum.)
```

**Do NOT pass `--benchmark-5410-out-root` when `--track 5410`** — the script ignores the flag in that mode (a 5410 run benchmarking against itself is meaningless). To overlay a 5410 run on an own-track figure, use the own-track skill's rerender form instead.

Runs in ~10–30 s on a login node, overwrites `figures/*.png` in place. Falls back to a single-row layout for bias maps (no own/5410 stacking, since the run *is* the 5410 result).

## Status check (when user asks "is the 5410 eval done?")

```bash
squeue -u $USER -o "%.10i %.20j %.8T %.10M %.20S"            # running/pending
ls -1 <run_root>/inference/upstream_raw/ 2>/dev/null | wc -l  # 0..96; 96 = done
du -sh <run_root>/inference/upstream_raw/ 2>/dev/null         # ~125 GB at completion
```

- If the inference job shows `CD` and `upstream_raw/` has 96 NetCDFs → inference done.
- If `<96` files and the inference job is `R`/`PD` → still running; ETA based on the SLURM `t` budget (4h).
- If the job shows `F` / `CA` / `TO` → look in `logs/5410_inf_packed_<id>.err`.

Before recommending a script path or job ID, verify it still matches the repo: `ls scripts/submit_eval_*5410*` and `squeue -u $USER` + `sacct -j <id>`.

## Cross-references

- 5410 plan: `docs/2026-05-06_group_sfno_5410_eval_plan.md`
- 5410 climatology prompt (Derecho-built): `docs/2026-05-06_group_sfno_5410_climatology_prompt_for_derecho.md`
- Investigation history (boundary-phase, epoch-50 vs 48, .venv vs packed env): `docs/2026-05-09_sfno_5410_investigation_history.md`
- Sibling skill (own emulator): `.claude/skills/eval-sfno-own/SKILL.md`
- Inference script (H100 + packed env): `scripts/infer_sfno5410_blocking_h100_packed.py`
- Inference SLURM (packed): `scripts/submit_eval_inference_5410_packed.slurm`
- Deprecated legacy chain (preserved for forensic reference): `scripts/submit_eval_5410.sh`, `scripts/submit_eval_inference_5410.slurm`, `scripts/eval_inference_5410.py`
