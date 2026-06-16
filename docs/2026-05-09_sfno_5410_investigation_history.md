# SFNO-5410 evaluation — investigation history (through 2026-05-09)

This is the dated diagnostic log that previously lived in
`.claude/skills/eval-sfno-5410/SKILL.md`. It was moved here on
2026-05-09 once the H100 + packed Derecho env path was confirmed as
THE valid Stampede3 production runtime for the blocking emulator.
The skill itself now describes only that runtime; this file preserves
the trail that led there in case the same diagnostics are ever needed
again.

Source: prior `.claude/skills/eval-sfno-5410/SKILL.md` (verbatim sections).

---

## Important 2026-05-09 correction

Do not quote metrics from `results/sfno_eval_5410/20260508_5410_vs_gb4/report.md`
as valid group-SFNO performance. That run used `nc_bc_offset=18`, so the
first forecast step consumed boundary forcing from `init+18h` instead of
IC time. Also do not quote the upstream 5410 epoch-50 validation CSVs
(`tas` day-14 ACC around 0.324, `zg500` around 0.414) as the expected
score for the local 96-IC year-121+ eval. Those CSVs were produced on
upstream validation year 11 in the March 17 Derecho-era runtime. After
transferring year 11 locally and rerunning upstream validation on
Stampede3, the current environment still failed to reproduce those CSVs.
A post-fix full rerun with `nc_bc_offset=0` also produced near-zero
day-14 ACC; see the investigation result below before interpreting that
as a scoring-only failure.

## Blocking-checkpoint correction, 2026-05-09

The Derecho extreme blocking forecast that motivated this comparison
did **not** use `ckpt_epoch_50.tar`. Derecho provenance showed the
runtime `training_checkpoints/best_ckpt.tar` resolved to
`/glade/work/awikner/PanguWeather/v2.0/results/SFNO/5410/checkpoints/ckpt_epoch_48.tar`
with SHA256
`1af4a89fc8a61e4d82008c601a9f434a31f9443c6d5723488fb772057db8ab09`.
The verified Stampede transfer lives at
`/work2/11114/zhixingliu/stampede3/SFNO_Climate_Emulator/artifacts/derecho_blocking/sfno5410_blocking_epoch48_20260509/checkpoints/ckpt_epoch_48.tar`.
Treat the epoch-50 full test run as valid only for epoch 50, not as the
blocking emulator. A K=1 one-IC probe submitted as job `3100509`
completed successfully and logged `Restored from epoch 48, iteration
219024`; its raw output is under
`/work2/11114/zhixingliu/stampede3/SFNO_Climate_Emulator/results/sfno_eval_5410/20260509_5410_blocking_epoch48_k1_probe_runroot/inference/upstream_raw/`.
For the first IC (`Y=121`, `s=0000`), the 6h lat-weighted `zg500` RMSE
was `20.7566`, essentially the same as the epoch-50 run's `20.7329` for
that IC. Do not launch a full epoch-48 production eval unless the user
explicitly asks for "full" or "production".

## Boundary-template correction, 2026-05-09

Derecho's smallest-case fingerprint package for the blocking runtime
showed the trusted path uses prescribed-boundary template years
`51/52`, not target years `121..128`: for `Y=121,s=0`, model step 1
reads `51_0000.h5` and step 56 reads `51_0055.h5`. The previous
Stampede 5410 evals generated per-Y yamls and in-process reconfigure
calls with `leap_year=no_leap_year=Y`, so they used target-year
boundary forcing and are not faithful reproductions of the blocking
emulator. Stampede currently lacks local `51_*.h5` and `52_*.h5` in
`/scratch/11114/zhixingliu/SFNO_Climate_Emulator/data/derecho_glade/sim52/h5/sigma_data/`;
transfer those years before rerunning. The local builder/orchestrator
has been patched to emit/preserve `boundary_no_leap_year=51`,
`boundary_leap_year=52`, while keeping `val_year_start/end=Y/Y+1`.
After transferring years 51/52, job `3100907` reran the smallest
`Y=121,s=0,K=56` epoch-48 probe with the patched boundary template. It
still produced the bad Stampede RMSE (`zg500`: 6h `20.7566`, 24h
`51.8316`, 336h `128.2027`) instead of the Derecho CPU reference (6h
`2.8062`, 24h `3.5408`, 336h `61.6215`). For the checked forcing fields
at `*_0000.h5`, years 51 and 121 were byte-identical, and raw NetCDF
time=0 matched the IC H5 exactly for `tas` and `zg500`. Do not claim
the boundary template alone fixes the evaluation. The next active
suspect is the exact blocking inference/data-loader source tree:
Derecho's direct script imports `ensemble_inference.py` and
`forecast_modules/PanguPlasim/utils/data_loader_multifiles.py` (hash
`d63e2e3f...`), while the Stampede orchestrator uses the patched
`long_inference.py` mirror and a different data loader.

## Fingerprint narrowing, 2026-05-09

After the Derecho fingerprint package for `Y=121,s=0` was transferred
to
`/work2/11114/zhixingliu/stampede3/SFNO_Climate_Emulator/artifacts/derecho_fingerprints/sfno5410_y121_s0_fingerprint_20260509`,
Stampede-side checks showed the IC/truth H5 fields match Derecho
byte-for-byte at leads 0, 6, 24, 72, 120, 240, and 336h. The local data
loader also matched Derecho's normalized inputs exactly:
`normalized_input_surface=6034841f...`,
`normalized_input_upper_air=263f9f23...`,
`normalized_varying_boundary=007b70aa...`, and
`normalized_constant_boundary=fb3f2552...`. The loaded epoch-48 model
state hash after stripping `module.` prefixes was
`81653ea1f4ecb37d051188cfb46794550b79a1ca8d8346d413b1ee594bc2a31f`,
matching the checkpoint. Despite identical inputs and weights, a local
CPU one-forward probe produced first-forward hashes
`surface=ff808b61...`, `upper_air=d2594804...`,
`diagnostic=0655f8b0...`, while the Derecho reference is
`surface=2f1ca2f0...`, `upper_air=9972a33c...`,
`diagnostic=4e160926...`. The remaining suspects are therefore the
exact blocking runtime source/import path and/or numerical stack
(`torch/torch_harmonics`), not scoring, H5 truth, plev selection, unit
conversion, boundary year selection, or checkpoint identity.

## Exact blocking source + hooks, 2026-05-09

Derecho later transferred the exact blocking source tree to
`/work2/11114/zhixingliu/stampede3/SFNO_Climate_Emulator/artifacts/derecho_blocking/source_trees/forecast_modules/PanguPlasim/`.
Key hashes match Derecho: `ensemble_inference.py=69c21851...`,
`utils/data_loader_multifiles.py=d63e2e3f...`,
`utils/YParams.py=d1856e1b...`, and `utils/integrate.py=6c90c17c...`.
Running the transferred hook script logic locally with this exact
source still does **not** reproduce Derecho. The concatenated
58-channel SFNO input before the encoder matches exactly
(`8e1fc4f...`), but the first forward diverges at the model operators.
In FP32/no-autocast: encoder differs only slightly (`rmse=3.88e-05`),
but after block 0 the local-vs-Derecho tensor difference jumps to
`rmse=0.687`, `meanabs=0.525`, `maxabs=2.73`. Feeding Derecho's exact
`after_positional_embedding` tensor into the local `model.blocks[0]`
still diverges by the same amount, so the active culprit is inside
block 0 / spectral-operator runtime, not the loader, source tree,
checkpoint, or pre-block inputs. Isolated local tests with
`torch_harmonics==0.7.4` and with `torch==2.6.0+cpu` plus
`torch_harmonics==0.7.4` did **not** change the local hashes. The
remaining requirement is an exact Derecho numerical environment (not
just matching source files), especially the PyTorch 2.6.0+cu124 CPU
build/linked libraries or a containerized copy of the Derecho runtime.

## Inverse-transform replay, 2026-05-09

Derecho transferred an exact `conda-pack` of
`/glade/work/zhil/conda_envs/aires`; after unpacking it locally,
Stampede still fails the one-operator replay. Script
`scripts/debug_sfno5410_inverse_replay.py` feeds Derecho's exact
`filter_after_contraction_full_spectrum_complex` from the block0 deep
hook NPZ directly into the local
`model.blocks[0].filter.filter.inverse_transform`. With the packed
Derecho env (`torch 2.6.0+cu124`, `torch_harmonics 0.7.4`) the replay
output hash is `71c78b74...`, while Derecho's saved
`filter_after_inverse_transform_pre_bias` hash is `484e799d...`
(`rmse=0.380084`, `mean_abs=0.225889`, `max_abs=6.35794`). Forcing
`ATEN_CPU_CAPABILITY=avx2`, `MKL_ENABLE_INSTRUCTIONS=AVX2`,
`MKL_CBWR=AVX2`, or `MKL_CBWR=COMPATIBLE` changes only byte hashes,
not the large mismatch. A manual float64 implementation of the same
`InverseRealSHT.forward` math reproduces the local result, not
Derecho's saved inverse output. Before launching more evaluations,
ask Derecho to run the same inverse-only replay against its own NPZ
and to dump the inner inverse stages (`rl`, `im`, stacked complex
pre-irfft, final irfft), plus
`inspect.getsource(torch_harmonics.sht.InverseRealSHT.forward)` and
`sha256sum` of `torch_harmonics/sht.py`.

## Stampede H100 reproduction, 2026-05-09 (THE valid path)

The same inverse replay on a Stampede H100 compute node with the
packed Derecho env matches Derecho byte-for-byte for the decisive
tensor: `inverse_from_derecho_contracted_pre_bias=484e799d...`,
`vs_derecho_rmse=0.0`. A smallest actual forecast/RMSE probe on H100
using the exact epoch-48 checkpoint, exact blocking source tree, exact
packed Derecho env, and local H5 data also reproduces Derecho's good
z500 scores. For `Y=121`, ICs `0,122,244`, mean RMSEs are: 6h
`2.4603`, 24h `3.7268`, 72h `9.2570`, 120h `17.3646`, 240h `44.7694`,
336h `67.5669`, matching the Derecho CPU diagnostic to roundoff.
Therefore the reliable Stampede path for SFNO-5410 is **GPU + packed
Derecho runtime + exact blocking source/checkpoint**; the earlier bad
Stampede evaluations from the project `.venv`/CPU diagnostics should
be treated as invalid for judging the blocking emulator.

## Valid 96-IC rerun launched, 2026-05-09

Invalid old score/report outputs were removed from the known bad
roots: `20260508_5410_vs_gb4`,
`20260509_5410_exact_epoch50_test_full`,
`20260509_5410_blocking_epoch48_test_full`, and
`20260509_eval-boundaryfix_5410_full`; the old full invalid
`inference/upstream_raw` directories were also removed from the known
bad run-roots to prevent accidental `SKIP_INF=1` reuse. The clean
valid run root is
`/work2/11114/zhixingliu/stampede3/SFNO_Climate_Emulator/results/sfno_eval_5410/20260509_blocking_96ic_h100_packed_derecho_env_valid`.
Jobs: inference `3101102`, scoring `3101103`, combined post
report+figures+summary `3101105`. The inference uses
`scripts/infer_sfno5410_blocking_h100_packed.py` with H100 GPU, packed
Derecho env
`/work2/11114/zhixingliu/stampede3/SFNO_Climate_Emulator/artifacts/derecho_runtime/aires_env_20260509/unpacked/bin/python`,
exact blocking source tree, and epoch-48 checkpoint. The post job
writes `completion_summary.md` with counts, sanity-gate status,
scorecard/report paths, and tas/zg500 ACC/RMSE at 6, 24, 120, and
336 h.

## Investigation result, 2026-05-09

The local low-skill result reproduces inside upstream `train.py
--just_validate` when pointed at local year 121. Diagnostic job
`3099015` used the same epoch-50 checkpoint and local mirror
(`val_year_start=121`, `num_inferences=128`, `WORLD_SIZE=1`) and
wrote CSVs under
`/work2/11114/zhixingliu/stampede3/SFNO_Climate_Emulator/results/sfno_eval_5410/20260509_val_y121_upstream/inference/SFNO/5410/plots/acc/`.
Its metrics were:

| Variable | 6h ACC/RMSE | 24h ACC/RMSE | 336h ACC/RMSE | 360h ACC/RMSE |
|---|---:|---:|---:|---:|
| `tas` | 0.899 / 1.358 | 0.575 / 3.352 | 0.019 / 7.292 | 0.019 / 7.332 |
| `zg500` | 0.957 / 19.801 | 0.730 / 55.710 | -0.038 / 138.577 | -0.036 / 138.909 |

This means the post-fix AI-RES score/report is not alone: upstream
validation itself scores badly on the local year-121 data.

After the original validation year 11 was transferred to Stampede3
(`/scratch/11114/zhixingliu/SFNO_Climate_Emulator/data/derecho_glade/sim52/h5/sigma_data/11_*.h5`
plus `12_0000.h5`), diagnostic job `3099245` reran upstream
`train.py --just_validate` with `val_year_start=11`, `val_year_end=12`,
`num_inferences=128`, `WORLD_SIZE=1`, `model.eval()`, and
`epsilon_factor=0.0`. It completed successfully and wrote CSVs under
`/work2/11114/zhixingliu/stampede3/SFNO_Climate_Emulator/results/sfno_eval_5410/20260509_val_y11_upstream_eps0/inference/SFNO/5410/plots/acc/`.
The result still did not reproduce the March 17 epoch-50 CSVs:

| Variable | 6h ACC/RMSE | 24h ACC/RMSE | 336h ACC/RMSE | 360h ACC/RMSE |
|---|---:|---:|---:|---:|
| `tas` | 0.900 / 1.353 | 0.590 / 3.323 | 0.047 / 7.254 | 0.044 / 7.289 |
| `zg500` | 0.959 / 19.693 | 0.748 / 55.011 | 0.021 / 135.895 | 0.021 / 136.213 |

The strongest reproducibility clue is the first forward diagnostic.
The March 17 `results/SFNO/5410/out.log` loaded `ckpt_epoch_50.tar`,
saw `val_input_surface[:1,:,0,0] = tensor([[0.4416, -1.8768]])`,
`Model training mode: False`, and produced
`val_output_surface[:1,:,0,0] = tensor([[0.4312, -1.9141]])`. The
current Stampede3 rerun loaded the same checkpoint, saw the same first
input/boundary scalars and `Model training mode: False`, but produced
`tensor([[0.3647, -1.8994]])`. That points upstream of downstream
scoring/reporting. Current `.venv` is also far newer than the artifact
environment files (`torch 2.11.0+cu130`, `torch_harmonics 0.8.1`,
`numpy 2.4.4` versus recorded envs around `torch 2.3.x`/`2.6.0` and
`torch_harmonics 0.6.2`). Before trusting any new 5410 score from the
project `.venv`, recreate the pinned training/validation environment
or obtain the exact March 17 runtime, then rerun a small upstream
validation probe and compare the first forward output against
`0.4312, -1.9141`. (The H100 + packed Derecho env path documented
above sidesteps this entirely and is the recommended runtime.)

## Boundary-lag rumor test, 2026-05-09

A follow-up diagnostic tested the rumored training mismatch where
autoregressive validation uses varying boundary forcing as
`[t0, t0-6h, t0-12h, ...]` instead of `[t0, t0+6h, t0+12h, ...]`.
A diagnostic-only knob `validation_boundary_reverse_lag_hours: 6`
was added to the upstream artifact data loader and job `3099349`
reran epoch-50 validation on local year 11. Output:
`/work2/11114/zhixingliu/stampede3/SFNO_Climate_Emulator/results/sfno_eval_5410/20260509_val_y11_bcrev6_upstream/inference/SFNO/5410/plots/acc/`.
It did not help. Step 0 is intentionally identical to the clean run;
by 12h and later the scores are same-or-worse:

| Variable | 12h clean -> reverse | 336h clean -> reverse | 360h clean -> reverse |
|---|---:|---:|---:|
| `tas` ACC/RMSE | 0.773/2.196 -> 0.677/2.696 | 0.047/7.254 -> 0.023/7.446 | 0.044/7.289 -> 0.019/7.514 |
| `zg500` ACC/RMSE | 0.889/33.615 -> 0.886/34.086 | 0.021/135.895 -> 0.021/136.646 | 0.021/136.213 -> 0.021/137.133 |

So the local low-skill result is not fixed by this backward 6h
boundary sequence.

---

## Pre-2026-05-09 architecture notes (now superseded)

These notes described the in-process orchestrator that ran on the
project `.venv` (`scripts/eval_inference_5410.py`,
`src/sfno_inference_5410/upstream_hydration.py`,
`src/sfno_inference_5410/preflight.py`). That path is preserved in
the repo for forensic reference but is **not** the valid production
runtime — it failed to reproduce the Derecho blocking forecast even
after fingerprint matching all inputs and weights. The H100 + packed
Derecho env path (`scripts/infer_sfno5410_blocking_h100_packed.py`)
is the authoritative runtime as of 2026-05-09.

Pre-2026-05-09 architecture references:
- Design plan: `docs/2026-05-08_sfno_5410_inproc_orchestrator_plan.md`
- LP-003 / LP-004 patches: `docs/2026-05-04_makani_local_patches.md`
- K=60 horizon plan: `docs/2026-05-08_sfno_5410_explicit_K_horizon_plan.md`
- Scoring chain plan: `docs/2026-05-08_sfno_5410_scoring_plan.md`
- Old standard driver: `scripts/submit_eval_5410.sh` (still references
  `scripts/eval_inference_5410.py`; deprecated for production use)
- Old inference SLURMs: `scripts/submit_eval_inference_5410.slurm`,
  `scripts/submit_eval_inference_5410_smoke.slurm` (still call
  `eval_inference_5410.py` — preserved for forensic comparison only)
