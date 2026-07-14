# Running the Polaris smokes from your own folder

**Audience:** any member of the `lighthouse-uchicago` project (written for **jesswan**).
**Covers:** PanguWeather-SFNO, SI, Makani-SFNO, PhysicsNeMo-SFNO — i.e. everything except
S2S / S2S-Lightning, which are blocked on an ERA5 Globus stage (see
`polaris_pbs_notes.md` §4).

All four are **GREEN on 4× A100** as of 2026-07-14. Nothing below is hypothetical — the
job ids are real and the logs are quoted in `polaris_pbs_notes.md` §5.

---

## 0. TL;DR

```bash
# once
cd /eagle/projects/lighthouse-uchicago/members/jesswan          # YOUR folder
git clone -b polaris-pbs-bringup git@github.com:rcc-uchicago/pedramh-profiling.git
cd pedramh-profiling

# then, from the repo:
cd PanguWeather/v2.0     && qsub HPC_scripts/polaris_train_e3sm_sfno.pbs   # Pangu-SFNO
cd si                    && qsub bench_polaris.pbs                          # SI
cd makani_sfno           && qsub polaris/polaris_sfno_smoke.pbs             # Makani
cd physicsnemo_sfno      && qsub polaris/polaris_sfno_smoke.pbs             # PhysicsNeMo
```

**You do not need to convert any data or build any env** — see §2.

---

## 1. How the scripts find *your* folder

Every script sources `polaris_env.sh`, which resolves:

| var | meaning | where it points for you |
|---|---|---|
| `MEMBER_ROOT` | **your writable dir** — all caches, logs, run dirs, CSVs | `members/jesswan` |
| `SHARED_ROOT` | group-readable artifacts to **reuse** | `members/mehta5` |
| `E3SM_ROOT` | the read-only E3SM archive | `jesswan/AI4SRM/data/E3SMv3_…` (yours!) |

It auto-detects `MEMBER_ROOT` as the `members/*` dir **you own**. (It can't just use
`$USER` — for `rmehta1987` the folder is `members/mehta5`.) If it guesses wrong:

```bash
export POLARIS_MEMBER=jesswan
```

Every job prints a `polaris_env` block at the top showing exactly what it resolved —
**check that first if anything looks odd.**

Nothing writes into another member's folder, so two people can run concurrently.

## 2. You can reuse ~75 GB of already-converted data (recommended)

The three converted datasets and the SFNO venv are group-readable, so the scripts pick
them up automatically and **skip conversion entirely**:

| artifact | size | saves you |
|---|---|---|
| `mehta5/si_e3sm_stage` | 56 GB | ~30 min h5 rename + npz→nc |
| `mehta5/data/e3sm_makani` | 18 GB | the multifiles pack |
| `mehta5/e3sm_seqzarr` | 1.7 GB | the zarr build |
| `mehta5/conda-envs/sfno-venv` | — | a torch_harmonics **source build** |

They're read-only to you — that's fine, the scripts only *read* them and write results to
your own dir.

**To build your own instead** (e.g. different years/variables), just create the same paths
under your folder and the resolver prefers yours:

```bash
export POLARIS_SHARED=/eagle/projects/lighthouse-uchicago/members/jesswan   # ignore mehta5's
bash polaris_setup_sfno_venv.sh        # LOGIN node; PASS = "SFNO_VENV_OK"  (makani/physicsnemo only)
```
The conversion steps inside each PBS script run automatically when the data is missing.

## 3. The four smokes

Submit **from the directory shown** (the scripts resolve paths relative to `$PBS_O_WORKDIR`).
Logs land as `<jobname>.o<jobid>` in that same directory.

| model | submit from | command | PASS looks like |
|---|---|---|---|
| **PanguWeather-SFNO** | `PanguWeather/v2.0` | `qsub HPC_scripts/polaris_train_e3sm_sfno.pbs` | `DONE ---- rank 0..3`, `rc=0`, finite `Loss:` |
| **SI** | `si` | `qsub bench_polaris.pbs` | `BENCH RESULT` block + a CSV row in `$MEMBER_ROOT/polaris_logs/` |
| **Makani-SFNO** | `makani_sfno` | `qsub polaris/polaris_sfno_smoke.pbs` | finite `loss=`, `Saving checkpoint`, `rc=0` |
| **PhysicsNeMo-SFNO** | `physicsnemo_sfno` | `qsub polaris/polaris_sfno_smoke.pbs` | `Epoch 0 Metrics: … loss = …`, `Saved training checkpoint`, `rc=0` |

Common options:
- `qsub -v NPROC=1 <script>` → the **1-GPU** rung. Also the best way to debug: it uses plain
  `python`, so a real traceback reaches the log (under `torchrun` the child stderr is
  swallowed and you only get a bare `ChildFailedError`).
- Queue: `debug` allows **one running job per user** and ≤1 h. Yours is separate from mine.

**Reference results** (mine, for comparison — not targets):

| model | job | result |
|---|---|---|
| Pangu-SFNO 4-GPU | 7252271 | train loss 0.3411, val 0.7049, 365 steps/rank, ~4 min |
| SI 4-GPU | 7252700 | step_med 0.400 s, **peak 30.98 GB**, 20 steps |
| Makani 4-GPU | 7252769 | train 2.19 / val 2.05, epoch 11.5 s |
| PhysicsNeMo 4-GPU | 7252933 | loss 0.889 / val err 0.541 |

## 4. Gotchas that will cost you an hour (all already handled in the scripts)

These are real bugs we hit; the scripts encode the fixes, but you'll meet them if you
deviate:

1. **Pangu `--debug` is single-GPU ONLY.** It hardcodes `world_size=1`, so under
   `torchrun --nproc_per_node=4` all 4 ranks init as rank-0-on-GPU-0 and OOM the same
   card. Bound a short run with **`--epochs 1`** instead.
2. **Lustre needs `HDF5_USE_FILE_LOCKING=FALSE`** or writing `.nc`/`.h5` dies with
   `BlockingIOError: unable to lock file`.
3. **SI: `calendar: 'noleap'` crashes the loader** (`TypeError: cannot compute the time
   difference between dates with year zero conventions`) — `noleap` is an *idealized*
   cftime calendar that forces `has_year_zero=True`. Use `standard`; it's correct for
   2015 + early-Jan 2016. A run crossing a leap year needs a loader fix.
4. **makani/physicsnemo: use the venv, and `python -m torch.distributed.run`, not
   `torchrun`.** The venv inherits torch from the conda base, so the bare `torchrun`
   resolves to the *base* python and the ranks die with `No module named 'makani'`.
   Also `PYTHONNOUSERSITE=1` — a `--system-site-packages` venv re-enables `~/.local`,
   which shadows the venv's torch_harmonics 0.9.x with the base's 0.7.4.
5. **makani `--batch_size` is GLOBAL** and must divide the rank count (`--batch_size 1` on
   4 ranks aborts).
6. **PBS appends to a fixed `-o` path**, so a log can contain several runs. **Anchor on the
   `PBS_JOBID=` header** of the run you care about — a naive `grep OutOfMemory` will report
   a green run as failed.
7. **Login node is resource-capped.** Heavy git/pack work dies with
   `unable to create thread` / `git-pack-objects died`; use `git -c pack.threads=1 push`.
   Run heavy compute as PBS jobs, never on login.

## 5. Known limitations (please don't be surprised)

- These are **tiny** models (makani: embed_dim 16, 2 layers, 4 samples/epoch; physicsnemo:
  10 iterations). They prove the loop closes — **not** performance.
- **SI numbers are not Midway-comparable**: warmup 5 / steps 20 (vs Midway's 20/80), and
  different channel counts (153 vs 151, 18 vs 26 levels).
- **SI validation/rollout on E3SM had a latent bug** (`disassemble_input` defaulted to
  `ndiagnostic=15`, we use 3). Fixed, but the same pattern remains in
  `combined_module.py`, `ae_module.py`, `bias.py` — those paths are untested here.
- **SI SST normalization is degenerate** — the converter copies upstream npz SST stats
  (mean ≈ 110) that don't describe the °C, land-filled data. Doesn't crash; scales SST badly.
- `S2S` / `S2S-Lightning` exit `ERROR ERA5_NOT_STAGED` by design until ERA5 is staged.

## 6. If something breaks

1. Read the **`polaris_env` block** at the top of the log — is `MEMBER_ROOT` your folder?
2. Re-run with `-v NPROC=1` to get a real traceback.
3. Check the job actually ran: `qstat -x -f <jobid> | grep Exit_status`.
4. `polaris_pbs_notes.md` has the full cluster facts, the GREEN matrix, every trap, and the
   conversion recipes.
