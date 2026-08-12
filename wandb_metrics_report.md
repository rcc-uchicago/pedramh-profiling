# SFNO-E3SM parity: PanguWeather vs. ai-rossby — per-variable metrics snapshot

Point-in-time snapshot, **2026-08-12**, pulled directly from each run's local wandb
datastore file (no wandb.ai access needed — see `wandb_local_history.py`, which every
number below was generated with). Re-run that script for fresher numbers; this file is
a snapshot, not a live dashboard.

Both runs share the SFNO-E3SM parity architecture (bitwise param-count match,
`compare_sfno_parity.py`) and the same 101-channel variable contract (§ below), but see
`ai_rossby_finegrained_wandb_handoff.md` for the full history of why their wandb
schemas diverged and how that gap was closed.

## Run identity

| | PanguWeather | ai-rossby |
|---|---|---|
| Job (at snapshot time) | `7368539` (queued on `preemptable`, was `7368237` on `capacity`) | `7368536` (running on `capacity`, was `7368547` on `preemptable`) |
| wandb run | `run-20260807_150638-j796bp1k` | `run-20260812_165521-03ha0747` (post-cutover; see below) |
| Local datastore path | `/eagle/projects/lighthouse-uchicago/members/mehta5/wandb/wandb/run-20260807_150638-j796bp1k/run-j796bp1k.wandb` | `/lus/eagle/projects/lighthouse-uchicago/members/mehta5/runs/ai_rossby_sfno_e3sm/sfno_e3sm_parity01/wandb/wandb/run-20260812_165521-03ha0747/run-03ha0747.wandb` |
| Epoch at snapshot | ~41 (last checkpoint; job idle in queue since — see Known issues in CHANGELOG.md re: `preemptable` `queue_tags`) | 31 complete, mid-epoch 32 |
| Schedule | 100 epochs, `LinearWarmupCosineAnnealingLR`, 5-epoch warmup, `eta_min=1e-5` | 100 epochs, same scheduler shape, `eta_min=1e-8` |

## ai-rossby's variable-coverage cutover — the key fact this file exists to record

ai-rossby's wandb history is **two different schemas stitched together at epoch 30**,
because the per-variable diagnostics work
(`ai_rossby_finegrained_wandb_handoff.md`) only landed in the code on 2026-08-12, at
the queue-rotation restart:

| | Epochs 1–30 (`run-20260809_172041-cz4pfmed`) | Epoch 30+ (`run-20260812_165521-03ha0747`) |
|---|---|---|
| Metric keys | **5**: `train/mini_batch_{loss,surface,upper_air,diagnostic,vae_kl}` (LaunchLogger aggregate buckets only) | **101**: `train_{var}_lwrmse` / `train_{var}_level{level:.4f}_lwrmse`, byte-identical key scheme to PanguWeather |

Before epoch 30 there is **no way** to ask "which ai-rossby variable is lagging" — only
the three coarse buckets (surface/upper_air/diagnostic) existed. From epoch 30 on it
has the same 101-channel resolution PanguWeather has always had. Any cross-run
per-variable comparison before epoch 30 is impossible by construction, not a data gap.

## PanguWeather — aggregate loss trajectory (epoch 1 → 41)

Epoch-end `Train loss:` / `Validation loss:` log lines (physical training-loss units,
not wandb's per-channel de-normalized RMSE):

| Epoch | Train loss | Validation loss |
|---|---|---|
| 1 | 0.02634 | 0.05258 |
| 10 | 0.01157 | 0.02815 |
| 17 | 0.00844 | 0.02680 |
| 20 | 0.00853 | 0.02663 |
| **25 (low point)** | **0.00775** | **0.02655** |
| 27 | 0.00741 | 0.02658 |
| 31 | 0.00706 | 0.02667 |
| 36 | 0.00685 | 0.02676 |
| 40 | 0.00650 | 0.02683 |
| **41 (latest)** | **0.00708** | **0.02685** |

Validation loss bottomed out around **epoch 25** (0.02655) and has drifted **up**
every few epochs since, to 0.02685 by epoch 41 — a small but consistent reversal, not
just noise around a floor.

## PanguWeather — per-channel training RMSE (101 channels, physical units)

All 101 tracked channels are **still improving** on the training set (none flat,
none regressing) as of epoch ~38.6, but at very different rates. Grouped by base
variable (upper-air groups average over their 18 levels):

| Group | Recent trend (last 20% of steps vs. the 20% before) |
|---|---|
| `PRECT` | **−10.4%** (fastest) |
| `PS`, `RELHUM` | −5.1% |
| `PSL` | −4.8% |
| `Z3` | −4.6% |
| `U10`, `RHREFHT`, `V`, `TREFHT` | −4.0 to −4.1% |
| `U`, `TMQ` | −4.0% |
| `TSOI_10CM` | −3.6% |
| `T` (all 18 levels) | −3.5% |
| `FSNT`, `FSNTOA` | −3.3% |
| `SOILWATER_10CM` | **−1.8%** (slowest) |

Reading the two tables together: **every training-side channel keeps improving, but
none of that is reaching the validation set anymore** — the textbook shape of
training-set overfitting, not one bad channel dragging the aggregate down.

## ai-rossby — aggregate loss trajectory

**Epochs 1–30** (`train/mini_batch_loss` reduction + `valid/val_loss`, log-derived):

| Epoch | Train loss | Validation loss |
|---|---|---|
| 1 | 0.5792 | 0.1257 |
| 10 | 0.0331 | 0.0358 |
| 19 | 0.0235 | 0.0328 |
| 24 | — | 0.03258 |
| **25 (first flat point)** | — | **0.03258** |
| 29 | — | 0.03268 |
| **30 (last epoch before the swap)** | — | **0.03270** |

**Epoch 31+** (post-resume, new run):

| Epoch | Train loss | Validation loss |
|---|---|---|
| 31 | 0.01857 | 0.03274 |

Validation loss decelerated from ~0.4%/epoch (around epoch 17-19) to ~flat by epoch
24-25, then — like PanguWeather — started **ticking up**: 0.03258 → 0.03268 → 0.03270
→ 0.03274 over epochs 25→31. Same qualitative pattern as PanguWeather, reached at a
slightly earlier fraction of the 100-epoch schedule (~epoch 25/100 vs. PanguWeather's
~epoch 25-27/100 — actually nearly identical fractional onset).

## ai-rossby — per-channel training RMSE (101 channels, epoch 30+ only)

Only ~1.5 epochs of data exist at this resolution so far (the schema cutover means
there is no earlier per-channel history to compare against). Grouped by base variable:

| Group | Recent trend (last 20% of logged steps vs. the 20% before) |
|---|---|
| `PRECT` | **−1.7%** (fastest) |
| `FSNT`, `FSNTOA` | −1.4% |
| `PSL` | −0.9% |
| `RHREFHT`, `RELHUM` | −0.7% |
| `U10`, `TREFHT`, `PS`, `SOILWATER_10CM`, `TMQ` | −0.5 to −0.7% |
| `V`, `U`, `T` | −0.5% |
| `TSOI_10CM` | −0.5% |
| `Z3` | **−0.4%** (slowest) |

Same overall ranking shape as PanguWeather at a comparable point in *its* schedule
(`PRECT` fastest, diagnostics near the top) — but every rate here is roughly 3-6× smaller
than PanguWeather's equivalent-window numbers. That's consistent with ai-rossby
already being past its own validation-plateau onset (epoch ~25) by the time this
window starts (epoch 30+), rather than mid-plateau the way this window caught it for
PanguWeather (epoch ~33-38).

## How to reproduce / go deeper

```bash
module use /soft/modulefiles && module load conda && conda activate base
python3 wandb_local_history.py <run_dir> --summary                # trend per key
python3 wandb_local_history.py <run_dir> --csv out.csv            # full per-step series
```

No wandb.ai login needed — every path above is `lighthouse-uchicago`-group-readable.
See `wandb_local_history.py`'s docstring for the full explanation of why this works
(wandb's local `.wandb` leveldb-log datastore is readable directly with the `wandb`
package's own `DataStore` reader, entirely offline).
