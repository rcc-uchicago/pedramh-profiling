# Pre-registration — does removing soil change A's ~500-lead blow-up? (surgical checkpoint screen)

Written 2026-09-30, **committed before the job was submitted** (operator: "okay submit the debug
job"). The job's `stime` must be later than this file's commit time.

## Why

F (7660250) is still queued, so no F checkpoint exists. The only soil-free checkpoint with real
training behind it is the surgical transfer **`surgical_nosoil_7646690/best_ckpt_mp0.tar`**: A epoch
243 with SOILWATER_10CM / TSOI_10CM sliced out (99 out / 105 in), then **20 steps at LR 1e-5**
(validation 0.01427 vs A's 0.01284; CHANGELOG 2026-09-23). It is "A without soil" with almost no
retraining — so, unlike F's own screen (F3), it is **not** confounded by 43 extra single-step epochs
(the confound the 2026-09-30 moderated analysis ruled on).

## Design

`polaris/polaris_surgical_soil_screen.pbs` (debug, 1 node) runs `polaris_climate_screen.pbs` twice,
unchanged: members **`Fsurg`** (the surgical checkpoint) and **`A_e243`** (control,
`prod1n_b32_sgdr:ckpt_mp0_v242.tar`), 1,460 leads (1 year), chunk 40, dry-air fix per config (off),
from **2044 f1092** (Oct 1) and **2044 f1156** (Oct 17). Screen metrics and ranking as
pre-registered in `docs/2026-09-24_climate_screen_prereg.md`; the 99-channel member is read through
the subset-aware stats path (`ec5ccfb7`, tested; full-width path bitwise, 7669103).

Sanity: `A_e243` from f1092 must truncate at **lead 595** again (7649647, 7649567). Anything else
means the pipeline changed, and the job is not read.

## Outcomes (per start, `L` = truncation lead; a survivor counts as 1,460)

| outcome | rule, at BOTH starts | reading |
|---|---|---|
| **S — soil implicated** | `L(Fsurg) ≥ 1.5 × L(A_e243)` | Removing soil delays the blow-up. F3 must confirm on F itself |
| **N — soil not the killer** | `0.75 ≤ L(Fsurg) / L(A_e243) < 1.5` | The blow-up does not depend on soil; stability must come from the multi-step arms |
| **D — slicing damage** | `L(Fsurg) < 0.75 × L(A_e243)` | Uninformative about soil: the model was never trained to run without soil inputs |
| **X — inconclusive** | the two starts disagree | Report both; no reading |

Also reported, not used to classify: first channel past 3σ and its lead, PS drift at 600 leads, and
the number of channels past 3σ.

## Strength

One checkpoint, two starts, single-step models only. At best this is a pointer for F3 and for the
G/H decision, not a result about F. The variable set and the fine-tunes are not tested here.

PASS = `SURGICAL_SOIL_SCREEN_OK` (both `CLIMATE_SCREEN_OK`, and the f1092 control at 595).
