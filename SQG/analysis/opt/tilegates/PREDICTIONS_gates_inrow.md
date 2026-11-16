# Gates A, B, E and D for `tilerows_inrow_if25`: predictions (written 2026-11-16, before any of these jobs)

**Ask (user, 2026-11-16):** "okay run it on capacity for gates A-D". This came after the o1.2 timing (job 7853365:
0.8885 s/step, 48.9 h on 14 A100s), whose prereg left admission open.

**The candidate.** `tilerows_inrow_if25` is `tilerows_if25` with θ's inverse real FFT moved into the row kernel
(`inrow/inrow_step.InRow`, scheme kind `tri` in `o12run/tilerows_advance.py`).
- It keeps IF-AB3 at 25 steps per frame (h = 8e-6).
- It differs from `tilerows_if25` only in float32 rounding: 6.7e-8 rel at 40 steps (job 7853317). In o1.2's in-situ
  gate the two arms score 2.718e-7 and 2.728e-7 at frame 1.
- **The reported arm is `tilerows_if25`** (in place of `if32_25`), run beside the candidate in each job. It went
  through this battery from 11-13 to 11-16: A PASS; B NOT_PASS on B3 detection in k2-4 only; E PASS_E; D PASS_D
  (15/15 predictions).

**Same gates, thresholds and reference.** Everything in `PREDICTIONS_gates_tilerows.md` holds, with
`tilerows_if25` → `tilerows_inrow_if25` (candidate) and `if32_25` → `tilerows_if25` (reported). That covers:
- the data and validity of A/B;
- gate E's design and arms;
- gate D's design, dose rule and PASS_D;
- what the result decides.

The rule files are unchanged: `taxonomy/ACCEPTANCE_gates_AB.md`, `energy/PREDICTIONS_energy.md`,
`gateD/PREDICTIONS_gateD.md`. Admitting `tilerows_inrow_if25` stays the user's call.

**Harness changes.** Their defaults reproduce the `tilerows_if25` battery.
- **A/B:** `tr_pairs.py --arms`, `gates_ab_tr.py --arms/--candidate`, and `TR_ARMS` / `TR_CAND` in
  `run_tr_ab_polaris.pbs`. Checked on the login node: with its defaults, `gates_ab_tr.py` on job 7841299's pairs
  reproduces that job's `gates_ab_tr.json` and bias pattern. Keys and verdicts are identical; floats differ by
  ≤ 1.6e-14 rel (numpy 1.17 against the job's).
- **E:** `energy_budget.py` registers `tilerows_inrow_if25` at 25 steps per call; `TRE_ARMS` / `TRE_CANDS` in
  `run_tr_energy_polaris.pbs`.
- **D:** `doses_from_trab.py` reads the doses when the job starts, by the same rule.
  - This lets D be submitted together with A/B (`-W depend=afterany:<A/B>`).
  - It refuses unless A/B was valid, judged this candidate, wrote its pattern with a finite R > 0, and ended
    `TRAB_JOB_DONE failures=0`.
  - Self-test OK. On job 7841299 it returns 860 / 8600, job 7843532's doses.
  - Run name `gateD_inrow_v1`. The production frames `gateD_frames` are reused: complete segments are skipped and
    checked bitwise.

**Gate D's design is unchanged, including N(w) from one α = 0 seed pair.** Part XVII H found that same-seed
training is not bitwise reproducible. N therefore understates the noise, which makes the rule stricter than written,
so a PASS stays valid. An N from ≥ 3 realizations would be a design change; it is not made here.

**Queue (the user's call): all three jobs on capacity.** A/B and E get 1 h walltime each, D 36 h (job 7843532 used
23.3 node-h). Capacity runs one job per project at a time.

## Predictions (P = probability)

| # | prediction | P |
|---|---|---|
| A1 | gate A/B validity holds (V_frames 64/64, V_sealed, V_floor) | 0.92 |
| A2 | gate A PASS for `tilerows_inrow_if25`, every band Var_b(e)/A_b ≤ 1e-5 | 0.92 |
| A3 | gate B: B1 and B2 hold in every band | 0.88 |
| A4 | gate B NOT_PASS on detection only: every non-PASS band fails B3 alone (B1, B2 and power hold), and these bands are `tilerows_if25`'s in the same job | 0.75 |
| A5 | `tilerows_inrow_if25` and `tilerows_if25` (same job) reach the same gate A verdict and the same gate B verdict in every band | 0.85 |
| A6 | size_ratio R within ±2% of job 7841299's (1.1643, `tilerows_if25`) | 0.8 |
| A7 | the doses equal job 7843532's: α_lo = 860 (so α_hi = 8,600) | 0.6 |
| A8 | vs ref160: in every band, the candidate's median e_b is within ±10% of `tilerows_if25`'s (same job) | 0.85 |
| A9 | `tilerows_if25`'s gate A verdict and per-band gate B verdicts equal job 7841299's | 0.95 |
| E1 | gate E valid (V1–V5) | 0.88 |
| E2 | **PASS_E** for `tilerows_inrow_if25` (max \|s\| ≤ 1e-4) | 0.88 |
| E3 | against `tilerows_if25` in the same job: max \|s_SPE\| within 0.8–1.25×, and max \|s_VIE\| ≤ 2× | 0.7 |
| E4 | DETECTED for at least one invariant | 0.85 |
| E5 | the candidate's ms/step (budget included) is below `tilerows_if25`'s in the same job | 0.85 |
| D1 | **PASS_D** on best.pt | 0.8 |
| D2 | no overfitting in the α = 0 arms (final / min < 1.10) | 0.88 |
| D3 | the α_hi positive control drifts beyond N(w) in ≥ 1 window (`power_drift`) | 0.65 |

## Tokens and outputs

- As `PREDICTIONS_gates_tilerows.md`: `TRAB_*` → `results/polaris_trab_<job>/`, `ENERGY_*` →
  `results/polaris_tre_<job>/`, `DOSE_*` → `results/polaris_trgateD_<job>/`.
- D also prints `TRD_DOSES <α_lo> <α_hi> <pattern> <R>`, or `ERROR TRD_*` when it refuses.
- Scoring: `score_gates_inrow.py --ab <A/B dir> --e <E dir> --d <D dir>/best/dose_summary.json`, written before the
  results are read.
