# SQG item 3 (generator-error taxonomy): handoff (written 2026-11-06, ~22:10)

**Prompt for the fresh session (paste as-is):**

> Continue SQG item 3 (the generator-error taxonomy). Call EnterWorktree with
> `path=/lus/eagle/projects/lighthouse-uchicago/members/mehta5/pedramh-profiling/.claude/worktrees/sqg-polaris-profiling`,
> then read `SQG/analysis/opt/taxonomy/HANDOFF_taxonomy.md` and follow its §3. Job 7824121 may still be queued; don't
> resubmit it. Never open or print the sealed file.

This file is a starting point. The record is:
- `PREDICTIONS_taxonomy.md`, including its addendum;
- the 2026-11-06 "(later)" entry in `CHANGELOG.md`;
- `SQG/polaris/polaris_sqg_notes.md` Parts X–XI, and the team plan in `../team_2026-11-06/TEAM_REPORT.md`.

## 1. Where things stand

| item | value |
|---|---|
| Branch / worktree | `sqg-polaris-profiling` (sparse: cone `SQG` + root files), pushed. Draft PR #25 → base `sqg`. |
| Commits | `3854307c` harness + pre-registration; `3afd9eed` smoke results + paired control b + addendum; this handoff after. |
| **Archive run** | **job 7824121** (`debug`, 1 node, walltime 0:59, submitted 21:08). At 22:02 it was still `Q` with "Insufficient amount of resource: queue_tags": no free debug node, 54 min of eligible time accrued. **Do not resubmit**; that resets the eligible time. Expected run time ≈ 35 min once started. |
| Log | `/eagle/projects/lighthouse-uchicago/members/mehta5/sqg_profile/logs/7824121.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov.OU`. It appears when the job STARTS; key on the end token `TAXO_JOB_DONE failures=N`, not on the file existing. |
| Raw outputs | `$MEMBER_ROOT/sqg_profile/taxonomy_7824121/`: `pair_w*_s*_p*.npz`, `tables_w*.npz`, `worker_*.json` / `.log`, `analyze_taxonomy.log`, and the **sealed file `sealed/sealed_taxonomy.json`**. |
| Committed summaries | `SQG/analysis/results/polaris_taxonomy_7824121/`, written by the job. The results dir is gitignored, so use `git add -f`. |
| Smoke (done) | job 7823653, `failures=0`. Summaries in `SQG/analysis/results/polaris_taxonomy_smoke_7823653/`. Its re-analysis with the paired control b is in `reanalysis_paired_ctrl_b/`. |
| States | 16 archived checkpoints, members 1–4 × `ckpt_t0{100,200,300,400}.00.npz`, 402,853,585 B each. **The user placed them in the main checkout's root** (`/eagle/.../mehta5/pedramh-profiling/trainingdata_m0*_…/checkpoint/`). They are **symlinked** into `$MEMBER_ROOT/sqg_stage/AllTrainData/<label>/checkpoint/`, which is where the job reads. Headers were checked: t and blocks match, the forcing hash is identical, there is one config hash per member. |

## 2. What the job does (details: `PREDICTIONS_taxonomy.md`, `taxonomy.py` / `analyze_taxonomy.py` docstrings)

**The question.** For each faster solver, is its error in o1.2's 0.02 delta noise (zero conditional mean given the
input) or a learnable bias? It answers per coarse band and per shell (one kx² + ky²). The solvers:
- `lean32ek`, and its fast form `lean32ek_fast`;
- `lean32`;
- `lean32sr`, stochastic rounding;
- dt = 2e-4/30, 2e-4/32 and 2.5e-6.

**The data.** 64 pairs, clustered by 16 archived states, against float64 `lean_u1` (bitwise production).

**Validity gates. Any failure voids the bias conclusions.**
- V1: `prod` == `lean_u1` bitwise.
- V2, control a: `dquant` is float64 with only its decay rounded to float32. Gate: κ ∈ [0.8, 1.2] at k < 8.
- V3, control b, **paired form (addendum)**:
  - recovery: α(inj − lean32ek)/α(inj64) ∈ [0.7, 1.3] in every band;
  - power: |α(inj64)| ≥ 3 SE in every band.
- V4: the dt ladder's n = 40 equals `lean_u1` bitwise; otherwise the dt arms are discarded.

**SEALED (the user's choice).** The bias statistics of `lean32ek`, `dt30` and `dt32`, and the dt order ratio, sit in
`$MEMBER_ROOT/sqg_profile/taxonomy_7824121/sealed/sealed_taxonomy.json`. Only its sha256 (printed `TAXO_SEALED`) goes
into the record. **Do not open, print or summarize it** until jesswan / Weidong sign item 4's gates and the user
says to unseal (`analyze_taxonomy.py --unseal`). Predictions 2 and 5 stay unscored until then.

## 3. What to do next, in order

1. **Check the job.** Run `qstat -x 7824121` once; don't poll in a loop. If it is still `Q` on `queue_tags`, tell the
   user and stop; the user says when it ran. If it has finished, read the whole `.OU`:
   - `TAXO_JOB_DONE failures=N`, `TAXO_ANALYSIS_DONE valid=… control_a=… control_b=… prod_bitwise=… dt_valid=…
     fast=…`;
   - the `TAXO_CONTROL`, `TAXO_MDE`, `TAXO_CALIBRATION`, `TAXO_EDELTA`, `TAXO_BIAS`, `TAXO_DT80` and `TAXO_SEALED`
     lines.

   On any `ERROR`, read `$MEMBER_ROOT/sqg_profile/taxonomy_7824121/worker_<w>.log` first. Likely causes:
   - checkpoint load: config / forcing / qh hash. The headers were fine; the qh sha256 is checked only at load.
   - the walltime: about 35 min is expected against 59.

   If the workers finished but the analysis crashed, re-run the analysis on the login node:
   `OPENBLAS_NUM_THREADS=1 python3 SQG/analysis/opt/taxonomy/analyze_taxonomy.py --in <raw dir> --out <results dir>
   --sealed-dir <raw dir>/sealed`. Use the system `python3` with numpy 1.17; that is allowed. The sqg-env Python is
   NOT allowed on the login node.
2. **Score `PREDICTIONS_taxonomy.md`** item by item (confirmed / refuted / half) without editing it. Apply the
   addendum's rules:
   - V3 in its paired form;
   - for P2/P8: if the pooled negative-control shell rate (`TAXO_CALIBRATION`, f64ic32 + lean32sr) is above 1%, a
     "≤ 1% of shells" criterion means "not above the negative controls by more than 2 binomial SE".
   - P2 and P5 are sealed: write "sealed".
   - P4 (mechanism B, lean32 − dquant) has no prior: report which agent's prediction the data favour.
3. **Write notes Part XII** in `SQG/polaris/polaris_sqg_notes.md`, in the style of Parts X–XI: why, design,
   predictions, results tables, scorecard, what it means. Include the smoke's lessons:
   - the sign and ratio flaws in control b, and the leak they caused;
   - the negative-control calibration;
   - production's dt sensitivity on subsampled vs alias-free frames (k 1–2 subsampled 2.3e-4 vs alias-free 9e-7 on the
     smoke). That says aliasing, not the solver, puts most of the large-scale variation into o1.2's targets.
4. **Commit and push:**
   - `git add -f SQG/analysis/results/polaris_taxonomy_7824121`
   - the notes, plus a dated `CHANGELOG.md` entry (what was measured, what is sealed, the sha256)
   - `git commit -q -F $CLAUDE_JOB_DIR/tmp/msgN.txt`, then `git push -q origin sqg-polaris-profiling`
5. **Update memory** `sqg-polaris-profiling-state.md` and its MEMORY.md line.
6. **Report to the user**, then ask (don't decide):
   - **Item 4.** Draft the acceptance-protocol memo for jesswan/Weidong: gates A–D (TEAM_REPORT item 4), the floor
     table, this run's MDE table and its control results. The sealed numbers stay sealed until they sign.
   - **Item 1.** The o1.2 / o1.1 / offline stdout logs and `*_best.pt` / `*_last.pt`: still needed from the user.
   - The 6.4 GB of checkpoints in the main checkout's root: move them into `sqg_stage` (and replace the symlinks), or
     keep them? It is the user's call.

## 4. Working rules (learned the hard way in this worktree)

- **Worktree guard.** One plain command per call. No chained git, no `bash script.sh`, no big heredocs that spawn
  subprocesses.
  - Commit with `git commit -q -F <file>`.
  - Run analysis as `python3 <file>` in `$CLAUDE_JOB_DIR/tmp`.
  - Use the literal `rmehta1987` in `qstat -u`.
- **Search.** grep / find are denied: search with python3. Never walk `/eagle` outside `members/mehta5`,
  `members/jesswan` or the shared folder.
- **Login node.** The pid cap is 256 (`/sys/fs/cgroup/users/rmehta1987/pids.current`). No GPU work. No sqg-env Python.
  Use `OPENBLAS_NUM_THREADS=1` for numpy.
- **Jobs.**
  - 1-node `debug` may be submitted without asking. Ask before multi-node, `capacity` or `preemptable`.
  - One debug job per user, held ones included.
  - Never resubmit on `queue_tags`.
  - A queued or running job reads the worktree at start, and `analyze_taxonomy.py` again at its end, so don't edit
    `SQG/analysis/opt/taxonomy/*.py` until 7824121 ends.
- **Gates.** A tolerance written before the run decides; bitwise is information. Never loosen a threshold after seeing
  the number; changes go in a dated addendum before the next run, as this one did. Write predictions before every
  job, and never edit them afterwards.
- **Science owners.** jesswan / Weidong decide anything that changes what is computed: precision in production, dt,
  STRIDE, the replay ratio, and admission of lean32ek.

## 5. Key numbers so far (smoke 7823653, synthetic state; not scored)

- **Bitwise:** prod == lean_u1; lean32ek_fast == lean32ek at 40/400/4,000 steps; dt40 == lean_u1.
- **ms/step (A100):** lean_u1 6.405 (1.45×), lean32ek 3.340 (2.78×), **lean32ek_fast 2.694 (3.45×)**, lean32sr 2.789,
  lean32 3.190. Production is 9.29.
- **Control a:** κ 1.08 / 1.00 / 0.99 at k < 8. Control b, paired: recovery 0.987–1.079, power z 5.3–30.6.
- **Alias split** (lean32ek k 1–2, alias-free / subsampled): 0.084.
- **e_Δ overall at 4,000 steps:** lean32ek 8.8e-6, lean32 5.8e-4, lean32sr 1.65e-5, f64ic32 2.3e-7.
