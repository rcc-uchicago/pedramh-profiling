# HANDOFF (monitor) — audit the makani `main` port

Written 2026-10-01. You watch the **worker** session executing `makani_port/HANDOFF_worker.md`
(branch `feat/makani-port-main`, worktree `.claude/worktrees/makani-port`). You **message, never
commit** to its branch (memory: monitor role). Keep your own notes outside git at
`$MEMBER_ROOT/runs/makani_port/monitor_notes.md` (dated, Progress/Surprises/Decisions/Next, like
PROGRESS.md). Read `makani_port/_papercuts.md` and the top of `makani_port/PROGRESS.md` at the start
of every check.

## 1. What you are protecting

1. **The old venv and everything running on it.** `sfno-venv`, `physicsnemo_sfno/` (editable,
   shared) and the worktrees with queued/running jobs (`makani-ace2-ports` for F 7660250;
   `f-finetune` while its jobs run). Any write there is an immediate STOP message.
2. **Checkpoint compatibility and outputs.** The port is only done if every listed checkpoint loads
   strict and reproduces its M0 golden outputs **bitwise** (or a tolerance pre-registered *and*
   operator-approved). "Close" is not green.
3. **The milestone discipline.** Small commits; one debug job per milestone gate; a
   `makani-port/mN-green` tag only on the commit the green job ran; `git revert` (not reset/force) to
   the last green tag on red.

## 2. The check you run (each time the worker reports, or on your own cadence)

Per CLAUDE.md shell rules: one Bash call per logical step, no polling loops, `qstat -x <id>` at most
once per check, stdlib `python3` (3.6) for reading files; grep/tail are denied — use python.

| check | how | red flag |
|---|---|---|
| gate is real | read the job's `.o` file: the milestone's PASS token, printed sha = tagged commit | tag without token; token from a job whose sha ≠ the tag; `rc=0` quoted as a result (#14); truncated log |
| prereg before job | `git log --format='%h %ci' -- <prereg>` vs the job's `stime` (`qstat -xf`) | prereg committed after `stime`, or edited after the result |
| golden is sound | M0 ran twice, bitwise equal; manifest sha256s committed; golden produced on the **old** venv | golden regenerated on the new venv; one run only |
| old venv untouched | sha256 of `sfno-venv/.../makani-0.2.0.dist-info/direct_url.json` + `RECORD` vs the value in PROGRESS M1 | any change |
| both venvs green | M2+ jobs report pass counts for **old and new** | new-only runs; pass count dropped on old |
| tolerance discipline | M4/M5 compare bitwise; any non-bitwise is reported as a STOP with the upstream bisect | `allclose`, raised atol/rtol, `xfail`/skip, "within noise" |
| checkpoint safety | new saves go under `$MEMBER_ROOT/runs/makani_port/`; no `RUN_NUM` reuse of an existing run | write into an existing run dir; resume of `nf4_prod_b16_r1` |
| scope | diff of each commit touches port code only | loss/channel/normalisation/output changes; edits to protected launchers (`submit_f_nosoil.sh`, `submit_finetune_stability_arm.sh`, `submit_nfuture_ladder.sh`, `polaris_setup_sfno_venv.sh`) |
| bookkeeping | dated PROGRESS entry per working block; new papercut for each repeated or costly mistake | a mistake already in `_papercuts.md` repeated (say which line) |
| queues | `debug` only | `debug-scaling`/`preemptable`/`capacity` without the operator's word |

## 3. Adversarial questions to ask at each milestone

- **M0:** Is the training trace actually deterministic (seed, cudnn flags, dataloader order)? Are all
  six checkpoint kinds covered (incl. EMA, 99-channel, multistep-trained)?
- **M1:** Does every package resolve from the new venv (not `~/.local`, not base conda)? Is
  torch-harmonics unchanged unless forced? Is physicsnemo non-editable?
- **M2:** Is anything version-branched on a string instead of feature-detected? Did a patch silently
  stop applying (e.g. the `compat.py` shim, the logger fix, the metric-handle rebuild — makani's
  ERA5 default names give **zero** metric handles on E3SM, silently)?
- **M3:** Does the pname checksum get bypassed globally to load old files? Are EMA files covered?
- **M4/M5:** Bitwise per lead and per channel, not a summary statistic? If not bitwise, did the
  worker bisect to a commit (`f9b6e787` fork-join matmul is the first suspect) instead of tuning?
- **M7:** Is the h2w4 comparison loss-trajectory against h1w1 at equal knobs (7669001 shape)?

## 4. Messaging

Message the worker (SendMessage) for: a red flag above (quote the line and the rule), a repeated
papercut, a gate you could not verify. Escalate to the operator for anything in the worker's §5
STOP list, or if the worker proceeds past a STOP. One message per issue, actionable, no praise.

## 5. Definition of done (yours)

Every milestone tag verified against its job output; your notes list each verification with job id
and sha; any unresolved doubt is in the operator summary for M8.
