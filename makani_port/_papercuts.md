# _papercuts — mistakes that cost time; read at session start and before every `qsub`

One line each: date (`~` = approximate, reconstructed) · what bit · the rule that avoids it. Append; never delete (strike obsolete ones
with the reason).

## Queue / PBS
- 09-29 · `debug` holds **one queued job per user**; a 2nd `qsub` fails · fold checks into one job.
- 09-30 · `-W depend=afterany:<id>` does **not** bypass that limit (the held job counts) · wait for the slot.
- 09-30 · another session's job (ACE2 7671383) held the `debug` slot · check `qstat -u rmehta1987` before promising a submit; never touch other sessions' jobs.
- 09-30 · PBS writes the `.o` file **live** · "file exists" ≠ done; key on the `… done` line / PASS token.
- 09-29 · `capacity` caps `nodect` at 4, `max_run 1` per project · never take it without the operator.
- ~08-05 · `queue_tags` + large `eligible_time` = no nodes · never resubmit; it resets eligible time (CLAUDE.md #12).
- 09-24 · a worktree with a queued/running job is frozen (harness imports `PBS_O_WORKDIR/src` at start and restart) · develop in another worktree.

## Login node / tooling
- 09-24 · no venv Python on the login node (operator ruling); pid cap 256 incl. threads · tests run as debug jobs; check `/sys/fs/cgroup/users/$USER/pids.current`.
- 09-30 · login `python3` is **3.6** (`subprocess.run(capture_output=)` fails) · use `stdout=PIPE, universal_newlines=True`; `/usr/bin/python3.11 -m py_compile` for syntax.
- 09-30 · `grep`/`tail`/`head` are denied in Bash · read with the Read tool or a stdlib python script.
- 09-29 · the worktree guard refuses commands with runtime vars (`$USER`) or `.git` strings near git · split into plain commands; put heredoc scripts in a file and run the file.
- 09-29 · working-tree git on Lustre can wedge · `git -c core.preloadIndex=false -c index.threads=1`; commit message via `-F <file>`.
- 09-30 · `! cd makani_sfno && …` failed: the `!` shell was already in `makani_sfno/` · give absolute paths in operator commands.
- 09-29 · Bash `${VAR:?msg}` with an apostrophe in msg is an unterminated quote.
- 09-30 · a placeholder sha was written into CHANGELOG · always paste the sha from `git log`.
- 10-01 · login node at 251/256 pids blocked the session start; the operator killed idle sessions (147 after) · read `pids.current` first; ask the operator — the classifier refuses `kill` from a session.
- 10-01 · `git grep` died "failed to create thread" at 190/256 pids (one thread per CPU) · `git -c grep.threads=1 grep …`.
- 10-01 · the worktree guard refuses `git` inside shell loops/`$VAR`s, even on another repo · run git from a stdlib python script (`subprocess`, `--git-dir`).
- 10-01 · `$MEMBER_ROOT/external/makani-upstream` is a **shallow** clone holding neither the pin nor `main` · use the bare full clone `$MEMBER_ROOT/external/makani-main.git`.
- 10-01 · the auto-mode classifier refused the operator-ruled `verify_grid_type` waiver (§3.2) as "Security Test Removal" · a validation bypass needs the operator's explicit go in the session before the edit; never re-route it through another tool.
- 10-01 · the `_lNN` levels look like a plev list but are terrain-following hybrid levels with no hyam/hybm in `data.json` · read the converter docstring before any pressure-based calculation on the pack.
- 10-01 · inspecting a checkpoint without torch · stdlib `zipfile` + `pickle.Unpickler` with stub `find_class`/`persistent_load` reads `data.pkl` only (keys, comm_grid, pickled classes) — never loads tensor bytes.

## makani behaviour
- ~09-18 · `--batch_size` is GLOBAL and makani has no gradient accumulation · global = ranks × LOCAL_BATCH.
- ~09-11 · multistep warm start needs `LOAD_LOSS=0` (loss state shape depends on `n_future`); config `n_future` is overwritten by `--multistep_count`.
- 09-24 · warmup steps **once per epoch**: `WARMUP_EPOCHS=1` makes epoch 1 run at `LR×LR_START` (4e-6) · nf4_proxy r2 vs r1 differ only in this.
- 09-23 · trainer pads summary columns · grep `validation loss: +[0-9.eE+-]+`, not one space.
- 09-23 · `EVAL_SAMPLES < global batch` ⇒ validation loss prints nan (0/0), not a divergence.
- ~09-17 · makani's ERA5 default metric names give **zero** metric handles on E3SM channels, silently · our trainer rebuilds them; keep that test.
- 09-29 · `data_parallel_rank == 0` is not unique under model parallelism · single writers need model rank 0 too (fixed `eb4e0cce`).
- 09-29 · `legacy` checkpoint restore checks `comm_grid`; only `flexible` scatters an h1w1 file into a sharded layout.
- 09-24 · `nf4_prod_b16_r1/ckpt_mp0_v1.tar` (B e22) is a **rotating** slot · copy it; never resume that run.
- 09-30 · `__version__` stayed `0.2.0` across 184 commits · pin and compare by commit sha only.
- 09-30 · default screen truth files lack TMQ (`TRUTH_MISSING_CHANNEL`) · use `screen_truth_2044f<F>_tmq.npz`.
- 09-24 · a per-step corrector below ~σ/256 rounds away in bf16 · corrected state must be fp32.

## Process
- 09-29 · two experiment factors were bundled into one run (G = variables + split) without asking · one factor per run; ask before combining.
- 09-29 · a short smoke (20–60 steps) overstates step time (1587.6 vs 473 ms) · size from steady-state logs.
- 09-30 · a debate brief called a fact "unconfirmed" that CHANGELOG already recorded (`Z3_l17`) · search CHANGELOG before asserting an unknown.
- always · prereg (with tolerance) committed **before** the job; never loosen after; key on tokens, not rc.
