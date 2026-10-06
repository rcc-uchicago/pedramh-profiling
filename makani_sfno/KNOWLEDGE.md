# makani_sfno — what we know, verified (E3SM on Polaris)

One file for what the makani track has established, what broke, and what was believed and then
retracted. Each entry is tagged and cites its **primary** artifact: a job log, a CSV row, or code.
A doc citation alone is not enough. Started 2026-10-06 from
`polaris_makani_docs_consolidation_handoff.md`. The doc inventory, unverified, is
[`docs/2026-10-06_knowledge_inventory.md`](docs/2026-10-06_knowledge_inventory.md).

## Status (update this block before you stop)

| § | section | tier | state |
|---|---|---|---|
| 1 | Checkpoint lineages | 1 | untouched |
| 2 | Hyperparameters | 1 | untouched |
| 3 | Depth / `n_future` | 2 | untouched |
| 4 | Mass conservation / dry-air | 2 (numbers Tier 1) | untouched |
| 5 | Multi-node scaling | 2 | untouched |
| 6 | Climate screening methodology | 2 | untouched |
| 7 | ACE2 comparison and ports | 2 | untouched |
| 8 | Known silent-failure traps | 1 | **started** (handoff §6 + §0a/§1/§3 verified; post-09-04 traps being collected) |
| 9 | Retired / contradicted claims | 1 | untouched |
| 10 | Dangling citations and contradictions found | — | **started** (running list) |

**Resuming after a context clear:** read this block, then §10, then continue the first section
that is not `done`. Verification rules: handoff §2. Tier 1 = checked against the primary
artifact. Tier 2 = a secondary citation is acceptable, tagged `❓ secondary-only`.

## How to read the tags

- `✅ CONFIRMED` — the primary artifact was opened on the date given and says this.
- `❌ BROKEN / LIVE TRAP` — the failure is real and still present in the code or system today.
- `🔧 FIXED` — the trap was real. The fix was checked in today's code, and a later job shows it working.
- `🔁 RETIRED` — once believed, now contradicted by later evidence. Do not resurrect it.
- `❓ secondary-only` — carried from a doc that cites it. The primary artifact was not opened.
- `❓ UNVERIFIED` — no primary source found. Listed so it stays visible.

Paths: `$MEMBER_ROOT` = `/eagle/projects/lighthouse-uchicago/members/mehta5`. "Installed makani"
means `$MEMBER_ROOT/conda-envs/sfno-venv/lib/python3.12/site-packages/makani` (0.2.0). "The fork"
means `makani_sfno/src/` at `44a3ea9c`. Line numbers drift (`.claude/comments.md`), so code is
cited by file and symbol. A line number, where given, is as of 2026-10-06.

---

## 1. Checkpoint lineages

*Untouched.*

## 2. Hyperparameters

*Untouched.*

## 3. Depth / `n_future`

*Untouched.*

## 4. Mass conservation / dry-air

*Untouched.*

## 5. Multi-node scaling

*Untouched.*

## 6. Climate screening methodology

*Untouched.*

## 7. ACE2 comparison and ports

*Untouched.*

---

## 8. Known silent-failure traps

Each of these cost at least one job and none announces itself. Source list:
`polaris_makani_analysis_ensemble_handoff.md` §6 (10 numbered) plus four described separately in
its §1 and §3. That is **14, counted by opening the file**. The handoff's own §0a calls
`SKIP_TRAIN` "the tenth trap" while its §6 numbers it 9 (see §10). Each was re-checked
2026-10-06.

### 8a. The 2026-09-04 list

| # | trap | state | primary evidence (checked 2026-10-06) |
|---|---|---|---|
| 1 | **`ckpt_mp0_v0.tar` or nothing.** `resuming` is true only if that exact file exists. Seed a scratch dir under any other name and the job trains from scratch, silently. Assert `resuming True` in the log. | ❌ LIVE (by design) | installed makani `train.py`: `resuming` is computed from `checkpoint_path.format(..., checkpoint_version=0)` existing (L101–105) |
| 2 | **A seeded/forked expdir needs `WANDB=0`.** With wandb on *and* resuming, makani reads `<wandb_dir>/wandb/makani_restart.yaml`. Only a fresh run writes that file, so every rank dies at construction. Don't copy the file in: the job would write into the parent run's wandb history. | ❌ LIVE | installed makani `utils/driver.py` `Driver._init_wandb`: `if not params.resuming:` writes `makani_restart.yaml`; else reads it (L248/270/275; the handoff's L237–248 has drifted). The rank-death log was not re-opened: symptom `❓ secondary-only` |
| 3 | **`LOAD_COUNTERS=0` for any validation-only run.** A restored epoch counter (243) against `EPOCHS=1` makes the epoch loop empty. Validation lives inside it, so the job exits 0 with no loss. | ✅ CONFIRMED, LIVE | `makani_mn_scaling.o7592332`: `max_epochs 1`, `resuming True`, and **no** `validation loss` line anywhere in the log |
| 4 | **`LOAD_LOSS=0` when `n_future` changes.** `LossHandler`'s running stats are shaped by `n_future`. | ✅ CONFIRMED, LIVE | `makani_mn_scaling.o7590350`: `size mismatch for running_mean: copying a param with shape torch.Size([101]) from checkpoint, the shape in current model is torch.Size([202])` |
| 5 | **A config-side `n_future` does nothing.** It is overwritten from `--multistep_count`. `MULTISTEP` is the only handle. | ✅ CONFIRMED, LIVE | both installed makani `train.py` (L119) and the fork's `train_plasim.py` (L343) set `params["n_future"] = args.multistep_count - 1` |
| 6 | **`pretrained` and `resuming` are mutually exclusive.** A fine-tune needs a **new** `RUN_NUM`, or resuming silently wins. | ✅ CONFIRMED, LIVE | installed makani `deterministic_trainer.py`: `if self.params.pretrained and not self.params.resuming:` (L237) |
| 7 | **`qalter` is refused on Polaris** for every attribute (rc=32). Walltime and dependencies are fixed at submit; stagger with `qhold`/`qrls`. | ❓ secondary-only | `polaris_pbs_notes.md` §1b (quotes the error text). No job log or transcript records it |
| 8 | ~~**`capacity` cannot hold a queued successor**~~ | 🔁 **RETIRED** (see §9) | `qstat -Qf capacity` on 2026-10-06: `max_queued = [p:PBS_GENERIC=2]`, `max_run = [p:PBS_GENERIC=1]`. In practice 7718436 (capacity) was queued behind 7715005: the `.o7715005` header shows it started 2026-10-05T17:40; 7718436's header shows it started 2026-10-06T01:44, after 7715005 ended. The cap is per *project*, so the second slot is free only if no other member holds it |
| 9 | **`SKIP_TRAIN=1` validated nothing before 2026-09-04.** The fork's entrypoint short-circuited before `trainer.train()`. makani handles `skip_training` *inside* `train()`. | 🔧 FIXED | fork `train_plasim.py` now calls `trainer.train()` under `skip_training`, with a comment citing 7592332/3/6. `makani_mn_scaling.o7598662` (va=3): `validation loss: 0.012838906608521938`, reproducing production's 0.01284 |
| 10 | **Never re-run a stuck job before diagnosing.** Read the queued job's `comment`. | rule, not a code trap | CLAUDE.md #12; `polaris_pbs_notes.md` §1b |
| 11 | **`validation loss` is single-step at every `valid_autoreg_steps`.** No validation loss in this project is a multi-step score, production's 0.01284 included. | ✅ CONFIRMED, LIVE | `.o7598662`/`.o7598663`/`.o7598664` (va = 3 / 10 / 20) all print `validation loss: 0.012838906608521938` |
| 12 | **Per-lead metrics were computed and discarded**: they went only to wandb, which must be off for a seeded expdir (#2). | 🔧 FIXED | fork `plasim_trainer.py` `log_epoch` prints `Per-lead validation metrics:`. First seen in `makani_mn_scaling.o7603089`, also 7603119/7603323/7603324/7603345 |
| 13 | **`_extract_truth_sic` read forcing channel 5 by position.** E3SM has 7 forcings, so the length guard passed and returned a different variable labelled `truth_sic`. | 🔧 FIXED | fork `sfno_inference/rollout_driver.py`: sea ice is looked up by name (`_SIC_ALIASES = ("sic", "ice", ...)`). truth_sic is disabled with a warning when names or stats are missing |
| 14 | **`save_raw_forecasts: True` is a dead key**: nothing reads it. | ❌ still no reader | no `save_raw_forecasts` reader anywhere in installed makani 0.2.0 or the fork's `src/` (searched 2026-10-06) |

### 8b. Traps found after 2026-09-04

*Being collected from the Decisions log (§9 pass). Verified entries so far:*

| trap | state | primary evidence |
|---|---|---|
| **A queued makani job freezes the worktree it was submitted from.** PBS copies the *script* at submit, but Python is imported from `$PBS_O_WORKDIR/src` when the job **starts**, and again on every preemption restart. Edit `src/` while a job is queued and the job runs the edit. | ✅ CONFIRMED, LIVE | `polaris_makani_multinode_scaling.pbs`: `MAKANI_ROOT` from `PBS_O_WORKDIR`, then `export PYTHONPATH="${MAKANI_ROOT}/src:..."`. Same in `polaris_climate_run.pbs` |
| **One `debug` job per user at a time, and a second submission can be refused at `qsub`.** Soil-fix screen 7719183 was first rejected this way. | ✅ `max_run` / ❓ `max_queued` | `qstat -Qf debug` 2026-10-06: `max_run = [u:PBS_GENERIC=1]`. It shows **no** queue-level `max_queued` (so the refusal comes from elsewhere, e.g. a server limit or hook). `polaris_pbs_notes.md` states `max_queued 1 per USER`; the rejection is CHANGELOG 2026-10-06 (`❓ secondary-only`) |

---

## 9. Retired / contradicted claims

*Untouched (Tier 1). Sources: handoff §5 table, CHANGELOG `## Known issues` (which, checked
2026-10-06, holds **no** makani-specific entries beyond the 1.18 B-parameter note and the
`torch_harmonics` version box), and the 85 makani-matched Decisions-log entries.*

| retired claim | what is true | evidence |
|---|---|---|
| "`capacity` cannot hold a queued successor while a job runs" (`polaris_pbs_notes.md` §1b #2, analysis handoff §6 #8) | `capacity` allows **2 queued+running per project**, with `max_run 1`. A successor can be pre-staged if no other project member holds the second slot. | `qstat -Qf capacity` 2026-10-06; 7718436 queued behind 7715005 (§8a #8) |

---

## 10. Dangling citations and contradictions found

A required deliverable (handoff §3.2). Each item says what was checked. **Nothing listed here
has been fixed.** Fixing is out of scope for a documentation task (handoff §4).

1. **`polaris_makani_b_dryair_findings_handoff.md` §3** cites
   `ACE2_retrain/ace_exp/fme/fme/core/corrector/atmosphere.py`. That path does not exist. The file
   is at `ACE2_retrain/ace_exp/fme/core/corrector/atmosphere.py` (one `fme/` fewer), in the main
   checkout only. `ACE2_retrain/ace_exp` is not in the `b-continuation-dryair` worktree's tree.
2. **`polaris_makani_analysis_ensemble_handoff.md` numbers its own traps inconsistently:** §0a
   calls `SKIP_TRAIN` "the tenth silent-failure trap", while §6 lists it as #9 of 10.
3. **`polaris_pbs_notes.md` §1b #2 vs the live queue:** "capacity cannot hold a queued
   successor… the earlier 'max_queued 2 per PROJECT' reading was wrong". On 2026-10-06 PBS reports
   `max_queued = [p:PBS_GENERIC=2]` for `capacity`, and 7718436 was queued behind 7715005. Either
   the limit changed or the original refusal was another member's job holding the second slot.
   `❓` which one.
4. **Line drift in trap citations** (claims still hold): `driver.py:237-248` → the
   `makani_restart.yaml` read is at L275; `train_plasim.py:382` → the fixed call is at L433.
5. **CHANGELOG's `## Known issues / failed approaches` section has no makani entries.** The
   section the handoff calls "the most direct what-doesn't-work source" covers PanguWeather, data
   prep, S2S and SI. makani's failed approaches live only in Decisions-log entries and handoffs.
