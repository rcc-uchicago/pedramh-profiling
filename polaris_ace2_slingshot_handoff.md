# HANDOFF — put ACE2 on Slingshot, and re-open what was measured without it

Companion to `polaris_ace2_multinode_handoff.md` (692 lines, 2026-09-02). **Read
this one first**: it does not replace that document, but it invalidates the
premise of two of its sections, and reading them in the old order will cost you
a day. Background: `polaris_nccl_debug_info.md`, `polaris_nccl_metrics.md`,
CHANGELOG 2026-09-17.

---

## 0. TL;DR

**Every ACE2 multi-node number on Polaris was measured over TCP, not Slingshot.**
The plugin ACE2 pinned (`aws-ofi-nccl 1.21.1`) cannot negotiate the CXI provider
and silently falls back to `tcp` with GPUDirect RDMA off, while still printing
`Using network AWS Libfabric` — the string the tooling checked.

The fabric half of the fix is **already applied** to
`ACE2_retrain/polaris/polaris_ace2_env.sh` (commits `698b867e`, `9c30e304`). What
is left is the guard, a re-measurement, and re-opening two conclusions that were
drawn on the wrong transport.

For the equivalent work already finished on makani, see §5's worked example:
4-node step time went **460.5 ms (tcp) → 186.4 ms (cxi)**, a 2.47x speedup, with
no model change.

## 1. The evidence

Six most recent ACE2 training logs, scanned for the provider line:

```
ace2_train.o7589852   provider=tcp   plugin=1.21.1   network=AWS Libfabric
ace2_train.o7598647   provider=tcp   plugin=1.21.1   network=AWS Libfabric
ace2_train.o7598648   provider=tcp   plugin=1.21.1   network=AWS Libfabric
ace2_train.o7602614   provider=tcp   plugin=1.21.1   network=AWS Libfabric
ace2_train.o7602650   provider=tcp   plugin=1.21.1   network=AWS Libfabric
ace2_train.o7602660   provider=tcp   plugin=1.21.1   network=AWS Libfabric
```

The discriminating line is `NET/OFI Selected Provider is cxi` versus
`NET/OFI Selected provider is tcp`. `Using network AWS Libfabric` names the
**plugin** and is byte-identical either way — which is why `parse_ace2_scaling.py`'s
`transport` column, which captures exactly that string, read `AWS Libfabric` on
every row regardless. A column with no failure mode is not a measurement
(CLAUDE.md #10).

Root cause: 1.21.1 omits `FI_MR_PROV_KEY` from its memory-registration hints and
CXI mandates it. It is plugin source, not a tunable — `FI_PROVIDER=cxi` and both
`OFI_NCCL_PROTOCOL` values fail identically (job 7629082).

## 2. What this re-opens in `polaris_ace2_multinode_handoff.md`

### 2a. §1a "THE RING/TREE DEFECT" — measured on TCP, so not established as a fabric property

That section reports that a **Tree** all-reduce between 25 MiB and 1000 MiB
*silently returns partially-reduced data and then hangs*, at 2 and 8 nodes, and
that `NCCL_ALGO=Ring` avoids it. It calls this "a fabric defect on Polaris".

Every one of those probes ran on the 1.21.1 plugin — i.e. **over Ethernet**. So
the finding is real but the attribution is not: it is a defect of *something* in
that path, and Slingshot was never in it. Three possibilities, none yet
distinguished:

1. it is a bug in aws-ofi-nccl 1.21.1's **tcp** path, and does not exist on cxi;
2. it is an NCCL Tree bug independent of transport, and still exists on cxi;
3. it exists on cxi too but at a different threshold.

**This must be re-run before `NCCL_ALGO=Ring` is carried forward.** Ring is not
free — it is the reason ACE2 cannot use Tree's latency advantage at small sizes.
If (1) holds, the pin can be dropped.

⚠ The silent-corruption property is the serious part and it is transport-agnostic
in principle. Keep `NCCL_ALGO=Ring` until the cxi re-run says otherwise. Do not
drop it speculatively.

### 2b. §1b "The fabric stack (unchanged, still mandatory)" — refuted

It states that self-built **v1.21.1 + `OFI_NCCL_PROGRESS_MODEL=AUTO`** is "the
only working combo of six" and that "everything else fails `fi_domain` with
ENOSYS". That was measured with a torch probe that could not tell cxi from tcp.
1.21.1 "worked" because it stopped using the fabric. The v1.6.0 plugin binds cxi
on the same nodes, today (jobs 7629082 M5, 7629096, 7630369).

`OFI_NCCL_PROGRESS_MODEL=AUTO` is neither proven nor disproven by this; it is
simply not the thing that made multi-node work.

## 3. What still stands — do not re-litigate

* **The LR sweep result (3e-4).** Jobs 7598647/7598648 were **1-node** runs
  (`nodes=1`), so no inter-node traffic existed and the transport is irrelevant.
  Loss is a numerical result; it does not change with the wire. The conclusion is
  untouched.
* **`NCCL_ALGO=Ring` was in force** on those runs (verified in both logs), so the
  silent-corruption path was avoided even where it could have applied.
* **§1f's economics** (more nodes at fixed global batch is slower; ask "fewest
  GPUs that hold the batch") is a *batch-size* argument, not a fabric one. It
  survives — but its magnitudes were computed against tcp step times, so the
  break-even node count will move once §5 T2 lands.

## 4. Already applied — do not redo

`ACE2_retrain/polaris/polaris_ace2_env.sh`:

* line ~145: `ACE2_OFI_PLUGIN` default flipped
  `$MEMBER_ROOT/sw/aws-ofi-nccl-1.21.1/lib` → `/soft/libraries/aws-ofi-nccl/v1.6.0-libfabric-1.22.0/lib`
* lines ~164-170: HPE's CXI rendezvous block behind `CXI_RDZV` (default on):
  `FI_CXI_RDZV_PROTO=alt_read`, `RDZV_EAGER_SIZE=0`, `RDZV_THRESHOLD=0`,
  `RDZV_GET_MIN=0`, `DEFAULT_TX_SIZE=2048`, `RX_MATCH_MODE=hybrid`.
  Measured on makani to unwedge an `all_gather` that hung at 512 KB
  (7630227 vs 7629096), at a cost of ~12-18% all_reduce bandwidth.

**No ACE2 job has yet been run on this env.** That is T2.

## 5. Tasks, in order

### T1 — the guard (no GPU needed, do this first)

`parse_ace2_scaling.py` has a `transport` column and no `provider` column, so it
still cannot fail on a tcp row. Port the guard from
`physicsnemo_ai_rossby/polaris/parse_ai_rossby_scaling.py` (commit `698b867e`):

* extract `provs = re.findall(r"Selected [Pp]rovider is (\w+)", text)` — **both
  spellings**: v1.6.0 prints capital `Provider`, v1.21.1 lowercase.
* add `"provider"` to `FIELDS` immediately after `"transport"`.
* in `check()`, gated on **nodes > 1** (a 1-node run never initialises the net
  plugin, so demanding a provider line would fail every single-node arm):
  `UNKNOWN` → `FABRIC_UNVERIFIED` rc=4; anything but `cxi` → `FABRIC_NOT_SLINGSHOT`
  rc=4, unless `ALLOW_TCP=1` is set, which downgrades it to a labelled warning.
* extend `test_parse_ace2_scaling.py` with the three cases the makani suite
  gained: tcp-is-rejected, unverified-is-rejected, single-node-needs-no-provider.
  The tcp test should assert that `transport` still reads `AWS Libfabric` — that
  is the property that made this invisible.

⚠ **`FIELDS` has drifted and it is my fault.** `parse_ace2_scaling.py`'s docstring
says its `FIELDS` is *"byte-identical"* to the ai-rossby file's, deliberately, so
the two tables can be concatenated. Commit `698b867e` added `provider` to
ai-rossby and not to ACE2, so they are now 28 vs 27 columns. T1 restores parity —
do it in the same commit as the guard, and re-read that docstring to confirm
nothing else diverged.

Run: `/soft/applications/conda/2025-09-25/mconda3/bin/python ACE2_retrain/polaris/test_parse_ace2_scaling.py`
(the system `python3` is 3.6 and cannot parse these files).

### T2 — re-measure the ladder on cxi

Re-run ACE2's 1/2/4/8-node ladder into a **fresh CSV**. The header change from T1
makes the old file refuse an append, which is correct: the existing rows are tcp
rows and must not be mixed.

```bash
qsub -q debug -l select=2:system=polaris \
     -v MAKANI_STYLE_KNOBS...,ACE2_SCALING_CSV=<fresh.csv> \
     ACE2_retrain/polaris/polaris_ace2_train.pbs
```

⚠ `polaris_ace2_train.pbs` had **uncommitted local modifications** as of
2026-09-17 and was deliberately not touched. Reconcile those before submitting,
and check whether it needs a `-v` knob for the CSV path the way makani's
`MAKANI_SCALING_CSV` does.

PASS = the run's normal token **plus** `provider=cxi` in the row. With T1 in
place the row cannot be written otherwise.

### T3 — re-test the Tree defect on cxi

The decisive experiment for §2a, and it is cheap. `read_nccl_trace.py` and the
app-free probe already exist. Re-run the §1a table on the v1.6.0 plugin:

| traffic | size | tcp result (recorded) | cxi result |
|---|---|---|---|
| `all_reduce`, Tree | 1000 / 2000 / 4700 MB | ❌ silent corruption + hang | **?** |
| `all_reduce`, Ring | 2000 / 4700 MB | ✅ | **?** |

If Tree passes on cxi, `NCCL_ALGO=Ring` can be dropped and ACE2 gets Tree's
small-message latency back. If it fails identically, the defect is NCCL-side and
the §1a table should be re-titled to say so.

Simplest instrument: `nccl-tests` `all_reduce_perf` with `-a` / `NCCL_ALGO=Tree`,
which needs no conda or torch —
`/eagle/projects/lighthouse-uchicago/members/mehta5/sw/src/nccl-tests-2.20.0/build/`.
Use it in preference to a torch probe; today both torch probes died on an
unrelated `libtorch_global_deps.so` permission error while the C binary ran fine.

### T4 — re-read §1f's economics against the new numbers

§1f concluded "the first question is the fewest GPUs that hold the batch". The
*shape* of that argument is batch-size-driven and survives, but every magnitude
in it came from tcp step times. Once T2 lands, recompute the break-even node
count. Do not quote the old node-hour projections after T2.

## 6. Prereg — write your prediction before running, score it after

Repo method (`makani_bench_report.md` §7). Suggested:

| # | prediction | why it is worth scoring |
|---|---|---|
| 1 | ACE2 2-node step time improves by >1.5x on cxi vs its tcp row | makani got 3.3x at 2 nodes; ACE2's collective is larger (~165-212 MB), so it should gain at least as much |
| 2 | Tree all-reduce still fails at 2000 MB on cxi | tests whether §1a is a fabric or an NCCL property — a MISS here is the more useful outcome |
| 3 | `wireup_s` increases vs the tcp rows | it did on makani (8.54 → 14.18 at 2 nodes); mechanism unknown, so this is a genuine guess |
| 4 | the LR-3e-4 conclusion is unchanged | it should be — 1-node, no fabric. Scoring it guards against quietly re-opening a settled result |

## 7. Definition of done

1. `test_parse_ace2_scaling.py` passes, including the three new guard cases, and
   `FIELDS` is identical to `parse_ai_rossby_scaling.py` again.
2. An ACE2 multi-node row exists with `provider=cxi`.
3. The ladder is re-measured on cxi in a fresh CSV, with the tcp ladder kept and
   labelled rather than deleted.
4. §1a and §1b of `polaris_ace2_multinode_handoff.md` are corrected in place with
   the T3 result — including if T3 shows the defect survives.
5. CHANGELOG entry: what moved, what was refuted, what is still n=1.

## 8. The one thing not to repeat

The reason this went unnoticed for three weeks is not that anyone was careless —
it is that the check that existed (`transport`) **could not fail**. It recorded a
string the fallback does not change, and a pre-registered prediction scored a HIT
against it. When adding any future fabric or hardware check, ask first: *what
value of this field would make the job fail?* If there is no such value, it is a
label, not a guard.
