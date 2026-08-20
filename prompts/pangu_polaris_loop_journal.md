# Loop journal — PanguWeather profiling on Polaris

The loop's own state, and the thing that survives a lost or compacted context. **Newest tick at the top.**
Driver: [`_live_session_pangu_polaris_loop.md`](_live_session_pangu_polaris_loop.md) · frozen plan:
[`../PANGU_POLARIS_PROFILING_PLAN.md`](../PANGU_POLARIS_PROFILING_PLAN.md) · setup:
[`_live_session_loop_README.md`](_live_session_loop_README.md).

Entry shape (keep it):

```markdown
## tick <N> — <YYYY-MM-DD HH:MM> — stage <id> — <one line: what this tick did>
- **in flight:** job <id> (`pploop-<stage>`, <queue>, submitted <ts>) | none
- **prereg:** <prediction + decision rule, written BEFORE the job ran> (commit <sha>)
- **result:** <value> — <measured | estimated | OPEN> — vs prediction: <hit | miss + why>
- **next:** <the single next action>
- **infra-failure count:** <n>/5
```

---

## tick 4 — 2026-08-20 — stage T0 item 4 — prereg for the kernel_census fix

- **in flight:** none (Tier 0 needs no `qsub`); **this is the last free item**
- **the broken behaviour, captured BEFORE the fix** (`kernel_census.py` as committed, on
  `nsys_pangu_sfno_7255503.sqlite`) — this is the bug the fix must reproduce-then-remove:

  | range | launches | % count | % time | avg µs |
  |---|---|---|---|---|
  | `(outside)` | 319,466 | **69.6%** | **71.4%** | 301.2 |
  | `forward_loss` | 124,595 | 27.1% | 23.8% | 258.0 |
  | `optimizer` | 14,753 | 3.2% | 4.7% | 433.3 |
  | **`backward`** | **203** | **0.0%** | **0.0%** | 10.2 |
  | `data_prep` | 71 | 0.0% | 0.0% | 426.0 |

  Header: "**459,088** kernel launches over ~**156** steps (2,943 per step), **134.8 s** GPU time".
  Three independent errors visible at once: the total is the **+29.4% phantom** join (true 354,720)
  and the time is **+31%** (true 102.911 s); **`backward` — 72.6% of GPU time — reads as 0.0%**
  because its launches come from `pt_autograd_*` and the tool looks the range up on the launching
  thread; and `data_prep`, which launches **zero** kernels, shows 71.
- **prereg (written BEFORE the fix was implemented):**
  - **P1.** Fixed, the header reads **354,720 launches** and **102.911 s** — identical to
    `nvtx_phase_attribution.py`, because it will be the *same* join, not a re-derivation.
  - **P2.** `(outside)` → **0.0%**. `backward` → **250,880** (1568/rank-step), `forward_loss` →
    **94,400** (590), `optimizer` → **9,440** (59), `data_prep` → **0** and absent.
  - **P3.** The "per step" denominator is also wrong: it counts distinct `step_%` **start**
    timestamps and gets **156** where the correct normaliser is **160 rank-steps** (40 × 4). Predict
    the fix has to change that too, and that nobody noticed because it is only a 2.5% error.
  - **P4 — the sharp one.** The census's own thesis is that a range with a **high launch share and a
    low time share** is a batching target ("many tiny kernels… fusing those buys launch-pipeline
    headroom"). **Predict it finds NO target on this capture:** every range's %count will sit within
    ~3 points of its %time, because §0d already established ~260 µs average kernels. That would make
    the tool's own headline advice inapplicable here — consistent with the **already-refuted** "~9%
    idle is launch latency" story its docstring still teaches, and the honest output is to say so
    rather than print the advice unconditionally.
  - **Decision rule.** P1+P2 are the gate: if the fixed census does not agree with
    `nvtx_phase_attribution.py` to the row, the two tools disagree and one is wrong — that is a
    blocker, not a pass. If P4 holds, the docstring's launch-pipeline framing is retired in the same
    commit and the heuristic is made conditional. If P4 *fails* — some range really is launch-heavy
    and time-light — that is a new finding and the batching advice earns its place.
  - **Stated limit:** this fixes attribution, not the underlying question. Per-range launch *counts*
    are what the census adds over §4.3; it does not localise a call site (item 8) or say anything
    about coalescing (item 7).
- **result:** **4/4 predictions HIT.** Prereg `a336c6fc` verified as an ancestor of HEAD and
  byte-identical before this line was written.
  - **P1** header → **354,720 launches / 102.911 s**, identical to `nvtx_phase_attribution.py`. HIT.
  - **P2** `(outside)` → **absent (0.0%)**; `backward` **250,880** (1568/rank-step), `forward_loss`
    **94,400** (590), `optimizer` **9,440** (59), `data_prep` absent. **Agrees with §4.3b row for
    row**, because it is now the same join and not a second implementation. HIT.
  - **P3** the per-step denominator was **also** wrong (156 distinct `step_%` starts vs **160**
    rank-steps) and had to be fixed. HIT — a third bug, predicted before it was looked for.
  - **P4** the census's own thesis finds **no batching target**: largest count-minus-time skew
    **+3.2 pt** (7255503) and **+2.0 pt** (7255557) against a +10 pt bar. HIT. ⇒ the docstring's
    "~9% of training time is idle gaps between launches" framing is **retired as refuted**, and the
    advice is now printed **conditionally on the data** rather than unconditionally.
  - **Decision rule fired:** P1+P2 were the gate and the two tools agree, so no blocker. P4 held, so
    the launch-pipeline framing was retired in the same commit.
- **gauntlet:** ONE focused adversary, not the pair — **deliberately proportionate.** Item 4's
  numbers are *reproductions* of §4.3b values already adversarially verified twice; the only
  genuinely new claim is the **negative** result (no batching target), which retires advice in three
  docs. The drift surface is 4 known files, so I swept it myself rather than spending an auditor.
- **result of the sweep:** plan item 4 ticked; `POLARIS_PROFILING_HANDOFF.md` §5 flipped from "BROKEN,
  TWO independent bugs" to FIXED, keeping the old numbers on the record *because three docs had cited
  them as an NVTX limitation*; `PROFILING_TABLES.md`'s superseded ACE2 table now says either tool can
  re-derive it — **and that nobody has yet run it on an ACE2 capture**, so it is superseded but not
  replaced.

---

## tick 4b — 2026-08-20 — **TIER 0 COMPLETE** — item 6 prepared, AWAITING SUBMISSION APPROVAL

- **in flight:** none. **Nothing has been submitted in this loop.**
- **Tier 0 is exhausted:** plan items **1, 2, 3, 4, 5 all done**, all from two captures already on
  disk, **zero GPU time and zero queue time spent**. Every remaining item needs compute.
- **prepared, not submitted — `polaris_topology_check.pbs`** (plan item **6**, plus the NUMA rows
  item **6b** needs):
  - **Static checks run:** `bash -n` clean; the PBS header matches the loop's required set (with
    walltime **00:10:00** rather than the template's 00:55:00 — the work is ~60 s and an honest
    request is better for queue position); the PASS gate was **dry-run against three synthetic
    matrices** (good / zero-cell / torch-unavailable) and catches all three.
  - **Design notes:** no `torchrun`/`mpiexec`/`srun` — `gpu_topology_check.py` is a single process
    that walks the device pairs itself. **`PYTHONPATH` is explicitly `unset`** (this job imports
    neither tree, and a stray entry could resolve `utils` to the wrong one). **No
    `$POLARIS_TOPUPS` and no `polaris_require_topups`** — it imports only `torch` from base conda,
    and `polaris_env.sh:139` reserves that gate for the 8 base-conda model jobs.
  - **Node-hour arithmetic:** 1 node × 10 min = **0.167 node-h requested**; realistic use ~3 min
    including `module load`/`conda activate` ≈ **0.05 node-h**, against **17,128 node-h available**
    — ~**0.001%** of the allocation. Queue: `debug`, currently 9 running / 3 queued; historically
    9/9 started with a median 19 s wait. `debug` is `max_run 1`/`max_queued 1` **per user**, so this
    would be the only job in flight.
- **prereg for item 6 (written BEFORE any submission):**
  - **P1.** 4 GPUs, all `NVIDIA A100-SXM4-40GB`.
  - **P2.** The measured matrix is **uniformly fast** — no 2×2 block structure, no pair at
    PCIe class (~20–30 GB/s), and all 12 off-diagonal cells within **±20%** of each other.
  - **P3.** Per-pair unidirectional bandwidth **80–200 GB/s**. Reasoning: A100 SXM4 has 12 NVLink3
    lanes at 25 GB/s; in a 4-GPU direct-connect mesh a pair gets ~4 lanes ⇒ ~100 GB/s per direction,
    and `copy_` measures one direction.
  - **P4.** This **confirms** the handoff §4 inference rather than overturning it: every pair
    measures **≥ 70 GB/s**, so a PCIe-class cross-pair hop is excluded. (Note §4.4's correction that
    the right NCCL anchor is the *minimum*, 59.67 ms ⇒ ≥79 GB/s, not the stall-carrying mean.)
  - **P5.** `numactl --hardware` reports **4 or 8** NUMA nodes (NPS4 is the common Polaris setting on
    a 32-core EPYC Milan, `nproc=64`), and the GPU→NUMA map is **not** the identity — the ai-rossby
    multinode script records that the ALCF helper assigns GPUs in *reverse* local-rank order. A
    non-identity map is what makes item 6b's affinity question real rather than hypothetical.
  - **Decision rule.** If P2+P3 hold, handoff §4's OPEN topology cell **closes with a measurement**
    and §0b's "comms are free inside a node" acquires a mechanism (real NVLink), leaving item 12's
    multi-node framing intact. **If any pair comes in at PCIe class, that is a major finding** — DDP's
    ring would be limited by the slow hop and §0b would partly re-open. P5 is a recorded cluster fact
    either way and feeds item 6b.
  - **Stated limit:** this measures *pairwise device-to-device copy*, which is the path a ring
    all-reduce uses, but it is **not** an all-reduce benchmark and says nothing about multi-node
    (item 12) or about the 1279 GB/s **intra**-device HBM figure of §4.3e, which is a different path.
- **next:** **STOP. Await explicit operator approval for the single `qsub` shown in the report.**
  Per the driver §0 there is no standing approval and approving the plan is not approving the
  submission.
- **infra-failure count:** 0/5
- **infra-failure count:** 0/5

---

## tick 3 — 2026-08-20 — stage T0 item 3 — prereg for the analytic bytes model

- **in flight:** none (Tier 0 needs no `qsub`)
- **built first, from the config + source only — not from the capture** (`ACE2_retrain/sfno_bytes_model.py`).
  Config `pangu_e3sm_sfno.nsys.rendered.yaml`: `horizontal_resolution: [180, 360]`, `embed_dim: 512`,
  `num_layers: 12`, `mlp_ratio: 2.0`, `hard_thresholding_fraction: 1.0`, `big_skip: True`, batch 1.
  Spectral shape grounded in source, not assumed: `modes_lat = int(h*thf) = 180`,
  `modes_lon = int((w//2+1)*thf) = 181` (`networks/modulus_sfno/sfnonet.py:481-482`).
  The tensor inventory (payload, one tensor, batch 1):

  | tensor | shape | elements | fp32 MB | complex64 MB |
  |---|---|---|---|---|
  | input/output field | 108x180x360 | 6,998,400 | 27.99 | — |
  | **latent** | 512x180x360 | 33,177,600 | **132.71** | — |
  | big_skip cat | 620x180x360 | 40,176,000 | 160.70 | — |
  | **MLP hidden** | 1024x180x360 | 66,355,200 | **265.42** | — |
  | **spectral** | 512x180x181 | 16,680,960 | — | **133.45** |

- **prereg (written BEFORE `--match` was run against either capture):**
  - **P1.** `direct_copy<complex64,nocast>` per-call payload ≈ **133.45 MB**, the spectral tensor
    (±5%). It is the only complex tensor at that scale.
  - **P2.** `conj<complex64>` per-call payload ≈ **133.45 MB** as well, and its **24 calls/rank-step
    = 2 x num_layers** — one conjugate per operand per spectral contraction.
  - **P3.** `direct_copy<float,nocast>` per-call payload ≈ **132.71 MB**, the fp32 latent (±5%).
  - **P4.** **No copy kernel's per-call payload exceeds 265.42 MB** (the largest real tensor, the MLP
    hidden at fp32). A copy above that is not moving one tensor.
  - **⚠ P1 and P3 are the ones I expect to be at risk.** Backing bytes out of §0d's published
    `est GB/s x us/call` gives ~**238 MB** for the complex64 copy (1.78x the spectral tensor) and
    ~**55 MB** for the float copy (0.41x the latent). If the geometry confirms those, **P1, P3 and
    possibly P4 all MISS** — and per the plan that mismatch *is* the deliverable: it localises a copy
    that does not correspond to any single tensor, which means either a fused/concatenated copy or a
    broken geometry estimate. I am predicting the clean-tensor case anyway, because that is what the
    model says should be there; recording the alternative so the miss cannot be re-narrated as a hit.
  - **Decision rule.** (a) If P1/P2/P3 hold, §0d's launch-geometry bytes are validated against an
    independent source and "17-27% of peak" becomes a **bound** rather than an estimate — and each copy
    has a named tensor, which is item 8's target handed over for free. (b) If they miss *high* (a copy
    bigger than its tensor), the excess is the finding: name the multiple and say what could produce it
    (a `cat`, a batched/strided copy over several tensors, or the `<128,2>` elements-per-block
    assumption being wrong for this kernel). (c) If they miss *low*, the copies are partial/tiled and
    the per-call figure is not a tensor at all. **Either way the number is recorded; §0d's caveat is
    only removed in case (a).**
  - **Stated limit:** this bounds *useful* bytes. It cannot distinguish "unused bandwidth" from
    "wasted uncoalesced traffic" — that is still item 7 (ncu), unchanged.
- **result:** **0/4 size predictions hit — and the prereg pre-registered that outcome as the
  deliverable (case b).** Prereg `45cbd7de` verified as an ancestor of HEAD and byte-identical before
  this line was written.
  - **P1** complex64 copy ≈ 133.45 MB → **MISS on the dominant kernel.** That copy exists and matches
    exactly, but it is 5.3% of copy time; the dominant complex64 copy is **377.49 MB** — the *weight*.
  - **P2** conj ≈ 133.45 MB, 24/rank-step = 2 × num_layers → **count HIT, size MISS**, and the stated
    rationale ("one conjugate per operand per contraction") was **also wrong**: 12 weight conj @377.49
    MB + 12 activation conj @66.72 MB.
  - **P3** float copy ≈ 132.71 MB (latent) → **MISS.** The dominant float copy is **66.72 MB** = one
    part (real or imag) of the complex spectral field.
  - **P4** nothing exceeds 265.42 MB → **MISS**, and the bound itself was wrong: 265.42 MB was the
    largest tensor *I had enumerated*.
  - **Why all four missed: the inventory omitted the weights.** For `dhconv` the spectral weight is
    `[in, out, modes_lat]` complex = **377.49 MB/layer**, and 12 of them are **95.8%** of the 1.18 B
    params. It is the largest tensor in the model. **Decision rule case (b) fired exactly as written.**
  - **The finding, after review:** **133.15 ms/rank-step** moves that weight in **four** places —
    forward permute 35.60, `ckpt3` recompute 35.61, adjoint `conj` 35.93, **`grad_w` → DDP bucket
    26.01** — all from one mismatch (stored `(in,out,lmax)`, contracted by `einsum("bixy,iox->boxy")`
    which permutes to `(x,i,o)` for `bmm`). **Invariant movement is 107.14 ms = 17.8% of the step.**
- **gauntlet: CLEARED, and it corrected my headline twice.** Both roles on the inherited tier
  (`claude-fable-5` still out of credits). **Both agents independently found the same refutation** —
  `spectral_layers` never reaches this module — which is the strongest signal yet that the gauntlet is
  not just echoing me.
  - **Adversary: 12 strikes, 2 FATAL.** (i) the `spectral_layers` mechanism was numerology; it then
    *measured* the real decomposition (12/12/12 by duration and phase). (ii) **26.01 ms of the 133 is
    the gradient, not an invariant weight** — proven by a **0.871 ms median gap to the next NCCL kernel
    with a 9 µs p10–p90 spread**, against 20.5 ms and scattered for the other population. I reproduced
    both before adopting them.
  - **It also found two bugs in my own tool:** elements-per-block is **launch-path dependent** and I
    under-counted the non-legacy paths by **exactly 4×** (a published "1.185×, no clean tensor" row is
    the fp32 latent exactly); and `C_in` was 108 when the parameter total forces **105**.
  - **Drift auditor: 16 items.** Highest-value: the handoff had shelved the spectral no-op-copy guard
    as "sub-1% class" on a figure that counted only the zero-fills against a 74.2%-NCCL denominator —
    §4.5b sizes the copy it removes at **2.4% of the step**, so it is **re-ranked as one of the
    cheapest levers**. And it established that the weight finding is **structural**: ACE2 has the same
    weight class (~93% of its params) and has never been checked.
  - **What I found that neither did:** with `factorization: None`, `use_tensorly=False` lands on
    `assert factorization == "ComplexDense"` (`s2convolutions.py:151`) — that assert **must** fire, yet
    both jobs ran 40 steps. So we do not actually know whether this weight is an `nn.Parameter` or a
    `FactorizedTensor`, and every *mechanism* claim depends on it. Recorded as OPEN; no measured number
    depends on it.
- **next:** plan item **4** (`kernel_census.py` — import the guarded, process-scoped join rather than
  re-deriving it). **That is the last free item; Tier 0 is then exhausted** and the next tick should
  prepare the first submission request (item **6**, `gpu_topology_check.py`, `debug`, ~1 min) and
  **stop for approval**. **Nothing has been submitted in this loop.**
- **infra-failure count:** 0/5

---

## tick 2 — 2026-08-20 — stage T0 item 2 — prereg for the n=2 re-derivation on job 7255557

- **in flight:** none (Tier 0 needs no `qsub`)
- **preconditions established from FILES, not from the capture** (so they are not part of the prediction):
  - `bench/bench_env_polaris_nsys_7255557.txt` records **the same workload** as 7255503's §0 description:
    `nettype: sfno_plasim`, `checkpointing: 3`, `world_size: 4`, `bench_warmup: 20`, `bench_steps: 40`,
    `num_data_workers: 1`, batch 1/GPU, bf16, `ddp_find_unused: false`, `use_ema: True`,
    `yaml_sha256_16: 47d632f85c84353a`, `git_sha: 9c3122e67d71`, torch 2.8.0 / CUDA 12.9, A100-SXM4-40GB.
    **There is no env file for 7255503**, so the two configs cannot be compared by sha — the structural
    check has to come from the captures themselves (rank count, phase-row count, kernel set).
  - **7255557 is the clean RE-RUN of 7255503, and the fix was to the clock, not the model** (CHANGELOG
    2026-07-15): `elapsed` was sampled *after* `cudaProfilerStop()`, so 7255503 read `elapsed=51.8 s` vs
    `sum=25.7 s` and its bench row was **refused** by the self-check. 7255557 records cleanly at rc=0.
    ⇒ the two should be the same workload; that is exactly what makes this a usable n=2.
  - **7255557 is much noisier at the harness level.** Its CSV row: `step_med 0.606010`, `step_p90 0.812770`,
    `step_mean 0.695946`, **`step_std 0.235865` = 39% of the median** — against 7255503's capture-side step
    std of **31.9 ms on 603.5 ms = 5.3%** (§4.1). `step_mean >> step_med` says there are outlier steps.
    Also `peak_mem_gb_max_rank` **28.762 GB** vs the sweep's 26.98 GB (§5) — the nsys run costs ~1.8 GB more.
- **prereg (written BEFORE any query against `nsys_pangu_sfno_7255557.sqlite`):**
  - **P1 (structure).** 4 ranks; `data_prep`/`forward_loss`/`backward`/`optimizer` at **160 rows each**;
    `(outside)` = **0.0%** again; the pid-guarded join returns exactly one row per kernel.
  - **P2 (the one that matters — ABSOLUTE work reproduces, SHARES may not).** `direct_copy`+`conj` should
    reproduce in **absolute ms/rank-step within ±5% of 271.19**, because that is config-determined compute.
    Its **share** of GPU kernel time may come in **below 42.2%**, because a stall inflates NCCL and therefore
    the denominator. Predicting the share moves *down*, not up.
  - **P3 (the split).** `backward` / `forward_loss` of the copy time reproduces **within ±2 points** of
    **72.9 / 27.1** — this is a ratio of two compute buckets and should be the most stable number here.
  - **P4 (stalls).** Given `step_std` = 39% and `step_mean >> step_med`, 7255557 contains **more or larger
    comms stalls than 7255503's single step-30 event**, and its NCCL ms/rank-step will exceed 7255503's
    67.82. Total GPU kernel time per rank-step will therefore come in **above** 643.19.
  - **P5 (bandwidth).** D2D above L2 within **±3 points of 82%** of peak.
  - **Decision rule.** If P2+P3 hold while P4 also holds, then **§0d's percentages are stall-sensitive and
    §0's numbers should be quoted as absolute ms/rank-step (or stall-excluded), not as shares** — that is a
    methodological finding worth landing in the plan, and it makes the n=2 stronger rather than weaker. If
    the *absolute* copy time does **not** reproduce within ±5%, then the two captures are not the same
    workload despite the matching env, and item 2's answer is "no usable n=2 exists" — a recorded null, not
    a failure. If P3 fails, the split from item 1 is not a property of the model and §4.3c must be
    de-generalised to job 7255503 alone.
  - **Stated limit:** this is n=2 on the **same node type, same day, same git sha**. It tests
    run-to-run reproducibility, **not** node-to-node (spread 10.5%) and not config sensitivity.
- **result:** **5/5 predictions HIT.** Prereg `952fcb8d` verified as an ancestor of HEAD and
  byte-identical before this line was written.
  - **P1 structure** → 4 ranks, 160 rows each phase, `(outside)` **0.0%**, exactly one join row per
    kernel. HIT.
  - **P2 absolute reproduces / share falls** → copies **271.19 → 270.94 ms/rank-step (−0.09%)**; share
    **42.16% → 37.39%**. HIT, both halves, and the share fell in the predicted direction.
  - **P3 the split ±2 pt** → **72.86 → 72.83%**, i.e. **0.03 pt**. HIT.
  - **P4 more stalls, higher total** → NCCL **+114.8%**, total **+12.7%**, stalled steps **1 → 17 of 40**.
    HIT.
  - **P5 bandwidth ±3 pt** → **82.2% → 82.4%** of peak. HIT.
  - **Decision rule fired as written:** §0d's percentages are stall-sensitive ⇒ the plan and the tables now
    say quote **ms/rank-step**, not a share. Also **stronger than predicted**: the two jobs ran on
    **different nodes**, so this is node-to-node, and it refines the repo's standing rule into *cross-job
    **compute** comparisons are sound; anything containing NCCL is not.*
- **gauntlet: CLEARED, and it changed the conclusion.** Both roles ran on the inherited tier again
  (`claude-fable-5` still out of credits).
  - **Drift auditor: 27 items.** Two mattered most: (a) I claimed "same `git_sha`" when **my own prereg
    two paragraphs above says no env file exists for 7255503** — a self-contradiction I should have caught;
    (b) §4.4d's per-rank table and the stall count had **no committed code behind them** (ad-hoc SQL), the
    same process violation as last tick's strike 15. Both fixed; `--per-rank` and `--stall-cause` now exist.
  - **Adversary: 11 strikes, 4 FATAL — and it found the actual cause I had mis-diagnosed.** The step-30
    stall is **CPython gen-2 GC** (`gc_collect_main`, 116/88/88 samples), not NUMA. I verified it myself
    before adopting it. It explains what my hypothesis could not: why the stall lands on the **same
    training iteration on two different nodes** — a gen-2 collection fires on allocation count, which is
    hardware-independent.
  - **What I got wrong, recorded so it is not re-derived** (full list in §4.4f): "dev1 is the straggler in
    both captures" (one event per capture, not a rank property — and my ranking summed the **rooted**
    broadcast, whose root is ~0 by construction, biasing every step); "`broadcast_buffers=False` would only
    move the wait" (refuted by **my own union column** — the forward broadcast wait is 0% overlapped while
    backward NCCL is 83–87% overlapped, so moving it would *hide* most of it); "same `git_sha`"; "every
    launch count identical" (cuDNN picked a different wgrad tile); "+20% warmup" (really +6.7% — I had
    compared one rank's step 0 against the all-rank median).
  - **Three tool bugs found by the review, all now tested:** the `Reduce` regex matched `AllReduce` and
    silently emptied the straggler ranking; SI and binary units were mixed in one table (12.56 "GB" was
    GiB); and a **cursor-reuse** bug made `--stall-cause` return no samples at all — indistinguishable from
    "this capture has no sampling data".
- **next:** plan item **3** (analytic bytes-per-step model — free, no `qsub`), then item **4**
  (`kernel_census.py`, free). Tier 0 is then exhausted; the first submission request will be item **6**
  (`gpu_topology_check.py`, `debug`, ~1 min) or the new **6b**. **Nothing has been submitted in this loop.**
- **infra-failure count:** 0/5

---

## tick 1 — 2026-08-20 — stage T0 item 1 — NVTX text path resolved; prereg for the fwd/bwd copy split

- **in flight:** none (Tier 0 needs no `qsub`)
- **prereg (written BEFORE the split was computed):**
  - **P1.** Launches falling **outside** all four house phases will be **< 2%** of all launches on rank 0.
    Rationale: the four phase windows sum to ~335 ms of a 603 ms step, but the missing ~266 ms sits between
    `optimizer` end and `step_N` end, which reads as the CPU blocking on a sync while the GPU drains — a wait,
    not a launch site. If instead >10% of launches land outside, there is an unranged launch site (EMA? logging
    sync? validation?) and the phase attribution is incomplete — that becomes the finding.
  - **P2.** `backward` will hold **>= 65%** of the `direct_copy` + `conj` GPU time. Rationale: `checkpointing: 3`
    recomputes the forward inside backward, so backward pays recompute (~1x forward) plus the adjoint
    (~1.5-2x forward) while forward pays 1x.
  - **P3.** `forward_loss` **15-30%**, `optimizer` **< 5%** (FusedAdam is its own kernel; EMA never fired),
    `data_prep` **< 1%** (0.16 ms/step of CPU window).
  - **Decision rule.** If P2 holds, activation **recompute** is a first-order share of the 271 ms/step and the
    `ckpt3 -> ckpt2 -> ckpt1` ladder (plan item 10) directly deletes a measurable part of §0d — the ladder is
    then the highest-value Tier-1 item after item 7. If instead `backward` < 50%, the copies are intrinsic to
    the SFNO spectral path (SHT transpose/`conj`), checkpointing barely touches them, and the lever is layout,
    not recompute. Either way the number is recorded; neither outcome edits a gate.
  - **Stated limit, pre-committed:** this split cannot separate *recompute* from *adjoint* inside `backward` —
    there is no NVTX range between them. That separation is plan item 17, not this item.
- **result:** **3/3 predictions HIT.** Prereg integrity asserted *before* writing this line:
  `985214b5` is an ancestor of HEAD, and the prereg text above is byte-identical to that commit
  (`git diff --quiet 985214b5 HEAD -- prompts/pangu_polaris_loop_journal.md` was clean at the time
  of measurement; re-check against `985214b5`, not against a later HEAD, since this `result:` line
  was added afterwards by design).
  - **P1** `(outside)` < 2% → **0.0%** (measured). Stronger than predicted: **all** 354,720 launches
    fall inside a house phase, so the step's "missing" 268 ms contains zero launches — it is pure
    GPU drain, which closes an open question in `polaris_bench_report.md` §4.1.
  - **P2** `backward` >= 65% of copy time → **72.9%** (measured; 197.6 ms/rank-step of 271.2).
  - **P3** `forward_loss` 15-30% → **27.1%**; `optimizer` < 5% → **0.0%** of copies; `data_prep`
    < 1% → **0.0%** (it launches zero kernels at all; its GPU cost is 2.18 ms/rank-step of H2D).
  - **Decision rule fired (P2 held):** activation recompute is first-order, so the `ckpt` ladder
    (plan item 10) is the highest-value Tier-1 item after item 7. The split gives a **ceiling**, not
    a decomposition: recompute <= forward's own 150.3 ms/rank-step ⇒ ckpt-off <= **24.9% of the
    step (<= 1.33x)** — and ai-rossby's measured full ladder (1.307x = 23.5%) sits **1.4 points**
    under it, so the ceiling is nearly saturated and essentially the whole forward is recomputed at
    `ckpt3`.
  - **Correction to the prereg's own pointer:** the recompute-vs-adjoint separation is plan item
    **16** (SFNO-internal NVTX ranges re-fire inside `backward` during checkpoint recompute), not
    item 17 (`--python-sampling`, which samples CPU stacks and cannot partition GPU time). The
    prereg text above is left unedited — it is frozen; this is the correction of record.
  - **Extra, not preregistered:** D2D `cudaMemcpyAsync` in the same capture sustains **1265 GB/s =
    81% of the A100's 1555 GB/s**, all four devices within 2 points. Intra-device HBM only — it does
    **not** close the interconnect/topology cell (item 6). It does kill the "81% is unreachable on
    this node" reading of §0d's estimated 17-27%, so it narrows item 7 without replacing it.
- **gauntlet: CLEARED — both roles returned, 42 findings between them, all triaged.** Ran on the
  inherited tier, not Fable 5: **`claude-fable-5` is out of credits on this account** (both first
  attempts died with "Usage credits are required for this model"). Worth knowing before the next tick
  plans a subagent.
  - **Drift auditor:** 27 items. It independently re-derived **every** §4.3 number from the capture and
    reproduced all of them. Three were real defects in my §4.3 (bytes column mixed per-rank-step with
    all-rank totals; a dropped `forward_loss` row so 128 != 131 launches; "26 points" mixed
    denominators). Two were interpretation errors (item 17 -> item **16**; "comfortably inside the
    ceiling" when ai-rossby's ladder actually sits 1.4 points under it). All applied.
  - **Adversary: 2 FATAL + 5 MATERIAL + 8 MINOR, and both FATALs held up.**
    1. **The removable/non-removable buckets were INVERTED.** Recompute happens *inside* `backward`, so
       `backward`'s 197.58 ms is the bucket that shrinks and `forward_loss`'s 73.61 ms is the one no
       checkpointing level can remove. My sentence said the opposite. The 27% magnitude was right only
       by coincidence (recompute ~ forward). **A reader would have concluded "73% is untouchable, skip
       the ckpt ladder" — the exact wrong decision.**
    2. **"Recompute cannot exceed the forward's GPU time" is not a bound.** Kernels with equal fwd/bwd
       launch counts run at **1.0136x, never below 1.0**, and recompute selects *different* GEMMs
       (+15%/call). So it is an estimate of ~150 ms. It also caught that my section contradicted itself
       in six lines: header said ESTIMATED, bullet said "measured bound".
    3. **My "backward is 77.4% of the step" was the sum-vs-union confusion** — the very framing the
       driver's §0 lists as already-refuted. Union is **408.63 ms = 67.7%**; only `backward` self-overlaps
       (12.5%, NCCL on `streamId 19`). I verified this myself by adding a union column to the tool: it
       reproduces the adversary's figure exactly.
    4. Also confirmed by my own re-derivation: the `2 x bytes` rule fails sub-L2 (**two** buckets compute
       to >100% of peak, 124.7% and 110.0% — the proof); the D2D stream claim (all on `streamId 7`, so
       serialized with compute, not concurrent — my hedge was the wrong hedge); 960 of 962 H2D, not 100%;
       step-30 comms stall worth -12.0% on the NCCL mean; and the `conj` warrant must come from **source**
       (no `conj` in `networks/modulus_sfno`, einsum over `view_as_complex`, 24/rs = 2 x `num_layers`),
       not from the phase split, since a recompute-only kernel would look identical.
    5. **Two claims it tried and could not break:** the +29.4% phantom join and the guard dropping
       nothing (it checked for orphaned/graph-launched kernels and found none), and `(outside)` = 0.0%
       (it went further and found the CPU inside `cudaDeviceSynchronize` for 10.72 s on rank 0).
  - **Strike 15 was a process finding against me:** the bandwidth table had **no committed code behind
    it** — it came from an ad-hoc heredoc, violating "analysis code that produces a load-bearing number
    must be re-runnable from the repo". Fixed by adding `--memcpy`, `--per-step` and the union column to
    the tool, so §4.3h now lists a command for every table in the section.
- **also landed this tick (commit 99378811), found by the new tool's test:** `parse_nsys.py` could
  not run on a Polaris login node **at all** — `sqlite3.connect(PosixPath)` needs Python >= 3.7 and
  the login default is **3.6.15**, and `statistics.fmean` is 3.8+. Both fixed with regression tests;
  the NVTX range list hoisted to one `RANGE_NAMES` constant, which also repairs the live CLAUDE.md
  #10 drift (`unstack` was in the SQL but not the print loop, so its rows were fetched and silently
  dropped). Verified behaviour-preserving: the NVTX table now reproduces §4.1 exactly.
- **FLAGGED FOR THE OPERATOR, not fixed:** `s2s/v2.0/HPC_scripts/parse_nsys.py` is a *different*
  copy carrying only **8 of the 19** range names, and it is the one the Polaris PBS scripts and
  `physicsnemo_ai_rossby/polaris/bench_instrumentation_test.py` actually invoke. So when plan item
  16 adds SFNO-internal ranges, the Polaris analysis path will print **nothing** and look like the
  instrumentation never fired. Fixing it touches the **live-coupled** s2s pair, which this loop must
  not do (driver §3.4) — it needs its own change with both S2S and s2s-lightning smokes run.
- **measured this tick already (not part of the prereg):**
  - **The NVTX text path is the inline `text` column, `domainId=0`, `eventType=59` (push/pop).** All four house
    ranges are present in `nsys_pangu_sfno_7255503.sqlite`: `data_prep`/`forward_loss`/`backward`/`optimizer`
    at **160 rows each** = 40 steps x 4 ranks, plus `step_20..step_59`. The `textId -> StringIds` path holds
    **only** NCCL's registered strings (`ncclAllReduce` 2402, `ncclBroadcast` 160, `domainId=1`). So the plan
    §1 item-1 symptom is explained, not merely worked around: nothing is missing from the capture.
    `parse_nsys.py`'s `WHERE text IN (...)` was already on the correct path.
  - **The `correlationId` join inflates by +29.4% on this capture** (naive 459,088 rows vs guarded 354,720 =
    exactly one row per kernel). The guard is `k.globalPid = (r.globalTid & -16777216)`; verified that clearing
    the low 24 bits of every RUNTIME `globalTid` reproduces the four KERNEL `globalPid` values exactly.
    Independently confirms the handoff §5 figure of +30.8% on a different capture. — measured
  - **`kernel_census.py` has a second, independent bug** beyond the missing guard: `enclosing(rs, tid)` looks up
    the NVTX range on the *launching* thread, but on rank 0 **62,680 of 88,680 launches come from the autograd
    worker thread** (`...2d149f`) while **all 201 NVTX events are on the main thread** (`...2d139d`). Same-thread
    attribution would therefore credit `(outside)` for the whole of `backward`. Attribution must be scoped to
    the **process** (globalPid), not the thread. — measured, plan item 4
- **STAGE_LANDED — plan items 1 AND 5, both ticked.** Item 5 fell out of the adversary's warmup
  attack: no warmup regime (first step 640.26 vs 634.36 ms median), but step index 30 is a comms stall
  worth -12.0% on the NCCL mean. Commits: `985214b5` (prereg) -> `17d2baf1` (tool+test) -> `99378811`
  (parse_nsys 3.6 fixes) -> `66572846` (verdict) -> `dfb36132` (item 1, §4.3) -> `5b48b176` (drift
  repair across 6 docs).
- **next:** plan item **2** — re-derive §0d on the **second** capture, job **7255557**, as a true n=2.
  The drift auditor was explicit that item 2 is **NOT** closed by this tick: §4.3 is an independent
  *query path* on the *same* capture. Then item **3** (analytic bytes model) and item **4**
  (`kernel_census.py`). Tier 0 remains `qsub`-free; the first submission request will be item **6**.
- **infra-failure count:** 0/5

---

## tick 0 — 2026-08-20 — setup — loop machinery written; nothing submitted

- **in flight:** none
- **prereg:** n/a (no measurement this tick)
- **result:** the three loop files exist (driver prompt, README, this journal) and the frozen plan is
  `PANGU_POLARIS_PROFILING_PLAN.md`. Design decision recorded: **live-session driver, not a batch
  orchestrator** — `debug` is `max_run 1`/`max_queued 1` per user and `capacity` is `max_run 1` per project, so
  a PBS orchestrator that submits nested PBS jobs deadlocks against itself (README §"Why this one is a LIVE
  session"). — measured (the queue limits are quoted from `polaris_pbs_notes.md` §1b, queried from PBS
  2026-08-05; not re-queried today)
- **starting state, so tick 1 does not re-derive it:**
  - Branch for the loop: `profile/pangu-polaris-profiling`, to be cut off `fix/tsoi-fill-270` (carries the
    plan). Not yet created.
  - Tier 0 is fully unblocked — both captures are on disk:
    `${MEMBER_ROOT}/bench/nsys_pangu_sfno_{7255503,7255557}.sqlite`.
  - Already derived from 7255503 (do **not** re-measure; see the plan §0): GPU-busy union 95.6–96.5%; NCCL
    88.7% overlapped / 1.2% exposed; `ncclDevKernel_Broadcast_RING_LL` present at 0.11%; `direct_copy` + `conj`
    = 42.2% of GPU kernel time at an **estimated** 17–27% of HBM peak.
  - Known-broken tool: `ACE2_retrain/kernel_census.py` (unguarded `correlationId` join) — plan item 4.
  - Known-open blocker for everything downstream: **no PanguWeather §4.1 baseline exists** — plan item 18.
- **next:** tick 1 = orient, create the branch, then plan item **1** (fix the NVTX↔kernel join so the 42% copy
  time can be split forward vs backward). No GPU, no `qsub`.
- **infra-failure count:** 0/5
