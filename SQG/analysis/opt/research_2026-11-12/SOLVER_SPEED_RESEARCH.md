# How much faster can the SQG solver step get? Research round 2 (2026-11-12)

**Ask (user, 2026-11-12):** "do more deep-research explore github and other code bases and other nascent strategies
of how to further solver work can be done to decrease it's step time" (and "use /workflows if necessary").

**Method.**
- **Workflow:** `wf_1a24fad4-309`, 10 agents. Five research tracks ran in parallel:
  1. open-source GPU pseudo-spectral codebases;
  2. FFT kernels;
  3. time stepping;
  4. changes reachable inside JAX/XLA;
  5. newer strategies (precision, tensor cores, Hopper hardware, scheduling).
- **Verification:** an adversarial verifier followed each track. It re-opened every cited source, re-did each gain's
  arithmetic against notes Part X's measured per-op table, and checked applicability and novelty against notes Part
  VII/X.
- **Raw output:** every finding with its verdict, sources, risks and test plan is in `research_findings.json` (this
  directory).
- **What I checked myself (main session):**
  - The production decay table carries the 2/3 mask, `decay = 1[kap <= KMAX] * exp(-(NU k^2 + MU/k^2) dt)`
    (`A-SQG_data.py` L326-329). So compact storage (XR-1) can be bitwise.
  - `A-SQG_data.py` L322: "DT=1e-5 reached a sampled CFL of 0.71 (abort at 0.72); DT=5e-6 stayed <= 0.42". This bounds
    the larger-dt options.
- **Status:** these are research results, not measurements. Every gain below is an estimate tied to the A100 per-op
  table. Nothing here has been run, and nothing was submitted.

## Bottom line

1. **The biggest lever is the time step, not the FFT.**
   - o1.2's measured dt error is first order (Part XII, P5). It comes from applying the decay once to all three AB3
     history terms.
   - An exact integrating-factor AB3 (IF-AB3) fixes that at ~0 cost per step. It multiplies the two older tendencies
     by `E` and `E^2`, where `E` is the decay table the step already reads.
   - With that fixed, dt can grow toward the CFL limit: 28 steps per frame (safe) or 25 (aggressive), instead of 40.
   - Gain: **1.30–1.40× per optimizer step** (94 h → 67–72 h on 14 A100s). Pure JAX, small code change, but a
     numerics change that must pass the statistical gates.
2. **The largest bitwise gain is compact storage of the kept modes (XR-1): −0.22 to −0.27 ms/step (~9%).** Pure
   JAX. Only the 2731 × 1366 kept-mode block is stored and updated, instead of the 4096 × 2049 half-plane.
3. **A custom fused FFT pipeline is the remaining per-step pot.** It brings the step to ~2.0–2.1 ms with cuFFT LTO
   callbacks, and ~1.6–1.7 ms with cuFFTDx column passes. That is the same "FFT roofline" Part VII/X estimated.
   - gyaradax's public code (LGPL-3.0) is a near-direct template.
   - **Blocker:** jax 0.4.28 has no FFI. This needs either the legacy custom-call API or a JAX upgrade. A JAX upgrade
     collides with torch 2.3.1's cuDNN pin in the same process.
4. **Stacked:** the pure-JAX route reaches **1.40–1.48×** per optimizer step. The custom-FFT route reaches **1.56–1.78×**,
   or **up to ~2.1×** with CFL-adaptive steps per frame. At that point the fixed 0.40 s ML step is half the optimizer
   step.
5. **The survey found no other open GPU QG/SQG/2-D turbulence code with a per-step technique we lack,** except the
   exact-linear time stepping in item 1. Tensor-core FFTs, FP64 tensor cores, Ozaki emulation, ETDRK4, AB4,
   parareal/SDC and temporal blocking are all dead ends for this step (see "Dead ends").

## Today's step (recap of notes Part X)

- `lean32ek_fast` = 2.698 ms/step on an A100 (2.73 ms inside o1.2). A frame is 40 steps; an optimizer step is 12
  frames plus a 0.40 s ML step, 1.695 s in total.
- The step is memory-bandwidth-bound: ideal 2,819 MB/step = 2.09 ms at 1.35 TB/s.
- cuFFT's column passes run at 41–52% of DRAM peak. Everything else runs at 78–89%.
- It does 5 transforms per step: the θ IRFFT, a full-plane C2C for u − iv, and 2 RFFTs of the products.

## The ranked plan

"Gain" is the verifier's corrected number. Items in the same pot (marked) must not be summed.

| # | change | gain | exactness | route, effort | verdict |
|---|---|---|---|---|---|
| 1 | **IF-AB3** (TS-2 / C1): `theta_{n+1} = E·[theta_n + dt·(23/12·N_n − 16/12·E·N_{n−1} + 5/12·E²·N_{n−2})]` | ~0 ms/step by itself; enables #2. Worst case +25 µs if XLA splits the fusion | numerics change: gated | pure JAX, S | corrected (formula re-derived: the −z/2 split deficit cancels through O(z²)) |
| 2 | **Larger fixed dt with IF-AB3:** 28 steps/frame (CFL ≈ 0.60) or 25 (CFL ≈ 0.67) (TS-3) | per frame 107.9 → 75.5 / 67.5 ms; per optimizer step 1.30× / 1.40× | gated | pure JAX, M (the gate campaign) | corrected (the error bound is loose but conclusions hold; worst CFL seen is 0.45 on a synthetic state) |
| 2a | without IF-AB3: n = 32 or 30 (TS-1), the arms already run in Part XII | 1.18× / 1.24× | gated; gates A and B1/B2 already pass | pure JAX, S | confirmed |
| 3 | **Compact kept-mode storage** (XR-1): state, compensation, AB3 history and build_w on the 2731 × 1366 block | −0.22 to −0.27 ms (2.698 → ~2.43–2.48) | **bitwise** (state is exactly 0 outside the mask) | pure JAX, M | confirmed |
| 4 | Batch the two product RFFTs into one batch-2 call (C4) | −0.03 to −0.06 ms (shared 4.2-wave tail) | likely bitwise (verify) | pure JAX, S | confirmed |
| 5 | Transposed layout + pruned forward transforms (XR-2), on top of #3 | −0.17 to −0.20 ms | rounding-only | pure JAX, L | corrected (build_w was double-counted with #3) |
| 6 | **Custom cuFFT call** without callbacks: removes the IRFFT input copy and the 1/N² `scal` (FFT-4) | −0.09 to −0.21 ms | bitwise candidate (1/N² = 2^−24 is an exact scaling) | custom call, M | confirmed; ⊂ #7 |
| 7 | **cuFFT LTO-callback pipeline**, gyaradax pattern (FFT-1 / C2 / C3): load callbacks build ψ̂ and u, v; products fused; store-skip of dealiased modes | −0.60 to −0.70 ms (→ ~2.0–2.1 ms) | rounding-only; gated if not bitwise | custom call + CUDA ≥ 12.6u2 callbacks, L | corrected (the 99 µs tendency copies were already removed by the ring) |
| 8 | **cuFFTDx** for the slow column passes (FFT-3), on top of #7 | up to −0.41 ms (→ ~1.6–1.7 ms) | rounding-only | CUDA C++, L (the hardest) | confirmed (bounded by Part X's measured headroom; no external 4096² number) |
| 8a | alternative to #8: padded-pitch half-plane plans (FFT-5) | −0.1 to −0.28 ms (same pot as #8's half-plane share) | rounding-only | custom call, S once #6 exists | confirmed; unmeasured anywhere |
| 9 | **CFL-adaptive steps per frame** (TS-4): n chosen per frame from the sampled umax | ≤ 1.62× per optimizer step at lean32ek_fast speed (upper bound) | gated; changes determinism structure | pure JAX, L | confirmed as an upper bound |
| 10 | Retest solver/ML overlap with the **fast** solver (FR4) | untested headroom up to ~22% of wall | bitwise (scheduling) | config, S | corrected: the old 21% was 21% *of the ML step* (~1.8% of wall), not of wall |

Cheap measurements that set the numbers above:
- **TS-5:** a wider dt ladder (n = 60, 40, 32, 28, 25, 20) per band. It pins AB3's h³ constant, and with it the
  admissible n for #2 and #9.
- **An offline scan of the archive's stored umax series,** for the worst production CFL. It decides 28 vs 25.

## Combined projections (A100; o1.2's 12 frames + 0.40 s ML step per optimizer step; 14-GPU schedule)

| variant | ms/step | steps/frame | ms/frame | s/optimizer step | speedup | full schedule |
|---|---|---|---|---|---|---|
| today: `lean32ek_fast` | 2.698 | 40 | 107.9 | 1.695 | 1.00× | 94 h |
| + #4 + #3 (bitwise, pure JAX) | 2.408 | 40 | 96.3 | 1.556 | 1.09× | 86 h |
| time step only: #2a, n = 32 | 2.698 | 32 | 86.3 | 1.436 | 1.18× | 80 h |
| time step only: #1 + #2, n = 28 | 2.698 | 28 | 75.5 | 1.307 | 1.30× | 72 h |
| time step only: #1 + #2, n = 25 | 2.698 | 25 | 67.5 | 1.209 | 1.40× | 67 h |
| pure-JAX stack: #4 + #3 + #1 + #2 (n = 28) | 2.408 | 28 | 67.4 | 1.209 | 1.40× | 67 h |
| + #5 | 2.223 | 28 | 62.2 | 1.147 | 1.48× | 64 h |
| custom FFT (#6–#7, ~2.05 ms) + n = 28 | 2.05 | 28 | 57.4 | 1.089 | 1.56× | 60 h |
| custom FFT + cuFFTDx (#8, ~1.65 ms) + n = 28 | 1.65 | 28 | 46.2 | 0.954 | 1.78× | 53 h |
| same, CFL-adaptive mean n ≈ 20 (#9, upper bound) | 1.65 | 20 | 33.0 | 0.796 | 2.13× | 44 h |

- The pure-JAX layout gains (#3, #5) and the custom-FFT gains (#6–#8) target overlapping bytes. The custom-FFT rows
  replace the pure-JAX rows; they are not added to them.
- In the last row the 0.40 s ML step is 50% of the optimizer step. Past that point, solver work buys less than
  ML-step work or decoupling.
- On H200 (production), a bandwidth-scaled step would be ~0.87 ms (FR2: 2.698 × 1.35/4.17). That is an optimistic
  bound: it assumes A100's attainable fraction and column-pass behaviour carry over. The anchored estimate in the
  2026-11-12 chat was 1.4–1.65 ms.

## The levers in more detail

### Time stepping (track TS; biggest)

- **Why the error is first order today.**
  - The step is `theta_{n+1} = E·(theta_n + inc)`, with `inc` the AB3 combination of `N_n`, `N_{n−1}`, `N_{n−2}`.
  - The exact solution weights an older tendency by the decay accumulated since it was computed. The current step
    applies a single `E` to all three.
  - The per-step deficit is ≈ (λ dt / 2)·dt·N. Part XII's ladder measured it: the D80/D30 ratio is −1.42 to −1.51,
    and theory gives −1.50 for first order.
- **The fix is the Lawson / integrating-factor AB3.** It is standard in the GPU spectral codes surveyed:
  - the Valadão/Boffetta group's exact-linear RK (arXiv 2408.15735);
  - FourierFlows' ETDRK4;
  - pyspectral's IF-RK4.

  `E` is already read by the AB3 fusion, so `E²` and `E³` are register math.
- **How far dt can go.**
  - The current CFL is 0.374–0.42 at dt 5e-6. AB3's imaginary-axis limit is ~0.72, and the code aborts at 0.72.
  - dt 1e-5 (n = 20) reached a sampled CFL of 0.71, so 20 is the edge. 25 gives CFL ≈ 0.67 and 28 gives ≈ 0.60.
  - With the split removed, AB3's own h³ error reaches production's current dt error only near h ≈ 1.4–1.6e-5
    (verifier's re-derivation). At n = 25 the cubic term is ≈ 0.12× production's dt error.
- **Risks the verifiers added.**
  - IF methods move the fixed points of a forced-dissipative system more than ETD does (arXiv 1709.06873 §II.C). The
    energy test (Part XIV) is the right check.
  - Rosa & Wang (2009) found a round-off feedback instability in an IF-AB scheme. That was 3-D, via the projection,
    but the float32 form of the decayed history deserves its own precision check.
  - If ETD3-AB is ever needed (TS-6, the fallback), its φ-coefficient tables cancel catastrophically at z ~ 1e-3 in
    float32. Build them in float64 and round once.
- **Ruled out:**
  - ETDRK4 / IF-RK4: 4 RHS evaluations per step at a per-evaluation stability of 0.707, below AB3's 0.724.
  - AB4: imaginary stability ~0.43.
  - Leapfrog-RAW: its filter adds damping the bias gates catch.
  - Kinnmark–Gray RK: at most 17% fewer evaluations, and accuracy binds first.
  - Parareal/PFASST/SDC, Richardson extrapolation, and non-integer steps per frame (that one is a science change).

### Bitwise layout work inside XLA (track XR)

- **#3, compact storage.**
  - The state is exactly zero outside |k| ≤ KMAX, because the decay table carries the mask. Tendencies outside the
    mask never reach the state: they are multiplied by 0.
  - The block |kx|, |ky| ≤ 1365 is 44.5% of the half-plane: 29.8 vs 67.1 MB in complex64.
  - The AB3/compensation fusion drops from ~629 MB to ~321 MB of traffic per step (−225 µs), and build_w by −85 µs.
- **#4, batched RFFTs.** The two product RFFTs share one half-plane column-pass tail. First check that XLA 0.4.28
  lowers a leading-dim-2 `rfft2` to one batch-2 plan; one nsys look answers it.
- **#5, pruned transposes.** These only pay when fused with the slice to the 1366 kept columns. A transpose alone costs
  more (99–130 µs) than the column-to-row gain (~84 µs per pass).
- **Not reachable in XLA:** the IRFFT input copy and the 1/N² `scal` are unconditional in XLA's FFT lowering (jax#9946,
  `fft_thunk.cc`). A pruned inverse θ transform therefore nets +3 µs. `jnp.fft.fft(axis=0)` always transposes, so
  cuFFT's strided column call is unreachable.

### Custom FFT pipeline (tracks FFT and C)

- **gyaradax** (github.com/gerkone/gyaradax, v0.2.0, **LGPL-3.0**) ships the full recipe:
  - cuFFT LTO load/store callbacks (`bracket_v5_*_cb.cu`);
  - Hermitian Z2Z packing;
  - a whole-bracket CUDA graph;
  - a cuFFTDx single-kernel variant;
  - a double-single (hi+lo float32) arithmetic backend, the CUDA twin of our compensated state.

  It integrates through `jax.ffi`. Copying its files makes them LGPL, so re-implementing the pattern is the clean
  option.
- **pyspectral** (github.com/walidhemati/pyspectral) reports 99–102% of an FFT roofline with two kernels per stage, via
  CuPy/NVRTC. Its numbers are on an RTX 3070 Laptop, so take the method, not the ms.
- **cuFFT LTO callbacks need CUDA ≥ 12.6u2.**
  - Coverage for our power-of-2 sizes predates CUDA 13.3. 13.3 only added sizes with large prime factors.
  - C2R load callbacks "may be called more than once per element" (cuFFT docs).
- **The JAX FFI blocker.**
  - `ffi_call` arrived in jax 0.4.32 (yanked; use ≥ 0.4.33 or ≥ 0.5). We run 0.4.28.
  - **Options:**
    - (a) the legacy custom-call API in jaxlib 0.4.28. One smoke job tells whether it is good enough.
    - (b) a solver side-env on a newer JAX. Any FFI-capable jaxlib needs cuDNN 9, while torch 2.3.1 pins cuDNN 8.
      o1.2 runs both in one process, and torch 2.4–2.6 pin cuDNN 9.1.0.70, below jax 0.11.2's ≥ 9.10.2. The clean
      route needs a newer torch, or a jax version whose cuDNN floor matches.
    - (c) measure callback gains offline first with nvmath-python or CuPy (`cupy.fft.config.set_cufft_callbacks`).
      Neither needs a JAX change.
- **Not levers:**
  - VkFFT: cuFFT is "significantly better" on Ampere-class 2-D fp32, and only its zero-padding could flip that.
  - explicit four-step transposes;
  - another cuFFT version: 11.4.1 was measured as a no-op in Part X;
  - pyvkfft through DLPack: the per-step sync dwarfs the gain;
  - CUDA graphs on their own: the GPU is 99.4% busy, so the gaps total ~19 µs/step.

### Precision below float32 (track FR; research only)

- **16-bit cuFFT transforms for the nonlinear term (FR1).**
  - Upper bound 2.698 → ~1.7–1.9 ms. It is likely less, because the column passes are limited by coalescing, not
    bytes.
  - It needs the FFI, and nvmath's callback helpers do not take fp16.
  - Given the user's rule that precision must not bias the training delta, its first gate is the existing delta
    harness, before any integration.
- **Stochastic rounding (FR3): refuted as new.**
  - `lean32sr` already exists (`taxonomy/lean32_fast.py`, mode `sr`), measured in job 7824121 at 2.793 ms.
  - Its e_Δ is 1.5× worse than `lean32ek`, with a suspected d-term of its own.
  - The only open item is a one-line `u1=True, ring=3` rebuild, if anyone wants its speed.

### Scheduling and hardware

- **Overlap of the solver and the ML step (FR4).**
  - Part III B's 21% was the overlap *efficiency*, the fraction of the ML step hidden. At `ref` speed that is ~1.8% of
    wall.
  - At `lean32ek_fast` speed the ceiling is 23.6% of wall (0.40 / 1.695), and it has not been tested.
  - The solver is bandwidth-bound and the UViT step compute-heavy, so a cheap streams/thread retest with the fast
    solver is worth one debug run.
  - Decoupling onto separate GPUs (round 1c: equal GPU-seconds, 3.8× shorter steps) stays the measured alternative.
- **H200 (FR2).** The step is near-linear in bandwidth: 0.87 ms is an optimistic bound. Hopper thread-block clusters,
  DSMEM and TMA only matter once we own the FFT kernels (#8).

## Dead ends (do not re-try)

| idea | why not |
|---|---|
| Tensor-core FFTs (tcFFT, FlashFFTConv, TurboFFT, error-corrected FP16/TF32) | the step is bandwidth-bound; FLOP throughput buys nothing, and hi/lo splits double the bytes |
| DFT-as-GEMM on FP64 tensor cores | one 4096² transform = 1.1e12 flops ≥ 56 ms at 19.5 TFLOPS, vs 1.2 ms in cuFFT |
| Ozaki-scheme FP64 emulation on INT8/FP8 tensor cores | accelerates GEMMs only, and multiplies them |
| Temporal blocking / fusing several steps on chip | the advection is nonlinear: physical-space products are needed every step |
| ETDRK4, IF-RK4, AB4, leapfrog-RAW, Kinnmark–Gray RK | see the time-stepping section |
| Parareal / PFASST / MGRIT / SDC | published evidence is negative for chaotic advection; the dt headroom is CFL-capped |
| Porting GeophysicalFlows.jl / FourierFlows.jl machinery | 20 transforms/step (RK4 × 5) vs our 4–5; their AB3 treats the linear term explicitly |
| Syncing vendored pyqg_jax with upstream | v0.1.0–v0.8.1 contain no performance changes |
| cuPSS, torchqg, dnsjax, spectralDNS/shenfun, GHOST, Dedalus GPU (merged into an unreleased branch 2026-11-10) | no per-step technique beyond ours |
| Filtered steppers, learned/ML solvers, coarser frames, interpolated frames | science changes |
| MIG partitioning | conserves total bandwidth; the solver slows proportionally |
| Scale-selective precision inside one FFT | an FFT mixes all scales within a pass |
| jax 0.10.0–0.11.1 | lowers multi-dimensional IRFFT decomposed (~1.7× the single C2R): a regression |

## Corrections the verifiers made (so they are not re-quoted wrongly)

- **Fused-FFT gain is −0.6 to −0.7 ms, not −0.75 to −0.85.** The 99 µs "tendency copies" were already removed by the
  3-slot ring in `lean32ek_fast`.
- **XR-2's on-top gain is −0.17 to −0.20 ms, not −0.26.** The build_w compaction was claimed by both XR-1 and XR-2.
- **The 21% overlap is 21% of the ML step,** not 21% of wall.
- **arXiv 2408.15735 is the group's Ekman–Navier–Stokes paper, not their SQG paper (2504.07914).** The "gTurbo2D" and
  "RK4EL" names are unconfirmed, and the code is not public.
- **The IF-AB scheme paper is Rosa & Wang, *Computers & Fluids* 38(10), 2009.**
- **The worst CFL seen is 0.45, on a synthetic spun-up state.** 0.42 is production-sampled, and 0.374 is from the 4
  archived states.
- **fp16 cuFFT needs SM ≥ 75.** Hatfield et al.'s 2.5× is half-vs-single GEMM on a CPU (A64FX), not a GPU.

## Decisions for the user

1. **Time step first?** Gate IF-AB3 at n = 28 (or 25, after TS-5 and the archive CFL scan)? Or first rescue the
   already-run n = 32/30 arms?
   - dt and scheme changes are deterministic per-mode shifts. Gate B's detection criterion (B3) will flag them as it
     flagged `lean32ek`, while B1/B2 pass.
   - Route them through gate D as before, or by your recorded post-hoc magnitude call?
   - The combination (`lean32ek_fast` + IF-AB3 + new n) must be gated as a unit.
2. **Bitwise layout work (#3, #4)?** No sign-off is needed if bitwise.
3. **Custom FFT pipeline?**
   - (a) the legacy custom call on jaxlib 0.4.28, or (b) a JAX/torch upgrade (re-baseline + the CXI libcudart shim);
   - clean-room vs LGPL code from gyaradax.
4. **Retest overlap with the fast solver?** Scheduling only, bitwise.

A single 1-node debug job could carry the first probes:
- IF-AB3 vs production at n = 40;
- the wider ladder;
- the archive umax scan;
- the batched-RFFT lowering check;
- the overlap retest.

Compact storage needs ~2–3 days of code first.

## Sources (opened by the agents and re-opened by the verifiers; "abstract" = only the abstract/README was read)

- gyaradax: https://github.com/gerkone/gyaradax (code; LICENSE LGPL-3.0); paper https://arxiv.org/abs/2604.06085
- pyspectral: https://github.com/walidhemati/pyspectral (README + repo tree)
- Valadão et al. (exact-linear RK on GPU, 2-D Ekman–NS): https://arxiv.org/html/2408.15735v1 (full); SQG paper
  arXiv 2504.07914
- FourierFlows.jl time steppers: https://raw.githubusercontent.com/FourierFlows/FourierFlows.jl/main/src/timesteppers.jl
- GeophysicalFlows.jl SurfaceQG: https://raw.githubusercontent.com/FourierFlows/GeophysicalFlows.jl/main/src/surfaceqg.jl
- pyqg-jax releases: https://github.com/karlotness/pyqg-jax/releases
- Kassam & Trefethen 2005 (ETDRK4, φ cancellation): https://people.maths.ox.ac.uk/trefethen/publication/PDF/2005_111.pdf
- IF vs ETD fixed points: https://arxiv.org/abs/1709.06873
- Rosa & Wang 2009: https://www.sciencedirect.com/science/article/abs/pii/S0045793009000814 (abstract)
- JAX FFI: https://docs.jax.dev/en/latest/changelog.html , https://github.com/jax-ml/jax/blob/main/docs/ffi.md ;
  IRFFT lowering commit https://github.com/jax-ml/jax/commit/2fe0e2b
- cuFFT (LTO callbacks, advanced layout, 16-bit): https://docs.nvidia.com/cuda/cufft/index.html ; cuFFTDx:
  https://docs.nvidia.com/cuda/cufftdx/examples.html
- VkFFT / pyvkfft: https://github.com/DTolm/VkFFT , https://raw.githubusercontent.com/vincefn/pyvkfft/master/README.rst
- CuPy cuFFT callbacks: https://docs.cupy.dev/en/stable/user_guide/fft.html
- jaxDecomp (permissive JAX custom-call example): https://github.com/DifferentiableUniverseInitiative/jaxDecomp
- Green contexts / GPU interference: https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/green-contexts.html ,
  https://anakli.inf.ethz.ch/papers/gpu_interference_socc25.pdf
- Dedalus GPU PR: https://github.com/DedalusProject/dedalus/pull/342 ; GHOST GPU: https://arxiv.org/abs/1808.01309 ;
  cuPSS: https://github.com/fcaballerop/cuPSS ; dnsjax: https://github.com/gokhanyalniz/dnsjax

The full per-finding evidence lists, with what each source shows and how much of it was read, are in
`research_findings.json`.
