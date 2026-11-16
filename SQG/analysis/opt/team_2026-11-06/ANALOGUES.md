# Online learning against DE solvers in other fields: what transfers to o1.2 (literature, 2026-11-06)

**Ask (user, 2026-11-06):** find analogues in other sciences that train ML models online against expensive
differential-equation solvers, for example cosmology, astronomy or applied physics. Any field with a similar focus
counts.

**Method.**
- **Search:** 14 web searches.
- **Primary sources opened** (marked ✔): Klöwer et al. 2022 (full text on PMC) and Meyer et al. 2023 (arXiv
  abstract).
- **Everything else** comes from search-result summaries. Treat it as a lead and open the paper before quoting a
  number from it.

## 1. Map: field → system → the idea that transfers

| field | system (solver) | what it does | o1.2 lever it informs |
|---|---|---|---|
| Weather/climate numerics | Klöwer et al. 2022 ✔, ShallowWaters.jl (Float16) | compensated time integration: one compensation array per prognostic variable | **= `lean32ek`** (independently derived; §2.1) |
| Weather/climate numerics | Paxton et al. 2022, Kimpson et al. 2023 (SPEEDY, Lorenz, shallow water) | stochastic rounding: zero-mean rounding, so a bias becomes a random walk | alternative to compensation; noise-vs-bias taxonomy (§2.1) |
| HPC surrogates | Melissa (Meyer et al. ICML 2023 ✔; SC'23 Reservoir) | ensemble runs stream to a data-parallel trainer, no disk; a buffer policy fights streaming bias | decoupled sims:trainer; buffer policy; **diversity is what pays** (§2.2) |
| CFD / turbulence (RL) | Relexi + FLEXI + SmartSim | solver ↔ learner over an in-memory database on HPC (PBS/Slurm) | decouple layout, orchestration (§2.2) |
| CFD in-situ training | SmartSim/SmartRedis in-situ framework (arXiv 2306.12900) | asynchronous in-memory training while the CFD runs; only the data loader changes | o1.2's host FIFO is the same pattern |
| Ocean/climate | Rasp 2020 "coupled online learning"; SmartSim + MOM6 (Partee et al. 2022, inference only) | pretrain offline, then keep training on the states the *ML model itself* visits | the one thing offline data cannot give (§2.5) |
| Geophysical turbulence (QG!) | Frezat et al. 2022 (JAMES); Yan et al. 2024 (two-layer QG, **pyqg-JAX**) | "a posteriori" learning: the loss is differentiated through several solver steps; a priori models went unstable in rollouts, a posteriori models stayed stable | rollout-bias concern; the same solver library as o1.2 (§2.5) |
| Neural PDE solvers | AL4PDE (Musekamp et al. ICLR 2025); Breed (MelissaDL); OGAS (arXiv 2606.09949) | active learning chooses which ICs/parameters to simulate; up to 71% lower error, or ~4× fewer samples | which checkpoints/segments to integrate (§2.3) |
| Materials / MD | FLARE (Vandermause et al. 2020), FLARE++ | the GP's epistemic uncertainty triggers a DFT call; ~100 DFT calls build a force field | uncertainty-triggered solver calls (§2.3) |
| Fusion plasma | ADEPT (Nucl. Fusion 2024); MIT/UKAEA simulator-in-the-loop AL (PoP 2025); TGLF-SINN/WINN | uncertainty-driven acquisition of gyrokinetic labels: ~10% of random sampling's compute; 75% less data | same (§2.3) |
| Cosmology | CONNECT (Nygaard et al. JCAP 2023); OLÉ (arXiv 2503.13183); Rogers et al. Ly-α Bayesian emulator optimisation; Bartlett et al. 2025 simulation budgeting | the emulator is trained alongside the sampler, and new solver calls go where the posterior or the error is; sim-count forecasts | trajectory-time scaling curve L11 (§2.4) |
| Gravitational waves | DINGO / GNPE (Dax et al. PRL 2021) | the *expensive* waveforms are precomputed once; the *cheap* extrinsic parameters and noise are re-drawn every epoch; an exact time-translation symmetry is factored out via proxies | tee + replay (L2), symmetry conditioning (§2.4) |
| Combustion | on-the-fly ANN chemistry in DNS (Combust. Flame ~2021); Neural-ISAM (arXiv 2605.10028); ISAT | the network learns the run's own state distribution while a reference solver stays in the loop for unseen states | a solver fallback / error trigger (§2.3) |
| RL with GPU physics | IMPALA, Parallel Q-Learning (ICML 2023, Isaac Gym), FastTD3 | decoupled actors/learners; the update-to-data (replay) ratio trades sample efficiency against wall-clock | S / reuse knob (L8), decoupling (§2.2) |

## 2. The five lessons, with what we could test

### 2.1 Low-precision solvers: two other groups hit our "mechanism B"

- **Klöwer et al. (✔, PMC9287017).**
  - The problem, in their words: "tendencies are often several times smaller than the prognostic variables …
    small increments can only be poorly resolved in low precision".
  - The fix: a compensation variable per prognostic variable (c_u, c_v, c_η) carries the lost bits to the next
    step. That is `lean32ek`'s hi + lo.
  - Without it, the Float16 error vs Float64 grows 100-fold in 150 days. With it, the error matches mixed
    precision.
  - Speed: Float16 3.8×, compensated 3.6×, Float16/32 mixed ≈ 2.75×. Compensation beat mixed precision there, the
    same ordering as our `lean32ek` 2.77× vs `mixed` 1.96× on A100.
  - *So `lean32ek` is not exotic: it is the published fix for exactly this failure.*
- **Stochastic rounding (Paxton 2022; Kimpson 2023).**
  - Round-to-nearest "may contribute to systemic biases". Stochastic rounding (SR) is unbiased, E[SR(x)] = x, so a
    systematic error becomes a zero-mean random walk.
  - Kimpson: 100-yr SPEEDY, half-precision mean bias in global mean temperature −3.5e-2 K, and +1.8e-2 K with SR.
- **For us** (our estimate, not run):
  - SR on `lean32`'s state update would turn mechanism B from **bias** into **noise**.
  - Per step the rms error is ≈ ulp/√6 ≈ 3.5e-8 of θ. After 4,000 steps that is √4000 × 3.5e-8 ≈ 2.2e-6 of θ.
  - At k = 1–2, Δ ≈ 3.9e-2 of θ, so e_Δ ≈ 6e-5. That is 60× better than `lean32` (3.6e-3) but ≈ 3× worse than
    `lean32ek` (2.2e-5). Overall it would be ≈ `lean32ek` (≈ 1e-5), where the float32 tendency dominates.
  - It saves `lean32ek`'s extra 67 MB array, but needs random bits (threefry in the fusion), so it is a similar
    cost class.
  - **Recommendation:** `lean32ek` stays the better candidate (u² vs u). An SR arm is worth adding to the "delta
    v2" job only as a calibrated *noise-only* control for the noise-vs-bias taxonomy (§3).

### 2.2 Online pays through *diversity*, and o1.2 does not buy any

- **Melissa (✔ abstract).** "exposing deep surrogate models to more dataset diversity, up to hundreds of GB, can
  increase model generalization": +68% FC, +16% FNO, +7% MP-PDE.
- **Their gain comes from new solver runs** (new parameters and ICs). o1.2 re-integrates the 3,500 archived
  trajectories bitwise (its assert), so its online data are the offline data at 2e-4 instead of 0.02 spacing.
- **Our numbers say the density adds ≈ 1.0× effective samples.** The m16 increment's C(0.02) = −0.066, and its
  integral time is 0.0096 time units (`decorrelation_multigrid_summary.json`).
- **Transfer:** spend the solver on **new trajectory time** (team item L11), not on re-generating the archive.
  Melissa's buffer design (FIFO vs a reservoir that mitigates "the inherent bias of streaming learning") is
  relevant if new-trajectory streaming is adopted. o1.2's FIFO is the plain-queue mode Melissa improves on.
- **Systems precedent for the decoupled layout we measured** (Part IX: equal GPU-seconds):
  - Relexi / SmartSim run simulators and learner as separate jobs over an in-memory store, on PBS and Slurm.
  - IMPALA / PQL do the same in RL, with a single learner fed by many actors.

### 2.3 Spend the solver where the model is uncertain

- **The pattern recurs everywhere:**
  - FLARE calls DFT only when the GP's epistemic uncertainty exceeds ~2× its noise.
  - ADEPT gets QuaLiKiz labels with ~10% of random sampling's compute.
  - AL4PDE: LCMD/SBAL/BAIT beat random by up to 71% error.
  - CONNECT / OLÉ / Rogers concentrate CLASS / N-body calls where the posterior or the error is.
  - Combustion keeps the reference integrator for out-of-distribution states.
- **The transfer to o1.2 is not direct; three cautions:**
  - **What can be chosen.** The pool is o1.2's choice of *which checkpoints / segments* to integrate (3,500
    candidates).
  - **Selection changes the training distribution.** o1.2 targets the attractor's invariant measure, so selecting
    by model error shifts it. Importance weights, or an unbiased val set, are needed. That makes it a science
    decision.
  - **Use epistemic uncertainty only.** The team's floor analysis puts ≈ 50% of val MSE in irreducible aliasing
    noise (Var(y_alias) ≈ 0.0021–0.0024), and that noise sits mostly at coarse |k| ≥ 64. FLARE and ADEPT use GP
    variance or deep ensembles for exactly this reason.
- **Possible test:** score the 3,500 checkpoints' segments by a small deep ensemble's disagreement (the epistemic
  part) on stored frames. If disagreement is nearly uniform across checkpoints, active selection has nothing to
  exploit. Cheap, offline, decisive.

### 2.4 Store the expensive part once, augment the cheap part every epoch (DINGO)

- **DINGO.** It precomputes 2.5e7 intrinsic waveforms (expensive: relativity solves) once. Extrinsic parameters,
  detector projection, noise and the noise spectrum are re-drawn every epoch. The exact symmetry (time
  translation) is handled by GNPE: it conditions on a proxy instead of hoping the network learns it.
- **For o1.2, the expensive part is the 4096² integration:**
  - Store its 256² stream once (team L2: 1.31 GB per segment, ≈ 4.3 TB).
  - Replay it in ≈ 21 h per later run.
- **Symmetry.**
  - The team found the forced A-SQG has no exact symmetry with the fixed random-phase forcing ring.
  - DINGO's route is to *condition* on the symmetry-breaking variable. For SQG that means feeding the forcing
    pattern as an input channel, then augmenting with D4 × translations, (θ, F) → (−θ∘R, −F∘R) for reflections.
  - That changes the architecture and the data: jesswan / Weidong's call.
- **Budget planning.** Cosmology forecasts how many simulations an emulator needs (Bartlett et al.: < 225
  N-body runs for an 8-parameter model). That is the same question as the team's trajectory-time scaling curve:
  how many time units of trajectory before val stops improving.

### 2.5 The one benefit only an online loop can give: the model's own states

- **Rasp's "coupled online learning".** Pretrain offline. Then run the ML model coupled, and keep training on
  high-resolution targets *for the states the ML model produces*. These lie off the training manifold, which is
  where instabilities and biases come from. DAgger-like.
- **Frezat et al. (QG turbulence).** A posteriori training, with the loss through several solver steps, gave
  stable rollouts where a priori training diverged. Yan et al. did it on two-layer QG with **pyqg-JAX**, the
  library o1.2's solver uses.
- **For o1.2:**
  - o1.2's "online" gives *more of the same distribution*, which the team argues is ≈ worthless.
  - The online loop that adds information is **model-visited states**, or a multi-step rollout loss.
  - **Obstacle:** the UViT emits 256², and the solver needs a 4096² state. Restarting the solver from a model
    state needs a small-scale re-spin-up, which changes the trajectory distribution.
  - A multi-step (unrolled) rollout loss on stored or replayed frames needs no solver at all. It targets exactly
    the low-k rollout bias that `MATH_delta.md` §9 warns about.
  - Both change the training objective: science owners' call.

### 2.6 Molecular dynamics on GPUs: the same precision problem, solved the same way (added 2026-11-10)

**Ask (user, 2026-11-10):** check MD as an analogue, since it is GPU-heavy with similar equations. The 11-06 pass had MD
only for active learning (FLARE).

**The shared problem.** MD adds tiny increments to a large state every step (x += v·Δt), the same as our mechanism B.
**What MD's GPU codes do** (each quote checked in the code's own documentation on 2026-11-10):

| code | computed in single precision | kept in higher precision | what the code says |
|---|---|---|---|
| AMBER pmemd.cuda, SPFP (default) | "single precision for calculation" | "fixed (integer) precision for accumulation" | "energy conservation is equivalent to the full double precision code". The all-single SPSP is deprecated. Bitwise reproducibility is used to validate cards. |
| OpenMM, `Precision=mixed` | "forces are computed in single precision" | "integration is done in double precision" | "much better energy conservation with only a slight decrease in speed". A "deterministic forces" option fixes the summation order. |
| GROMACS, mixed (default) | "particle coordinates, velocities and forces" | "critical variables", e.g. the virial | Double is needed for "long-term energy conservation, especially for large systems". Mixed-precision trajectories diverge faster, since MD is chaotic. |
| LAMMPS GPU package, mixed | "pairwise forces" | "accumulated into double-precision force vectors" | The docs recommend checking that single or mixed is accurate enough. |

**How it maps onto ours:**
- **Ours `mixed`** (float32 FFTs, float64 state) is OpenMM's and LAMMPS's mixed mode.
- **`lean32ek`** keeps the tendencies (our "forces") in float32 and makes the *accumulation* exact by compensation
  (hi + lo). AMBER's SPFP does the same with 64-bit fixed point.
  - MD found accumulation-only precision as good as double for energy conservation.
  - We found `lean32ek` equal to `mixed` at the 0.02 lag, and faster (2.77× vs 1.96×).
- **Plain float32 (`lean32`)** is AMBER's SPSP or OpenMM's single. Both are fastest and least accurate, and AMBER
  deprecated SPSP. That matches our rejection of plain float32.
- **Fixed point vs compensation, for us.** Fixed point needs a fixed scale. Our spectral state spans many orders of
  magnitude between k = 1 and k = 1,365, so it would need a scale per mode. TwoSum adapts automatically, so
  compensation is the more natural choice for a spectral solver. This is our reasoning, not tested.

**What MD does that we have not yet done.** MD accepts a precision model on **long-run energy conservation** (NVE drift
against double precision), not on short paired errors. Our system is forced and dissipative, so nothing is conserved.
The analogue is the **energy budget in statistical steady state**: injection P = ε_ν + ε_μ, with the mean energies
SPE / VIE.
- `A-SQG_data.py`'s `diagnostics()` already returns SPE, VIE, eps_nu, eps_mu and P.
- A long `lean32ek_fast` run against `lean_u1` (≥ 10 time units from the archived states), comparing those budget terms
  within their statistical error, is MD's acceptance test in our terms. It is the solver-side half of gate C.

**Sources:**
- Le Grand, S., Götz, A. W., & Walker, R. C. (2013). SPFP: Speed without compromise -- A mixed precision model for
  GPU accelerated molecular dynamics simulations. *Comput. Phys. Commun.*, 184(2), 374–380.
  https://doi.org/10.1016/j.cpc.2012.09.022 (citation from ambermd.org/GPUSupport.php; the model's description from
  ambermd.org/gpus12/)
- Eastman, P., et al. (2017). OpenMM 7: Rapid development of high performance algorithms for molecular dynamics.
  *PLOS Comput. Biol.*, 13(7), e1005659. https://doi.org/10.1371/journal.pcbi.1005659 (precision modes from
  docs.openmm.org, platform-specific properties)
- GROMACS reference manual, "Mixed or Double precision": https://manual.gromacs.org/current/reference-manual/definitions.html
- Brown, W. M., Wang, P., Plimpton, S. J., & Tharrington, A. N. (2011). Implementing molecular dynamics on hybrid high
  performance computers -- short range forces. *Comput. Phys. Commun.*, 182(4), 898–911.
  https://doi.org/10.1016/j.cpc.2010.12.021 (precision modes from docs.lammps.org/Speed_gpu.html)
- Abstracts were not available through the APIs used. The quotes come from the codes' own documentation pages, so the
  papers' contents beyond the titles were not read.

## 3. What this adds to the team's plan (for the PS report)

1. **Cite Klöwer et al. in the precision memo to jesswan / Weidong.** Compensated time integration is the
   established fix for small-tendency rounding. `lean32ek` is that method. Its residual error is the float32
   tendency, not state rounding.
2. **Delta v2 job:** add a stochastic-rounding `lean32` arm as a *noise-only* calibration control. Predicted:
   k1–2 e_Δ ≈ 3e-5–1e-4 with zero signed mean; overall ≈ `lean32ek`.
3. **Before any active-sampling idea:** run the ensemble-disagreement scan over checkpoint segments (offline,
   stored frames). It decides whether "where to simulate" has any signal beyond the aliasing noise.
4. **The online loop worth paying for is on-policy:** model-visited states (Rasp) or an unrolled rollout loss
   (Frezat). Neither is "more of the archive at higher density". Raise it with the owners as a research direction.
5. **If new trajectories are streamed:** use a reservoir-style buffer (Melissa SC'23), not a pure FIFO.

## Sources

*The three low-precision entries below were re-checked against Crossref on 2026-11-10, and Klöwer et al. also against
its full text. The first version of this list had abbreviated or slightly wrong titles for all three.*
- Klöwer, M., Hatfield, S., Croci, M., Düben, P. D., & Palmer, T. N. (2022). Fluid simulations accelerated with 16
  bits: Approaching 4x speedup on A64FX by squeezing ShallowWaters.jl into Float16. *Journal of Advances in
  Modeling Earth Systems*, 14(2), e2021MS002684. https://doi.org/10.1029/2021MS002684 (open access:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC9287017/) ✔
  - Checked in the full text on 2026-11-10:
    - one compensation variable per prognostic variable (c_u, c_v, c_η);
    - uncompensated Float16 error "exponentially increases 100-fold in the first 150 days";
    - compensated error "matches well with the error growth of mixed-precision";
    - speed on A64FX: Float16 3.8×, compensated 3.6×, mixed ≈ 2.75×;
    - the method is credited to Kahan (1965), Møller (1965) and Higham (1993).
- Paxton, E. A., Chantry, M., Klöwer, M., Saffin, L., & Palmer, T. (2022). Climate modeling in low precision:
  Effects of both deterministic and stochastic rounding. *Journal of Climate*, 35(4), 1215–1229.
  https://doi.org/10.1175/JCLI-D-21-0343.1
- Kimpson, T., Paxton, E. A., Chantry, M., & Palmer, T. (2023). Climate-change modelling at reduced floating-point
  precision with stochastic rounding. *Quarterly Journal of the Royal Meteorological Society*, 149(752), 843–855.
  https://doi.org/10.1002/qj.4435
- The error-free transformations used by `lean32ek` (`f32_update.compensated_update`), re-checked against Crossref on
  2026-11-10:
  - Kahan, W. (1965). Pracniques: Further remarks on reducing truncation errors. *Communications of the ACM*, 8(1),
    40. https://doi.org/10.1145/363707.363723 (compensated summation)
  - Møller, O. (1965). Quasi double-precision in floating point addition. *BIT*, 5(1), 37–50.
    https://doi.org/10.1007/BF01975722 (TwoSum)
  - Dekker, T. J. (1971). A floating-point technique for extending the available precision. *Numerische
    Mathematik*, 18(3), 224–242. https://doi.org/10.1007/BF01397083 (Fast2Sum)
  - Knuth, D. E. (1997). *The Art of Computer Programming, Vol. 2: Seminumerical Algorithms* (3rd ed.), §4.2.2.
    Addison-Wesley. (TwoSum. A book, no DOI)
  - Higham, N. J. (1993). The accuracy of floating point summation. *SIAM Journal on Scientific Computing*, 14(4),
    783–799. https://doi.org/10.1137/0914050
- Meyer et al. (2023) ICML, *Training deep surrogate models with large scale online learning*:
  https://arxiv.org/abs/2306.16133 ✔; SC'23 Reservoir: https://arxiv.org/abs/2309.16743; Breed:
  https://arxiv.org/abs/2410.05860; OGAS: https://arxiv.org/html/2606.09949v1
- Relexi: https://github.com/flexi-framework/relexi; Kurz et al. (2023) IJHFF, *Deep RL for turbulence modeling in
  LES*: https://www.researchgate.net/publication/366485150
- SmartSim in-situ CFD training: https://arxiv.org/pdf/2306.12900; Partee et al. (2022) SmartSim + MOM6:
  https://arxiv.org/abs/2104.09355
- Rasp (2020) GMD, *Coupled online learning*: https://gmd.copernicus.org/articles/13/2185/2020/
- Frezat et al. (2022) JAMES, *A posteriori learning for QG turbulence parametrization*:
  https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2022MS003124; Yan et al. (2024), two-layer QG with
  pyqg-JAX: https://arxiv.org/pdf/2411.14106
- Musekamp et al. (2025) ICLR, *Active learning for neural PDE solvers* (AL4PDE): https://arxiv.org/abs/2408.01536
- Vandermause et al. (2020) npj Comput. Mater., FLARE: https://www.nature.com/articles/s41524-020-0283-z
- ADEPT (2024) Nucl. Fusion: https://iopscience.iop.org/article/10.1088/1741-4326/ad240d; simulator-in-the-loop
  AL (2025) PoP: https://arxiv.org/abs/2507.15976; TGLF-WINN: https://arxiv.org/abs/2509.07024
- CONNECT (Nygaard et al. 2023) JCAP: https://arxiv.org/abs/2205.15726; OLÉ: https://arxiv.org/pdf/2503.13183;
  Rogers et al., Ly-α emulator optimisation: https://arxiv.org/html/1812.04631; Bartlett et al., simulation
  budgeting: https://arxiv.org/pdf/2510.13962
- Dax et al. (2021) PRL, DINGO: https://link.aps.org/doi/10.1103/PhysRevLett.127.241103; GNPE docs:
  https://dingo-gw.readthedocs.io/en/latest/gnpe.html
- On-the-fly ANN chemistry in DNS: https://www.sciencedirect.com/science/article/abs/pii/S0010218020305939;
  Neural-ISAM: https://arxiv.org/abs/2605.10028
- IMPALA: https://proceedings.mlr.press/v80/espeholt18a/espeholt18a.pdf; Parallel Q-Learning (ICML 2023):
  https://proceedings.mlr.press/v202/li23f/li23f.pdf; FastTD3: https://arxiv.org/abs/2505.22642
