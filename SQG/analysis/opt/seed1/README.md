# Seed 1: a lean float64 A-SQG solver step in plain JAX (same mathematics)

**Status:** written on the login node, not yet run on a GPU. Only the stdlib ring test (`test_lean_ring.py`) has
run. Every number below is an estimate until the job reports.

## Files

| file | what |
|---|---|
| `lean_solver.py` | `make_advance(GD, nsteps, ncol=..., unroll1=False, split=False)`, a drop-in replacement for o1.2's `advance`. It takes and returns pyqg_jax's `AB3State`. It also holds `preconditions(GD)` and `build_consts(GD, ncol)`. |
| `run_seed1.py` | Entry script. It runs the gate, then times each variant against production and writes `seed1.json`. |
| `test_lean_ring.py` | Stdlib unit test of the 3-slot history ring (no JAX). Prints `LEAN_RING_TEST_OK`. A mutation check showed it fails on wrong rotations. |

Job command, from the repo root, after the SQG env bootstrap block (`SQG_ASQG_PY`, `SQG_TRAINING_ROOT`; one GPU):

```bash
"$PY" SQG/analysis/opt/seed1/run_seed1.py --out "$OUT/seed1"
```

It takes about 10 minutes. `--budget-min 14` shrinks the optional phases to fit. No install and no build are needed.

The run prints these tokens:
- `SEED1_GATE_OK <v>` or `ERROR SEED1_GATE_FAIL <v> <reason>`
- `SEED1_BENCH <v> ms_ref= ms_cand= speedup= gate=`
- `SEED1_OPTSTEP`
- `SEED1_LONG`
- `SEED1_DONE`

## Technique

The step is `A-SQG_data.inner` with the work removed that cannot change a value in this configuration, plus layout
changes that XLA can use. `A-SQG_data.py` is imported unmodified. `preconditions` re-checks every claim on the GPU
and refuses to build if one fails.

| change | why the values stay the same | what it saves (job A/E nsys, job F ncu) |
|---|---|---|
| pyqg's filter `filtr` is not applied | `filterfac = 0` makes it exactly 1 (checked), and x·1 = x | the per-element f64 `sqrt` + `exp` in the AB3 fusion |
| the background-gradient term −i k_x Q_y ψ̂ is dropped | Q_y = β = 0 (checked), so the term only added ±0 | one 67 MB read (`wv2i`) and the `sqrt` |
| F̂₀ is added from its 5 nonzero k_x columns, zero-padded inside the kernel | the padded array equals `F0_hat` bit for bit (checked) | one dense 134 MB complex read per step (24 nonzero modes) |
| `decay` is multiplied as real f64, passed at run time | (a+ib)(d+0i) = (a+ib)d | 67 MB per step (XLA had folded it into a 134 MB complex constant) |
| the ψ̂ inversion factor (f₀/N_b)·k⁻¹ is evaluated once | same IEEE ops (exact integers, correctly rounded sqrt/div) | `loop_select_fusion` 0.07 ms + a sqrt per element |
| AB3 is fixed at level 2 | the same coefficients (job A: bitwise) | the per-step host sync and 3 tiny copies |
| the history is a 3-slot ring with the scan body unrolled ×3 | the same values, different buffers (`test_lean_ring.py`) | the 2 full-field rotation copies, **0.39 ms/step** |

There are three variants (all float64, all the same arithmetic):

- **`lean`**: the table above.
- **`lean_u1`**: `lean`, plus `x·cos(±0)` (exactly x) in each complex128 loop fusion. A fusion that contains a cosine is
  emitted by XLA with 1 element per thread instead of 4 (XLA's `MayPreventVectorization`).

  Why it should help: job F's Nsight Compute shows the AB3 fusion is **DRAM-bound, not FP64-bound**. It runs at 88% of
  DRAM peak, but reads **4.0 GB from DRAM for ≈1.0 GB of unique data**: 52% of its L2 sectors are excess and the L1
  hit rate is 10%. With 4 consecutive c128 elements per thread (64 B per operand), ~9 operands and ~110 k resident
  threads, about 60 MB of sectors are in flight. That is more than the 40 MB L2, so every 16 B access re-fetches a
  DRAM burst. The f32 build (32 B per operand) did not thrash, which is why it was 5.1× faster, not 2×. The f64
  products (`wrapped_multiply`) have 3 operands at 32 B and already move ideal bytes.
- **`lean_split`**: `lean`, plus an optimization barrier between the tendency T and the AB3 update. That gives two
  fusions with 3 and 6 operands. It is an independent route to less L2 pressure, in case the `cos` trigger does not
  take effect.

## Expected ms/step on an A100 (production 9.30)

These are estimates. Only the AB3 fusion's bytes (≈1.007 GB minimum) and the removed copies are firm.

| variant | estimate | reasoning |
|---|---|---|
| `lean` | ≈ 7.4 (6.8–7.9) | −0.39 copies, −0.07 select, −0.1 uv build. The AB3 fusion still has the 4-element pattern with 7 c128 + 1 f64 operands, ≈1.5–2.6 ms. |
| `lean_u1` | **≈ 6.0 (5.9–6.5)** | AB3 at ≈1.3 TB/s: 1.007 GB → 0.77 ms (now 3.28). uv build 0.47 GB → 0.36 ms (now 0.64). |
| `lean_split` | ≈ 6.9 (6.5–7.3) | T fusion 0.40 GB with few operands; update fusion 0.74 GB, partly thrashing. |

One optimizer step (480 steps + 12 `coarse`) would take ≈ 2.9 s with `lean_u1`, against 4.47 s.

**Floor at the JAX level:** cuFFT 3.06 + C2R input copies 0.59 + `scal` 0.59 + u·θ, v·θ products 0.62 = 4.86 ms per
step. These are inside XLA's FFT thunk or already at bandwidth, and no JAX-level change touches them. `lean_u1`'s
estimate is that floor plus ≈1.1 ms of minimal pointwise traffic. Going lower needs a custom FFT path or float32.

## Gate (fixed before any run; float64-only changes)

Test states:
- `random`: the production spin-up IC at AB3 level 2, where θ ≈ 1e-3 and advection is negligible. Jobs A and E timed
  this state.
- `synthetic`: a random-phase θ with |θ_k| ∝ k^(-4/3) exp(−(k/(KMAX/2))²) on 1 ≤ |k| ≤ KMAX. It is scaled to a
  spectral CFL (max|u|·KMAX·dt) of 0.3 and stepped to level 2 by the production step. Every term of the step is O(1)
  there. It is not an attractor state.

| check | requirement |
|---|---|
| **G1** (the policy gate) | After 1 step from each state, every value is finite and the relative L2 error of **each** AB3 field (θ̂ and both stored tendencies) vs `GD.inner` is ≤ **1e-12**. |
| **G2** (structural) | After the timed 40-step call from each state: finite, each field ≤ **1e-9**, t/tc identical, level 2. A wrong ring rotation is invisible to a 1-step test and gives O(1) errors here, while rounding stays near 1e-15. |
| **G3** | Finite after 480 steps wherever the reference is finite. |
| information only | Bitwise equality (signed zeros included) and values equality after 480 steps. Error at the 24 forced modes relative to \|F₀\|. After 4,000 steps from `synthetic`: relative L2 of o1.2's 256² `coarse()` field, the relative energy (0.5⟨θ²⟩) difference, and θ̂'s relative L2. |

Bitwise identity is plausible for `lean` and `lean_u1`, because the step keeps pyqg's operand order and complex
arithmetic. It is not guaranteed: FMA contraction and the sign of zeros may differ.

## Not verified (until the job runs)

- Nothing has been compiled. The JAX 0.4.28 APIs were checked by reading the installed sources:
  `optimization_barrier` is private there, and the code falls back to `jax._src.ad_checkpoint._optimization_barrier`.
- **The `lean_u1` trigger relies on XLA internals I recalled, not read:** that `MayPreventVectorization` covers `kCos`,
  and that the unroll is 4 when the element count is divisible by 4. The cause of the over-fetch is inferred from NCU
  counters. If the trigger does not fire, `lean_u1` ≈ `lean`. `hlo_40step_call.cosine` in the JSON shows whether the
  cosines exist; only a kernel profile shows the unroll.
- XLA must also (a) remove the ring's carry copies, (b) not hoist the forcing pad out of the loop, and (c) fuse as
  intended. The JSON reports `full_field_copy`, `pad` and fusion counts per variant. (b) is guarded by a carry-derived
  factor that is exactly 1.0.
- The synthetic state tests the arithmetic, not the climate. **Long-run statistics need a Midway twin run** from an
  archived checkpoint, production vs `lean_u1`, for ≥ 1–5 time units (50–250 blocks). Compare the SPE/VIE/ε/P time
  series, time-mean E(k), the umax/CFL distribution and the coarse-field PDFs and spectra. The pointwise divergence
  should grow no faster than a 1-ulp twin (`SQG/analysis/decorrelation.py`).
- Production use assumes AB3 level 2 at entry, which holds two steps after any (re)start. o1.2 is not edited here.
  Wiring it in is `advance = lean_solver.o12_advance(GD, STRIDE, unroll1=True)` in place of line 214, and needs the
  o1.2 smoke plus this gate.
- The ring adds one 134 MB carry buffer. `jax_memory_GiB` in the JSON records the peak (job A's reference: 4.49 GiB).
