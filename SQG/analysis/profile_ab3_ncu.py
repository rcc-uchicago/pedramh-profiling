"""profile_ab3_ncu.py -- the program Nsight Compute launches for the kernel-level profile of the A-SQG solver step
(profile_ab3_ncu_polaris.pbs). It is job A's harness without the measurements: build the production solver, warm it
up, then run a few jitted 40-step calls between cudaProfilerStart and cudaProfilerStop so that ncu, started with
--profile-from-start off, sees only steady-state launches of the kernel its --kernel-name filter selects.

  ncu --profile-from-start off --kernel-name regex:^loop_add_multiply_fusion_1 --launch-skip 3 --launch-count 3 \
      --set full -o <rep> python profile_ab3_ncu.py --variant ref --calls 2 --out <dir>

The solver, its import (SQG_ASQG_PY / SQG_TRAINING_ROOT), the start state (SQG_INIT=random -> the production spin-up
stepped to AB3 level 2; else the archived checkpoint) and the `ref` / `fixed` variants are profile_solver.py's, reused
by import (importing it runs nothing; its main() is guarded) or copied line for line where they live inside its main().
Compile + warm-up = 2 optimizer-equivalents (12 x 40 steps + coarse each, as job A's opt_step) BEFORE the profiled
range, so XLA autotuning, the AB3 level switch and the first-call allocations are never profiled. Inside the range:
--calls jitted 40-step calls (no coarse), then block_until_ready. Under ncu every launch is serialised and the
selected ones are replayed tens of times, so the wall time printed here is not a timing. Prints NCU_HARNESS_OK.
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import profile_solver as PS   # noqa: E402  -- import_asqg, patch_training_root, STRIDE, CALLS, M, CKPT_NAME

STRIDE, CALLS, M = PS.STRIDE, PS.CALLS, PS.M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["ref", "fixed"], default="ref")
    ap.add_argument("--calls", type=int, default=2, help="jitted 40-step calls inside the profiled range")
    ap.add_argument("--warm", type=int, default=2, help="optimizer-equivalents (12 calls + coarse) before the range")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    t_import = time.time()
    GD = PS.import_asqg()
    import jax
    import jax.numpy as jnp
    from jax import lax
    from pyqg_jax import steppers as pst
    t_import = time.time() - t_import

    # ---- start state: as profile_solver.py main() ----------------------------------------------------------------
    N = GD.N
    dev = jax.devices()[0]
    template = GD.sqg.init(GD.p, jax.random.key(GD.SEED_INITIAL_CONDITION))   # stepper-state template, as GD.main
    init = os.environ.get("SQG_INIT", "checkpoint")
    if init == "random":
        # DT is fixed and the CFL checks are host-side, so every step runs the same 4096^2 FFTs whatever the field;
        # the only state that changes which code runs is the AB3 level, 2 after two steps (PROFILING_PROMPT.md §2b).
        ckpt = None
        st0 = GD.advance(template, 2)
    elif init == "checkpoint":
        ckpt = GD.CHECKPOINT_DIR / PS.CKPT_NAME
        st0, _ = GD.load_checkpoint(template, ckpt)
    else:
        raise ValueError(f"SQG_INIT={init!r}: expected random or checkpoint")
    assert int(st0._ablevel) == 2, f"checkpoint AB level {int(st0._ablevel)} != 2: the fixed variant is not valid"

    # ---- variants: as profile_solver.py main() (ref = production; fixed = AB3 level pinned at 2) ------------------
    dt = GD.sm.stepper.dt
    c1, c2, c3 = (23 / 12) * dt, (-16 / 12) * dt, (5 / 12) * dt      # pyqg_jax AB3Stepper level-2 branch, verbatim

    def ab3_level2(ss, updates):
        """pyqg_jax AB3Stepper.apply_updates with the lax.switch replaced by its level-2 branch."""
        up, upp = ss._updates
        new_state = pst._nostep_tree_map(lambda v, u, u_p, u_pp: v + (c1 * u) + (c2 * u_p) + (c3 * u_pp),
                                         ss.state, updates, up, upp)
        return pst.AB3State(state=new_state, t=ss.t + jnp.float32(dt), tc=ss.tc + 1, _ablevel=jnp.uint8(2),
                            _updates=(pst._map_state_remove_nostep(updates), up))

    def inner_fixed(state, _):
        """GD.inner with ab3_level2 instead of sm.stepper.apply_updates."""
        updates = GD.sm.model.get_updates(state.state)
        updates = updates.update(qh=updates.qh + GD.F0_hat)
        state = ab3_level2(state, updates)
        qh = GD.sm.model.postprocess_state(state.state).qh * GD.decay
        return state.update(state=state.state.update(qh=qh)), None

    def make(inner):
        return jax.jit(lambda st: lax.scan(inner, st, None, length=STRIDE)[0])

    adv = make({"ref": GD.inner, "fixed": inner_fixed}[args.variant])
    coarse = jax.jit(lambda st: jnp.fft.irfft2(st.state.qh[0], s=(N, N))[::M, ::M].astype(jnp.float32))
    block = jax.block_until_ready

    def opt_step(st):
        """Production pattern of one optimizer step: 12 x (40 steps, coarse frame)."""
        frames = []
        for _ in range(CALLS):
            st = adv(st)
            frames.append(coarse(st))
        return st, frames

    # ---- compile + warm (outside the profiled range), then the profiled calls -----------------------------------
    import torch                                        # only for cudaProfilerStart/Stop, as profile_solver --mode nsys
    st = jax.tree_util.tree_map(lambda x: jnp.array(x, copy=True), st0)
    t_warm = time.time()
    for _ in range(args.warm):
        st, fr = opt_step(st)
        block((st, fr))
    t_warm = time.time() - t_warm
    torch.cuda.profiler.start()
    t_prof = time.time()
    for _ in range(args.calls):
        st = adv(st)
    block(st)
    t_prof = time.time() - t_prof
    torch.cuda.profiler.stop()

    info = {"device": str(dev), "device_kind": dev.device_kind, "jax": jax.__version__, "init": init,
            "checkpoint": None if ckpt is None else str(ckpt), "variant": args.variant, "stride": STRIDE,
            "warm_optsteps": args.warm, "profiled_calls": args.calls, "profiled_steps": args.calls * STRIDE,
            "import_s": round(t_import, 2), "warm_s": round(t_warm, 2), "profiled_range_s_under_profiler": round(t_prof, 2)}
    with open(os.path.join(args.out, f"ncu_harness_{args.variant}.json"), "w") as fh:
        json.dump(info, fh, indent=1)
    print(f"device {dev.device_kind}, jax {jax.__version__}, init {init}, import {info['import_s']} s, "
          f"warm {args.warm} optsteps {info['warm_s']} s, profiled range {info['profiled_range_s_under_profiler']} s")
    print(f"NCU_HARNESS_OK {args.variant} {args.calls}")


if __name__ == "__main__":
    main()
