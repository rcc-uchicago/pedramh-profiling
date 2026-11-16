"""test_lean_ring.py -- unit test of lean_solver.ring_advance with plain Python floats (no JAX, no GPU).

The 3-slot ring must give exactly the state and the two stored tendencies that pyqg_jax's AB3 history rotation
(`new_updates = (updates, updates_p)`) gives, for every step count, and must never read the third slot's initial
content (it is NaN here, so any read would show).
    python3 SQG/analysis/opt/seed1/test_lean_ring.py        -> LEAN_RING_TEST_OK
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lean_solver as LS  # noqa: E402

DT = 5e-6
C1, C2, C3 = (23 / 12) * DT, (-16 / 12) * DT, (5 / 12) * DT


def tendency(qh, tc):
    """Any deterministic, state-dependent tendency."""
    return math.sin(3.0 * qh) + 1e-3 * tc


def step(qh, Tp, Tpp, t, tc):
    T = tendency(qh, tc)
    return qh + C1 * T + C2 * Tp + C3 * Tpp, T, t + DT, tc + 1


def pyqg_like(qh, newest, prev, t, tc, nsteps):
    """The production bookkeeping: one AB3 step, then rotate the history tuple."""
    hist = (newest, prev)
    for _ in range(nsteps):
        T = tendency(qh, tc)
        qh = qh + C1 * T + C2 * hist[0] + C3 * hist[1]
        hist = (T, hist[0])
        t, tc = t + DT, tc + 1
    return qh, hist[0], hist[1], t, tc


def pyscan(body, init, xs, length):
    carry = init
    for _ in range(length):
        carry, _ = body(carry, None)
    return carry, None


def nan_like(_):
    return float("nan")


def main():
    for nsteps in range(0, 50):
        got = LS.ring_advance(step, 0.7, 0.11, -0.05, 0.0, 0, nsteps, pyscan, nan_like)
        want = pyqg_like(0.7, 0.11, -0.05, 0.0, 0, nsteps)
        if got != want:
            raise SystemExit(f"ERROR LEAN_RING_MISMATCH nsteps={nsteps}: ring {got} vs pyqg {want}")
    print("LEAN_RING_TEST_OK ring == pyqg history rotation for 0..49 steps, dead slot never read")


if __name__ == "__main__":
    main()
