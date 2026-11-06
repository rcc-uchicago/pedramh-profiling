"""Standard-library tests of dt_ladder.py's arithmetic (runs on a login node: python3 test_dt_ladder.py).

1. Exact (fractions): uniform spacing gives pyqg_jax's (23/12, -16/12, 5/12) h; the weights integrate every quadratic
   exactly for arbitrary spacings, and not a cubic.
2. The scheme production uses, y' = -r y + g(y, t) as y' = (y + sum_j b_j g_j) exp(-r h), on a scalar ODE, after a
   switch from spacing h0 to h with dt_ladder's transition weights:
   - the order ratio R = D(h0/2) / D(4 h0/3), D(h) = y_h(T) - y_h0(T), is -0.64 (third order, AB3) when r = 0 and
     -1.50 (first order: the integrating factor damps the whole increment, the splitting error) when r h is the
     dominant error. These are the two values the taxonomy job's pre-registered band [-1.6, -0.55] brackets;
   - the variable-step transition is much more accurate than reusing uniform weights at the switch.
Prints DT_LADDER_TEST_OK.
"""
import math
import os
import sys
from fractions import Fraction as Fr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dt_ladder as DL                                               # noqa: E402


def test_exact():
    h = Fr(1, 7)
    assert DL.ab3_coeffs(h, h, h) == (Fr(23, 12) * h, Fr(-16, 12) * h, Fr(5, 12) * h)
    for h, h1, h2 in [(Fr(4, 3), Fr(1), Fr(1)), (Fr(1, 2), Fr(1), Fr(1)), (Fr(4, 3), Fr(4, 3), Fr(1)),
                      (Fr(3, 5), Fr(2, 7), Fr(9, 11))]:
        b = DL.ab3_coeffs(h, h1, h2)
        nodes = (Fr(0), -h1, -(h1 + h2))
        for a, bb, c in [(1, 0, 0), (0, 1, 0), (0, 0, 1), (Fr(2, 3), Fr(-5), Fr(7, 2))]:
            f = [a + bb * t + c * t * t for t in nodes]
            exact = a * h + bb * h * h / 2 + c * h * h * h / 3
            assert sum(w * v for w, v in zip(b, f)) == exact, (h, h1, h2)
        cubic = sum(w * t ** 3 for w, t in zip(b, nodes))
        assert cubic != h ** 4 / 4
    (t1, t2) = DL.transition(Fr(4, 3), Fr(1))
    assert t1 == DL.ab3_coeffs(Fr(4, 3), 1, 1) and t2 == DL.ab3_coeffs(Fr(4, 3), Fr(4, 3), 1)
    assert DL.steps_per_lag(30) == 3000 and DL.steps_per_lag(32) == 3200 and DL.steps_per_lag(80) == 8000
    assert DL.steps_per_lag(40) == 4000


def g_nl(y, t):
    return -0.5 * y * y + math.cos(3.0 * t)


def rk4_full(r, g, y, t, h):
    f = lambda yy, tt: -r * yy + g(yy, tt)                           # noqa: E731
    k1 = f(y, t)
    k2 = f(y + 0.5 * h * k1, t + 0.5 * h)
    k3 = f(y + 0.5 * h * k2, t + 0.5 * h)
    k4 = f(y + h * k3, t + h)
    return y + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0


def reference(r, g, y0, t0, t1, n=200000):
    h = (t1 - t0) / n
    y, t = y0, t0
    for _ in range(n):
        y = rk4_full(r, g, y, t, h)
        t += h
    return y


T_HIST0, Y_HIST0 = -0.05, 0.3                                        # the trajectory every run's history lies on


def y_at(r, g, t):
    """y(t) on the reference trajectory through (T_HIST0, Y_HIST0), by fine RK4."""
    return reference(r, g, Y_HIST0, T_HIST0, t, max(1, int(round((t - T_HIST0) / 2.5e-6))))


def run_switched(r, g, h0, h, t_end, naive=False):
    """History y(-2 h0), y(-h0), y(0) on the reference trajectory, then steps of h to t_end with the production
    scheme: dt_ladder's transition weights (or, naive=True, uniform weights at h on the h0-spaced history)."""
    y_m2, y_m1, y0 = y_at(r, g, -2 * h0), y_at(r, g, -h0), y_at(r, g, 0.0)
    hist = [g(y0, 0.0), g(y_m1, -h0), g(y_m2, -2 * h0)]                # newest first
    n = int(round(t_end / h))
    if naive:
        w = [DL.ab3_coeffs(h, h, h)] * 2
    else:
        w = DL.transition(h, h0) if h != h0 else [DL.ab3_coeffs(h, h, h)] * 2
    y, t = y0, 0.0
    d = math.exp(-r * h)
    for i in range(n):
        b = w[i] if i < 2 else DL.ab3_coeffs(h, h, h)
        y = (y + b[0] * hist[0] + b[1] * hist[1] + b[2] * hist[2]) * d
        t = (i + 1) * h
        hist = [g(y, t), hist[0], hist[1]]
    return y


def test_order_ratio():
    h0, t_end = 0.01, 1.0
    # r = 0: AB3 truncation only -> third order
    yb = run_switched(0.0, g_nl, h0, h0, t_end)
    d_half = run_switched(0.0, g_nl, h0, h0 / 2, t_end) - yb
    d_big = run_switched(0.0, g_nl, h0, h0 * 4 / 3, t_end) - yb
    r3 = d_half / d_big
    # r h dominant: the integrating factor's splitting error -> first order (forcing only, linear)
    g_lin = lambda y, t: math.cos(3.0 * t)                           # noqa: E731
    yb = run_switched(5.0, g_lin, h0, h0, t_end)
    d_half = run_switched(5.0, g_lin, h0, h0 / 2, t_end) - yb
    d_big = run_switched(5.0, g_lin, h0, h0 * 4 / 3, t_end) - yb
    r1 = d_half / d_big
    print(f"order ratio D(h0/2)/D(4h0/3): AB3 only {r3:+.3f} (theory -0.639); splitting-dominated {r1:+.3f} "
          f"(theory -1.500)")
    assert -0.70 <= r3 <= -0.58, r3
    assert -1.60 <= r1 <= -1.40, r1
    return r3, r1


def test_transition_accuracy():
    """What the switch itself adds: each switched run vs the same scheme started from an h-spaced history on the
    same trajectory (no switch). dt_ladder's weights add only a local O(h^4) error at 2 steps; uniform weights on the
    h0-spaced history add a much larger one."""
    h0, t_end = 0.01, 1.0
    out = []
    for h in (h0 / 2, h0 * 4 / 3):
        ideal = run_switched(0.0, g_nl, h, h, t_end)                   # history at spacing h: no switch
        e_var = abs(run_switched(0.0, g_nl, h0, h, t_end) - ideal)
        e_naive = abs(run_switched(0.0, g_nl, h0, h, t_end, naive=True) - ideal)
        out.append((h / h0, e_var, e_naive))
        assert e_var < e_naive / 20, (h, e_var, e_naive)
    print("error added by the switch (h/h0: variable-step vs naive uniform weights): "
          + "; ".join(f"{a:.3f}: {b:.2e} vs {c:.2e}" for a, b, c in out))


def main():
    test_exact()
    test_order_ratio()
    test_transition_accuracy()
    print("DT_LADDER_TEST_OK")


if __name__ == "__main__":
    main()
