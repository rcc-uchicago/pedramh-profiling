"""test_if_ab3.py -- standard-library checks of if_ab3.py's scheme arithmetic (no jax). Prints IF_AB3_TEST_OK.

  1. uniform weights == pyqg_jax's (23/12, -16/12, 5/12) h;
  2. IF-AB3 is exact (to float64 rounding) for dtheta/dt = -lam theta + N when e^{lam t} N is a quadratic in t, for
     uniform and for the probe's two transition spacings (DT, DT) and (h, DT);
  3. order: integrating a constant forcing over a fixed interval, halving h divides prod's error by ~2 (first order:
     the missing factors) and IF-AB3's by ~8 (third order).
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import if_ab3 as IF                                                  # noqa: E402


def check(name, ok, detail=""):
    if not ok:
        raise SystemExit(f"ERROR IF_AB3_TEST {name} {detail}")
    print(f"ok {name} {detail}")


def main():
    h = 0.125
    b = IF.ab3_coeffs(h, h, h)
    want = (23 / 12 * h, -16 / 12 * h, 5 / 12 * h)
    check("uniform_weights", all(abs(x - y) < 1e-15 for x, y in zip(b, want)), str(b))

    # 2. exactness: N(t) = e^{-lam t} (a + b t + c t^2) => theta(t) = e^{-lam t} (theta0 + a t + b t^2/2 + c t^3/3)
    lam, a, bb, c, th0 = 37.0, 0.7, -1.3, 2.1, 0.4

    def N(t):
        return math.exp(-lam * t) * (a + bb * t + c * t * t)

    def theta(t):
        return math.exp(-lam * t) * (th0 + a * t + bb * t * t / 2 + c * t ** 3 / 3)

    worst = 0.0
    for (hh, h1, h2) in ((1e-2, 1e-2, 1e-2), (7e-3, 5e-3, 5e-3), (7e-3, 7e-3, 5e-3), (3e-3, 1e-2, 2e-2)):
        tn = 0.3
        hist = (N(tn), N(tn - h1), N(tn - h1 - h2))
        got = IF.step_scalar("if", theta(tn), hist, hh, h1, h2, lam)
        err = abs(got - theta(tn + hh)) / abs(theta(tn + hh))
        worst = max(worst, err)
    check("if_exact_quadratic", worst < 1e-13, f"max rel err {worst:.1e}")
    tn, hh = 0.3, 1e-2
    hist = (N(tn), N(tn - hh), N(tn - 2 * hh))
    perr = abs(IF.step_scalar("prod", theta(tn), hist, hh, hh, hh, lam) - theta(tn + hh)) / abs(theta(tn + hh))
    check("prod_not_exact", perr > 1e-6, f"prod one-step rel err {perr:.1e}")

    # 3. order: constant forcing N0, theta(0) = 0 with exact start-up history, integrate to T = 0.4
    lam, N0, T = 20.0, 1.0, 0.4

    def exact(t):
        return N0 * (1 - math.exp(-lam * t)) / lam

    def integrate(scheme, nsteps):
        hh = T / nsteps
        th = exact(0.0)
        hist = [N0, N0, N0]                                           # N is constant: the history is exact
        for _ in range(nsteps):
            th = IF.step_scalar(scheme, th, hist, hh, hh, hh, lam)
        return abs(th - exact(T))

    for scheme, lo, hi in (("prod", 1.8, 2.2), ("if", 6.5, 9.5)):
        e1, e2 = integrate(scheme, 200), integrate(scheme, 400)
        check(f"order_{scheme}", lo <= e1 / e2 <= hi, f"err(h)/err(h/2) = {e1 / e2:.3f} ({e1:.2e} -> {e2:.2e})")
    check("if_beats_prod", integrate("if", 200) < 1e-2 * integrate("prod", 200))

    # steps per lag: transitions only off DT
    check("steps_after_switch", IF.steps_after_switch(40) == (0, 4000) and IF.steps_after_switch(28) == (2, 2798))
    print("IF_AB3_TEST_OK")


if __name__ == "__main__":
    main()
