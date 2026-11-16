"""Emulated-float32 test of f32_update.py (standard library only; runs on a login node).

  python3 SQG/analysis/opt/delta/test_f32_update.py          # prints F32_UPDATE_TEST_OK
One Fourier mode at the weakest damping of A-SQG (k = (MU/NU)^(1/4) = 8.4, rate 0.0283, rate*DT = 1.4e-7), forced by
a slowly varying increment of ~2 float32 ulps per step, integrated 100,000 steps (0.5 time units) four ways:
float64 truth; lean32's update (s * float32(decay)); plain_update (s - float32(e) * s); compensated_update (hi + lo).
float32 is emulated by rounding every +, -, * through struct ("f"). Checks: storing decay in float32 biases theta
(rate quantized by ~16% here); the compensated update tracks float64 to <= 1e-7 in theta and in the 4,000-step delta
(o1.2's lag); identities: e = 0 with inc = 0 leaves the state exact, e = 1 (a mode beyond KMAX) gives exactly 0.
First run (polaris-login, 2026-11-05): theta rel err decay32 3.84e-3, plain 3.84e-3, compensated 3.81e-8; 4,000-step
delta rel err 0.307 / 0.307 / 8.8e-8. plain_update alone does not help: the sub-ulp increments and decrements are
rounded onto a float32 state's grid whatever the damping's form.
"""
import math
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import f32_update as FU                                              # noqa: E402


def r32(x):
    return struct.unpack("f", struct.pack("f", x))[0]


class F32:
    """A float32 scalar: every result of + - * is rounded to float32 (round to nearest even)."""
    __slots__ = ("v",)

    def __init__(self, x):
        self.v = r32(x.v if isinstance(x, F32) else x)

    @staticmethod
    def _v(o):
        return o.v if isinstance(o, F32) else r32(o)

    def __add__(self, o):
        return F32(self.v + self._v(o))

    __radd__ = __add__

    def __sub__(self, o):
        return F32(self.v - self._v(o))

    def __rsub__(self, o):
        return F32(self._v(o) - self.v)

    def __mul__(self, o):
        return F32(self.v * self._v(o))

    __rmul__ = __mul__


def main():
    NU, MU, DT = 2.0e-4, 1.0, 5.0e-6
    rate = NU * math.sqrt(MU / NU) + MU / math.sqrt(MU / NU)          # at k^2 = sqrt(MU/NU): 2 sqrt(MU NU)
    e64, d64 = -math.expm1(-rate * DT), math.exp(-rate * DT)
    e32, d32 = F32(e64), F32(d64)
    rate_d32 = -math.log(d32.v) / DT
    assert abs(rate_d32 / rate - 1) > 0.1, rate_d32 / rate           # the float32 decay is >10% off at this k
    assert abs(-math.log1p(-e32.v) / DT / rate - 1) < 1e-6

    n, lag = 100000, 4000
    th64 = 0.6
    th_d, th_p, hi, lo = F32(0.6), F32(0.6), F32(0.6), F32(0.0)
    hist = {"64": [], "d": [], "p": [], "c": []}
    for i in range(n):
        inc = rate * DT * (1.0 + 0.5 * math.sin(2 * math.pi * i / 7919.0))
        inc32 = F32(inc)
        th64 = (th64 + inc) * d64
        th_d = (th_d + inc32) * d32                                   # lean32: (theta + increments) * decay
        th_p = FU.plain_update(th_p, inc32, e32)
        hi, lo = FU.compensated_update(hi, lo, inc32, e32)
        if i >= n - lag - 1:
            hist["64"].append(th64)
            hist["d"].append(th_d.v)
            hist["p"].append(th_p.v)
            hist["c"].append(hi.v + lo.v)
    err = {k: abs(hist[k][-1] - th64) / abs(th64) for k in ("d", "p", "c")}
    dlt = {k: hist[k][-1] - hist[k][0] for k in hist}
    derr = {k: abs(dlt[k] - dlt["64"]) / abs(dlt["64"]) for k in ("d", "p", "c")}
    print(f"rate {rate:.5f}/time, float32 decay applies {rate_d32:.5f} ({rate_d32 / rate - 1:+.1%}); after {n} steps: "
          f"theta rel err  decay32 {err['d']:.2e}  plain {err['p']:.2e}  compensated {err['c']:.2e}; "
          f"{lag}-step delta rel err  decay32 {derr['d']:.2e}  plain {derr['p']:.2e}  compensated {derr['c']:.2e}")
    assert err["d"] > 1e-3, err                                      # the quantized damping biases theta
    assert err["c"] < 1e-7 and derr["c"] < 1e-6, (err, derr)         # compensated tracks float64
    assert err["c"] < err["p"] / 100 and err["c"] < err["d"] / 1000, err

    u, lo2 = FU.compensated_update(F32(0.75), F32(0.0), F32(0.0), F32(0.0))
    assert u.v == 0.75 and lo2.v == 0.0
    u, lo2 = FU.compensated_update(F32(0.75), F32(1e-9), F32(3e-7), F32(1.0))
    assert u.v == 0.0 and lo2.v == 0.0, (u.v, lo2.v)
    assert FU.plain_update(F32(0.75), F32(3e-7), F32(1.0)).v == 0.0
    print("F32_UPDATE_TEST_OK")


if __name__ == "__main__":
    main()
