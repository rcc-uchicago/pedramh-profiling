"""f32_update.py -- the AB3 state update of lean32_precise.py, written with + - * only, so the same code runs on JAX
arrays (inside the jitted step) and on test_f32_update.py's emulated float32 scalars. Standard library only.

  theta' = (theta + inc) * (1 - e),  inc = c1 T + c2 T_-1 + c3 T_-2,  e = 1 - decay = -expm1(-rate DT)

plain_update: one rounding at theta's scale for the sum, and the damping as s - e*s, so the damping RATE keeps
float32 relative precision (storing decay itself rounds rate*DT ~ 1e-7..5e-6 to a multiple of 6e-8).
compensated_update: theta is carried as hi + lo. TwoSum(hi, inc + lo) and Fast2Sum(a, -e*a) (|e*a| <= |a| since
0 <= e <= 1) put each rounding error into lo, which is added back next step.
"""


def plain_update(th, inc, e):
    s = th + inc
    return s - e * s


def compensated_update(hi, lo, inc, e):
    """(hi', lo') with hi' + lo' = (hi + lo + inc)(1 - e) up to the rounding of e*a and e*r (relative ~6e-8 of
    those products, i.e. ~6e-8 * e of theta)."""
    y = inc + lo
    a = hi + y                                    # TwoSum(hi, y): a + r == hi + y exactly
    bp = a - hi
    r = (hi - (a - bp)) + (y - bp)
    ea = e * a
    u = a - ea                                    # Fast2Sum(a, -ea): u + r2 == a - ea exactly
    r2 = (a - u) - ea
    return u, r2 + (r - e * r)
