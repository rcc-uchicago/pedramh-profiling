"""How float32 quantizes A-SQG's per-step damping factor decay = exp(-(NU k^2 + MU/k^2) DT).

Per distinct n = kx^2 + ky^2 (k = sqrt(n)), the damping rate actually applied when decay is stored in float32 is
-ln(f32(decay)) / DT. Relative error vs the true rate, per |k| octave band (unweighted over n ~ mode-weighted, since
modes per dn is ~constant), and exactly for every low-k mode. Also the expm1 form: store e = 1 - decay in float32
(theta' = s - e*s), whose rate error is ~ float32 epsilon.
"""
import math
import struct

NU, MU, DT, N = 2.0e-4, 1.0, 5.0e-6, 4096
KMAX = N / 3.0


def f32(x):
    return struct.unpack("f", struct.pack("f", x))[0]


def rate(n):
    return NU * n + MU / n


def err_decay32(n):
    r = rate(n)
    d = math.exp(-r * DT)
    r32 = -math.log(f32(d)) / DT
    return r32 / r - 1.0


def err_expm1_32(n):
    r = rate(n)
    e = -math.expm1(-r * DT)
    r32 = -math.log1p(-f32(e)) / DT
    return r32 / r - 1.0


# low-k modes exactly (half plane: kx > 0, or kx == 0 and ky > 0)
print("low-k modes (|k| <= 12): k, rate per time unit, rate*DT, f32 decay rel rate error, expm1 rel rate error")
seen = {}
for kx in range(0, 13):
    for ky in range(-12, 13):
        if (kx == 0 and ky <= 0):
            continue
        n = kx * kx + ky * ky
        if n > 144 or n in seen:
            continue
        seen[n] = (math.sqrt(n), rate(n), rate(n) * DT, err_decay32(n), err_expm1_32(n))
for n in sorted(seen)[:40]:
    k, r, rd, e32, em = seen[n]
    print(f"  k={k:6.3f} n={n:4d} rate={r:8.4f} rate*DT={rd:.3e}  f32 decay: {e32:+8.3%}   expm1 f32: {em:+.1e}")

bands = [(1, 2), (2, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 128), (128, 256), (256, 512), (512, 1024),
         (1024, KMAX + 1)]
print("\nper |k| band, over every integer n = k^2 in the band (~mode-weighted):")
print(f"  {'band':>12s} {'max |err|':>10s} {'rms err':>9s} {'mean err':>9s}  {'expm1 max':>9s}")
for lo, hi in bands:
    mx = s2 = s1 = em = 0.0
    cnt = 0
    for n in range(max(1, int(math.ceil(lo * lo))), int(hi * hi)):
        if n > KMAX * KMAX:
            break
        e = err_decay32(n)
        mx = max(mx, abs(e))
        s2 += e * e
        s1 += e
        em = max(em, abs(err_expm1_32(n)))
        cnt += 1
    print(f"  {lo:5d}-{int(hi):<6d} {mx:10.3%} {math.sqrt(s2 / cnt):9.3%} {s1 / cnt:+9.4%}  {em:9.1e}  (n={cnt})")
kmin = (MU / NU) ** 0.25
print(f"\nweakest damping at k = (MU/NU)^(1/4) = {kmin:.2f}: rate {2 * math.sqrt(MU * NU):.4f}/time, "
      f"rate*DT = {2 * math.sqrt(MU * NU) * DT:.2e}; float32 spacing below 1 = {2 ** -24:.2e}")
