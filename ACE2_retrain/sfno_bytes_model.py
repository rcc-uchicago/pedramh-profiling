#!/usr/bin/env python3
"""Analytic bytes-per-step model for an SFNO, and what each copy kernel matches.

    python3 sfno_bytes_model.py                       # the tensor inventory
    python3 sfno_bytes_model.py --match <nsys.sqlite> # match copy kernels to tensors

Plan item 3. `direct_copy` and `conj` perform no arithmetic, so they are not in
any minimal implementation of this model -- they are pure data movement. The
question this answers is *which tensor* each one is moving, because that is what
turns "42% of GPU time is copies" into a call site.

The method is deliberately independent of the launch-geometry estimate in
`PANGU_POLARIS_PROFILING_PLAN.md` §0d: the inventory below comes from the config
and the source, and the geometry comes from CUPTI. Agreement bounds §0d's
estimate; disagreement localises a copy that is larger than any tensor in the
model, which is itself a finding.

Shapes are grounded in the source, not assumed:
  * `modes_lat = int(h * hard_thresholding_fraction)`,
    `modes_lon = int((w // 2 + 1) * hard_thresholding_fraction)`
    -- networks/modulus_sfno/sfnonet.py:481-482
  * the spectral path is complex64 over `(B, E, modes_lat, modes_lon)`
  * `big_skip` concatenates the input to the latent before the decoder
    -- sfnonet.py:727 (`torch.cat((x, residual), dim=1)`)
"""
import argparse
import collections
import re
import sqlite3
import sys

# The capture under study: pangu_e3sm_sfno.nsys.rendered.yaml, jobs 7255503/7255557.
CFG = dict(
    batch=1, embed_dim=512, num_layers=12, h=180, w=360,
    hard_thresholding_fraction=1.0, mlp_ratio=2.0, scale_factor=1,
    # channels: upper_air 5 x 18 levels + surface 6 + diagnostic 3 + land 2,
    # plus boundary (constant 4 + varying 3) on the input side.
    n_prognostic=5 * 18 + 6 + 3 + 2, n_boundary=4 + 3,
)
DTYPE_BYTES = {'float': 4, 'fp32': 4, 'bf16': 2, 'fp16': 2, 'complex64': 8}


def inventory(cfg=CFG):
    """[(name, n_elements, dtype-agnostic)] — every distinct tensor shape the step touches."""
    b, e, h, w = cfg['batch'], cfg['embed_dim'], cfg['h'], cfg['w']
    thf = cfg['hard_thresholding_fraction']
    modes_lat, modes_lon = int(h * thf), int((w // 2 + 1) * thf)
    hidden = int(e * cfg['mlp_ratio'])
    c_in = cfg['n_prognostic'] + cfg['n_boundary']
    out = [
        ('input/output field  (B,C_in,h,w)', b * c_in * h * w, f'{c_in}x{h}x{w}'),
        ('latent              (B,E,h,w)', b * e * h * w, f'{e}x{h}x{w}'),
        ('big_skip cat        (B,E+C_in,h,w)', b * (e + c_in) * h * w,
         f'{e + c_in}x{h}x{w}'),
        ('MLP hidden          (B,E*ratio,h,w)', b * hidden * h * w, f'{hidden}x{h}x{w}'),
        ('spectral            (B,E,lmax,mmax)', b * e * modes_lat * modes_lon,
         f'{e}x{modes_lat}x{modes_lon}'),
        ('spectral as_real    (B,E,lmax,mmax,2)', b * e * modes_lat * modes_lon * 2,
         f'{e}x{modes_lat}x{modes_lon}x2'),
    ]
    return out, dict(modes_lat=modes_lat, modes_lon=modes_lon, hidden=hidden, c_in=c_in)


def show_inventory():
    inv, meta = inventory()
    print(f"config: batch={CFG['batch']} embed_dim={CFG['embed_dim']} "
          f"grid={CFG['h']}x{CFG['w']} num_layers={CFG['num_layers']} "
          f"modes={meta['modes_lat']}x{meta['modes_lon']} "
          f"mlp_hidden={meta['hidden']} C_in={meta['c_in']}\n")
    print(f"{'tensor':<38}{'shape':>22}{'elements':>14}"
          f"{'bf16 MB':>10}{'fp32 MB':>10}{'cplx64 MB':>11}")
    print("  " + "-" * 103)
    for name, n, shape in inv:
        print(f"{name:<38}{shape:>22}{n:>14,}"
              f"{n * 2 / 1e6:>10.2f}{n * 4 / 1e6:>10.2f}{n * 8 / 1e6:>11.2f}")
    big = max(n * 8 for _, n, _ in inv)
    print(f"\n  largest single tensor at any dtype in this config: "
          f"**{big / 1e6:.2f} MB**")
    print(f"  a copy whose per-call payload exceeds that is NOT moving one tensor.")
    return inv, meta


# --- the geometry side -------------------------------------------------------
# elementwise_kernel<nt, vt, ...>: each block handles nt*vt elements.
_ELEMS_PER_BLOCK = re.compile(r'elementwise_kernel<\(int\)(\d+), \(int\)(\d+)')


def _dtype_of(name):
    if 'complex<float>' in name:
        return 'complex64'
    if 'c10::BFloat16' in name:
        return 'bf16'
    if 'c10::Half' in name:
        return 'fp16'
    if re.search(r'\bfloat\b', name):
        return 'float'
    return None


def match(path, regex='direct_copy|conj', top=10):
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    rx = re.compile(regex)
    rows = collections.defaultdict(lambda: [0, 0, set()])
    for name, gx, gy, gz, bx, by, bz, n, ns in con.execute("""
            SELECT s.value, k.gridX, k.gridY, k.gridZ, k.blockX, k.blockY, k.blockZ,
                   COUNT(*), SUM(k.end - k.start)
            FROM CUPTI_ACTIVITY_KIND_KERNEL k
            JOIN StringIds s ON s.id = k.demangledName
            GROUP BY s.value, k.gridX, k.gridY, k.gridZ, k.blockX, k.blockY, k.blockZ"""):
        if not rx.search(name):
            continue
        m = _ELEMS_PER_BLOCK.search(name)
        # nt*vt per block for the TensorIterator path; a vectorized kernel still
        # covers nt*vt elements per block, the vectorization is within a thread.
        per_block = (int(m.group(1)) * int(m.group(2))) if m else bx * by * bz
        elems = gx * gy * gz * per_block
        dt = _dtype_of(name)
        key = (_short(name), dt)
        cell = rows[key]
        cell[0] += n
        cell[1] += ns
        cell[2].add(elems)

    inv, meta = inventory()
    print(f"\n{'kernel':<34}{'calls':>7}{'elem/call':>13}{'payload MB':>12}"
          f"{'r+w MB':>9}{'us/call':>9}  best analytic match")
    print("  " + "-" * 118)
    for (lbl, dt), (n, ns, elemset) in sorted(rows.items(), key=lambda kv: -kv[1][1]):
        for elems in sorted(elemset, reverse=True):
            sz = DTYPE_BYTES.get(dt, 4)
            payload = elems * sz
            # the closest tensor in the inventory, by element count
            best, ratio = None, None
            for nm, ne, shape in inv:
                r = elems / ne
                if best is None or abs(r - 1) < abs(ratio - 1):
                    best, ratio = (nm, shape), r
            verdict = (f"{best[0].split()[0]} x{ratio:.3f}"
                       if abs(ratio - 1) < 0.02 else
                       f"{best[0].split()[0]} x{ratio:.3f}  <-- NOT a clean tensor")
            print(f"{lbl + '<' + str(dt) + '>':<34}{n:>7}{elems:>13,}"
                  f"{payload / 1e6:>12.2f}{2 * payload / 1e6:>9.1f}"
                  f"{ns / n / 1000:>9.1f}  {verdict}")
    biggest = max(ne * 8 for _, ne, _ in inv)
    over = [(lbl, dt, e) for (lbl, dt), (_, _, es) in rows.items() for e in es
            if e * DTYPE_BYTES.get(dt, 4) > biggest]
    print(f"\n  largest analytic tensor: {biggest / 1e6:.2f} MB. "
          f"Copies exceeding it: {len(over)}")
    for lbl, dt, e in over:
        print(f"    ⚠ {lbl}<{dt}> moves {e * DTYPE_BYTES.get(dt, 4) / 1e6:.2f} MB/call "
              f"— more than any single tensor in the model")


def _short(name):
    for pat in ('direct_copy_kernel_cuda', 'conj_kernel_cuda'):
        if pat in name:
            base = pat.replace('_kernel_cuda', '')
            tag = ('vec' if 'vectorized_elementwise' in name else
                   'nocast' if 'gpu_kernel_impl_nocast' in name else
                   'unrolled' if 'unrolled_elementwise' in name else '')
            return f"{base}.{tag}" if tag else base
    return name[:30]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--match', metavar='SQLITE', default=None)
    ap.add_argument('--regex', default='direct_copy|conj')
    a = ap.parse_args(argv)
    show_inventory()
    if a.match:
        match(a.match, a.regex)


if __name__ == '__main__':
    main()
