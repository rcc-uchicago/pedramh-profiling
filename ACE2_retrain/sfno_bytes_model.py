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
    # Channels, from data_loader_multifiles.py:457,641-655 -- NOT re-counted from
    # the YAML variable lists, which is how the first version got 108. `surface`
    # absorbs `land` (6+2=8); the 3 diagnostics are OUTPUT-only; `varying_boundary`
    # (3) is input-only alongside `constant_boundary` (4):
    #   in_chans  = 90 upper-air + 8 surface + 3 varying + 4 constant = 105
    #   out_chans = 90 + 8 + 3 diagnostic                             = 101
    # Cross-check that pins it: reconstructing the parameter count forces
    # 2*in + out = 311 = 2(105) + 101 against the logged 1,182,108,160.
    in_chans=105, out_chans=101,
)
DTYPE_BYTES = {'float': 4, 'fp32': 4, 'bf16': 2, 'fp16': 2, 'complex64': 8}


def inventory(cfg=CFG):
    """[(name, n_elements, dtype-agnostic)] — every distinct tensor shape the step touches."""
    b, e, h, w = cfg['batch'], cfg['embed_dim'], cfg['h'], cfg['w']
    thf = cfg['hard_thresholding_fraction']
    modes_lat, modes_lon = int(h * thf), int((w // 2 + 1) * thf)
    hidden = int(e * cfg['mlp_ratio'])
    c_in, c_out = cfg['in_chans'], cfg['out_chans']
    out = [
        # --- activations (per step) ---
        ('act  input             (B,C_in,h,w)', b * c_in * h * w, f'{c_in}x{h}x{w}', 4),
        ('act  output            (B,C_out,h,w)', b * c_out * h * w, f'{c_out}x{h}x{w}', 4),
        ('act  latent            (B,E,h,w)', b * e * h * w, f'{e}x{h}x{w}', 4),
        ('act  big_skip cat      (B,E+C_in,h,w)', b * (e + c_in) * h * w,
         f'{e + c_in}x{h}x{w}', 4),
        ('act  MLP hidden        (B,E*r,h,w)', b * hidden * h * w, f'{hidden}x{h}x{w}', 4),
        ('act  spectral          (B,E,lmax,mmax)', b * e * modes_lat * modes_lon,
         f'{e}x{modes_lat}x{modes_lon}', 8),
        ('act  spectral 1 part   (B,E,lmax,mmax)', b * e * modes_lat * modes_lon,
         f'{e}x{modes_lat}x{modes_lon}', 4),
        ('act  spectral as_real  (...,2)', b * e * modes_lat * modes_lon * 2,
         f'{e}x{modes_lat}x{modes_lon}x2', 4),
        # --- WEIGHTS. Omitting these was the flaw in the first version of this
        # model, and it is the whole finding: for dhconv the spectral weight is
        # [in_channels, out_channels, modes_lat] complex (s2convolutions.py:107-135),
        # which at E=512 is 377 MB -- almost 3x the largest activation -- and 12 of
        # them are 95.8% of this model's 1,182,108,160 parameters.
        ('WGT  spectral / layer   (E,E,lmax)', e * e * modes_lat, f'{e}x{e}x{modes_lat}', 8),
        ('WGT  spectral as_real   (E,E,lmax,2)', e * e * modes_lat * 2,
         f'{e}x{e}x{modes_lat}x2', 4),
        ('WGT  MLP fc1/fc2        (E*r,E,1,1)', hidden * e, f'{hidden}x{e}', 4),
    ]
    return out, dict(modes_lat=modes_lat, modes_lon=modes_lon, hidden=hidden, c_in=c_in,
                     n_spectral_params=2 * e * e * modes_lat * cfg['num_layers'])


def show_inventory():
    inv, meta = inventory()
    print(f"config: batch={CFG['batch']} embed_dim={CFG['embed_dim']} "
          f"grid={CFG['h']}x{CFG['w']} num_layers={CFG['num_layers']} "
          f"modes={meta['modes_lat']}x{meta['modes_lon']} "
          f"mlp_hidden={meta['hidden']} C_in={meta['c_in']}\n")
    print(f"{'tensor':<40}{'shape':>22}{'elements':>14}{'dtype':>10}{'MB':>10}")
    print("  " + "-" * 96)
    for name, n, shape, sz in inv:
        dt = {4: 'fp32', 8: 'complex64', 2: 'bf16'}[sz]
        print(f"{name:<40}{shape:>22}{n:>14,}{dt:>10}{n * sz / 1e6:>10.2f}")
    big_name, big = max(((nm, n * sz) for nm, n, _, sz in inv), key=lambda kv: kv[1])
    print(f"\n  largest single tensor: **{big / 1e6:.2f} MB** ({big_name.strip()})")
    print(f"  a copy whose per-call payload exceeds that is not moving one tensor.")
    print(f"  spectral weights are {meta['n_spectral_params']:,} params over "
          f"{CFG['num_layers']} layers = "
          f"{100 * meta['n_spectral_params'] / 1182108160:.1f}% of the 1,182,108,160 total.")
    return inv, meta


# --- the geometry side -------------------------------------------------------
# Elements per block, per launch path. Getting this wrong silently rescales every
# byte figure, and it did: the first version matched only the legacy pattern and
# fell back to `blockX` for the other two, under-counting them by exactly 4x and
# turning an exact match to the fp32 latent into a spurious "1.185x, no clean
# tensor" row.
#
#   legacy      `elementwise_kernel<(int)nt, (int)vt, ...>`
#               launch_legacy_kernel: grid = ceil(N / (nt*vt))  => nt*vt per block
#   vectorized  `vectorized_elementwise_kernel<(int)vec_size, ...>`
#   unrolled    `unrolled_elementwise_kernel<..., (int)elems_per_thread, ...>`
#               both from launch_vectorized_kernel: grid = ceil(N / (num_threads *
#               thread_work_size)) => blockX * 4 per block. NOTE the leading (int)
#               on the vectorized name is vec_size, NOT nt -- reading it as nt is
#               how the 4x error happens.
_LEGACY = re.compile(r'(?<!unrolled_)(?<!vectorized_)elementwise_kernel<\(int\)(\d+), \(int\)(\d+)')
_THREAD_WORK_SIZE = 4


def _elems_per_block(name, block_threads):
    m = _LEGACY.search(name)
    if m:
        return int(m.group(1)) * int(m.group(2))
    if 'vectorized_elementwise_kernel' in name or 'unrolled_elementwise_kernel' in name:
        return block_threads * _THREAD_WORK_SIZE
    return block_threads


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
    # Key on the GEOMETRY too. Grouping only by (name, dtype) and printing every
    # distinct geometry against the GROUP's aggregate call count and us/call is
    # actively misleading -- you cannot tell which geometry carries the time.
    rows = collections.defaultdict(lambda: [0, 0])
    for name, gx, gy, gz, bx, by, bz, n, ns in con.execute("""
            SELECT s.value, k.gridX, k.gridY, k.gridZ, k.blockX, k.blockY, k.blockZ,
                   COUNT(*), SUM(k.end - k.start)
            FROM CUPTI_ACTIVITY_KIND_KERNEL k
            JOIN StringIds s ON s.id = k.demangledName
            GROUP BY s.value, k.gridX, k.gridY, k.gridZ, k.blockX, k.blockY, k.blockZ"""):
        if not rx.search(name):
            continue
        per_block = _elems_per_block(name, bx * by * bz)
        elems = gx * gy * gz * per_block
        cell = rows[(_short(name), _dtype_of(name), elems, gx * gy * gz, per_block)]
        cell[0] += n
        cell[1] += ns

    inv, meta = inventory()
    tot_ns = sum(v[1] for v in rows.values())
    print(f"\n{'kernel':<28}{'calls':>7}{'blocks':>9}{'elem/call':>13}"
          f"{'MB/call':>9}{'us/call':>9}{'% copy':>8}  analytic match")
    print("  " + "-" * 116)
    for (lbl, dt, elems, blocks, per_block), (n, ns) in sorted(
            rows.items(), key=lambda kv: -kv[1][1])[:top]:
        sz = DTYPE_BYTES.get(dt, 4)
        best, ratio = None, None
        for nm, ne, shape, nsz in inv:
            if nsz != sz:      # a float copy is not a match for a complex tensor
                continue
            r = elems / ne
            if best is None or abs(r - 1) < abs(ratio - 1):
                best, ratio = nm.strip(), r
        if best is None:
            print(f"{lbl:<28}{n:>7}  (no inventory tensor at dtype {dt})")
            continue
        verdict = (f"= {best}" if abs(ratio - 1) < 0.001
                   else f"{ratio:.4f} x {best}  <-- no clean tensor")
        print(f"{lbl + '<' + str(dt) + '>':<28}{n:>7}{blocks:>9,}{elems:>13,}"
              f"{elems * sz / 1e6:>9.2f}{ns / n / 1000:>9.1f}"
              f"{100 * ns / tot_ns:>7.1f}%  {verdict}")

    biggest = max(ne * nsz for _, ne, _, nsz in inv)
    over = [(lbl, dt, e) for (lbl, dt, e, _, _) in rows
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
