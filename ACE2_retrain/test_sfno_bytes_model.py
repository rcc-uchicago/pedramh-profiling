#!/usr/bin/env python3
"""Tests for sfno_bytes_model — the arithmetic and the dtype-aware matching.

    python3 test_sfno_bytes_model.py     # prints PASS or ERROR <reason>

No capture and no GPU needed: the matcher runs against a synthetic sqlite whose
launch geometry is constructed to land exactly on known tensors.
"""
import os
import sqlite3
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sfno_bytes_model as m  # noqa: E402


def check(cond, msg):
    if not cond:
        print(f"ERROR {msg}")
        sys.exit(1)


def build(path, launches):
    """launches: [(demangled_name, n_elements_per_call, elems_per_block, n_calls)]"""
    db = sqlite3.connect(path)
    db.executescript("""
        CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT);
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INT, end INT,
            demangledName INT, gridX INT, gridY INT, gridZ INT,
            blockX INT, blockY INT, blockZ INT);
    """)
    for i, (name, elems, per_block, n) in enumerate(launches, start=1):
        db.execute("INSERT INTO StringIds VALUES (?,?)", (i, name))
        grid = elems // per_block
        for k in range(n):
            db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES "
                       "(?,?,?,?,1,1,128,1,1)", (k * 1000, k * 1000 + 100, i, grid))
    db.commit()
    db.close()


def main():
    inv, meta = m.inventory()
    by_name = {n.strip(): (e, s, sz) for n, e, s, sz in inv}

    # -- the arithmetic that carries the finding
    wgt = next(v for k, v in by_name.items() if k.startswith('WGT  spectral / layer'))
    check(wgt[0] == 512 * 512 * 180,
          f"spectral weight elements wrong: {wgt[0]} != {512*512*180}")
    check(abs(wgt[0] * 8 / 1e6 - 377.49) < 0.01,
          f"spectral weight MB wrong: {wgt[0]*8/1e6}")
    check(meta['n_spectral_params'] == 2 * 512 * 512 * 180 * 12,
          "spectral param count wrong")
    # 95.8% of the recorded 1,182,108,160 -- if this drifts, the config changed
    share = 100 * meta['n_spectral_params'] / 1182108160
    check(95.0 < share < 96.5, f"spectral param share {share:.1f}% off 95.8%")
    check(meta['modes_lat'] == 180 and meta['modes_lon'] == 181,
          f"modes wrong: {meta['modes_lat']}x{meta['modes_lon']}")

    # -- the weight is bigger than every activation. That is the finding, so pin it.
    acts = [e * sz for k, (e, s, sz) in by_name.items() if k.startswith('act')]
    check(wgt[0] * 8 > max(acts),
          "the spectral weight is no longer the largest tensor — re-read §4.5")

    # -- the channel counts are pinned by the parameter total, not re-counted.
    check(m.CFG['in_chans'] == 105 and m.CFG['out_chans'] == 101,
          f"channel counts drifted: {m.CFG['in_chans']}/{m.CFG['out_chans']}")
    check(2 * m.CFG['in_chans'] + m.CFG['out_chans'] == 311,
          "2*in + out must be 311 — that is what the logged 1,182,108,160 forces")

    # -- elements-per-block differs by LAUNCH PATH. Reading the vectorized name's
    # leading (int) as `nt` under-counts by 4x and silently rescales every byte
    # figure; that bug shipped once because the test only had the legacy path.
    for name, block, expect, why in (
            ("void at::native::elementwise_kernel<(int)128, (int)2, ...>",
             128, 256, "legacy: nt*vt"),
            ("void at::native::unrolled_elementwise_kernel<f, a, (int)4, ...>",
             128, 512, "unrolled: blockX * thread_work_size"),
            ("void at::native::vectorized_elementwise_kernel<(int)2, f, a>",
             128, 512, "vectorized: leading (int) is vec_size, NOT nt"),
            ("some_other_kernel<float>", 256, 256, "unknown: fall back to block")):
        got = m._elems_per_block(name, block)
        check(got == expect, f"{why}: got {got}, expected {expect}")

    NOCAST = ("void at::native::elementwise_kernel<(int)128, (int)2, void "
              "at::native::gpu_kernel_impl_nocast<at::native::%s_kernel_cuda"
              "(at::TensorIteratorBase &)::{lambda()#1}, %s>>")
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "synth.sqlite")
        build(path, [
            (NOCAST % ('direct_copy', 'c10::complex<float>'), 512 * 512 * 180, 256, 3),
            (NOCAST % ('conj', 'c10::complex<float>'), 512 * 512 * 180, 256, 1),
            (NOCAST % ('direct_copy', 'float'), 512 * 180 * 181, 256, 5),
        ])
        import io
        buf, out = io.StringIO(), sys.stdout
        sys.stdout = buf
        try:
            m.match(path)
        finally:
            sys.stdout = out
        txt = buf.getvalue()
        # each synthetic launch was built to BE a known tensor exactly
        check(txt.count('= WGT  spectral / layer') == 2,
              f"weight-sized copies not matched to the weight:\n{txt}")
        check('= act  spectral 1 part' in txt,
              f"a float copy of 16,680,960 elems should match one part of the "
              f"spectral activation, not the complex tensor:\n{txt}")
        # dtype awareness: the float copy must NOT be credited to the complex tensor
        for line in txt.splitlines():
            if 'direct_copy.nocast<float>' in line:
                check('complex' not in line.split('=')[-1],
                      f"float copy matched to a complex tensor:\n{line}")
        check('Copies exceeding it: 0' in txt,
              f"nothing here exceeds the largest tensor:\n{txt}")

    print("PASS sfno_bytes_model: weight shape + param share + modes\n"
          "     + channel counts pinned by the param total\n"
          "     + elems-per-block per launch path + dtype-aware matching")


if __name__ == '__main__':
    main()
