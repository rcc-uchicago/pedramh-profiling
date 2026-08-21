#!/usr/bin/env python3
"""Tests for attrib_copies — synthetic 2-rank captures, stdlib only (login node = 3.6).

The load-bearing test is `test_cross_rank_mis_join_is_prevented`: it builds a capture
where rank 1 reuses rank 0's correlationId (which is what really happens — the id is
per-PROCESS) and asserts the unguarded join over-counts while the real query does not.
Reproduce the bug, then prove the fix — the same discipline the NVTX join fix used.
"""
import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import attrib_copies as ac

TOOL = str(Path(__file__).parent / "attrib_copies.py")
F32 = ("void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast"
       "<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda]<float>>>")
C64 = F32.replace("<float>", "<c10::complex<float>>")


def build(path, rows, chains, extra_table=None):
    """rows: (globalPid, tid, correlationId, kernelName, gridX, callchainId)"""
    db = sqlite3.connect(path)
    c = db.cursor()
    c.execute("CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT)")
    c.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL "
              "(globalPid INT, correlationId INT, demangledName INT, gridX INT)")
    c.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME "
              "(globalTid INT, correlationId INT, callchainId INT)")
    c.execute("CREATE TABLE CUDA_CALLCHAINS "
              "(id INT, symbol INT, module INT, stackDepth INT)")
    if extra_table:                      # a decoy with the same shape
        c.execute("CREATE TABLE SAMPLING_CALLCHAINS "
                  "(id INT, symbol INT, module INT, stackDepth INT)")
        for i, (cid, sym, mod, d) in enumerate(extra_table):
            c.execute("INSERT INTO SAMPLING_CALLCHAINS VALUES (?,?,?,?)", (cid, sym, mod, d))
    sids, nxt = {}, [1]

    def sid(v):
        if v not in sids:
            sids[v] = nxt[0]
            c.execute("INSERT INTO StringIds VALUES (?,?)", (nxt[0], v))
            nxt[0] += 1
        return sids[v]

    for pid, tid, corr, kname, grid, ccid in rows:
        c.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?)",
                  (pid, corr, sid(kname), grid))
        c.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?,?,?)",
                  (pid | tid, corr, ccid))
    for ccid, sym, mod, depth in chains:
        c.execute("INSERT INTO CUDA_CALLCHAINS VALUES (?,?,?,?)",
                  (ccid, sid(sym), sid(mod), depth))
    db.commit()
    db.close()


def tmpdb():
    fh = tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False)
    fh.close()
    os.unlink(fh.name)
    return fh.name


P0, P1 = 0x1000000, 0x2000000          # two ranks; tid lives in the low 24 bits
CHAIN = [(7, "forward", "s2convolutions.py", 0), (7, "_contract_dhconv", "contractions.py", 1)]


def run(db, *args):
    return subprocess.run([sys.executable, TOOL, db] + list(args),
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          universal_newlines=True)


class T(unittest.TestCase):
    def test_cross_rank_mis_join_is_prevented(self):
        # Both ranks use correlationId 42 — which is what really happens, because the
        # id is per-process. Unguarded, the join yields 2x2=4 rows for 2 launches.
        db = tmpdb()
        build(db, [(P0, 11, 42, F32, 64800, 7), (P1, 22, 42, F32, 64800, 7)], CHAIN)
        con = sqlite3.connect(db)
        bad = con.execute(
            "SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL k "
            "JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON r.correlationId = k.correlationId"
        ).fetchone()[0]
        good = con.execute(
            "SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL k "
            "JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON r.correlationId = k.correlationId "
            "AND k.globalPid = (r.globalTid & %d)" % ac.PID_MASK).fetchone()[0]
        self.assertEqual(bad, 4, "the bug must reproduce before the fix is meaningful")
        self.assertEqual(good, 2)
        out = run(db).stdout
        self.assertIn("attributed launches: 2", out, out)

    def test_grid_filter_isolates_the_target_population(self):
        db = tmpdb()
        build(db, [(P0, 11, 1, F32, 64800, 7), (P0, 11, 2, C64, 184320, 7)], CHAIN)
        self.assertIn("attributed launches: 1", run(db, "--grid", "64800").stdout)
        self.assertIn("attributed launches: 2", run(db).stdout)

    def test_frames_print_in_stack_order(self):
        db = tmpdb()
        build(db, [(P0, 11, 1, F32, 64800, 7)], CHAIN)
        out = run(db).stdout
        self.assertLess(out.index("forward"), out.index("_contract_dhconv"))

    def test_zero_callchains_fails_loudly_not_silently(self):
        # tick 17's exact trap: the column exists, and is all zeros.
        db = tmpdb()
        build(db, [(P0, 11, 1, F32, 64800, 0)], CHAIN)
        r = run(db)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("NO_CALLCHAINS", r.stdout)

    def test_decoy_callchain_table_is_rejected_by_key_overlap(self):
        # SAMPLING_CALLCHAINS has an identical schema but holds CPU samples. Picking
        # it would print plausible-looking frames that are simply the wrong data.
        db = tmpdb()
        build(db, [(P0, 11, 1, F32, 64800, 7)], CHAIN,
              extra_table=[(999, 1, 1, 0)])
        out = run(db).stdout
        self.assertIn("callchain table: CUDA_CALLCHAINS", out, out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
