#!/usr/bin/env python3
"""Tests for nvtx_phase_attribution — both bugs it exists to avoid, reproduced.

    python3 test_nvtx_phase_attribution.py     # prints PASS or ERROR <reason>

Runs on a synthetic 2-rank capture built in a temp file, so it needs no
120 MB artifact and no GPU. The synthetic capture is deliberately shaped like
the real one: two processes whose `correlationId` spaces COLLIDE, and backward
kernels launched from a thread that carries no NVTX ranges.
"""
import collections
import os
import sqlite3
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nvtx_phase_attribution as npa  # noqa: E402

PID_A, PID_B = 0x111000000, 0x222000000
MAIN_A, WORK_A = PID_A | 0x10, PID_A | 0x20   # worker carries NO nvtx ranges
MAIN_B, WORK_B = PID_B | 0x10, PID_B | 0x20

# Per rank: forward_loss [0,100) launches 1 copy of 10 ns from the MAIN thread;
# backward [100,300) launches 3 copies of 20 ns from the WORKER thread.
# Truth: forward 10 ns, backward 60 ns, i.e. backward = 85.7% of copy time.
FWD_NS, BWD_NS = 10, 60


def build(path):
    db = sqlite3.connect(path)
    db.executescript("""
        CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT);
        CREATE TABLE NVTX_EVENTS (start INT, end INT, eventType INT, text TEXT,
                                  globalTid INT, textId INT);
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (start INT, end INT,
                                  globalTid INT, correlationId INT);
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INT, end INT,
                                  correlationId INT, globalPid INT,
                                  demangledName INT);
    """)
    copy_name = ("void at::native::elementwise_kernel<(int)128, (int)2, void "
                 "at::native::gpu_kernel_impl_nocast<at::native::"
                 "direct_copy_kernel_cuda(at::TensorIteratorBase &)::"
                 "{lambda()#1}::operator()() const::{lambda()#2}, "
                 "c10::complex<float>>>")
    db.execute("INSERT INTO StringIds VALUES (1, ?)", (copy_name,))
    for main, work, pid in ((MAIN_A, WORK_A, PID_A), (MAIN_B, WORK_B, PID_B)):
        for text, start, end in (('forward_loss', 0, 100), ('backward', 100, 300)):
            db.execute("INSERT INTO NVTX_EVENTS VALUES (?,?,59,?,?,NULL)",
                       (start, end, text, main))
        # correlationId 1..4 in BOTH processes -- this is the collision.
        launches = [(1, 50, main, 10), (2, 150, work, 20),
                    (3, 200, work, 20), (4, 250, work, 20)]
        for cid, ts, tid, dur in launches:
            db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?,?,?,?)",
                       (ts, ts + 1, tid, cid))
            db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?,1)",
                       (ts + 2, ts + 2 + dur, cid, pid))
    db.commit()
    db.close()


def rollup(agg):
    out = collections.defaultdict(int)
    for (phase, _), (_, ns) in agg.items():
        out[phase] += ns
    return dict(out)


def check(cond, msg):
    if not cond:
        print(f"ERROR {msg}")
        sys.exit(1)


def main():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "synth.sqlite")
        build(path)
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)

        # -- bug 1, reproduced: the unguarded join cross-products the ranks.
        naive = con.execute(
            "SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_RUNTIME r "
            "JOIN CUPTI_ACTIVITY_KIND_KERNEL k "
            "ON k.correlationId = r.correlationId").fetchone()[0]
        n_kernels = con.execute(
            "SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchone()[0]
        check(naive == 2 * n_kernels,
              f"synthetic capture does not reproduce the collision "
              f"(naive={naive}, kernels={n_kernels})")

        by_pid = npa.load_phases(con.cursor())
        check(set(by_pid) == {PID_A, PID_B},
              f"pid mask wrong: got {[hex(p) for p in by_pid]}")

        agg = npa.attribute(con.cursor(), by_pid)
        # -- bug 1 fixed: exactly one row per kernel, no phantom time.
        check(sum(v[0] for v in agg.values()) == n_kernels,
              "guarded join did not return exactly one row per kernel")
        # -- bug 2 fixed: worker-thread launches land in `backward`, not (outside).
        r = rollup(agg)
        check('(outside)' not in r,
              f"worker-thread launches fell outside a phase: {r}")
        check(r == {'forward_loss': 2 * FWD_NS, 'backward': 2 * BWD_NS},
              f"attribution wrong: {r} != forward {2*FWD_NS} / backward {2*BWD_NS}")

        # -- the label keeps the dtype and the non-vectorized path.
        labels = {lab for _, lab in agg}
        check(labels == {'direct_copy<complex64,nocast>'},
              f"label lost dtype or path: {labels}")

        # -- exec-time bucketing is the sensitivity check, and here agrees.
        check(rollup(npa.attribute(con.cursor(), by_pid, by_exec=True)) == r,
              "by-exec disagreed on a capture with no CPU run-ahead")

        # -- overlapping ranges must be refused, not silently double-counted.
        con2 = sqlite3.connect(path)
        con2.execute("INSERT INTO NVTX_EVENTS VALUES (0,300,59,'ema',?,NULL)",
                     (MAIN_A,))
        con2.commit()
        try:
            npa.load_phases(con2.cursor())
        except SystemExit:
            pass
        else:
            check(False, "nested/overlapping ranges were not refused")

    print("PASS nvtx_phase_attribution: pid guard + process-scoped windows "
          "+ dtype labels + overlap refusal")


if __name__ == '__main__':
    main()
