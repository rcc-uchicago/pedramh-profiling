#!/usr/bin/env python3
"""Tests for nvtx_phase_attribution — both bugs it exists to avoid, reproduced.

    python3 test_nvtx_phase_attribution.py     # prints PASS or ERROR <reason>

Runs on a synthetic 2-rank capture built in a temp file, so it needs no
120 MB artifact and no GPU. The synthetic capture is deliberately shaped like
the real one: two processes whose `correlationId` spaces COLLIDE, and backward
kernels launched from a thread that carries no NVTX ranges.
"""
import collections
import io
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


def build_stall_cause_case(path):
    """One forward_loss window 10x the others, with GC frames sampled inside it."""
    db = sqlite3.connect(path)
    db.executescript("""
        CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT);
        CREATE TABLE NVTX_EVENTS (start INT, end INT, eventType INT, text TEXT,
                                  globalTid INT, textId INT);
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (start INT, end INT,
                                  globalTid INT, correlationId INT);
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INT, end INT,
                                  correlationId INT, globalPid INT,
                                  demangledName INT, deviceId INT);
        CREATE TABLE COMPOSITE_EVENTS (id INTEGER PRIMARY KEY, start INT,
                                  globalTid INT);
        CREATE TABLE SAMPLING_CALLCHAINS (id INT, stackDepth INT, symbol INT);
    """)
    for i, v in ((1, 'gc_collect_main'), (2, 'visit_reachable'),
                 (3, '_PyEval_EvalFrameDefault'), (9, 'compute_kernel')):
        db.execute("INSERT INTO StringIds VALUES (?,?)", (i, v))
    tid = MAIN_A
    ev = 0
    for step in range(4):
        base = step * 1_000_000
        db.execute("INSERT INTO NVTX_EVENTS VALUES (?,?,59,'data_prep',?,NULL)",
                   (base, base + 10, tid))
        # step 2's forward_loss is 10x longer and full of GC samples
        span = 100_000 if step == 2 else 10_000
        db.execute("INSERT INTO NVTX_EVENTS VALUES (?,?,59,'forward_loss',?,NULL)",
                   (base + 20, base + 20 + span, tid))
        db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?,?,?,?)",
                   (base + 30, base + 31, tid, step + 1))
        db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES "
                   "(?,?,?,?,9,0)", (base + 40, base + 140, step + 1, PID_A))
        for k in range(20 if step == 2 else 2):
            ev += 1
            sym = 1 if step == 2 and k % 2 == 0 else (2 if step == 2 else 3)
            db.execute("INSERT INTO COMPOSITE_EVENTS VALUES (?,?,?)",
                       (ev, base + 50 + k * 100, tid))
            db.execute("INSERT INTO SAMPLING_CALLCHAINS VALUES (?,0,?)", (ev, sym))
    db.commit()
    db.close()


def build_straggler_case(path):
    """2 ranks... 4 ranks: one stalled step where dev1 waits far LESS than the rest.

    Shaped like Pangu job 7255503 step 30 (251/65/612/614 ms): the straggler's
    own NCCL time is near-median while the others' balloons.
    """
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
    db.execute("INSERT INTO StringIds VALUES (1, 'compute_kernel')")
    db.execute("INSERT INTO StringIds VALUES (2, 'ncclDevKernel_AllReduce_Sum_f32')")
    pids = [0x100000000 * (i + 1) for i in range(4)]
    cid = 0
    for rank, pid in enumerate(pids):
        tid = pid | 0x10
        for step in range(4):
            base = step * 100_000
            db.execute("INSERT INTO NVTX_EVENTS VALUES (?,?,59,'data_prep',?,NULL)",
                       (base, base + 10, tid))
            db.execute("INSERT INTO NVTX_EVENTS VALUES (?,?,59,'backward',?,NULL)",
                       (base + 20, base + 90_000, tid))
            # step 2 stalls: everyone waits 600 except rank 1, which waits 65
            nccl = 100 if step != 2 else (65 if rank == 1 else 600)
            for name_id, dur in ((1, 500), (2, nccl)):
                cid += 1
                ts = base + 30 + name_id
                db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?,?,?,?)",
                           (ts, ts + 1, tid, cid))
                db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?,?)",
                           (ts + 2, ts + 2 + dur, cid, pid, name_id))
    # deviceId mapping, which per_rank_report reads back for its labels
    db.execute("ALTER TABLE CUPTI_ACTIVITY_KIND_KERNEL ADD COLUMN deviceId INT")
    for rank, pid in enumerate(pids):
        db.execute("UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET deviceId=? WHERE globalPid=?",
                   (rank, pid))
    db.commit()
    db.close()


def build_warmup_case(path):
    """4 steps, flat compute, and step 0 carrying 3x the NCCL of the others.

    Deliberately shaped like Pangu job 7255557: nothing about the compute is
    special about step 0, but its collective waits far longer.
    """
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
    db.execute("INSERT INTO StringIds VALUES (1, 'compute_kernel')")
    db.execute("INSERT INTO StringIds VALUES (2, "
               "'ncclDevKernel_AllReduce_Sum_f32_RING_LL(ncclDevKernelArgsStorage)')")
    cid = 0
    for step in range(4):
        base = step * 10_000
        # every step opens with data_prep (the step delimiter) then backward
        db.execute("INSERT INTO NVTX_EVENTS VALUES (?,?,59,'data_prep',?,NULL)",
                   (base, base + 10, MAIN_A))
        db.execute("INSERT INTO NVTX_EVENTS VALUES (?,?,59,'backward',?,NULL)",
                   (base + 20, base + 9_000, MAIN_A))
        for name_id, dur in ((1, 100), (2, 300 if step == 0 else 100)):
            cid += 1
            ts = base + 30 + name_id
            db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?,?,?,?)",
                       (ts, ts + 1, MAIN_A, cid))
            db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?,?)",
                       (ts + 2, ts + 2 + dur, cid, PID_A, name_id))
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

        agg, union = npa.attribute(con.cursor(), by_pid)
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
        check(rollup(npa.attribute(con.cursor(), by_pid, by_exec=True)[0]) == r,
              "by-exec disagreed on a capture with no CPU run-ahead")

        # -- union == sum here (the synthetic kernels do not overlap), which is
        # what makes the overlapping case below a real test rather than a tautology.
        check(union == r, f"union != sum on non-overlapping kernels: {union} vs {r}")

        # -- and the union must count overlap ONCE. This is the arithmetic that
        # caught a sum-vs-union error in polaris_bench_report.md §4.3: a phase
        # whose kernels run concurrently on two streams has sum > union.
        check(npa._union_ns([(0, 10), (5, 20), (30, 40)]) == 30,
              "union mis-handles overlapping intervals")
        check(npa._union_ns([(0, 10), (10, 20)]) == 20, "union drops a touching pair")
        check(npa._union_ns([(0, 100), (10, 20)]) == 100, "union mis-handles nesting")
        check(npa._union_ns([(50, 60), (0, 10)]) == 20, "union requires pre-sorted input")
        check(npa._union_ns([]) == 0, "union of nothing is not 0")

        # -- parse_nsys.py must run on a Path under Python 3.6 (the Polaris
        # login node's default), where sqlite3.connect() rejects os.PathLike.
        import parse_nsys
        argv, out = sys.argv, sys.stdout
        try:
            sys.argv = ['parse_nsys.py', path]
            sys.stdout = open(os.devnull, 'w')
            parse_nsys.main()
        except TypeError as exc:
            sys.stdout = out
            check(False, f"parse_nsys.py cannot open a Path: {exc}")
        finally:
            sys.stdout.close() if sys.stdout is not out else None
            sys.argv, sys.stdout = argv, out

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

    # -- a step whose NCCL is inflated but whose COMPUTE is flat is comms noise,
    # NOT a warmup regime. Judging warmup on the total confuses the two, which is
    # what made --per-step print "WARMUP REGIME PRESENT" for Pangu job 7255557
    # when step 0 there is +0.5% compute and +61% NCCL.
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "warmup.sqlite")
        build_warmup_case(path)
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        out, sys.stdout = sys.stdout, open(os.devnull, 'w')
        try:
            st = npa.per_step_report(con.cursor(), npa.load_phases(con.cursor()))
        finally:
            sys.stdout.close()
            sys.stdout = out
        check(st['n_steps'] == 4, f"step indexing wrong: {st['n_steps']} != 4")
        check(st['compute_warmup'] is False,
              f"comms noise on step 0 was misread as a compute warmup regime: {st}")
        check(st['step0_nccl_pct'] > 50,
              f"step 0's NCCL inflation was not detected: {st}")
        check(abs(st['step0_compute_pct']) < 1,
              f"compute was reported as inflated when it is flat: {st}")

    # -- the straggler is the rank with the LOWEST NCCL time on a stalled step:
    # it never waited, everyone waited for it. Getting that backwards would
    # accuse the wrong rank, so it is pinned here.
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "straggler.sqlite")
        build_straggler_case(path)
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        buf = io.StringIO()
        out, sys.stdout = sys.stdout, buf
        try:
            npa.per_rank_report(con.cursor(), npa.load_phases(con.cursor()))
        finally:
            sys.stdout = out
        txt = buf.getvalue()
        check("1 of 4 steps" in txt, f"stall count wrong:\n{txt}")
        check("dev1 is the straggler in 1/1" in txt,
              f"named the wrong rank as straggler (dev1 has the LOWEST nccl):\n{txt}")

    # -- stall_cause_report must actually return samples. It silently returned
    # NONE for every window until a cursor-reuse bug was fixed (the inner query
    # invalidated the outer globalTid iteration) -- a failure mode that looks
    # exactly like "this capture has no sampling data".
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "stall.sqlite")
        build_stall_cause_case(path)
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        buf = io.StringIO()
        out, sys.stdout = sys.stdout, buf
        try:
            npa.stall_cause_report(con.cursor(), npa.load_phases(con.cursor()))
        finally:
            sys.stdout = out
        txt = buf.getvalue()
        check("no samples in this window" not in txt,
              f"cursor reuse regression — samples silently lost:\n{txt}")
        check("gc_collect_main" in txt, f"leaf symbol not surfaced:\n{txt}")
        check("GARBAGE COLLECTION" in txt,
              f"GC signature not recognised, so the fix hint is missing:\n{txt}")

    # -- rootedness does NOT follow from substring matching: a regex on `Reduce`
    # classifies AllReduce as rooted, which silently empties the straggler ranking.
    for name, rooted in (
            ('ncclDevKernel_AllReduce_Sum_f32_RING_LL(x)', False),
            ('ncclDevKernel_Broadcast_RING_LL(x)', True),
            ('ncclDevKernel_ReduceScatter_Sum_f32_RING_LL(x)', False),
            ('ncclDevKernel_AllGather_RING_LL(x)', False),
            ('ncclDevKernel_Reduce_Sum_f32_RING_LL(x)', True)):
        check(npa._is_rooted(name) is rooted,
              f"_is_rooted({name.split('(')[0]}) should be {rooted}")

    # -- the range-name contract has ONE source of truth (CLAUDE.md #10).
    import parse_nsys
    check(npa.PHASES is parse_nsys.RANGE_NAMES,
          "PHASES is a copy of the contract, not the contract itself")
    check('unstack' in parse_nsys.RANGE_NAMES,
          "'unstack' fell out of the contract again")
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'parse_nsys.py')).read()
    check(src.count("('preprocess',") == 1,
          "a second literal range list reappeared in parse_nsys.py — the "
          "drift that silently dropped 'unstack' is back")

    print("PASS nvtx_phase_attribution: pid guard + process-scoped windows + dtype\n"
          "     labels + overlap refusal + one-source range contract + union arithmetic\n"
          "     + comms-noise-is-not-warmup + straggler id + stall-cause sampling\n"
          "     + rooted-collective classification")


if __name__ == '__main__':
    main()
