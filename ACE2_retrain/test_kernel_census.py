#!/usr/bin/env python3
"""Tests for kernel_census — the two attribution bugs, and the conditional thesis.

    python3 test_kernel_census.py     # prints PASS or ERROR <reason>

Runs on a synthetic 2-rank capture, so no 120 MB artifact and no GPU. The
capture is shaped like the real one: colliding `correlationId`s across ranks,
and backward launches issued from a thread that carries no NVTX ranges.
"""
import collections
import io
import os
import sqlite3
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kernel_census as kc  # noqa: E402
import nvtx_phase_attribution as npa  # noqa: E402

PID_A, PID_B = 0x111000000, 0x222000000


def check(cond, msg):
    if not cond:
        print(f"ERROR {msg}")
        sys.exit(1)


def build(path, fwd_kernels=1, bwd_kernels=3, bwd_us=10, fwd_us=10):
    """Two ranks; backward launches from a worker thread with NO nvtx ranges."""
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
    db.execute("INSERT INTO StringIds VALUES (1, 'some_kernel')")
    for pid in (PID_A, PID_B):
        main, work = pid | 0x10, pid | 0x20
        for text, s, e in (('data_prep', 0, 5), ('forward_loss', 10, 100),
                           ('backward', 100, 300)):
            db.execute("INSERT INTO NVTX_EVENTS VALUES (?,?,59,?,?,NULL)",
                       (s, e, text, main))
        cid = 0
        # correlationId 1..N in BOTH ranks -- the collision.
        for k in range(fwd_kernels):
            cid += 1
            ts = 20 + k
            db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?,?,?,?)",
                       (ts, ts + 1, main, cid))
            db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?,1)",
                       (ts + 2, ts + 2 + fwd_us, cid, pid))
        for k in range(bwd_kernels):
            cid += 1
            ts = 150 + k
            db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?,?,?,?)",
                       (ts, ts + 1, work, cid))   # worker thread: no NVTX
            db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?,1)",
                       (ts + 2, ts + 2 + bwd_us, cid, pid))
    db.commit()
    db.close()


def build_prize(path, small_time_share):
    """One phase; N tiny kernels holding `small_time_share` of total GPU time.

    Sizes the tiny population's TIME share directly, because that -- not the
    launch count -- is what the verdict is driven by.
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
    db.execute("INSERT INTO StringIds VALUES (1, 'tiny_kernel')")
    db.execute("INSERT INTO StringIds VALUES (2, 'big_kernel')")
    n_tiny, tiny_ns = 100, 5_000              # 5 us each => under the 10 us bar
    # big kernel absorbs the rest of the time so the tiny share is as requested
    big_ns = int(n_tiny * tiny_ns * (1 - small_time_share) / small_time_share)
    for pid in (PID_A,):
        main = pid | 0x10
        for text, s, e in (('data_prep', 0, 5), ('forward_loss', 10, 10**9)):
            db.execute("INSERT INTO NVTX_EVENTS VALUES (?,?,59,?,?,NULL)",
                       (s, e, text, main))
        cid = 0
        for k in range(n_tiny):
            cid += 1
            db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?,?,?,?)",
                       (100 + k, 101 + k, main, cid))
            db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?,1)",
                       (200 + k, 200 + k + tiny_ns, cid, pid))
        cid += 1
        db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?,?,?,?)",
                   (500, 501, main, cid))
        db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?,2)",
                   (600, 600 + big_ns, cid, pid))
    db.commit()
    db.close()


def run(path):
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    by_pid = npa.load_phases(con.cursor())
    rows = kc.census(con.cursor(), by_pid, 2)
    buf, out = io.StringIO(), sys.stdout
    sys.stdout = buf
    try:
        kc.report(rows, 2)
    finally:
        sys.stdout = out
    return rows, buf.getvalue()


def main():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "synth.sqlite")
        build(path)
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        n_kernels = con.execute(
            "SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchone()[0]

        # -- BUG 1 reproduced: the unguarded join cross-products the ranks.
        naive = con.execute(
            "SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_RUNTIME r "
            "JOIN CUPTI_ACTIVITY_KIND_KERNEL k "
            "ON k.correlationId = r.correlationId").fetchone()[0]
        check(naive == 2 * n_kernels,
              f"synthetic capture does not reproduce the collision: {naive} vs {n_kernels}")

        # -- BUG 2 reproduced: thread-scoped lookup loses ALL of backward.
        by_pid = npa.load_phases(con.cursor())
        by_tid = collections.defaultdict(list)
        for pid, iv in by_pid.items():
            for s, e, name in iv:
                by_tid[pid | 0x10].append((s, e, name))
        lost = 0
        for ts, tid in con.execute(
                "SELECT start, globalTid FROM CUPTI_ACTIVITY_KIND_RUNTIME"):
            if npa._enclosing(sorted(by_tid.get(tid, [])), ts) == '(outside)':
                lost += 1
        check(lost == 6, f"thread-scoped lookup should lose all 6 backward launches, "
                         f"lost {lost}")

        # -- FIXED: agrees with the attribution tool, exactly.
        rows, txt = run(path)
        check(sum(v[0] for v in rows.values()) == n_kernels,
              f"census total {sum(v[0] for v in rows.values())} != {n_kernels} kernels")
        check('(outside)' not in rows, f"(outside) is not empty: {dict(rows)}")
        check(rows['backward'][0] == 6,
              f"backward should have 6 launches, got {rows['backward'][0]}")
        check(rows['forward_loss'][0] == 2,
              f"forward_loss should have 2, got {rows['forward_loss'][0]}")
        check('data_prep' not in rows, "data_prep launches no kernels and must be absent")
        agg, _ = npa.attribute(con.cursor(), by_pid)
        check(sum(n for n, _ in agg.values()) == sum(v[0] for v in rows.values()),
              "census disagrees with nvtx_phase_attribution -- one of them is wrong")

        # -- the phase skew is reported but must NOT carry the verdict. Here every
        # kernel is the same duration, so count-share EQUALS time-share by
        # construction (skew_r = pc_r*(1 - 1) = 0 identically) -- which is why the
        # skewed fixture below is what pins the sign, not this one.
        check('+0.0 pt' in txt, f"balanced capture should show zero skew:\n{txt}")
        check('cannot see a tiny-kernel' in txt,
              f"the report must warn that the phase metric is blind to tiny kernels:\n{txt}")

        # -- `(outside)` is an instrumentation gap, not a code site, and must never
        # be offered as a batching target. The PRE-FIX version of this tool
        # reported it at 69.6%, so this is the exact shape that would have fired.
        rows2 = dict(rows)
        rows2['(outside)'] = [10_000, 1]
        buf2, out2 = io.StringIO(), sys.stdout
        sys.stdout = buf2
        try:
            kc.report(rows2, 2)
        finally:
            sys.stdout = out2
        t2 = buf2.getvalue()
        check('instrumentation-coverage gap' in t2,
              f"(outside) must be flagged as a coverage gap:\n{t2}")
        check('(outside)' not in t2.split('largest phase skew:')[1][:40],
              f"(outside) must be excluded from the skew candidates:\n{t2}")

    # -- the VERDICT is driven by the prize (share of GPU time recoverable), not
    # by the phase skew. Brackets on both sides of the 5%-of-GPU-time bar, so the
    # constant is pinned rather than merely non-zero.
    for share_pct, expect_target in ((0.5, False), (40.0, True)):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "prize.sqlite")
            build_prize(path, small_time_share=share_pct / 100.0)
            con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
            by_pid = npa.load_phases(con.cursor())
            small = kc.small_kernel_census(con.cursor())
            buf, out = io.StringIO(), sys.stdout
            sys.stdout = buf
            try:
                kc.report(kc.census(con.cursor(), by_pid, 2), 2, small)
            finally:
                sys.stdout = out
            txt = buf.getvalue()
            got = 'BATCHING TARGET' in txt
            check(got == expect_target,
                  f"prize {share_pct}% of GPU time: expected target={expect_target}, "
                  f"got {got}\n{txt}")
            if not expect_target:
                check('EXISTS and is large by count' in txt,
                      f"a small prize must still ADMIT the population exists:\n{txt}")

    # -- the rank-step normaliser: main() derives it, and must ERROR rather than
    # silently substitute the rank count when there is no step anchor. (ACE2
    # deliberately emits no `data_prep` -- ace2_nvtx.py:30 -- so this path is live.)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "anchored.sqlite")
        build(path)
        buf, out = io.StringIO(), sys.stdout
        sys.stdout = buf
        try:
            kc.main(path)
        finally:
            sys.stdout = out
        txt = buf.getvalue()
        check('2 rank-steps' in txt,
              f"main() should derive 2 rank-steps from the 2 data_prep ranges:\n{txt}")

        noanchor = os.path.join(d, "noanchor.sqlite")
        build(noanchor)
        db = sqlite3.connect(noanchor)
        db.execute("UPDATE NVTX_EVENTS SET text='sfno_net' WHERE text='data_prep'")
        db.commit(); db.close()
        try:
            buf, out = io.StringIO(), sys.stdout
            sys.stdout = buf
            try:
                kc.main(noanchor)
            finally:
                sys.stdout = out
        except SystemExit as exc:
            check('NO_STEP_ANCHOR' in str(exc),
                  f"wrong error for a missing step anchor: {exc}")
        else:
            check(False, "a capture with no step anchor must ERROR, not guess")

    print("PASS kernel_census: both attribution bugs reproduced then fixed,\n"
          "     agrees with nvtx_phase_attribution, (outside) excluded from targets,\n"
          "     verdict bracketed on both sides of the prize bar,\n"
          "     rank-step anchor derived and its absence ERRORs")


if __name__ == '__main__':
    main()
