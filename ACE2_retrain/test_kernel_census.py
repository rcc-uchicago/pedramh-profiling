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

        # -- the thesis is CONDITIONAL. Here every kernel is the same duration, so
        # count-share EQUALS time-share by construction and the skew is 0 --
        # exactly the situation on the real capture (+3.2 pt), where the advice
        # must not print.
        check('NO batching target' in txt, f"expected no target on a balanced capture:\n{txt}")
        check('+0.0 pt' in txt, f"balanced capture should show zero skew:\n{txt}")

    # -- and it fires when a range really is launch-heavy / time-light.
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "skewed.sqlite")
        build(path, fwd_kernels=200, bwd_kernels=1, fwd_us=1, bwd_us=10_000)
        _, txt = run(path)
        check('BATCHING TARGET' in txt,
              f"a 200-launch/1us range vs a 1-launch/10ms range must trip it:\n{txt}")
        check('forward_loss' in txt.split('BATCHING TARGET')[1][:80],
              f"named the wrong range as the target:\n{txt}")

    print("PASS kernel_census: both attribution bugs reproduced then fixed,\n"
          "     agrees with nvtx_phase_attribution, thesis is conditional both ways")


if __name__ == '__main__':
    main()
