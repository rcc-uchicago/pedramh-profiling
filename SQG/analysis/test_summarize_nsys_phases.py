"""Synthetic-capture test of summarize_nsys_phases.py (stdlib only, no nsys / torch; runs on a login node).

  python3 SQG/analysis/test_summarize_nsys_phases.py          # prints PHASES_TEST_OK
Builds a two-step nsys-shaped SQLite file and checks: API calls go to the innermost harness range open on their own
thread; syncs and DtoH copies name the XLA thunk (another NVTX domain) open when they were issued; a "step" range in
XLA's domain is not a step; GPU activities follow their API call by correlationId; copy and memory kinds are read
from nsys's enum tables; slow launches; worker / unlabelled threads; and a sync outside every step still counts.
"""
import json
import os
import sqlite3
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
MAIN, WORKER, XLA = 100, 200, 300
THUNK = "Thunk:#name=while/body/cond[linear=()],hlo_op=conditional.277.0#"
US = 1000                                               # timestamps are ns; the layout below is in us


def build(path):
    db = sqlite3.connect(path)
    c = db.cursor()
    c.execute("create table StringIds (id integer, value text)")
    c.execute("create table NVTX_EVENTS (start integer, end integer, eventType integer, text text, textId integer, "
              "globalTid integer, domainId integer)")
    c.execute("create table CUPTI_ACTIVITY_KIND_RUNTIME (start integer, end integer, globalTid integer, "
              "correlationId integer, nameId integer)")
    c.execute("create table CUPTI_ACTIVITY_KIND_MEMCPY (start integer, end integer, correlationId integer, "
              "bytes integer, copyKind integer, srcKind integer, dstKind integer)")
    c.execute("create table CUPTI_ACTIVITY_KIND_KERNEL (start integer, end integer, correlationId integer)")
    c.execute("create table ENUM_CUDA_MEMCPY_OPER (id integer, name text, label text)")
    c.execute("create table ENUM_CUDA_MEM_KIND (id integer, name text, label text)")
    c.executemany("insert into ENUM_CUDA_MEMCPY_OPER values (?, ?, ?)",
                  [(1, "HTOD", "Host-to-Device"), (2, "DTOH", "Device-to-Host"), (8, "DTOD", "Device-to-Device")])
    c.executemany("insert into ENUM_CUDA_MEM_KIND values (?, ?, ?)",
                  [(0, "PAGEABLE", "Pageable"), (1, "PINNED", "Pinned"), (2, "DEVICE", "Device")])
    names = {}

    def sid(s):
        if s not in names:
            names[s] = len(names) + 1
            c.execute("insert into StringIds values (?, ?)", (names[s], s))
        return names[s]

    def rng(a, b, text, tid, dom=0):
        if dom == 0:                                    # torch.cuda.nvtx: inline ASCII text
            c.execute("insert into NVTX_EVENTS values (?, ?, 59, ?, NULL, ?, ?)", (a * US, b * US, text, tid, dom))
        else:                                           # XLA: registered string
            c.execute("insert into NVTX_EVENTS values (?, ?, 59, NULL, ?, ?, ?)", (a * US, b * US, sid(text), tid, dom))

    cid = [0]

    def call(t, dur, name, tid):
        cid[0] += 1
        c.execute("insert into CUPTI_ACTIVITY_KIND_RUNTIME values (?, ?, ?, ?, ?)",
                  (t * US, (t + dur) * US, tid, cid[0], sid(name)))
        return cid[0]

    def copy(k, nbytes, kind, src, dst):
        c.execute("insert into CUPTI_ACTIVITY_KIND_MEMCPY values (0, 2000, ?, ?, ?, ?, ?)", (k, nbytes, kind, src, dst))

    def kernel(k, dur_ns=1000):
        c.execute("insert into CUPTI_ACTIVITY_KIND_KERNEL values (0, ?, ?)", (dur_ns, k))

    c.execute("insert into NVTX_EVENTS values (-5, NULL, 75, 'TSL', NULL, ?, 1)", (MAIN,))   # domain create: ignored
    for k in range(2):
        o = 1000 * k
        rng(o, o + 1000, "step", MAIN)
        rng(o + 10, o + 500, "advance", MAIN)
        rng(o + 10, o + 100, "flush", MAIN)
        rng(o + 100, o + 500, "dispatch", MAIN)
        rng(o + 100, o + 400, "solver", MAIN)
        rng(o + 200, o + 300, THUNK, MAIN, dom=1)
        rng(o + 800, o + 850, "step", MAIN, dom=1)       # a "step" in XLA's domain is not an optimizer step
        copy(call(o + 20, 30, "cudaMemcpyAsync", MAIN), 65536, 2, 2, 0)            # flush: DtoH to pageable
        call(o + 60, 10, "cudaStreamSynchronize", MAIN)
        kernel(call(o + 150, 2, "cuLaunchKernel", MAIN))                           # solver: fast launch
        kernel(call(o + 160, 300, "cuLaunchKernel", MAIN))                         # solver: slow launch (> 100 us)
        copy(call(o + 210, 3, "cuMemcpyDtoHAsync_v2", MAIN), 4, 2, 2, 1)          # inside the thunk: DtoH to pinned
        call(o + 220, 70, "cuStreamSynchronize", MAIN)                             # inside the thunk: the sync
        kernel(call(o + 600, 5, "cudaLaunchKernel_v7000", MAIN))                   # directly in step
        call(o + 700, 1, "cudaStreamWaitEvent", MAIN)
        rng(o + 50, o + 900, "dispatch", WORKER)
        g = call(o + 60, 5, "cuGraphLaunch", WORKER)                               # worker: graph = kernel + DtoD
        kernel(g)
        copy(g, 134283264, 8, 2, 2)
        call(o + 70, 1, "cuEventQuery", XLA)                                       # thread with no harness range
    call(2500, 40, "cuCtxSynchronize", MAIN)                                       # after the steps: main, no range
    kernel(99999)                                                                  # no API call: unmatched
    db.commit()
    db.close()


def main():
    with tempfile.TemporaryDirectory() as d:
        db, out = os.path.join(d, "c.sqlite"), os.path.join(d, "s.json")
        build(db)
        script = os.path.join(HERE, "summarize_nsys_phases.py")
        r = subprocess.run([sys.executable, script, db, "--out", out],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
        assert r.returncode == 0 and "PHASES_OK" in r.stdout, r.stdout
        with open(out) as fh:
            s = json.load(fh)
        P = s["phases"]
        assert s["steps"] == 2 and s["step_wall_ms_mean"] == 1.0, (s["steps"], s["step_wall_ms_mean"])

        fl = P["main:step/advance/flush"]
        assert fl["syncs"] == {"cudaStreamSynchronize": 1.0} and fl["dtoh"] == 1.0, fl
        assert fl["dtoh_dst"] == {"Pageable": 1.0} and fl["dtoh_bytes"] == 65536.0, fl
        assert fl["memcpy_api_ms"] == 0.03 and fl["wall_ms"] == 0.09 and fl["sync_issuers"] == {"-": 1.0}, fl

        so = P["main:step/advance/dispatch/solver"]
        iss = "conditional.277.0 (while/body/cond)"
        assert so["syncs"] == {"cuStreamSynchronize": 1.0} and so["sync_ms"] == 0.07, so
        assert so["sync_issuers"] == {iss: 1.0} and so["dtoh_issuers"] == {iss: 1.0}, so
        assert so["dtoh_dst"] == {"Pinned": 1.0} and so["dtoh_bytes"] == 4.0, so
        assert so["launches"] == 2.0 and so["slow_launches"] == 1.0 and so["wall_ms"] == 0.3, so

        assert P["main:step/advance/dispatch"]["wall_ms"] == 0.4 and P["main:step/advance/dispatch"]["api_calls"] == 0
        st = P["main:step"]
        assert st["launches"] == 1.0 and st["stream_waits"] == 1.0 and st["wall_ms"] == 1.0, st

        wk = P["thread1:dispatch"]
        assert wk["launches"] == 1.0 and wk["dtod"] == 1.0 and wk["dtod_bytes"] == 134283264.0, wk
        assert wk["gpu_ms"] == 0.003, wk                                           # 1 us kernel + 2 us copy
        assert P["other threads:(no range)"]["polls"] == 1.0
        assert P["main:(no range)"]["syncs"] == {"cuCtxSynchronize": 0.5}

        t = s["per_step"]
        assert t["syncs"] == 2.5 and t["dtoh"] == 2.0 and t["launches"] == 4.0 and t["slow_launches"] == 1.0, t
        assert t["stream_waits"] == 1.0 and t["dtod"] == 1.0, t
        assert s["api_calls_per_step"]["cuMemcpyDtoHAsync"] == 1.0 and "cudaLaunchKernel" in s["api_calls_per_step"]
        assert s["unmatched_gpu_activities"] == 1 and s["threads"]["main"] == str(MAIN)

        r = subprocess.run([sys.executable, script, db, "--out", out, "--step-regex", "^nope$"],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
        assert r.returncode != 0 and "ERROR NO_STEP_RANGES" in r.stdout, r.stdout
    print("PHASES_TEST_OK")


if __name__ == "__main__":
    main()
