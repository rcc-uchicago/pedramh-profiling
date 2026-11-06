"""Synthetic-capture test of summarize_nsys_ops.py (stdlib only, no nsys / GPU; runs on a login node).

  python3 SQG/analysis/test_summarize_nsys_ops.py          # prints NSYS_OPS_TEST_OK
Builds an nsys-shaped SQLite file with a warm-up range and two captured ranges ("v_rep_0", "v_rep_1", 2 solver steps
each) and checks: GPU work goes to the innermost XLA thunk open on the launching thread (a body op, the while op, or
"(no thunk)"); an FFT op's copy + kernels stay together, with its FFT type; the same op name in two XLA modules gives
two entries; work launched in the warm-up range, or with no API call, is left out; per-step division; the busy /
span / idle arithmetic over a gap; execution order; and a regex that matches nothing gives a greppable ERROR line.
"""
import json
import os
import sqlite3
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
MAIN, OTHER = 100, 200
US = 1000                                                            # timestamps are ns; the layout is in us
MOD7 = "XlaModule:#prefix=jit(<lambda>)/jit(main),hlo_module=jit__lambda_,program_id=7#"
MOD8 = "XlaModule:#prefix=jit(<lambda>)/jit(main),hlo_module=jit__lambda_,program_id=8#"
IRFFT = ("Thunk:#name=while/body/jit(fft)/fft[fft_type=jaxlib.xla_extension.FftType.IRFFT fft_lengths=(4096, 4096)],"
         "hlo_op=fft.1.0#")
IRFFT8 = "Thunk:#name=jit(fft)/fft[fft_type=jaxlib.xla_extension.FftType.IRFFT fft_lengths=(4096, 4096)],hlo_op=fft.1.0#"


def build(path):
    db = sqlite3.connect(path)
    c = db.cursor()
    c.execute("create table StringIds (id integer, value text)")
    c.execute("create table NVTX_EVENTS (start integer, end integer, eventType integer, text text, textId integer, "
              "globalTid integer, domainId integer)")
    c.execute("create table CUPTI_ACTIVITY_KIND_RUNTIME (start integer, end integer, globalTid integer, "
              "correlationId integer, nameId integer)")
    c.execute("create table CUPTI_ACTIVITY_KIND_KERNEL (start integer, end integer, correlationId integer, "
              "shortName integer, gridX integer, gridY integer, gridZ integer, blockX integer, blockY integer, "
              "blockZ integer)")
    c.execute("create table CUPTI_ACTIVITY_KIND_MEMCPY (start integer, end integer, correlationId integer, "
              "bytes integer, copyKind integer)")
    c.execute("create table ENUM_CUDA_MEMCPY_OPER (id integer, name text, label text)")
    c.executemany("insert into ENUM_CUDA_MEMCPY_OPER values (?, ?, ?)",
                  [(1, "HTOD", "Host-to-Device"), (2, "DTOH", "Device-to-Host"), (8, "DTOD", "Device-to-Device")])
    names = {}

    def sid(s):
        if s not in names:
            names[s] = len(names) + 1
            c.execute("insert into StringIds values (?, ?)", (names[s], s))
        return names[s]

    def rng(a, b, text, tid=MAIN, dom=0):
        if dom == 0:
            c.execute("insert into NVTX_EVENTS values (?, ?, 59, ?, NULL, ?, 0)", (a * US, b * US, text, tid))
        else:
            c.execute("insert into NVTX_EVENTS values (?, ?, 59, NULL, ?, ?, ?)", (a * US, b * US, sid(text), tid, dom))

    cid = [0]

    def call(t, tid=MAIN):
        cid[0] += 1
        c.execute("insert into CUPTI_ACTIVITY_KIND_RUNTIME values (?, ?, ?, ?, ?)",
                  (t * US, (t + 1) * US, tid, cid[0], sid("cuLaunchKernel")))
        return cid[0]

    def kernel(k, a, b, name, grid=(1, 1, 1), block=(1, 1, 1)):
        c.execute("insert into CUPTI_ACTIVITY_KIND_KERNEL values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                  (a * US, b * US, k, sid(name)) + tuple(grid) + tuple(block))

    def copy(k, a, b, nbytes, kind):
        c.execute("insert into CUPTI_ACTIVITY_KIND_MEMCPY values (?, ?, ?, ?, ?)", (a * US, b * US, k, nbytes, kind))

    rng(0, 900, "warm")
    kernel(call(500), 500, 800, "loop_add_multiply_fusion_1")         # warm-up: excluded
    for k in range(2):
        o = 1000 * (k + 1)
        rng(o, o + 1000, f"v_rep_{k}")
        rng(o + 5, o + 810, MOD7, dom=1)
        rng(o + 10, o + 800, "Thunk:#name=,hlo_op=while.1.0#", dom=1)
        rng(o + 20, o + 30, "Thunk:#name=while/body,hlo_op=loop_add_multiply_fusion.1#", dom=1)
        rng(o + 40, o + 60, IRFFT, dom=1)
        rng(o + 850, o + 890, MOD8, dom=1)
        rng(o + 855, o + 870, IRFFT8, dom=1)
        kernel(call(o + 22), o + 100, o + 600, "loop_add_multiply_fusion_1", (16392, 1, 1), (128, 1, 1))
        copy(call(o + 42), o + 600, o + 700, 67108864, 8)              # the IRFFT's input copy
        kernel(call(o + 45), o + 700, o + 900, "regular_fft_factor", (513, 1, 1), (4, 256, 1))
        kernel(call(o + 50), o + 910, o + 960, "scal_kernel_val", (65536, 1, 1), (256, 1, 1))   # after a 10 us gap
        kernel(call(o + 400), o + 960, o + 962, "loop_compare_fusion")  # in the while thunk, outside the body ops
        kernel(call(o + 860), o + 962, o + 972, "regular_fft_factor", (513, 1, 1), (4, 256, 1))  # module 8
        kernel(call(o + 950), o + 972, o + 975, "torch_kernel")         # no XLA range open
        copy(call(o + 500, OTHER), o + 975, o + 976, 4, 1)              # another thread, no XLA ranges
    kernel(99999, 5000, 5001, "orphan")                                # no API call
    db.commit()
    db.close()


def main():
    with tempfile.TemporaryDirectory() as d:
        db, out = os.path.join(d, "c.sqlite"), os.path.join(d, "s.json")
        build(db)
        script = os.path.join(HERE, "summarize_nsys_ops.py")
        r = subprocess.run([sys.executable, script, db, "--out", out, "--steps-per-range", "2"],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
        assert r.returncode == 0 and "NSYS_OPS_OK" in r.stdout, r.stdout
        with open(out) as fh:
            s = json.load(fh)
        assert s["ranges"] == 2 and s["steps"] == 4, (s["ranges"], s["steps"])
        ops = {(o["module"], o["op"]): o for o in s["ops"]}
        assert set(ops) == {("jit__lambda_#7", "loop_add_multiply_fusion.1"), ("jit__lambda_#7", "fft.1.0"),
                            ("jit__lambda_#7", "while.1.0"), ("jit__lambda_#8", "fft.1.0"), ("-", "(no thunk)")}, set(ops)
        ab3 = ops[("jit__lambda_#7", "loop_add_multiply_fusion.1")]
        assert ab3["us_per_step"] == 250.0 and ab3["kernels"][0]["mean_us"] == 500.0, ab3
        assert ab3["kernels"][0]["grid_x_block"] == "16392x128" and ab3["kernels"][0]["per_step"] == 0.5, ab3
        fft = ops[("jit__lambda_#7", "fft.1.0")]
        assert fft["fft_type"] == "IRFFT" and fft["us_per_step"] == 175.0 and fft["activities_per_step"] == 1.5, fft
        assert [k["name"] for k in fft["kernels"]] == ["regular_fft_factor", "scal_kernel_val"], fft["kernels"]
        assert fft["copies"] == [{"kind": "memcpy_dtod", "per_step": 0.5, "bytes_per_step": 33554432.0,
                                  "us_per_step": 50.0}], fft["copies"]
        assert ops[("jit__lambda_#7", "while.1.0")]["us_per_step"] == 1.0
        assert ops[("jit__lambda_#8", "fft.1.0")]["us_per_step"] == 5.0
        assert ops[("-", "(no thunk)")]["us_per_step"] == 2.0
        p = s["per_step"]
        assert p["gpu_sum_us"] == 433.0 and p["busy_us"] == 433.0 and p["span_us"] == 438.0, p
        assert p["idle_us"] == 5.0 and p["gpu_busy_fraction"] == 0.9886 and p["range_wall_us"] == 500.0, p
        assert s["execution_order_range0"][:2] == ["loop_add_multiply_fusion.1", "fft.1.0"], s["execution_order_range0"]
        assert s["api_calls_in_ranges"] == 16 and s["activities_outside_ranges"] == 2, s
        assert s["kernels_by_name"][0] == {"name": "loop_add_multiply_fusion_1", "per_step": 0.5,
                                           "us_per_step": 250.0}, s["kernels_by_name"][0]
        assert "[jit__lambda_#8]" in r.stdout, r.stdout                 # several modules: shown in the table

        r = subprocess.run([sys.executable, script, db, "--out", out, "--steps-per-range", "2", "--range-regex", "^x$"],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
        assert r.returncode != 0 and "ERROR NSYS_OPS_NO_RANGES" in r.stdout, r.stdout
    print("NSYS_OPS_TEST_OK")


if __name__ == "__main__":
    main()
