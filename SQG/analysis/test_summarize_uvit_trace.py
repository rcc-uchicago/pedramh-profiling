"""Synthetic-trace test of summarize_uvit_trace.py (pure python, no torch; runs on a login node).

  python3 SQG/analysis/test_summarize_uvit_trace.py          # prints TRACE_TEST_OK
Checks: only GPU-side events count (a long cpu_op is ignored), steps come from the ProfilerStep# annotations, the busy
time is the union of overlapping kernels, and families / percentages are shares of GPU time.
"""
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
EVENTS = [
    {"ph": "X", "cat": "user_annotation", "name": "ProfilerStep#5", "ts": 0, "dur": 1000},
    {"ph": "X", "cat": "user_annotation", "name": "ProfilerStep#6", "ts": 1000, "dur": 1000},
    {"ph": "X", "cat": "kernel", "name": "void at::native::conv_depthwise2d_forward_kernel<0>", "ts": 10, "dur": 300},
    {"ph": "X", "cat": "kernel", "name": "ampere_sgemm_128x64_tn", "ts": 200, "dur": 200},      # overlaps the first
    {"ph": "X", "cat": "gpu_memcpy", "name": "Memcpy DtoD (Device -> Device)", "ts": 1100, "dur": 100},
    {"ph": "X", "cat": "kernel", "name": "void at::native::conv_depthwise2d_forward_kernel<0>", "ts": 1300, "dur": 300},
    {"ph": "X", "cat": "cpu_op", "name": "aten::mm", "ts": 0, "dur": 5000},                     # host side: ignored
]


def main():
    with tempfile.TemporaryDirectory() as d:
        trace, out = os.path.join(d, "t.json"), os.path.join(d, "s.json")
        with open(trace, "w") as fh:
            json.dump({"traceEvents": EVENTS}, fh)
        r = subprocess.run([sys.executable, os.path.join(HERE, "summarize_uvit_trace.py"), trace, "--out", out],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
        assert r.returncode == 0 and "UVIT_TRACE_OK" in r.stdout, r.stdout
        with open(out) as fh:
            s = json.load(fh)
    assert s["profiled_steps"] == 2, s
    assert s["gpu_ms_per_step"] == 0.45, s                 # (300 + 200 + 100 + 300) us / 2 steps
    assert s["busy_ms_per_step"] == 0.4, s                 # union: 10-400, 1100-1200, 1300-1600 = 800 us / 2
    fam = s["families_pct_of_gpu_time"]
    assert abs(fam["depthwise_conv_native"] - 66.67) < 0.01 and abs(fam["gemm"] - 22.22) < 0.01, fam
    assert abs(fam["memcpy_memset"] - 11.11) < 0.01, fam
    assert s["top_kernels"][0]["calls_per_step"] == 1.0 and s["max_single_kernel_pct"] == 66.67, s
    print("TRACE_TEST_OK")


if __name__ == "__main__":
    main()
