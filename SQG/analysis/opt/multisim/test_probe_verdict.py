"""test_probe_verdict.py -- probe_verdict.py on job 7815096's real failed probe and on a synthetic passing one.

    python3 SQG/analysis/opt/multisim/test_probe_verdict.py        -> PROBE_VERDICT_TEST_OK
"""
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import probe_verdict as PV  # noqa: E402

SQG = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
FAILED_LOG = os.path.join(SQG, "analysis", "results", "polaris_opt_round1b_ms_7815096", "multisim_ncclprobe_ref.log")

OK_LOG = """\
h 0: x:1:1 [0] NCCL INFO NET/Plugin: Loaded net plugin AWS Libfabric (v6)
h 0: x:1:2 [0] NCCL INFO NET/OFI Selected Provider is cxi (found 2 nics)
h 1: x:1:2 [0] NCCL INFO NET/OFI Selected Provider is cxi (found 2 nics)
h 0: x:1:2 [0] NCCL INFO Channel 00/0 : 0[0] -> 1[1] via P2P/CUMEM/read
h 1: x:1:2 [0] NCCL INFO Channel 00/0 : 5[1] -> 1[1] [receive] via NET/AWS Libfabric/1/GDRDMA
h 1: x:1:2 [0] NCCL INFO Channel 01/0 : 1[1] -> 5[1] [send] via NET/AWS Libfabric/1/GDRDMA
h 2: x:1:2 [0] NCCL INFO Channel 01/0 : 2[2] -> 6[2] [send] via NET/AWS Libfabric/0
h 0: x:1:2 [0] NCCL INFO Connected all rings, use ring PXN 0 GDR 1
h 0: x:1:1 [0] NCCL INFO NET/Plugin: Failed to find ncclNetPlugin_v8 symbol.
h 0: libfabric:1:1::core:core:cuda_hmem_verify_devices():123<info> Number of CUDA devices detected: 1
"""


def main():
    # 1. the real 7815096 failure: no JSON was written; the signature counts must match Part VIII
    ok, line = PV.verdict("/nonexistent.json", FAILED_LOG, "r1b")
    c, gdr = PV.scan(FAILED_LOG)
    assert not ok and "reason=no_json" in line, line
    assert c["cxi"] == 8 and c["plugin"] == 8 and c["gdrdma"] == 16 and c["p2p"] == 24 and c["shm"] == 0, c
    assert c["regfail"] == 8 and c["cxil_map"] > 0 and c["dlopen_cudart_fail"] == 0 and gdr == "?", (c, gdr)
    assert c["hmem_cuda_ok"] == 0 and c["failed_to_find_sym"] == 0, c   # no FI_LOG; NCCL's plugin-symbol lines excluded
    with tempfile.TemporaryDirectory() as d:
        log = os.path.join(d, "ok.log")
        with open(log, "w") as fh:
            fh.write(OK_LOG)
        js = os.path.join(d, "ok.json")

        def write(**kw):
            res = {"probe_ok_all": True, "busbw_GBps_min": 17.9, "errors": [], "nccl_version": "2.20.5"}
            res.update(kw)
            with open(js, "w") as fh:
                json.dump(res, fh)

        # 2. a passing arm
        write()
        ok, line = PV.verdict(js, log, "B_shim")
        c, gdr = PV.scan(log)
        assert ok and line.startswith("NCCL_PROBE_OK B_shim busbw_min=17.9"), line
        assert c["gdrdma"] == 2 and c["net_nogdr"] == 1 and gdr == "1" and c["cxi"] == 2, (c, gdr)
        assert c["hmem_cuda_ok"] == 1 and c["failed_to_find_sym"] == 0, c
        # 3. provider found but wrong values / no bandwidth / rank errors -> FAIL (the 7815096 token bug)
        for kw, why in (({"probe_ok_all": False}, "values"), ({"busbw_GBps_min": None}, "no_busbw"),
                        ({"errors": [3]}, "rank_errors")):
            write(**kw)
            ok, line = PV.verdict(js, log, "X")
            assert not ok and f"reason={why}" in line, (kw, line)
    print("PROBE_VERDICT_TEST_OK 7815096 log -> FAIL (cxi 8, gdrdma 16, regfail 8); synthetic pass + 3 fail modes")


if __name__ == "__main__":
    main()
