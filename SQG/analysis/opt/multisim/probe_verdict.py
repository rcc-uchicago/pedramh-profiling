"""probe_verdict.py -- the PASS/FAIL token of one NCCL probe arm, from its JSON and its log (standard library only).

    python3 probe_verdict.py <multisim_ncclprobe_ref_<tag>.json> <log> <tag>
prints one line and exits 0 / 1:
    NCCL_PROBE_OK <tag> busbw_min=<GB/s> nccl=<v> cxi=<n> gdrdma=<n> rings_gdr=<0|1|?> p2p=<n> shm=<n> ...
    ERROR NCCL_PROBE_FAILED <tag> reason=<no_json|values|no_busbw|rank_errors> cxi=<n> cxil_map=<n> regfail=<n> ...

PASS needs the all_reduce to have run: the JSON exists, every rank's values were right (probe_ok_all) and a busbw was
measured. Round 1b's NCCL_CXI_OK (job 7815096) only counted "Selected Provider is cxi", which NCCL prints before the
connection setup, so it fired on a probe whose all_reduce never ran (combine_handoffs, 2026-11-05). The counts are
information: the transport (cxi provider lines, NET/GDRDMA channel lines, the GDR flag of "Connected all rings" where
NCCL prints it), and the 7815096 failure signature (cxil_map write errors, "Unable to register memory", and libfabric's
"Failed to dlopen libcudart.so", which needs FI_LOG_LEVEL=info).
"""
import json
import os
import re
import sys

COUNTS = {
    "cxi": r"Selected Provider is cxi",
    "plugin": r"Loaded net plugin AWS Libfabric",
    "gdrdma": r"via NET/.*/GDRDMA",                                   # the net name has a space ("AWS Libfabric")
    "net_nogdr": r"via NET/(?!.*/GDRDMA)",
    "p2p": r"via P2P/",
    "shm": r"via SHM/",
    "cxil_map": r"cxil_map: write error",
    "regfail": r"Unable to register memory",
    "dlopen_cudart_fail": r"Failed to dlopen libcudart\.so",
    # FI_LOG_LEVEL=info markers (combine_handoffs' working 7629082 M1 log): libfabric's CUDA HMEM came up, or a cudart
    # symbol was missing (a risk with the env's 12.1 runtime). NCCL's own "Failed to find ncclNetPlugin_v8 symbol" /
    # "ncclCollNetPlugin" INFO lines are harmless and excluded.
    "hmem_cuda_ok": r"Number of CUDA devices detected",
    "failed_to_find_sym": r"Failed to find (?!ncclNetPlugin|ncclCollNetPlugin)",
    "rank_failed": r"ERROR MULTISIM_RANK_FAILED",
}


def scan(log_path):
    c = {k: 0 for k in COUNTS}
    gdr = set()
    if not os.path.isfile(log_path):
        return c, "?"
    with open(log_path, errors="replace") as fh:
        for line in fh:
            for k, pat in COUNTS.items():
                if re.search(pat, line):
                    c[k] += 1
            m = re.search(r"Connected all rings.*\bGDR (\d)", line)
            if m:
                gdr.add(m.group(1))
    return c, ",".join(sorted(gdr)) if gdr else "?"


def verdict(json_path, log_path, tag):
    c, gdr = scan(log_path)
    info = " ".join(f"{k}={v}" for k, v in c.items())
    if not os.path.isfile(json_path):
        return False, f"ERROR NCCL_PROBE_FAILED {tag} reason=no_json rings_gdr={gdr} {info}"
    with open(json_path) as fh:
        res = json.load(fh)
    bw, ok, errs = res.get("busbw_GBps_min"), res.get("probe_ok_all"), res.get("errors")
    reason = ("values" if not ok else "no_busbw" if bw is None else "rank_errors" if errs else None)
    if reason:
        return False, f"ERROR NCCL_PROBE_FAILED {tag} reason={reason} busbw_min={bw} rings_gdr={gdr} {info}"
    return True, (f"NCCL_PROBE_OK {tag} busbw_min={bw} GB/s nccl={res.get('nccl_version')} rings_gdr={gdr} {info}")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 3:
        print("usage: probe_verdict.py <json> <log> <tag>")
        return 2
    ok, line = verdict(*argv)
    print(line)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
