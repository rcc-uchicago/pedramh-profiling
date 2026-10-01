#!/usr/bin/env python3
"""port_ckpt_load_compare.py — M3 gate: every legacy checkpoint loads on the new venv as it did on the old.

    port_ckpt_load_compare.py LOAD.json REF.json [REF.json ...] [--expect N]

LOAD.json is ``port_golden_infer.py --load-only`` from the new venv; REF.json are the golden
manifests (``makani_port/golden/infer.json``, ``infer_ext.json``). For every checkpoint in the refs
the load must carry the same ``file_sha256``, ``params_sha256``, ``params_struct_sha256``,
``state_sha256`` and ``state_struct_sha256`` (tolerance ruling 7697688 row T1-d: equality, none of
these ignorable). Strictness of the load itself is ``build_wrapper_from_checkpoint``'s
``strict=True``. Prints ``PORT_CKPT_LOAD_OK n=<k>`` or one ``ERROR PORT_CKPT_LOAD_MISMATCH`` line
per differing field. Stdlib only.
"""
from __future__ import annotations

import argparse
import json
import sys

FIELDS = ("file_sha256", "params_sha256", "params_struct_sha256", "state_sha256", "state_struct_sha256")


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("load")
    p.add_argument("refs", nargs="+")
    p.add_argument("--expect", type=int, default=None, help="number of checkpoints required")
    a = p.parse_args(argv)

    with open(a.load) as f:
        got = json.load(f)["checkpoints"]
    ref = {}
    for path in a.refs:
        with open(path) as f:
            for tag, entry in json.load(f)["checkpoints"].items():
                if tag in ref:
                    print("ERROR PORT_CKPT_LOAD_DUPLICATE_TAG %s" % tag)
                    return 1
                ref[tag] = entry

    bad, n_ok = 0, 0
    for tag in sorted(ref):
        if tag not in got:
            print("ERROR PORT_CKPT_LOAD_MISMATCH %s missing_from_load" % tag)
            bad += 1
            continue
        diff = [k for k in FIELDS if got[tag].get(k) != ref[tag].get(k) or ref[tag].get(k) is None]
        for k in diff:
            print("ERROR PORT_CKPT_LOAD_MISMATCH %s %s ref=%s new=%s"
                  % (tag, k, str(ref[tag].get(k))[:12], str(got[tag].get(k))[:12]))
        bad += bool(diff)
        n_ok += not diff
    if a.expect is not None and len(ref) != a.expect:
        print("ERROR PORT_CKPT_LOAD_COUNT refs=%d expected=%d" % (len(ref), a.expect))
        bad += 1
    if bad:
        print("ERROR PORT_CKPT_LOAD_FAILED ok=%d/%d" % (n_ok, len(ref)))
        return 1
    print("PORT_CKPT_LOAD_OK n=%d" % n_ok)
    return 0


if __name__ == "__main__":
    sys.exit(main())
