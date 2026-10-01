#!/usr/bin/env python3
"""port_golden_compare.py — bitwise comparison of two golden JSON files (stdlib only, any python 3).

Deep-compares REF and NEW, ignoring keys passed with ``--ignore`` (dotted paths). Every leaf must
be identical — losses and grad norms are stored as ``float.hex`` strings, tensors as sha256, so
equality here is bitwise equality of the underlying numbers. Prints one line per difference
(first 20) and ``GOLDEN_MATCH <label>`` or ``ERROR GOLDEN_MISMATCH <label> n=<k>``.
"""
import argparse
import json
import sys


def walk(a, b, path, ignore, diffs):
    if path in ignore:
        return
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            sub = "%s.%s" % (path, k) if path else str(k)
            if k not in a or k not in b:
                if sub not in ignore:
                    diffs.append("%s: only in %s" % (sub, "ref" if k in a else "new"))
                continue
            walk(a[k], b[k], sub, ignore, diffs)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            diffs.append("%s: length %d != %d" % (path, len(a), len(b)))
        for i, (x, y) in enumerate(zip(a, b)):
            walk(x, y, "%s[%d]" % (path, i), ignore, diffs)
    elif a != b:
        diffs.append("%s: %r != %r" % (path, a, b))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ref")
    p.add_argument("new")
    p.add_argument("--label", default="golden")
    p.add_argument("--ignore", nargs="*", default=[])
    a = p.parse_args()
    diffs = []
    walk(json.load(open(a.ref)), json.load(open(a.new)), "", set(a.ignore), diffs)
    for d in diffs[:20]:
        print("  DIFF " + d)
    if diffs:
        print("ERROR GOLDEN_MISMATCH %s n=%d ref=%s new=%s" % (a.label, len(diffs), a.ref, a.new))
        return 1
    print("GOLDEN_MATCH %s" % a.label)
    return 0


if __name__ == "__main__":
    sys.exit(main())
