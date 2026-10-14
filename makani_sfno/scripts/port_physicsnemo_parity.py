#!/usr/bin/env python3
"""port_physicsnemo_parity.py — are the two venvs' physicsnemo installs the same source? (stdlib only)

The old venv's physicsnemo is an EDITABLE install of some checkout's ``physicsnemo_sfno`` working
tree; the new venv's is a plain copy of ``git archive`` tree ``1674e93e``. Resolves each install's
``physicsnemo/`` package directory from its dist-info (``direct_url.json`` for the editable one,
``RECORD`` for the plain one), sha256-hashes every file except ``__pycache__``/``*.pyc``, and lists
files that differ or exist on one side only. ``PHYSICSNEMO_PARITY_OK`` iff both sets are equal.
"""
import glob
import hashlib
import json
import os
import sys


def site(venv):
    return glob.glob(os.path.join(venv, "lib", "python3.*", "site-packages"))[0]


def pkg_dir(venv):
    s = site(venv)
    di = sorted(glob.glob(os.path.join(s, "nvidia_physicsnemo-*.dist-info")))[0]
    du = os.path.join(di, "direct_url.json")
    if os.path.exists(du):
        d = json.load(open(du))
        if d.get("dir_info", {}).get("editable"):
            root = d["url"][len("file://"):]
            return os.path.join(root, "physicsnemo"), "editable:" + root
    return os.path.join(s, "physicsnemo"), "installed"


def hashes(root):
    out = {}
    for d, dirs, files in os.walk(root):
        dirs[:] = [x for x in dirs if x != "__pycache__"]
        for f in files:
            if f.endswith(".pyc"):
                continue
            p = os.path.join(d, f)
            out[os.path.relpath(p, root)] = hashlib.sha256(open(p, "rb").read()).hexdigest()
    return out


def main():
    old_venv, new_venv = sys.argv[1], sys.argv[2]
    (od, okind), (nd, nkind) = pkg_dir(old_venv), pkg_dir(new_venv)
    ho, hn = hashes(od), hashes(nd)
    print("PHYSICSNEMO old=%s (%s, %d files) new=%s (%s, %d files)" % (od, okind, len(ho), nd, nkind, len(hn)))
    only_old = sorted(set(ho) - set(hn))
    only_new = sorted(set(hn) - set(ho))
    differ = sorted(k for k in set(ho) & set(hn) if ho[k] != hn[k])
    for label, lst in (("only_old", only_old), ("only_new", only_new), ("differ", differ)):
        for k in lst[:15]:
            print("  %s %s" % (label, k))
        if len(lst) > 15:
            print("  %s ... %d more" % (label, len(lst) - 15))
    agg = lambda h: hashlib.sha256("".join("%s %s\n" % kv for kv in sorted(h.items())).encode()).hexdigest()
    if not (only_old or only_new or differ):
        print("PHYSICSNEMO_PARITY_OK files=%d tree_sha256=%s" % (len(ho), agg(ho)))
        return 0
    print("ERROR PHYSICSNEMO_PARITY_MISMATCH only_old=%d only_new=%d differ=%d old_sha256=%s new_sha256=%s"
          % (len(only_old), len(only_new), len(differ), agg(ho), agg(hn)))
    return 1


if __name__ == "__main__":
    sys.exit(main())
