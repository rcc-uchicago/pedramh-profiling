#!/usr/bin/env python3
"""port_venv_check.py — makani-port M1 gate: provenance + symbol imports of a venv (``VENV_MAIN_OK``).

Run with the venv's own python. Checks (1) every package resolves where it must — makani,
torch_harmonics, physicsnemo and zarr from the venv itself, torch from the base conda; (2) the
makani and torch_harmonics commits equal the pins; (3) physicsnemo is a plain (non-editable)
install; (4) every ``module:name`` we import (``--symbols``, one per line) imports, printed as
``OK``/``MISSING``; (5) the old venv's makani dist-info sha256s equal ``--old-direct-url`` /
``--old-record``. Imports only makani, never our sfno_* packages (those are M2's business).
"""
import argparse
import glob
import hashlib
import importlib
import json
import os
import sys


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def dist_info(site, name):
    hits = sorted(glob.glob(os.path.join(site, name + "-*.dist-info")))
    return hits[0] if len(hits) == 1 else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", required=True)
    p.add_argument("--makani-sha", required=True)
    p.add_argument("--th-sha", required=True)
    p.add_argument("--old-venv", required=True)
    p.add_argument("--old-direct-url", required=True)
    p.add_argument("--old-record", required=True)
    a = p.parse_args()

    venv = os.path.realpath(os.environ["VIRTUAL_ENV"])
    site = glob.glob(os.path.join(venv, "lib", "python3.*", "site-packages"))[0]
    fails = []

    print("== provenance (venv=%s)" % venv)
    must_be_in_venv = ["makani", "torch_harmonics", "physicsnemo", "zarr"]
    for m in ["torch", "torch_harmonics", "makani", "physicsnemo", "zarr", "numpy", "h5py", "wandb",
              "nvidia.dali", "warp", "tensorly", "tltorch", "ruamel.yaml", "xarray", "netCDF4"]:
        try:
            mod = importlib.import_module(m)
        except Exception as e:  # noqa: BLE001
            fails.append("import %s" % m)
            print("  FAIL %-14s %s: %s" % (m, type(e).__name__, str(e)[:100]))
            continue
        f = os.path.realpath(getattr(mod, "__file__", "") or "")
        where = "venv" if f.startswith(venv + os.sep) else "base"
        print("  %-14s %-24s %s %s" % (m, getattr(mod, "__version__", "?"), where, f))
        if m in must_be_in_venv and where != "venv":
            fails.append("%s not from venv" % m)
        if m == "torch" and where != "base":
            fails.append("torch not from base conda")

    for name, want in (("makani", a.makani_sha), ("torch_harmonics", a.th_sha)):
        di = dist_info(site, name)
        got = json.load(open(os.path.join(di, "direct_url.json")))["vcs_info"]["commit_id"] if di else None
        ok = got == want
        print("  %-14s commit %s %s" % (name, got, "OK" if ok else "WANT " + want))
        if not ok:
            fails.append("%s commit" % name)
    di = dist_info(site, "nvidia_physicsnemo")
    du = os.path.join(di, "direct_url.json") if di else None
    editable = bool(du and os.path.exists(du) and json.load(open(du)).get("dir_info", {}).get("editable"))
    tree = open(os.path.join(venv, "physicsnemo_sfno_tree.txt")).read().strip() \
        if os.path.exists(os.path.join(venv, "physicsnemo_sfno_tree.txt")) else None
    print("  physicsnemo    editable=%s physicsnemo_sfno_tree=%s" % (editable, tree))
    if editable or tree is None:
        fails.append("physicsnemo editable or unrecorded")

    print("== symbols")
    n_ok = n_all = 0
    for line in open(a.symbols):
        line = line.split()[0] if line.strip() else ""
        if not line or line.endswith(":..."):
            continue
        n_all += 1
        mod, _, name = line.partition(":")
        try:
            m = importlib.import_module(mod)
            if name and not hasattr(m, name):
                importlib.import_module(mod + "." + name)
            n_ok += 1
            print("  OK      %s" % line)
        except Exception as e:  # noqa: BLE001
            print("  MISSING %s  (%s: %s)" % (line, type(e).__name__, str(e)[:120]))
    if n_ok != n_all:
        fails.append("symbols %d/%d" % (n_ok, n_all))

    old_site = glob.glob(os.path.join(a.old_venv, "lib", "python3.*", "site-packages"))[0]
    odi = dist_info(old_site, "makani")
    od, orc = sha(os.path.join(odi, "direct_url.json")), sha(os.path.join(odi, "RECORD"))
    old_ok = (od == a.old_direct_url) and (orc == a.old_record)
    print("== old venv makani dist-info direct_url=%s RECORD=%s %s" % (od, orc, "UNCHANGED" if old_ok else "CHANGED"))
    if not old_ok:
        fails.append("OLD VENV CHANGED")

    if fails:
        print("ERROR VENV_MAIN_FAILED %s" % fails)
        return 1
    print("VENV_MAIN_OK makani=%s torch_harmonics=%s symbols=%d/%d old_venv_unchanged=1"
          % (a.makani_sha[:12], a.th_sha[:12], n_ok, n_all))
    return 0


if __name__ == "__main__":
    sys.exit(main())
