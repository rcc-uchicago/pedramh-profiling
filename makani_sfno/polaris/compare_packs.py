#!/usr/bin/env python3
"""Compare two pack output roots and report every difference.

Used to check that a change to the E3SM converters leaves their output unchanged
(polaris_pack_equivalence.pbs). For each file under either root:
    .h5    the same datasets with the same shape, dtype, chunking and bytes, the same
           attributes, and the same dimension scales attached to each dimension
    .npy   the same shape, dtype and bytes
    .npz   the same arrays, byte for byte
    .json  the same parsed content
    other  the same bytes
Log files (.log) are skipped.

Usage:
    python compare_packs.py REFERENCE_ROOT NEW_ROOT

Prints PACKS_IDENTICAL, or ERROR PACKS_DIFFER followed by the first 20 differences.
"""

import json
import os
import sys

import h5py
import numpy as np

# Object references that differ between any two files. The dimension scales they
# point to are compared by name instead.
SKIP_ATTRS = {"DIMENSION_LIST", "REFERENCE_LIST"}


def same_array(a, b):
    """Return True if two arrays have the same shape, dtype and contents, bit for bit."""
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.dtype.kind == "O":  # h5py strings come back as Python objects
        return a.tolist() == b.tolist()
    return a.tobytes() == b.tobytes()


def compare_attrs(where, a, b, diffs):
    """Append a message to diffs for each attribute that differs."""
    names_a = sorted(k for k in a.keys() if k not in SKIP_ATTRS)
    names_b = sorted(k for k in b.keys() if k not in SKIP_ATTRS)
    if names_a != names_b:
        diffs.append(f"{where}: attribute names {names_a} != {names_b}")
        return
    for k in names_a:
        if not same_array(a[k], b[k]):
            diffs.append(f"{where}: attribute {k}: {a[k]!r} != {b[k]!r}")


def compare_h5(rel, path_a, path_b, diffs):
    """Compare two HDF5 files object by object."""
    with h5py.File(path_a, "r") as fa, h5py.File(path_b, "r") as fb:
        compare_attrs(rel, fa.attrs, fb.attrs, diffs)
        names_a, names_b = [], []
        fa.visit(names_a.append)
        fb.visit(names_b.append)
        if sorted(names_a) != sorted(names_b):
            diffs.append(f"{rel}: objects {sorted(names_a)} != {sorted(names_b)}")
            return
        for name in sorted(names_a):
            da, db = fa[name], fb[name]
            where = f"{rel}:{name}"
            compare_attrs(where, da.attrs, db.attrs, diffs)
            if not isinstance(da, h5py.Dataset):
                continue
            if (da.shape, da.dtype, da.chunks) != (db.shape, db.dtype, db.chunks):
                diffs.append(f"{where}: shape/dtype/chunks {(da.shape, da.dtype, da.chunks)} "
                             f"!= {(db.shape, db.dtype, db.chunks)}")
                continue
            dims_a = [sorted(da.dims[i].keys()) for i in range(da.ndim)]
            dims_b = [sorted(db.dims[i].keys()) for i in range(db.ndim)]
            if dims_a != dims_b:
                diffs.append(f"{where}: dimension scales {dims_a} != {dims_b}")
            if da.ndim == 4:  # one sample at a time keeps memory small
                for t in range(da.shape[0]):
                    a, b = da[t], db[t]
                    if not same_array(a, b):
                        diffs.append(f"{where}: sample {t} differs, max |diff| "
                                     f"{np.nanmax(np.abs(a.astype(np.float64) - b)):.3e}")
                        break
            elif not same_array(da[...], db[...]):
                diffs.append(f"{where}: values differ")


def compare_file(rel, path_a, path_b, diffs):
    """Compare one file present under both roots."""
    if rel.endswith(".h5"):
        compare_h5(rel, path_a, path_b, diffs)
    elif rel.endswith(".npy"):
        if not same_array(np.load(path_a), np.load(path_b)):
            diffs.append(f"{rel}: values differ")
    elif rel.endswith(".npz"):
        za, zb = np.load(path_a), np.load(path_b)
        if sorted(za.files) != sorted(zb.files):
            diffs.append(f"{rel}: arrays {sorted(za.files)} != {sorted(zb.files)}")
        else:
            diffs += [f"{rel}:{k}: values differ" for k in za.files if not same_array(za[k], zb[k])]
    elif rel.endswith(".json"):
        with open(path_a) as fa, open(path_b) as fb:
            if json.load(fa) != json.load(fb):
                diffs.append(f"{rel}: content differs")
    else:
        with open(path_a, "rb") as fa, open(path_b, "rb") as fb:
            if fa.read() != fb.read():
                diffs.append(f"{rel}: bytes differ")


def list_files(root):
    """Return the paths of all files under root, relative to it, except logs."""
    out = set()
    for dirpath, _, files in os.walk(root):
        for name in files:
            if not name.endswith(".log"):
                out.add(os.path.relpath(os.path.join(dirpath, name), root))
    return out


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    ref, new = sys.argv[1], sys.argv[2]
    files_ref, files_new = list_files(ref), list_files(new)
    diffs = [f"only in {ref}: {p}" for p in sorted(files_ref - files_new)]
    diffs += [f"only in {new}: {p}" for p in sorted(files_new - files_ref)]
    for rel in sorted(files_ref & files_new):
        compare_file(rel, os.path.join(ref, rel), os.path.join(new, rel), diffs)
    if not files_ref:
        diffs.append(f"no files under {ref}")
    if diffs:
        print(f"ERROR PACKS_DIFFER: {len(diffs)} difference(s) between {ref} and {new}")
        for d in diffs[:20]:
            print("  " + d)
        sys.exit(1)
    print(f"PACKS_IDENTICAL {len(files_ref)} files: {ref} == {new}")


if __name__ == "__main__":
    main()
