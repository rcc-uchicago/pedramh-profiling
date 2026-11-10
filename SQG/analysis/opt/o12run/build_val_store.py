"""build_val_store.py -- a small validation store for o12_polaris.py, from the taxonomy job's production frames.

o1.2 validates on val_m16.zarr (held-out trajectories, on Weidong's machine only). For a TIMING run any frames of the
right shape will do; these are real ones: per archived state (16, job 7824121), production's (lean_u1, bitwise) o1.2
frames theta[::16, ::16] at 0, 0.02, 0.04, 0.06, 0.08 -> 5 rows = 4 pairs per trajectory, 64 pairs in all.
They come from the TRAINING members' first segments, so the val loss of a run on them is in-sample: not a science
number. Layout read by o12_polaris.py's zarr stand-in: <out>/theta.npy (rows, 256, 256) float32 and <out>/attrs.json
{"sources": [{"id", "name", "offset", "n_records", "t_first"}, ...]} as make_pairs() reads them.

  python3 build_val_store.py --ref-dir <taxonomy raw dir> --out <dir>      # numpy only; prints VAL_STORE_OK
"""
import argparse
import json
import os

import numpy as np

THETA_SCALE = np.float32(8.0)
PAIR_T = 0.02


def build(ref_dir, out):
    rows, sources = [], []
    for w in range(4):
        fn = os.path.join(ref_dir, f"worker_{w}.json")
        if not os.path.exists(fn):
            continue
        with open(fn) as fh:
            states = json.load(fh)["states"]
        for s, st in enumerate(states):
            if st.get("offset_calls", 0):
                continue
            frames = []
            for j in range(len(st["pairs"])):
                with np.load(os.path.join(ref_dir, f"pair_w{w}_s{s}_p{j}.npz")) as z:
                    a0, a1 = z["lean_u1__sub__0"], z["lean_u1__sub__100"]
                if j == 0:
                    frames.append(a0)
                elif not np.array_equal(a0, frames[-1]):                  # pairs chain: p_j starts where p_j-1 ends
                    raise SystemExit(f"ERROR VAL_STORE_CHAIN w{w} s{s} p{j}: start frame != previous end frame")
                frames.append(a1)
            sources.append({"id": len(sources), "name": f"m{int(st['member']):02d}_t{float(st['t']):g}",
                            "member": int(st["member"]), "t_first": float(st["t"]), "dt": PAIR_T,
                            "offset": len(rows), "n_records": len(frames)})
            rows.extend((f * THETA_SCALE).astype(np.float32) for f in frames)      # sub = theta / 8 -> theta (exact)
    if not rows:
        raise SystemExit(f"ERROR VAL_STORE_EMPTY no worker_*.json states in {ref_dir}")
    os.makedirs(out, exist_ok=True)
    theta = np.stack(rows)
    np.save(os.path.join(out, "theta.npy"), theta)
    with open(os.path.join(out, "attrs.json"), "w") as fh:
        json.dump({"sources": sources, "built_from": os.path.abspath(ref_dir)}, fh, indent=1)
    n_pairs = sum(s["n_records"] - 1 for s in sources)
    print(f"VAL_STORE_OK trajectories={len(sources)} rows={theta.shape[0]} pairs={n_pairs} shape={theta.shape[1:]} "
          f"dtype={theta.dtype} -> {out}")
    return theta, sources


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    build(a.ref_dir, a.out)


if __name__ == "__main__":
    main()
