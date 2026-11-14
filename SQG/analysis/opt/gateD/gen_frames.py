"""gen_frames.py -- stored production frames for gate D (item 4). One process per GPU (run_gateD_polaris.pbs).

  python gen_frames.py --out <frames dir> --worker <w> --ckpt <member>:<ckpt_t*.npz> [--ckpt ...] [--seg-t 1.0]
                       [--ref-dir <taxonomy raw dir>]

Per checkpoint:
- Load it as o1.2 does (GD.load_checkpoint under the member's CONFIG_HASH, o1.2 L192-204).
- Run lean_u1 (float64, bitwise production: jobs 7815637, 7824121, 7835278) for SEG_T time units.
- Keep o1.2's frame, float32(theta[::16, ::16]) / 8, every 40 solver steps (o1.2's ring stride). Frame 0 is the
  checkpoint. A 0.02 pair is frames (k, k + 100).
- Write m<MM>_t<tttt.tt>.npy (n_frames, 256, 256) float32 via <name>.tmp.npy and a rename. A complete file is skipped
  on rerun, so a requeued job resumes.
- Bitwise check: frames 0 and k = 100, 200, 300, 400 against job 7824121's production frames of the same checkpoint,
  when --ref-dir has them.

Prints GEN_SEGMENT lines and GEN_WORKER_DONE w=<w> failures=<n>.
"""
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse                                                      # noqa: E402
import hashlib                                                       # noqa: E402
import json                                                          # noqa: E402
import re                                                            # noqa: E402
import sys                                                           # noqa: E402
import time                                                          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OPT = os.path.dirname(HERE)
ANALYSIS = os.path.dirname(OPT)
for p in (os.path.join(OPT, "seed1"), ANALYSIS):
    sys.path.insert(0, p)
import profile_solver as PS                                          # noqa: E402  (import_asqg: the vendored solver)
import lean_solver as LS                                             # noqa: E402  (lean_u1)

STRIDE, M, THETA_SCALE = 40, 16, 8.0


def seg_name(member, t_c):
    return f"m{member:02d}_t{t_c:07.2f}"


def ref_index(ref_dir):
    """(member, t) -> {frame k: (npz path, key)} from the taxonomy job's worker_*.json and pair npz."""
    idx = {}
    if not ref_dir:
        return idx
    for w in range(4):
        fn = os.path.join(ref_dir, f"worker_{w}.json")
        if not os.path.exists(fn):
            continue
        with open(fn) as fh:
            states = json.load(fh)["states"]
        for s, st in enumerate(states):
            if st.get("offset_calls", 0):
                continue
            fr = {0: (os.path.join(ref_dir, f"pair_w{w}_s{s}_p0.npz"), "lean_u1__sub__0")}
            for j in range(len(st["pairs"])):
                fr[100 * (j + 1)] = (os.path.join(ref_dir, f"pair_w{w}_s{s}_p{j}.npz"), "lean_u1__sub__100")
            idx[(int(st["member"]), round(float(st["t"]), 6))] = fr
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--worker", type=int, required=True)
    ap.add_argument("--ckpt", action="append", required=True, help="<member>:<path of ckpt_t*.npz>")
    ap.add_argument("--seg-t", type=float, default=1.0)
    ap.add_argument("--ref-dir", default="")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    GD = PS.import_asqg()
    import jax
    import jax.numpy as jnp
    import numpy as np
    N, DT = GD.N, GD.DT
    n_calls = int(round(a.seg_t / (STRIDE * DT)))
    assert abs(n_calls * STRIDE * DT - a.seg_t) < 1e-12, (n_calls, a.seg_t)
    template = GD.sqg.init(GD.p, jax.random.key(GD.SEED_INITIAL_CONDITION))

    def config_hash(member):                                         # as o1.2 L192-195
        cfg = dict(GD.CONFIG, member=member, seed_initial_condition=member)
        return hashlib.sha256(json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def load(path, member):                                          # as o1.2 L198-204
        own, GD.CONFIG_HASH = GD.CONFIG_HASH, config_hash(member)
        try:
            return GD.load_checkpoint(template, path)[0]
        finally:
            GD.CONFIG_HASH = own

    _, ncol = LS.preconditions(GD)
    cu1 = LS.build_consts(GD, ncol)
    u1 = LS.make_advance(GD, STRIDE, ncol=ncol, unroll1=True)
    coarse = jax.jit(lambda st: jnp.fft.irfft2(st.state.qh[0], s=(N, N))[::M, ::M].astype(jnp.float32))
    refs = ref_index(a.ref_dir)
    failures = 0
    for spec in a.ckpt:
        m_s, path = spec.split(":", 1)
        member = int(m_s)
        t_c = float(re.search(r"ckpt_t([0-9.]+)\.npz", path).group(1))
        name = seg_name(member, t_c)
        final = os.path.join(a.out, name + ".npy")
        meta_fn = os.path.join(a.out, name + ".json")
        if os.path.exists(final) and os.path.exists(meta_fn):
            with open(meta_fn) as fh:
                meta = json.load(fh)
            if meta.get("n_frames") == n_calls + 1 and meta.get("complete"):
                print(f"GEN_SEGMENT {name} skipped (complete, {meta['n_frames']} frames)", flush=True)
                continue
        t0 = time.time()
        st = load(path, member)
        tmp = os.path.join(a.out, name + ".tmp.npy")
        frames = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32, shape=(n_calls + 1, N // M, N // M))
        frames[0] = np.asarray(coarse(st), dtype=np.float32) / np.float32(THETA_SCALE)
        for i in range(1, n_calls + 1):
            st = u1(st, cu1)
            frames[i] = np.asarray(coarse(st), dtype=np.float32) / np.float32(THETA_SCALE)
        frames.flush()
        checks = {}
        for k, (fn, key) in sorted(refs.get((member, round(t_c, 6)), {}).items()):
            if k <= n_calls:
                with np.load(fn) as z:
                    checks[k] = bool(np.array_equal(frames[k], z[key]))
        del frames
        os.replace(tmp, final)
        verdict = "NO_REF" if not checks else ("PASS" if all(checks.values()) else "FAIL")
        failures += verdict == "FAIL"
        meta = {"member": member, "t_c": t_c, "n_frames": n_calls + 1, "seg_t": a.seg_t, "stride_steps": STRIDE,
                "solver": "lean_u1", "seconds": round(time.time() - t0, 1), "bitwise_vs_taxonomy": checks,
                "verdict": verdict, "complete": True, "ckpt": path}
        with open(meta_fn, "w") as fh:
            json.dump(meta, fh, indent=1)
        print(f"GEN_SEGMENT {name} frames={n_calls + 1} seconds={meta['seconds']} bitwise={verdict} "
              f"checked={sorted(checks)}", flush=True)
    print(f"GEN_WORKER_DONE w={a.worker} failures={failures}", flush=True)


if __name__ == "__main__":
    main()
