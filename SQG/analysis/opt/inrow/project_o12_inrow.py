"""project_o12_inrow.py -- a PROJECTION (not a measurement) of o1.2's full-schedule A100 wall-clock with the in-tile
C2R step (tilerows_inrow_if25): option C's measured tilerows_if25 pieces (job 7840909, o12_extrapolate.py's formula)
with the measured per-solver-step saving of job 7853317 applied to s_step (12 frames x 25 = 300 solver steps per
optimizer step) and s_frame (25 solver steps). Everything else (ML step, init, loads, val, saves) is unchanged.
Stdlib only.

  python3 project_o12_inrow.py
"""
import json
import os

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results")
DDP = 0.005                                                            # o12_extrapolate.py: s/step over CXI, nodes > 1
STEPS_PER_FRAME, FRAMES_PER_OPT = 25, 12


def main():
    X = json.load(open(os.path.join(RES, "polaris_o12tr_7840909", "o12_extrapolation.json")))["solvers"]["tilerows_if25"]
    P = json.load(open(os.path.join(RES, "polaris_inrow_7853317", "inrow_probe_prod.json")))["bench_ms_per_step"]
    Q = json.load(open(os.path.join(RES, "polaris_inrow_7853317", "inrow_probe_cufft114.json")))["bench_ms_per_step"]
    deltas = {"IF arms, production cuFFT, eager (the o1.2 case)": P["tilerows32_if_eager"] - P["inrow32_if_eager"],
              "lean32ek-numerics arms, production cuFFT, eager": P["tilerows32_eager"] - P["inrow32_eager"],
              "lean32ek-numerics arms, production cuFFT, CUDA graph": P["tilerows32_graph"] - P["inrow32_graph"],
              "IF arms, cuFFT 11.4.1, eager": Q["tilerows32_if_eager"] - Q["inrow32_if_eager"]}
    s_step, s_frame = X["pieces"]["s_step"], X["pieces"]["s_frame"]
    print(f"tilerows_if25 measured in o1.2 (job 7840909): s_step {s_step} s, s_frame {s_frame} s")
    for name, d_ms in deltas.items():
        d = d_ms * 1e-3
        s2, f2 = s_step - FRAMES_PER_OPT * STEPS_PER_FRAME * d, s_frame - STEPS_PER_FRAME * d
        print(f"--- saving {d_ms:.4f} ms/solver step ({name}): s_step {s2:.4f} s ({s2 / s_step - 1:+.2%})")
        for cfg, pr in X["projections"].items():
            t = pr["terms_s"]
            assert abs(sum(t.values()) / 3600 - pr["hours"]) < 1e-6
            nodes = int(cfg.split(" node")[0])
            gpus = int(cfg.split(" GPUs")[0].split()[-1])
            new = dict(t, fill=t["fill"] / s_frame * f2, steps=pr["steps"] * (s2 + (DDP if nodes > 1 else 0.0)))
            h = sum(new.values()) / 3600
            print(f"  {cfg}: {pr['hours']:.1f} h -> {h:.1f} h ({h / pr['hours'] - 1:+.1%}); node-h {h * nodes:.0f}, "
                  f"GPU-h {h * gpus:.0f}")
    print("PROJECT_O12_INROW_OK")


if __name__ == "__main__":
    main()
