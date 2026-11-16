"""o12_extrapolate.py -- parse o1.2 logs (o1.2's own lines + o12_polaris.py's O12_* lines) and extrapolate a FULL
o1.2 run's A100 wall-clock from the measured pieces. Stdlib only (login node OK). Formula and configurations are fixed
in PREDICTIONS_o12.md; the constants below are o1.2's.

  python3 o12_extrapolate.py --log <solver>=<rank-0 log> [--log ...] --out <dir>      # prints O12_EXTRAPOLATE_OK

Per solver, from its log:
  s_step     steady-state seconds per optimizer step = median over epochs >= 1 of the per-epoch median step interval
  s_frame    steady fill seconds per ring frame (40 solver steps + o1.2's frame handling, no ML step)
  t_init     process start -> fill done, minus the fill itself (imports, JIT, model, DDP)
  t_load     checkpoint load seconds (first segment)
  t_val      validation seconds per epoch, per val batch per rank;  t_save  save_last (+ save_best) seconds
Full run (configuration: n GPUs, batch b per GPU, nodes):
  steps   = N_EPOCH x ((N_OFFLINE_PAIRS // n) // b)                    (o1.2 L230-231)
  T       = t_init + t_load + (LAG + START_SEGS x PAIRS_PER_SEG) x s_frame
            + steps x (s_step + ddp_xnode [nodes > 1])
            + N_EPOCH x (t_val x ceil(ceil(N_VAL / n) / b) + t_save_last + t_save_best)
            + ceil(steps x FRAMES_PER_STEP / FRAMES_PER_SEG) x t_load
ddp_xnode = 0.005 s/step (DDP over CXI, job 7815637, notes Part IX). No restarts are modelled.
"""
import argparse
import json
import math
import os
import re
import statistics

N_EPOCH, N_OFFLINE_PAIRS, N_VAL = 250, 174993, 200
START_SEGS, PAIRS_PER_SEG, LAG, FRAMES_PER_SEG, FRAMES_PER_STEP = 4, 4901, 100, 5000, 12
DDP_XNODE = 0.005
CONFIGS = (("1 node, 4 GPUs, batch 16", 4, 16, 1), ("4 nodes, 14 GPUs, batch 16 (o1.2's global batch 224)", 14, 16, 4),
           ("4 nodes, 16 GPUs, batch 16", 16, 16, 4))

RX = {
    "params": re.compile(r"ConvNextUViT ([\d.]+)M params"),
    "peak": re.compile(r"peak GPU memory \(PyTorch\) ([\d.]+) GiB"),
    "gpus": re.compile(r": (\d+) GPU\(s\), (.+)$"),
    "filled": re.compile(r"buffer filled: (\d+) pairs from (\d+) segments in (\d+) s"),
    "epoch": re.compile(r"^epoch (\d+): train ([\d.]+)\s+val ([\d.]+) \(persistence ([\d.]+)\)\s+lr (\S+)\s+"
                        r"([\d.]+) s/step\s+\((\d+) s\)"),
    "resume": re.compile(r"resuming from (\S+): epoch (\d+) done, step (\d+), best val ([\d.]+), (\d+) segments"),
    "done": re.compile(r"^done: best val ([\d.]+)"),
    "kv": re.compile(r"(\w+)=(\S+)"),
}


def num(v):
    try:
        return float(v)
    except ValueError:
        return v


def parse(text):
    """One rank-0 log -> dict of everything o1.2 and the O12 lines report."""
    r = {"knobs": {}, "epochs": [], "o12_epochs": [], "saves_best": [], "resumes": []}
    for line in text.split("\n"):
        line = line.split("\r")[-1].strip()               # tqdm bars (stderr) may share a line with a message
        for key in ("params", "peak"):
            m = RX[key].search(line)
            if m:
                r[key] = float(m.group(1))
        m = RX["gpus"].search(line)
        if m and "GPU(s)" in line:
            r["world_size"], r["device"] = int(m.group(1)), m.group(2)
        m = RX["filled"].search(line)
        if m:
            r["filled"] = {"pairs": int(m.group(1)), "segments": int(m.group(2)), "seconds": int(m.group(3))}
        m = RX["epoch"].match(line)
        if m:
            r["epochs"].append({"epoch": int(m.group(1)), "train": float(m.group(2)), "val": float(m.group(3)),
                                "persistence": float(m.group(4)), "s_step": float(m.group(6)),
                                "seconds": int(m.group(7))})
        m = RX["resume"].search(line)
        if m:
            r["resumes"].append({"from": m.group(1), "epoch": int(m.group(2)), "step": int(m.group(3))})
        m = RX["done"].match(line)
        if m:
            r["best_val"] = float(m.group(1))
        if line.startswith("O12_"):
            tag, _, rest = line.partition(" ")
            kv = {k: num(v) for k, v in RX["kv"].findall(rest)}
            if tag == "O12_KNOB":
                r["knobs"].update({k: v for k, v in kv.items()})
            elif tag == "O12_EPOCH":
                r["o12_epochs"].append(kv)
            elif tag == "O12_SAVE":
                r["saves_best"].append(kv.get("seconds"))
            elif tag == "O12_BITWISE":
                r["bitwise"] = dict(kv, verdict=rest.split()[-1])
            elif tag in ("O12_FILL", "O12_PHASES"):
                r[tag[4:].lower()] = kv
            elif tag == "O12_RUN_DONE":
                r["run_done"] = True
    return r


def pieces(r):
    """The measured inputs of the formula; None where the log lacks them."""
    ep = [e for e in r["o12_epochs"] if e.get("epoch", 0) >= 1] or r["o12_epochs"]
    fill, ph = r.get("fill", {}), r.get("phases", {})
    s_frame = fill.get("s_per_frame_steady")
    s_frame = s_frame if isinstance(s_frame, float) else None
    p = {"s_step": statistics.median(e["step_median"] for e in ep) if ep else None,
         "s_step_o12_epoch": statistics.median(e["s_step"] for e in r["epochs"][1:]) if len(r["epochs"]) > 1 else None,
         "s_frame": s_frame, "t_load": fill.get("load_s"),
         "t_val": statistics.median(e["val_s"] for e in ep) if ep else None,
         "t_save_last": statistics.median(e["save_last_s"] for e in ep) if ep else None,
         "t_save_best": statistics.median(r["saves_best"]) if r["saves_best"] else None,
         "batch": r["knobs"].get("BATCH"), "world_size": r.get("world_size"), "n_val_run": r["knobs"].get("N_VAL")}
    if ph.get("startup_to_fill_done_s") is not None and fill.get("seconds") is not None:
        p["t_init"] = ph["startup_to_fill_done_s"] - fill["seconds"]
    else:
        p["t_init"] = None
    if p["t_val"] is not None and p["n_val_run"] and p["world_size"] and p["batch"]:
        p["val_batches_run"] = math.ceil(math.ceil(p["n_val_run"] / p["world_size"]) / p["batch"])
    return p


def project(p, n, b, nodes):
    """Full-run hours for one configuration, with the term breakdown (seconds)."""
    need = ("s_step", "s_frame", "t_load", "t_init", "t_val", "t_save_last")
    if any(p.get(k) is None for k in need):
        return None
    steps = N_EPOCH * ((N_OFFLINE_PAIRS // n) // b)
    val_batches = math.ceil(math.ceil(N_VAL / n) / b)
    t_val_batch = p["t_val"] / max(p.get("val_batches_run") or 1, 1)
    terms = {"init": p["t_init"], "fill": p["t_load"] + (LAG + START_SEGS * PAIRS_PER_SEG) * p["s_frame"],
             "steps": steps * (p["s_step"] + (DDP_XNODE if nodes > 1 else 0.0)),
             "epoch_overhead": N_EPOCH * (t_val_batch * val_batches + p["t_save_last"] + (p["t_save_best"] or 0.0)),
             "segment_loads": math.ceil(steps * FRAMES_PER_STEP / FRAMES_PER_SEG) * p["t_load"]}
    total = sum(terms.values())
    return {"steps": steps, "terms_s": terms, "hours": total / 3600, "node_hours": total / 3600 * nodes,
            "gpu_hours": total / 3600 * n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", action="append", required=True, help="<solver>=<path of the rank-0 log>")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    res = {"formula": "see module docstring", "ddp_xnode_s": DDP_XNODE, "solvers": {}}
    for spec in a.log:
        name, _, path = spec.partition("=")
        with open(path) as fh:
            r = parse(fh.read())
        p = pieces(r)
        res["solvers"][name] = {"parsed": r, "pieces": p,
                                "projections": {c[0]: project(p, *c[1:]) for c in CONFIGS}}
        bw = r.get("bitwise", {})
        print(f"O12_PIECES {name}: s_step={p['s_step']} s_frame={p['s_frame']} t_init={p['t_init']} "
              f"t_load={p['t_load']} t_val={p['t_val']} t_save_last={p['t_save_last']} t_save_best={p['t_save_best']} "
              f"peak={r.get('peak')} GiB params={r.get('params')}M bitwise={bw.get('verdict')} "
              f"run_done={r.get('run_done', False)}")
        for c in CONFIGS:
            pr = res["solvers"][name]["projections"][c[0]]
            if pr:
                print(f"O12_PROJECTION {name} [{c[0]}]: {pr['steps']} steps, {pr['hours']:.1f} h "
                      f"({pr['node_hours']:.0f} node-h, {pr['gpu_hours']:.0f} GPU-h); "
                      + " ".join(f"{k}={v / 3600:.2f}h" for k, v in pr["terms_s"].items()))
            else:
                print(f"O12_PROJECTION {name} [{c[0]}]: MISSING_PIECES")
    with open(os.path.join(a.out, "o12_extrapolation.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    ok = all(all(v["projections"].values()) for v in res["solvers"].values())
    print(f"O12_EXTRAPOLATE_{'OK' if ok else 'INCOMPLETE'} solvers={len(res['solvers'])} -> "
          f"{os.path.join(a.out, 'o12_extrapolation.json')}")


if __name__ == "__main__":
    main()
