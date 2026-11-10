"""o12_polaris.py -- run o1.2-TrainUViT_m16_tendency.py ITSELF on Polaris A100s, to time the real online loop (buffer
fill, steady-state s/step, validation, checkpoint saves) and extrapolate a full run's A100 wall-clock. One process per
GPU under torchrun (run_o12_polaris.pbs). Predictions: PREDICTIONS_o12.md (written before the first job).

The file on disk is not edited. Its source is compiled with the exact-match patches in PATCHES below, each of which
must match exactly once and keeps the line count (tracebacks point at the real lines of o1.2). They do three things:
  environment  paths; the torch 2.3.1 spelling of strict IEEE fp32 (o1.2's own check_fp32() still runs and asserts);
               the vendored solver's names (pair spacing BLOCK_T = 0.02, CHECKPOINT_DIR layout); the staged members.
  stand-ins    sqg-env has no zarr / wandb / matplotlib: a read-only zarr stand-in for the val store (theta.npy +
               attrs.json, build_val_store.py), wandb -> a jsonl file, no plot (it only went to wandb). o1.2's
               first-segment check against train_m16.zarr becomes the same check against the taxonomy job's
               production frames of the same checkpoints (lean_u1 == production bitwise, job 7824121).
  knobs        O12_<NAME> env overrides, each logged; unset = o1.2's value. BATCH (A100 40 GB: 16), START_PAIRS,
               N_EPOCH, STEPS_PER_EPOCH, N_VAL, BUF; O12_SOLVER = ref (o1.2's own call) | lean_u1 | lean32ek_fast.
Timing lines (rank 0): O12_FILL, O12_BITWISE, O12_EPOCH (per-step median / p10 / p90, train, val, save), O12_PHASES.

  O12_OUT=<run dir> O12_VAL_STORE=<dir> O12_REF_DIR=<taxonomy raw dir> SQG_ASQG_PY=... SQG_TRAINING_ROOT=... \
      python -m torch.distributed.run --standalone --nproc_per_node=4 o12_polaris.py
"""
import json
import os
import statistics
import sys
import time
import types

HERE = os.path.dirname(os.path.abspath(__file__))
OPT = os.path.dirname(HERE)
ANALYSIS = os.path.dirname(OPT)
SQG = os.path.dirname(ANALYSIS)
O12 = os.path.join(SQG, "o1.2-TrainUViT_m16_tendency.py")
SOLVERS = ("ref", "lean_u1", "lean32ek_fast")
BITWISE_SOLVERS = ("ref", "lean_u1")            # production bit for bit (jobs 7815637, 7824121)

# (name, old, new): old must occur exactly once in o1.2's source; new has the same number of lines
PATCHES = [
    ("data_root", 'DATA_ROOT = "/data/weidong/SQG_data/GenData_deterForcing/coarse_grained"',
     'DATA_ROOT = __O12__.cfg["out"]'),
    ("train_store", 'TRAIN_STORE = f"{DATA_ROOT}/train_m16.zarr"     # only for the bitwise check of the first segment',
     'TRAIN_STORE = None                                 # Polaris: no train_m16.zarr; see _push'),
    ("val_store", 'VAL_STORE = f"{DATA_ROOT}/val_m16.zarr"', 'VAL_STORE = __O12__.cfg["val_store"]'),
    ("ckpt_dir", 'CKPT_DIR = "/data/weidong/SQG_data/GenData_deterForcing/model_ckpt"',
     'CKPT_DIR = __O12__.cfg["ckpt_dir"]'),
    ("asqg_py", 'ASQG_PY = "/home/nvidia/weidong/SQG/SQG_deterForcing/fixed_ring/A-SQG_data.py"',
     'ASQG_PY = __O12__.cfg["asqg_py"]'),
    ("buf_size", 'BUF_SIZE = 262144                  #', 'BUF_SIZE = __O12__.knob("BUF", 262144)  #'),
    ("batch_size", 'BATCH_SIZE = 32                    #', 'BATCH_SIZE = __O12__.knob("BATCH", 32)  #'),
    ("n_epoch", 'N_EPOCH = 250\n', 'N_EPOCH = __O12__.knob("N_EPOCH", 250)\n'),
    ("n_val", 'N_VAL = 200                        #', 'N_VAL = __O12__.knob("N_VAL", 200)  #'),
    ("wandb_dir", 'WANDB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")',
     'WANDB_DIR = os.path.join(__O12__.cfg["out"], "wandb")'),
    # torch 2.3.1 has no fp32_precision API: the same strict IEEE fp32 by its older switches; check_fp32() asserts it
    ("fp32", 'torch.backends.fp32_precision = "ieee"\ntorch.backends.cuda.matmul.fp32_precision = "ieee"\n'
             'torch.backends.cudnn.fp32_precision = "ieee"\ntorch.backends.cudnn.conv.fp32_precision = "ieee"\n'
             'torch.backends.cudnn.rnn.fp32_precision = "ieee"',
     'torch.backends.cuda.matmul.allow_tf32 = False      # Polaris torch 2.3.1: strict fp32 by the pre-2.6 switches\n'
     'torch.backends.cudnn.allow_tf32 = False\ntorch.set_float32_matmul_precision("highest")\npass\npass'),
    ("import_asqg", "GD = import_asqg()", "GD = __O12__.import_asqg()"),
    ("lag", "LAG = int(round(K * GD.DT_SAVE / DT)) // STRIDE",
     "LAG = int(round(K * GD.BLOCK_T / DT)) // STRIDE"),
    ("lag_assert", "assert LAG * STRIDE * DT == K * GD.DT_SAVE and",
     "assert LAG * STRIDE * DT == K * GD.BLOCK_T and"),           # BLOCK_T = 8 x 0.0025 == 0.02 exactly
    ("advance", "advance = jax.jit(lambda st: lax.scan(GD.inner, st, None, length=STRIDE)[0])",
     "advance = __O12__.make_advance(GD, jax.jit(lambda st: lax.scan(GD.inner, st, None, length=STRIDE)[0]), STRIDE)"),
    ("members", "for member in range(1, 8):", "for member in __O12__.members:"),
    ("ckpt_layout", '    d = GD.RUN_CHECKPOINT_DIR.parent / GD.RUN_LABEL.replace("_m01_", f"_m{member:02d}_")',
     '    d = GD.TRAINING_ROOT / GD.RUN_LABEL.replace("_m01_", f"_m{member:02d}_") / "checkpoint"'),
    ("steps_per_epoch", "steps_per_epoch = (N_OFFLINE_PAIRS // world_size) // BATCH_SIZE",
     'steps_per_epoch = __O12__.knob("STEPS_PER_EPOCH", (N_OFFLINE_PAIRS // world_size) // BATCH_SIZE)'),
    ("seg_assert", 'assert len(my_segs) >= n_seg_needed, f"{len(my_segs)} checkpoints per rank < {n_seg_needed} segments needed"',
     "__O12__.segments(len(my_segs), n_seg_needed)"),
    ("train_group", 'train_group = zarr.open_group(TRAIN_STORE, mode="r")', "train_group = None"),
    ("train_src", 'train_src = {int(re.search(r"_m(\\d\\d)_", s["name"]).group(1)): s for s in train_group.attrs["sources"]}',
     "train_src = {}"),
    ("push_check",
     '            s = train_src[self.member]\n'
     '            row = s["offset"] + int(round((self.t_c + self.k * STRIDE * DT - s["t_first"]) / GD.DT_SAVE))\n'
     '            ref = torch.from_numpy(train_group["theta"][row]).to(device)\n'
     '            self.check.append((f[0] * THETA_SCALE - ref).abs().max().item())',
     '            ref = __O12__.ref_frame(self.member, self.t_c, self.k)   # production theta[::16, ::16], or None\n'
     '            if ref is not None:\n'
     '                ref = torch.from_numpy(ref).to(device)\n'
     '                self.check.append((f[0] * THETA_SCALE - ref).abs().max().item())'),
    ("start_pairs", "start_pairs = START_SEGS * PAIRS_PER_SEG",
     'start_pairs = __O12__.knob("START_PAIRS", START_SEGS * PAIRS_PER_SEG)'),
    ("fill_mark", "while online.n_pairs < start_pairs:                 # fill the buffer before the first optimizer step\n"
                  "    online.advance()",
     "while online.n_pairs < start_pairs:                 # fill the buffer before the first optimizer step\n"
     "    online.advance(); __O12__.fill_mark(online)"),
    ("check_max", "check = torch.tensor(max(online.check), device=device)",
     "check = torch.tensor(max(online.check, default=-1.0), device=device)"),
    ("check_assert", 'assert check.item() == 0.0, f"online frames differ from train_m16.zarr: max |diff| {check.item():.3e}"',
     "__O12__.check_first_segment(check.item(), len(online.check), t0, online)"),
    ("check_log", 'f"first segment bitwise equal to train_m16.zarr on all ranks ({len(online.check)} frames each)")',
     'f"first segment checked against the taxonomy production frames ({len(online.check)} frames each; O12_BITWISE)")'),
    ("step_mark", "        loss_sum, n_steps, global_step = loss_sum + loss.item(), n_steps + 1, global_step + 1",
     "        loss_sum, n_steps, global_step = loss_sum + loss.item(), n_steps + 1, global_step + 1; __O12__.step_mark()"),
    ("train_end", "    t_train = time.time() - t0", "    t_train = time.time() - t0; __O12__.mark('train_end')"),
    ("val_end", "    persistence = (sums[1].sum() / sums[2].sum()).item()",
     "    persistence = (sums[1].sum() / sums[2].sum()).item(); __O12__.mark('val_end')"),
    ("no_plot", '        fig = plot_sample(x_viz, y_viz, pred_viz, f"epoch {epoch}  val {val_loss:.4f}")',
     "        fig = None                                     # Polaris: no matplotlib; the plot only went to wandb"),
    ("no_close", "        plt.close(fig)", "        pass"),
    ("save_last", "        save(last_path, epoch, val_loss)",
     "        __O12__.timed('save_last', save, last_path, epoch, val_loss); __O12__.epoch_done(epoch, n_steps, t_train)"),
    ("save_best", "            save(best_path, epoch, val_loss)",
     "            __O12__.timed('save_best', save, best_path, epoch, val_loss)"),
]


def apply_patches(src, patches=PATCHES):
    """o1.2's source with every patch applied; raises if a patch does not match exactly once or changes the line
    count."""
    for name, old, new in patches:
        n = src.count(old)
        if n != 1:
            raise RuntimeError(f"O12 patch {name}: expected 1 match in {os.path.basename(O12)}, found {n}")
        if old.count("\n") != new.count("\n"):
            raise RuntimeError(f"O12 patch {name}: changes the line count")
        src = src.replace(old, new, 1)
    return src


def stand_ins(out, is_main):
    """Minimal zarr (read-only theta.npy + attrs.json), wandb (jsonl) and matplotlib modules for sys.modules."""
    import numpy as np

    zarr = types.ModuleType("zarr")

    class Group:
        def __init__(self, store):
            with open(os.path.join(store, "attrs.json")) as fh:
                self.attrs = json.load(fh)
            self._theta = np.load(os.path.join(store, "theta.npy"), mmap_mode="r")

        def __getitem__(self, key):
            if key != "theta":
                raise KeyError(key)
            return self._theta

    def open_group(store, mode="r"):
        if mode != "r":
            raise ValueError("O12 zarr stand-in is read-only")
        return Group(store)

    zarr.open_group = open_group

    wandb = types.ModuleType("wandb")
    path = os.path.join(out, "wandb_shim.jsonl")

    def init(**kw):
        return types.SimpleNamespace(id=kw.get("id") or "o12-offline", url=f"(no wandb on Polaris: {path})")

    def log(d, step=None):
        if is_main:
            row = {k: v for k, v in d.items() if isinstance(v, (int, float, str))}
            with open(path, "a") as fh:
                fh.write(json.dumps({"step": step, **row}) + "\n")

    wandb.init, wandb.log, wandb.finish, wandb.Image = init, log, (lambda *a, **k: None), (lambda *a, **k: None)

    mpl = types.ModuleType("matplotlib")
    mpl.use = lambda *a, **k: None
    plt = types.ModuleType("matplotlib.pyplot")
    plt.close = lambda *a, **k: None
    mpl.pyplot = plt
    return {"zarr": zarr, "wandb": wandb, "matplotlib": mpl, "matplotlib.pyplot": plt}


class Helper:
    """The __O12__ object the patched source calls."""

    def __init__(self, env):
        self.rank = int(env.get("RANK", "0"))
        self.is_main = self.rank == 0
        self.solver = env.get("O12_SOLVER", "ref")
        if self.solver not in SOLVERS:
            raise SystemExit(f"ERROR O12_SOLVER={self.solver!r} not in {SOLVERS}")
        out = env["O12_OUT"]
        self.cfg = {"out": out, "val_store": env["O12_VAL_STORE"], "ckpt_dir": os.path.join(out, "model_ckpt"),
                    "asqg_py": env["SQG_ASQG_PY"]}
        self.members = [int(m) for m in env.get("O12_MEMBERS", "1,2,3,4").split(",")]
        self.env, self.ref_dir, self._ref_index = env, env.get("O12_REF_DIR"), None
        self.t_start, self.marks, self.fill, self.steps, self.knobs = time.time(), {}, [], [], {}

    def log(self, msg):
        if self.is_main:
            print(msg, flush=True)

    def knob(self, name, default):
        raw = self.env.get(f"O12_{name}")
        value = default if raw in (None, "") else int(raw)
        self.knobs[name] = value
        self.log(f"O12_KNOB {name}={value} (o1.2: {default})")
        return value

    def import_asqg(self):
        sys.path.insert(0, ANALYSIS)
        import profile_solver as PS
        return PS.import_asqg()

    def make_advance(self, GD, ref_call, stride):
        self.log(f"O12_SOLVER {self.solver}")
        if self.solver == "ref":
            return ref_call                                       # o1.2's own line, unchanged
        for p in (os.path.join(OPT, "seed1"), os.path.join(OPT, "delta"), os.path.join(OPT, "taxonomy")):
            sys.path.insert(0, p)
        if self.solver == "lean_u1":                              # as taxonomy.py L133-135
            import lean_solver as LS
            _, ncol = LS.preconditions(GD)
            cu1 = LS.build_consts(GD, ncol)
            u1 = LS.make_advance(GD, stride, ncol=ncol, unroll1=True)
            return lambda st: u1(st, cu1)
        import lean32_fast as LF                                  # lean32ek_fast, as taxonomy.py L137-139
        import lean32_precise as LP
        from pyqg_jax import steppers as pst
        c32 = LP.tables(GD)
        fast = LF.make_call(pst, GD.N, GD.N // 2 + 1, GD.DT, stride, mode="ek", u1=True, ring=3)
        return lambda st: fast(st, c32)

    def segments(self, have, need):
        self.log(f"O12_SEGMENTS {have} staged checkpoints per rank, {need} needed by the full schedule"
                 + ("" if have >= need else " -> o1.2 reuses them (wrap-around): fine for timing, not for science"))

    def _index(self):
        """(member, t) -> {ring frame k: production theta[::16, ::16] (float32)}, from the taxonomy run's pair npz."""
        import numpy as np
        idx = {}
        for w in range(4):
            fn = os.path.join(self.ref_dir, f"worker_{w}.json")
            if not os.path.exists(fn):
                continue
            with open(fn) as fh:
                states = json.load(fh)["states"]
            for s, st in enumerate(states):
                if st.get("offset_calls", 0):
                    continue
                frames = {}
                for j in range(len(st["pairs"])):
                    frames[100 * (j + 1)] = os.path.join(self.ref_dir, f"pair_w{w}_s{s}_p{j}.npz")
                idx[(int(st["member"]), round(float(st["t"]), 6))] = frames
        self._np = np
        return idx

    def ref_frame(self, member, t_c, k):
        if not self.ref_dir:
            return None
        if self._ref_index is None:
            self._ref_index = self._index()
        fn = self._ref_index.get((int(member), round(float(t_c), 6)), {}).get(int(k))
        if fn is None:
            return None
        with self._np.load(fn) as z:
            return (z["lean_u1__sub__100"] * self._np.float32(8.0)).astype(self._np.float32)   # sub = theta / 8

    def fill_mark(self, online):
        self.fill.append((time.time(), online.k))

    def check_first_segment(self, maxdiff, n_frames, t0, online):
        t_fill = time.time() - t0
        rate = None
        if len(self.fill) >= 4:                                   # steady fill rate: second half of the fill
            (ta, ka), (tb, kb) = self.fill[len(self.fill) // 2], self.fill[-1]
            rate = (tb - ta) / (kb - ka) if kb > ka else None
        self.marks["fill_done"] = time.time()
        self.log(f"O12_FILL pairs={online.n_pairs} frames={online.k} seconds={t_fill:.1f} load_s={online.load_time:.1f}"
                 f" s_per_frame_steady={rate if rate is None else f'{rate:.5f}'} solver={self.solver}")
        if n_frames == 0 or maxdiff < 0:
            verdict = "NO_REFERENCE"
        elif self.solver in BITWISE_SOLVERS:
            verdict = "PASS" if maxdiff == 0.0 else "FAIL"
        else:
            verdict = "INFO"
        self.log(f"O12_BITWISE solver={self.solver} frames_per_rank={n_frames} max_abs_diff={maxdiff:.3e} {verdict}")
        if verdict == "FAIL":
            raise AssertionError(f"O12: {self.solver} frames differ from production: max |diff| {maxdiff:.3e}")

    def step_mark(self):
        self.steps.append(time.time())

    def mark(self, name):
        self.marks[name] = time.time()

    def timed(self, name, fn, *a):
        t = time.time()
        fn(*a)
        self.marks[name + "_s"] = time.time() - t
        if name == "save_best":
            self.log(f"O12_SAVE best seconds={self.marks[name + '_s']:.2f}")

    def epoch_done(self, epoch, n_steps, t_train):
        st = self.steps[-n_steps:]
        dts = sorted(b - a for a, b in zip(st, st[1:]))
        q = (lambda f: dts[min(len(dts) - 1, int(f * len(dts)))]) if dts else (lambda f: float("nan"))
        med = statistics.median(dts) if dts else float("nan")
        t_val = self.marks.get("val_end", float("nan")) - self.marks.get("train_end", float("nan"))
        self.log(f"O12_EPOCH epoch={epoch} steps={n_steps} step_median={med:.4f} step_p10={q(0.1):.4f} "
                 f"step_p90={q(0.9):.4f} train_s={t_train:.1f} val_s={t_val:.2f} "
                 f"save_last_s={self.marks.get('save_last_s', float('nan')):.2f} solver={self.solver}")

    def phases(self):
        fill = self.marks.get("fill_done", float("nan")) - self.t_start
        self.log(f"O12_PHASES solver={self.solver} startup_to_fill_done_s={fill:.1f} "
                 f"total_s={time.time() - self.t_start:.1f}")


def main():
    helper = Helper(os.environ)
    # o1.2 resumes from <RUN_TAG>_best.pt if present: a timing run must start from scratch. Rank 0 checks before its
    # own first write (segments.json); the PBS script also refuses an existing run dir.
    if helper.is_main and os.path.isdir(helper.cfg["ckpt_dir"]) and os.listdir(helper.cfg["ckpt_dir"]):
        raise SystemExit(f"ERROR O12_CKPT_DIR_NOT_EMPTY {helper.cfg['ckpt_dir']} (o1.2 would resume from it)")
    sys.modules.update(stand_ins(helper.cfg["out"], helper.is_main))
    with open(O12) as fh:
        src = apply_patches(fh.read())
    g = {"__name__": "__main__", "__file__": O12, "__O12__": helper, "__builtins__": __builtins__}
    exec(compile(src, O12, "exec", dont_inherit=True), g)
    helper.phases()
    helper.log(f"O12_RUN_DONE solver={helper.solver}")


if __name__ == "__main__":
    main()
