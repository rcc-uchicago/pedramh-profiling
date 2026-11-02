#!/usr/bin/env python3
"""extract_configs.py -- the configuration receipt for the makani SFNO runs on E3SM.

Reads the primary records of every training run (the trainer's own rank-0 log
`out.log`, the PBS stdout `makani_mn_scaling.o<job>`, the rendered yaml, the
launcher's provenance sidecars and the bench CSVs) plus the rollout records
(climate_run stdouts, per-member logs, K=56 / lagged campaign files, 1-yr screen
dirs) and writes `configs.json` next to this file. Every number in
2026-11-02_makani_configurations.md is read from that JSON.

Stdlib only; runs under the login node's python3 (3.6): no walrus, no f-string
`=` specifiers, no `from __future__ import annotations`.

Usage:
    python3 extract_configs.py            # writes configs.json, prints a summary
    python3 extract_configs.py --render   # also prints the markdown tables

Rules it applies (the doc's Limits section repeats them):
  * The trainer's printed value is the configuration. The rendered yaml carries
    the template's n_future 0 / batch_size 8 / max_epochs; the launcher overrides
    them, so yaml-vs-log disagreements are recorded, not used.
  * A resumed run is several jobs. out.log is cumulative (one "Versions" header
    per job), so each job is one segment; training time is summed over segments.
  * Training time = the trainer's "Total training time" line. A segment that has
    none (killed by walltime mid-epoch) falls back to the sum of its logged
    "epoch time [s]" values, flagged `derived_from_epoch_times`.
  * Epoch N's checkpoint is ckpt_mp0_v(N-1) unless a run rotates its slots
    (B keeps 4 rotating versions); the member logs' ckpt_epoch is the check.
"""
import csv
import json
import os
import re
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
M = "/lus/eagle/projects/lighthouse-uchicago/members/mehta5"
RUNS = M + "/runs/makani_mn_scaling"
EXP = RUNS + "/e3sm_mn_scaling"
EVAL = M + "/runs/makani_eval"
BENCH = M + "/bench"
REPO = M + "/pedramh-profiling"
WT = REPO + "/.claude/worktrees"
KNOW = WT + "/makani-knowledge/makani_sfno/KNOWLEDGE.md"
PREREG_PIPE = WT + "/makani-knowledge/makani_sfno/docs/2026-10-09_pipeline_comparison_prereg.md"
GPUS_PER_NODE = 4

# ---------------------------------------------------------------------------
# The runs. Job ids and where their PBS stdout lives. `job_source` says how the
# job id was established; anything not read from a primary record is flagged
# `unverified` and the doc tags it UNVERIFIED.
# ---------------------------------------------------------------------------
RUN_DEFS = [
    ("128-node", "prod128_alldata_v2",
     "batch-512 throughput run used as a training baseline",
     [("7566145", REPO + "/makani_sfno")]),
    ("A", "prod1n_b32_sgdr",
     "single-step production run from scratch",
     [("7585080", REPO + "/makani_sfno")]),
    ("B", "nf4_prod_b16_r1",
     "depth-5 fine-tune of A",
     [("7630639", REPO + "/makani_sfno")]),
    ("DRYAIR", "fs_anneal_dryair_nf4_b16_rb01",
     "depth-5 fine-tune of B e01 with dry air trained in",
     [("7709268", WT + "/b-continuation-dryair/makani_sfno")]),
    ("SOILFIX", "fs_anneal_soilfix_nf4_b16_rb01",
     "depth-5 fine-tune of B e01 with the soil-moisture fix trained in",
     [("7715005", WT + "/b-continuation-dryair/makani_sfno")]),
    ("F-warm", "f_nosoil_2n_b32_e43_warm",
     "A's recipe on 99 outputs, warm-started from A",
     [("7660250", None)]),
    ("F-scratch", "f_nosoil_2n_b32_e23_scratch",
     "A's recipe on 99 outputs, from scratch, with EMA",
     [("7718436", None), ("7721577", None)]),
    ("F-depth4", "fsF_anneal_nf4_b16_rtmax100_scratch",
     "B's depth-5 recipe applied to F-scratch e23",
     [("7719538", WT + "/arm-fsF-nf4-bsched/makani_sfno"),
      ("7725884", WT + "/arm-fsF-nf4-bsched/makani_sfno")]),
    ("F-depth4-dry", "fsF_dryair_nf4_b16_rtmax100_scratch",
     "F-depth4's recipe plus trained-in dry air, stop-early chain",
     [("7732625", WT + "/arm-fd-dryair/makani_sfno"),
      ("7733190", WT + "/arm-fd-dryair/makani_sfno"),
      ("7733283", WT + "/arm-fd-dryair/makani_sfno"),
      ("7733346", WT + "/arm-fd-dryair/makani_sfno")]),
    ("G1", "f_nosoil_2n_b32_e23_scratch_fp32",
     "F-scratch's recipe in fp32",
     [("7731784", WT + "/arm-g1-fp32/makani_sfno")]),
    ("G1-dry", "f_nosoil_2n_b32_e63_scratch_fp32_dryair",
     "G1 plus trained-in dry air",
     [("7735732", WT + "/arm-g1-dryair/makani_sfno")]),
    ("G4E", "f_nosoil_4n_b32_e23_scratch_fp32_nf4",
     "G1's recipe at depth 5, from scratch",
     [("7740992", WT + "/arm-g1-fp32/makani_sfno")]),
    ("G4D", "f_nosoil_4n_b32_e23_scratch_fp32_dryair_nf4",
     "G4E plus trained-in dry air",
     [("7742231", WT + "/arm-g1-dryair/makani_sfno"),
      ("7771394", WT + "/arm-g1-dryair/makani_sfno")]),
]

# rollout-arm name prefix -> training run label, for arms whose stdout does not
# print a checkpoint path (rollbatch / precision-cross jobs). Longest prefix wins.
ARM_PREFIX = [
    ("DRYAIRF", "DRYAIR"), ("DRYAIR", "DRYAIR"), ("SOILFIX", "SOILFIX"),
    ("FSEMA", "F-scratch"), ("FSEF", "F-scratch"), ("FSEQ", "F-scratch"), ("FSE", "F-scratch"),
    ("FS", "F-scratch"), ("FDEF", "F-depth4"), ("FDE", "F-depth4"), ("FD", "F-depth4"),
    ("DAE", "F-depth4-dry"), ("DAR", "F-depth4-dry"),
    ("G1EB", "G1"), ("G1E", "G1"), ("G1D", "G1-dry"), ("G4E", "G4E"), ("G4D", "G4D"),
    ("BF", "B"), ("B", "B"), ("A", "A"),
]

TS_RE = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d+ - (\S+) - (\w+) - (.*)$")
NUM_RE = re.compile(r"^-?\d+(\.\d+)?([eE]-?\d+)?$")


def num(s):
    s = s.strip()
    if NUM_RE.match(s):
        return int(s) if re.match(r"^-?\d+$", s) else float(s)
    if s in ("True", "False"):
        return s == "True"
    return s


# ---------------------------------------------------------------------------
# out.log: one segment per job
# ---------------------------------------------------------------------------
def parse_outlog(tag):
    path = EXP + "/" + tag + "/out.log"
    segs = []
    cur = None
    ep = None
    with open(path, errors="replace") as f:
        for line in f:
            m = TS_RE.match(line.rstrip("\n"))
            if not m:
                continue
            ts, logger, _lvl, msg = m.groups()
            if "--------------- Versions" in msg:
                cur = {"start_ts": ts, "params": {}, "main": {}, "epochs": [], "total_training_s": None,
                       "total_ts": None, "started_loop": False, "resume_from": None, "loaded_pretrained": None,
                       "ema_trainer": None, "n_params": None, "git_hash": None, "last_ts": ts}
                segs.append(cur)
                ep = None
                continue
            if cur is None:
                continue
            cur["last_ts"] = ts
            if msg.startswith("git hash:"):
                h = re.search(r"([0-9a-f]{40})", msg)
                cur["git_hash"] = h.group(1) if h else None
                continue
            if "Starting Training Loop" in msg:
                cur["started_loop"] = True
                continue
            if logger == "root" and not cur["started_loop"]:
                mm = re.match(r"^(\w+) (.*)$", msg)
                if mm and mm.group(1) not in cur["params"]:
                    cur["params"][mm.group(1)] = mm.group(2)
            if logger == "__main__":
                mm = re.match(r"^(\S+)\s+=\s+(.*)$", msg)
                if mm:
                    cur["main"][mm.group(1)] = mm.group(2).strip()
            mm = re.search(r"Resuming from checkpoint (\S+)", msg)
            if mm:
                cur["resume_from"] = mm.group(1)
            mm = re.search(r"Loading pretrained checkpoint (\S+)", msg)
            if mm:
                cur["loaded_pretrained"] = mm.group(1)
            mm = re.search(r"EMA enabled: decay=([0-9.]+)", msg)
            if mm:
                cur["ema_trainer"] = float(mm.group(1))
                cur["ema_line"] = msg
            mm = re.search(r"Number of trainable model parameters: (\d+)", msg)
            if mm:
                cur["n_params"] = int(mm.group(1))
            mm = re.search(r"Epoch (\d+) summary", msg)
            if mm:
                ep = {"epoch": int(mm.group(1))}
                cur["epochs"].append(ep)
                continue
            if ep is not None:
                for key, rx in (("steps", r"^\s*training steps: (\d+)"),
                                ("epoch_time_s", r"^\s*epoch time \[s\]: ([0-9.]+)"),
                                ("train_time_s", r"^\s*training time \[s\]: ([0-9.]+)"),
                                ("val_loss", r"^\s*validation loss:\s+([0-9.eE-]+)"),
                                ("step_ms", r"^\s*training step time \[ms\]: ([0-9.]+)")):
                    if key not in ep:
                        mm = re.match(rx, msg)
                        if mm:
                            ep[key] = num(mm.group(1))
            mm = re.search(r"Total training time is ([0-9.]+) sec", msg)
            if mm:
                cur["total_training_s"] = float(mm.group(1))
                cur["total_ts"] = ts
                ep = None
    return path, segs


# ---------------------------------------------------------------------------
# PBS stdout: header (first lines) and tail only -- the 128-node stdout is 485 MB
# ---------------------------------------------------------------------------
def parse_stdout(job, wdir):
    out = {"path": None, "found": False}
    if wdir is None:
        return out
    path = wdir + "/makani_mn_scaling.o" + job
    out["path"] = path
    if not os.path.exists(path):
        return out
    out["found"] = True
    with open(path, "rb") as f:
        for i, raw in enumerate(f):
            if i > 5000:
                break
            s = raw.decode("utf-8", "replace").rstrip()
            mm = re.match(r"^=== makani_mn_scaling: (\S+) ===", s)
            if mm:
                out["start"] = mm.group(1)
            mm = re.match(r"^jobid=(\d+)\s+nodes=(\d+)\s+ranks=(\d+)", s)
            if mm:
                out["jobid"], out["nodes"], out["ranks"] = mm.group(1), int(mm.group(2)), int(mm.group(3))
            mm = re.match(r"^local_batch=(\d+)\s+global_batch=(\d+)", s)
            if mm:
                out["local_batch"], out["global_batch"] = int(mm.group(1)), int(mm.group(2))
            mm = re.match(r"^h_par=(\d+)\s+w_par=(\d+)", s)
            if mm:
                out["h_par"], out["w_par"] = int(mm.group(1)), int(mm.group(2))
            mm = re.match(r"^run_num=(\S+)", s)
            if mm:
                out["run_num"] = mm.group(1)
            mm = re.search(r"\.yaml: n_train_samples_per_epoch=(\d+) max_epochs=(\d+)", s)
            if mm:
                out["rendered_max_epochs"] = int(mm.group(2))
                break
        size = os.path.getsize(path)
        f.seek(max(0, size - 400000))
        tail = f.read().decode("utf-8", "replace")
    mm = re.search(r"Total training time is ([0-9.]+) sec", tail)
    out["total_training_s"] = float(mm.group(1)) if mm else None
    mm = re.search(r"job killed: walltime (\d+) exceeded limit (\d+)", tail)
    if mm:
        out["killed_walltime_elapsed_s"] = int(mm.group(1))
        out["walltime_limit_s"] = int(mm.group(2))
    out["train_done_rc0"] = bool(re.search(r"TRAIN_DONE .* rc=0", tail))
    out["scaling_ok"] = "MAKANI_MN_SCALING_OK" in tail
    out["csv_schema_drift"] = "SCALING_CSV_SCHEMA_DRIFT" in tail
    return out


def rendered_yaml(tag):
    path = RUNS + "/e3sm_mn_scaling." + tag + ".yaml"
    keys = {}
    with open(path, errors="replace") as f:
        for line in f:
            mm = re.match(r"^\s+(batch_size|n_future|max_epochs|lr|scheduler|scheduler_T_max|scale_factor)\s*:\s*([^#]+)", line)
            if mm and mm.group(1) not in keys:
                keys[mm.group(1)] = num(mm.group(2).strip().strip('"'))
    return path, keys


def csv_rows():
    rows = {}
    for fn in sorted(os.listdir(BENCH)):
        if not (fn.startswith("makani_") and fn.endswith(".csv")):
            continue
        with open(BENCH + "/" + fn, errors="replace") as f:
            for r in csv.DictReader(f):
                rows.setdefault(r.get("jobid", ""), []).append(
                    {"file": fn, "log": r.get("log"), "total_train_s": r.get("total_train_s"),
                     "step_ms": r.get("step_ms"), "nodes": r.get("nodes"),
                     "global_batch": r.get("global_batch")})
    return rows


def provenance_jobs():
    """F-scratch's job ids: the resume sidecar names both jobs."""
    p = RUNS + "/f_nosoil_2n_b32_e23_scratch.resume_e43_provenance.txt"
    txt = open(p).read()
    first = re.search(r"identical to (\d+) except EPOCHS (\d+)->(\d+)", txt)
    second = re.search(r"job: (\d+)\.polaris", txt)
    return p, {
        "7718436": {"named": first.group(1) == "7718436", "epochs": int(first.group(2)), "text": first.group(0)},
        "7721577": {"named": second.group(1) == "7721577", "epochs": int(first.group(3)), "text": second.group(0)},
    }


def knowledge_job(tag):
    """Secondary source (KNOWLEDGE.md checkpoint-lineage table) for a job id."""
    with open(KNOW, errors="replace") as f:
        for line in f:
            if ("`" + tag + "`") in line and line.startswith("|"):
                cells = [c.strip() for c in line.split("|")]
                for c in cells:
                    if re.match(r"^\*{0,2}\d{7}\*{0,2}$", c):
                        return c.strip("*"), line.strip()[:200]
    return None, None


def first_float(params, key):
    v = params.get(key)
    return num(v) if v is not None else None


# ---------------------------------------------------------------------------
# Rollout arms
# ---------------------------------------------------------------------------
def rollout_arms():
    """Every 5-yr-protocol arm, keyed NAME@job (an arm re-rolled by a later job is a new row)."""
    arms = {}
    # (1) rollout stdouts: checkpoint path, weights and flags per arm
    stdouts = []
    for d in (WT + "/eval-protocol/makani_sfno", WT + "/b-continuation-dryair/makani_sfno"):
        for fn in sorted(os.listdir(d)):
            mm = re.match(r"^makani_(climate_run|rollbatch|pxc)\.o(\d+)$", fn)
            if mm:
                stdouts.append((d + "/" + fn, mm.group(2)))
    for fn in ("jc_climate.log", "jc16_climate.log"):
        stdouts.append((EVAL + "/g4d_e63_2026-11-02/" + fn, None))
    for path, job in stdouts:
        ema = {}
        with open(path, "rb") as f:
            for i, raw in enumerate(f):
                if i > 600:
                    break
                s = raw.decode("utf-8", "replace").rstrip()
                j = re.search(r"PBS_JOBID=(\d+)", s)
                if j and job is None:
                    job = j.group(1)
                w = re.match(r"^weights (\S+): ema from (\S+)", s)
                if w:
                    ema[w.group(1)] = w.group(2)
                c = re.match(r"^ckpt (\S+): (\S+) flags='(.*)'$", s)
                if c:
                    a = c.group(1)
                    src = ema.get(a, c.group(2))
                    mt = re.search(r"/e3sm_mn_scaling/([^/]+)/training_checkpoints/([^/]+)$", src)
                    arms.setdefault(a + "@" + job, {"name": a}).update({
                        "protocol_job": job, "stdout": path,
                        "weights": "ema" if a in ema else "raw",
                        "weights_source": "rollout stdout",
                        "run_tag": mt.group(1) if mt else None,
                        "ckpt_file": mt.group(2) if mt else src,
                        "flags": c.group(3)})
    # (2) submit scripts of the jobs whose stdout prints no checkpoint lines
    ep = WT + "/eval-protocol/makani_sfno/polaris/"
    sub_w = {}
    for line in open(ep + "submit_fp32_reroll.sh", errors="replace"):
        for mm in re.finditer(r'v "?([A-Z]+)(?:\$\{e\}|\d\d)?([a-z]*)"? "\$\{\w+\}" (?:"[^"]*"|\S+) (ema|raw) \'([^\']*)\'', line):
            sub_w[(mm.group(1), mm.group(2))] = (mm.group(3), mm.group(4), ep + "submit_fp32_reroll.sh")
    pc = open(ep + "submit_precision_cross.sh", errors="replace").read()
    if "WEIGHTS_%s=ema" in pc:
        for pre, fl in (("FSEQ", "--amp bf16"), ("G1EB", "--amp bf16"), ("FSEF", "--amp off")):
            sub_w.setdefault((pre, ""), ("ema", fl, ep + "submit_precision_cross.sh"))
    # (3) member logs: what each rollout actually did (all climate_protocol dirs)
    for dn in sorted(os.listdir(EVAL)):
        mm = re.match(r"^climate_protocol_(\d+)(_n2)?$", dn)
        if not mm:
            continue
        job = mm.group(1)
        logs = sorted(x for x in os.listdir(EVAL + "/" + dn) if re.match(r"^member_.+\d\d\.log$", x))
        names = sorted(set(re.sub(r"\d\d\.log$", "", x[len("member_"):]) for x in logs))
        for a in names:
            p = EVAL + "/" + dn + "/member_" + a + "00.log"
            if not os.path.exists(p):
                continue
            hdr = {}
            with open(p, errors="replace") as f:
                for s in f:
                    if s.startswith("member="):
                        for kv in s.split():
                            if "=" in kv:
                                k, v = kv.split("=", 1)
                                hdr[k] = v
                        break
            n_members = sum(1 for x in logs if re.match("^member_" + re.escape(a) + r"\d\d\.log$", x))
            rec = arms.setdefault(a + "@" + job, {"name": a, "protocol_job": job})
            rec["protocol_dir"] = dn
            rec["members"] = n_members
            rec["ckpt_epoch"] = int(hdr["ckpt_epoch"]) if "ckpt_epoch" in hdr else None
            rec["feedback_dtype"] = hdr.get("feedback_dtype")
            rec["dry_air_fix"] = hdr.get("dry_air_fix")
            rec["force_positive"] = hdr.get("force_positive")
            rec["feedback_fp32"] = hdr.get("feedback_fp32")
            rec["amp"] = hdr.get("amp")
    for key, rec in arms.items():
        a = rec["name"]
        lab = None
        if rec.get("run_tag"):
            for d in RUN_DEFS:
                if d[0] and d[1] == rec["run_tag"]:
                    lab = d[0]
        if lab is None:
            for pre, l in ARM_PREFIX:
                if a.startswith(pre):
                    lab = l
                    rec["run_from_name_prefix"] = True
                    break
        rec["run_label"] = lab
        if "weights" not in rec:
            m2 = re.match(r"^([A-Z]+?)(\d\d)?([a-z]*)$", a.replace("G1", "g1").replace("G4", "g4"))
            for (pre, suf), (w, fl, src) in sub_w.items():
                if re.match("^" + pre + r"\d\d" + suf + "$", a):
                    rec["weights"], rec["flags"], rec["weights_source"] = w, fl, src
            if "weights" not in rec and a in ("A", "B"):
                rec["weights"] = "raw"
                rec["weights_source"] = "polaris_climate_run.pbs defaults CKPT_A/CKPT_B = best_ckpt_mp0.tar"
                rec["ckpt_file"] = "best_ckpt_mp0.tar"
        if (rec.get("ckpt_file") or "").startswith("best_ckpt_ema"):
            rec["weights"] = "ema (EMA-best file)"
        v = re.match(r"^ckpt_mp0_v(\d+)\.tar$", rec.get("ckpt_file") or "")
        rec["ckpt_version"] = int(v.group(1)) if v else None
        if rec["ckpt_version"] is not None and rec.get("ckpt_epoch") is not None:
            rec["version_plus_one_is_epoch"] = (rec["ckpt_version"] + 1 == rec["ckpt_epoch"])
    return arms


def k56_arms():
    """K=56 and lagged sweeps: the arm definitions as written in the campaign files."""
    out = {}
    pre = EVAL + "/depth_vs_length_2026-10-21/prereg.md"
    for line in open(pre, errors="replace"):
        mm = re.match(r"^\| \*\*(\w+)\*\* \| (.+?) \|", line)
        if mm:
            out[mm.group(1) + "_K56"] = {"definition": mm.group(2), "source": pre}
    pre2 = EVAL + "/g4d_e63_2026-11-02/prereg.md"
    for line in open(pre2, errors="replace"):
        mm = re.match(r"^\| \*\*(G4D\d\d)\*\* \| `(ckpt_mp0_v(\d+))` \| (\d+) \| (.+?) \|", line)
        if mm:
            out[mm.group(1) + "_K56"] = {"definition": "EMA of G4D " + mm.group(2) + " (epoch " +
                                         str(int(mm.group(3)) + 1) + "); " + mm.group(5),
                                         "ckpt_version": int(mm.group(3)), "source": pre2}
    drv = EVAL + "/g4d_e63_2026-11-02/chain_driver_v3.sh"
    for line in open(drv, errors="replace"):
        mm = re.search(r"OUT_ROOT=\$\{E\}/(\w+rr_K56)", line)
        if mm:
            ck = re.search(r"CKPT=(\S+?),", line)
            out[mm.group(1)] = {"definition": "re-roll for maps after the NetCDFs were freed; CKPT=" +
                                (ck.group(1) if ck else "?"), "source": drv}
    pre3 = EVAL + "/g4d16_k56_lagged_2026-10-19/prereg.md"
    txt = open(pre3, errors="replace").read()
    lf = re.search(r"LIMIT_FILES=(\d+)`? \((\d{4})\.h5\)", txt)
    mm = re.search(r"That is (\d+)\s+rollouts and (\d+) targets × (\d+) members at depths (\d+)…(\d+)", txt)
    if mm:
        out["G4D16_lagged_K56_d4"] = {"definition": "lagged sweep of G4D16", "rollouts": int(mm.group(1)),
                                      "targets": int(mm.group(2)), "members": int(mm.group(3)),
                                      "depth_min": int(mm.group(4)), "depth_max": int(mm.group(5)),
                                      "limit_files": int(lf.group(1)) if lf else None,
                                      "test_file_year": int(lf.group(2)) if lf else None,
                                      "source": pre3}
    present = sorted(x for x in os.listdir(EVAL) if x.endswith("_K56") or "_lagged_K56" in x)
    return out, present


def screens():
    res = {}
    for dn in sorted(os.listdir(EVAL)):
        mm = re.match(r"^climate_screen_(\d+)$", dn)
        if not mm:
            continue
        labs = sorted(x[len("screen_"):-4] for x in os.listdir(EVAL + "/" + dn)
                      if x.startswith("screen_") and x.endswith(".log"))
        res[mm.group(1)] = labs
    return res


def pbs_default(path, var):
    rx = re.compile(r"^" + var + r'="\$\{' + var + r":-([^}]+)\}")
    for line in open(path, errors="replace"):
        mm = rx.match(line.strip())
        if mm:
            return num(mm.group(1))
    return None


def protocols():
    ep = WT + "/eval-protocol/makani_sfno/polaris/"
    cr = ep + "polaris_climate_run.pbs"
    ei = ep + "polaris_eval_inference.pbs"
    sc = WT + "/b-continuation-dryair/makani_sfno/polaris/polaris_climate_screen.pbs"
    sw = ep + "score_window_protocol.py"
    p = {"five_year": {"source": cr}, "k56": {"source": ei}, "lagged": {"source": ei},
         "screen": {"source": sc}, "score": {"source": sw}}
    for v in ("N_MEMBERS", "MEMBER_STRIDE", "START_FRAME0", "N_STEPS0"):
        p["five_year"][v] = pbs_default(cr, v)
    fy = p["five_year"]
    fy["last_member_steps"] = fy["N_STEPS0"] - fy["MEMBER_STRIDE"] * (fy["N_MEMBERS"] - 1)
    fy["member_start_frames"] = [fy["START_FRAME0"] + fy["MEMBER_STRIDE"] * i for i in range(fy["N_MEMBERS"])]
    fy["member_pairs"] = fy["N_MEMBERS"] * (fy["N_MEMBERS"] - 1) // 2
    txt = open(PREREG_PIPE, errors="replace").read()
    mm = re.search(r"score window is\s+\*\*(\d{4})-01-01 … (\d{4})-12-31\*\*", txt)
    fy["selection_window"] = [int(mm.group(1)), int(mm.group(2))]
    mm = re.search(r"window is \*\*(\d{4})-01-01 … (\d{4})-12-31\*\*, on S and on a new start set", txt)
    fy["reporting_window_sealed"] = [int(mm.group(1)), int(mm.group(2))]
    mm = re.search(r"\(frames (\d+) \+ (\d+)i\)", txt.split("**Reporting:**")[1])
    fy["reporting_set_R_frames"] = [int(mm.group(1)), int(mm.group(2))]
    fy["prereg"] = PREREG_PIPE
    # gates (prereg section 4)
    g = txt.split("## 4. Gates")[1].split("## 5.")[0]
    fy["gate2_neg_tmq_area_max_pct"] = num(re.search(r"negative-TMQ area ≤ (\d+) %", g).group(1))
    fy["gate3_abs_dps_max_hpa"] = num(re.search(r"\|dPS\| ≤ (\d+) hPa", g).group(1))
    fy["gate4_max_drifting_groups"] = num(re.search(r"at most (\d+) drifting groups", g).group(1))
    fy["gate4_drift_threshold_sd"] = num(re.search(r"median channel \|drift\| > (\d+)", g).group(1))
    crt = open(cr, errors="replace").read()
    mm = re.search(r'N_STEPS0="\$\{N_STEPS0:-\d+\}"\s+# Oct 1 (\d{4}) 00:00 -> (\d{4})-(\d\d)-(\d\d) (\d\d):00', crt)
    fy["start_year"] = int(mm.group(1))
    fy["end_date"] = [int(mm.group(2)), int(mm.group(3)), int(mm.group(4)), int(mm.group(5))]
    # groups in the score (score_window_protocol.py)
    st = open(sw, errors="replace").read()
    surf = re.search(r"SURFACE = \[(.*?)\]", st).group(1)
    n_surface = len([x for x in surf.split(",") if x.strip()])
    bands = re.findall(r'\("(top|mid|low)", (\d+), (\d+)\)', st)
    fams = re.search(r"\(T\|U\|V\|Z3\|RELHUM\)", st)
    n_fam = 5 if fams else None
    p["score"]["n_surface_groups"] = n_surface
    p["score"]["n_bands"] = len(bands)
    p["score"]["n_level_families"] = n_fam
    p["score"]["n_groups"] = n_surface + n_fam * len(bands)
    p["score"]["bands"] = [[b, int(lo), int(hi)] for b, lo, hi in bands]
    for v in ("K", "N_IC", "IC_MODE", "STRIDE", "N_TARGETS", "FIRST_START"):
        p["k56"][v] = pbs_default(ei, v)
    test_dir = M + "/data/e3sm_makani_alldata_production/test"
    p["k56"]["test_files"] = sorted(os.listdir(test_dir))
    p["k56"]["n_test_files"] = len(p["k56"]["test_files"])
    p["k56"]["n_ics"] = p["k56"]["N_IC"] * len(p["k56"]["test_files"])
    p["k56"]["horizon_h"] = p["k56"]["K"] * 6
    p["k56"]["horizon_days"] = p["k56"]["horizon_h"] // 24
    p["lagged"]["K"] = p["k56"]["K"]
    p["lagged"]["STRIDE"] = p["k56"]["STRIDE"]
    p["lagged"]["N_TARGETS"] = p["k56"]["N_TARGETS"]
    p["lagged"]["members_per_target"] = p["k56"]["K"] // p["k56"]["STRIDE"]
    for v in ("START_YEAR", "START_FRAME", "N_LEADS"):
        p["screen"][v] = pbs_default(sc, v)
    p["screen"]["second_start_frame"] = 1156 if "START_FRAME=1156" in open(sc).read() else None
    p["data_split"] = {}
    for part in ("train", "valid", "test"):
        yrs = sorted(int(x[:4]) for x in os.listdir(M + "/data/e3sm_makani_alldata_production/" + part)
                     if re.match(r"^\d{4}\.h5$", x))
        p["data_split"][part] = {"first": yrs[0], "last": yrs[-1], "n_years": len(yrs)}
    return p


# ---------------------------------------------------------------------------
def build():
    csvr = csv_rows()
    prov_path, prov = provenance_jobs()
    configs = {}
    for label, tag, role, jobs in RUN_DEFS:
        logpath, segs = parse_outlog(tag)
        ypath, ykeys = rendered_yaml(tag)
        if len(segs) != len(jobs):
            raise SystemExit("ERROR SEGMENT_JOB_COUNT %s: %d segments vs %d jobs" % (label, len(segs), len(jobs)))
        s0 = segs[0]
        P = s0["params"]
        Mn = s0["main"]
        rec = {"run_tag": tag, "role": role}
        # --- jobs and segments
        seg_out = []
        for (job, wdir), s in zip(jobs, segs):
            so = parse_stdout(job, wdir)
            ep_nums = [e["epoch"] for e in s["epochs"]]
            seg = {"job": job, "start_ts": s["start_ts"], "end_ts": s["total_ts"] or s["last_ts"],
                   "max_epochs_printed": num(s["params"].get("max_epochs", "nan")),
                   "epoch_first": min(ep_nums) if ep_nums else None,
                   "epoch_last": max(ep_nums) if ep_nums else None,
                   "n_epochs": len(ep_nums),
                   "total_training_s": s["total_training_s"],
                   "git_hash": (s["git_hash"] or "")[:8],
                   "resume_from": s["resume_from"], "loaded_pretrained": s["loaded_pretrained"],
                   "world_size": num(s["params"].get("world_size", "nan")),
                   "stdout": so}
            if s["total_training_s"] is None:
                seg["training_s"] = round(sum(e["epoch_time_s"] for e in s["epochs"]), 2)
                seg["training_s_basis"] = "derived_from_epoch_times"
            else:
                seg["training_s"] = s["total_training_s"]
                seg["training_s_basis"] = "Total training time line"
            seg["nodes"] = seg["world_size"] // GPUS_PER_NODE
            # how the job id was tied to this segment
            if so["found"]:
                checks = [so.get("run_num") == tag, so.get("rendered_max_epochs") == seg["max_epochs_printed"],
                          so.get("jobid") == job]
                if so.get("total_training_s") is not None and s["total_training_s"] is not None:
                    checks.append(abs(so["total_training_s"] - s["total_training_s"]) < 0.01)
                seg["job_source"] = "PBS stdout " + so["path"]
                seg["job_verified"] = all(checks)
                seg["job_checks"] = checks
            elif job in prov:
                seg["job_source"] = "launcher provenance sidecar " + prov_path + " ('" + prov[job]["text"] + "')"
                seg["job_verified"] = prov[job]["named"] and prov[job]["epochs"] == seg["max_epochs_printed"]
            else:
                kj, kline = knowledge_job(tag)
                seg["job_source"] = "SECONDARY: KNOWLEDGE.md lineage table (" + KNOW + "): " + (kline or "")
                seg["job_verified"] = False
                seg["unverified"] = True
                if kj != job:
                    raise SystemExit("ERROR JOB_ID_MISMATCH %s %s vs KNOWLEDGE %s" % (label, job, kj))
            seg["csv_rows"] = csvr.get(job, [])
            seg_out.append(seg)
        rec["segments"] = seg_out
        rec["jobs"] = [int(s["job"]) for s in seg_out]
        rec["jobs_unverified"] = [int(s["job"]) for s in seg_out if s.get("unverified")]
        # --- identity / data / loss
        n_state = num(P["n_state_channels"])
        n_diag = num(P["n_diagnostic_channels"])
        n_forc = num(P["n_forcing_channels"])
        rec["n_state_channels"] = n_state
        rec["n_diagnostic_channels"] = n_diag
        rec["n_forcing_channels"] = n_forc
        rec["output_channels"] = n_state + n_diag
        rec["input_channels"] = n_state + n_forc
        rec["dropped_channel_names"] = P.get("dropped_channel_names")
        rec["soil_outputs"] = "SOILWATER_10CM" in P.get("channel_names", "")
        rec["multistep_count"] = num(P["multistep_count"])
        rec["n_future"] = num(P["n_future"])
        rec["target_hours_printed"] = None
        rec["conserve_dry_air"] = P.get("conserve_dry_air") == "True"
        rec["conserve_soil_moisture"] = P.get("conserve_soil_moisture") == "True"
        rec["amp_mode"] = P["amp_mode"]
        emaline = P.get("ema")
        if emaline:
            mm = re.search(r"'enabled': (True|False), 'decay': ([0-9.]+)", emaline)
            rec["ema_enabled"] = mm.group(1) == "True"
            rec["ema_decay"] = float(mm.group(2))
        else:
            rec["ema_enabled"] = False
            rec["ema_decay"] = None
        rec["ema_trainer_line_decay"] = s0["ema_trainer"]
        # --- optimisation
        rec["optimizer_type"] = P["optimizer_type"]
        rec["optimizer_beta1"] = num(P["optimizer_beta1"])
        rec["optimizer_beta2"] = num(P["optimizer_beta2"])
        rec["optimizer_max_grad_norm"] = num(P["optimizer_max_grad_norm"])
        rec["weight_decay"] = num(P["weight_decay"])
        rec["lr"] = num(P["lr"])
        rec["lr_start"] = first_float(P, "lr_start")
        rec["scheduler"] = P["scheduler"]
        rec["scheduler_T_0"] = first_float(P, "scheduler_T_0")
        rec["scheduler_T_mult"] = first_float(P, "scheduler_T_mult")
        rec["scheduler_T_max"] = first_float(P, "scheduler_T_max")
        rec["scheduler_min_lr"] = first_float(P, "scheduler_min_lr")
        rec["scheduler_factor"] = first_float(P, "scheduler_factor")
        rec["scheduler_patience"] = first_float(P, "scheduler_patience")
        rec["lr_warmup_steps"] = num(P["lr_warmup_steps"])
        rec["local_batch"] = num(Mn.get("per_rank_batch_size", P["batch_size"]))
        rec["gpus"] = num(P["world_size"])
        rec["nodes"] = rec["gpus"] // GPUS_PER_NODE
        rec["global_batch"] = num(P["global_batch_size"])
        rec["global_batch_check"] = rec["local_batch"] * rec["gpus"] == rec["global_batch"]
        rec["h_parallel_size"] = num(P["h_parallel_size"])
        rec["w_parallel_size"] = num(P["w_parallel_size"])
        rec["scale_factor"] = num(P["scale_factor"])
        rec["embed_dim"] = num(P["embed_dim"])
        rec["num_layers"] = num(P["num_layers"])
        rec["n_params"] = s0["n_params"]
        rec["n_train_samples_per_epoch"] = num(P["n_train_samples_per_epoch"])
        rec["n_eval_samples"] = num(P["n_eval_samples"])
        rec["checkpoint_num_versions"] = first_float(P, "checkpoint_num_versions")
        rec["ema_warmup"] = ("warmup=True" in (s0.get("ema_line") or "")) if s0.get("ema_line") else None
        rec["valid_autoreg_steps"] = num(P["valid_autoreg_steps"])
        rec["pretrained"] = P.get("pretrained")
        rec["pretrained_checkpoint_path"] = P.get("pretrained_checkpoint_path")
        # consistency of the training knobs across segments
        knobs = ("lr", "scheduler", "scheduler_T_0", "scheduler_T_mult", "scheduler_T_max", "scheduler_min_lr",
                 "lr_warmup_steps", "batch_size", "global_batch_size", "world_size", "amp_mode", "multistep_count",
                 "n_future", "optimizer_beta2", "optimizer_max_grad_norm", "ema", "conserve_dry_air",
                 "n_state_channels", "scale_factor")
        diffs = {}
        for s in segs[1:]:
            for k in knobs:
                if s["params"].get(k) != P.get(k):
                    diffs.setdefault(k, [P.get(k)]).append(s["params"].get(k))
        rec["segment_knob_differences"] = diffs
        rec["segment_max_epochs"] = [s["max_epochs_printed"] for s in seg_out]
        # --- epochs, updates, time
        allep = [e for s in segs for e in s["epochs"]]
        epn = [e["epoch"] for e in allep]
        if len(set(epn)) != len(epn):
            raise SystemExit("ERROR DUPLICATE_EPOCHS %s" % label)
        rec["epochs_trained"] = len(epn)
        rec["epoch_range"] = [min(epn), max(epn)]
        rec["steps_per_epoch"] = sorted(set(e["steps"] for e in allep))
        rec["updates"] = sum(e["steps"] for e in allep)
        rec["updates_check"] = rec["updates"] == rec["epochs_trained"] * rec["steps_per_epoch"][0] if len(rec["steps_per_epoch"]) == 1 else None
        rec["samples_per_epoch_over_steps"] = rec["n_train_samples_per_epoch"] // rec["steps_per_epoch"][0]
        rec["training_s"] = round(sum(s["training_s"] for s in seg_out), 2)
        rec["training_h"] = rec["training_s"] / 3600.0
        rec["node_hours"] = sum(s["training_s"] * s["nodes"] for s in seg_out) / 3600.0
        rec["median_epoch_time_s"] = statistics.median([e["epoch_time_s"] for e in allep])
        rec["median_step_ms"] = statistics.median([e["step_ms"] for e in allep if "step_ms" in e])
        rec["training_s_has_derived_segment"] = any(s["training_s_basis"] != "Total training time line" for s in seg_out)
        # lowest raw validation loss epoch = what best_ckpt_mp0.tar holds (used as an init)
        vl = [(e["val_loss"], e["epoch"]) for e in allep if "val_loss" in e]
        rec["best_ckpt_epoch_end_of_run"] = min(vl)[1] if vl else None
        vl1 = [(e["val_loss"], e["epoch"]) for e in segs[0]["epochs"] if "val_loss" in e]
        rec["best_ckpt_epoch_end_of_first_job"] = min(vl1)[1] if vl1 else None
        # yaml vs trainer
        yd = {}
        printed = {"batch_size": rec["local_batch"], "n_future": rec["n_future"],
                   "max_epochs": num(segs[-1]["params"].get("max_epochs")), "lr": rec["lr"],
                   "scheduler": rec["scheduler"], "scheduler_T_max": rec["scheduler_T_max"],
                   "scale_factor": rec["scale_factor"]}
        for k, v in ykeys.items():
            if printed.get(k) != v:
                yd[k] = {"yaml": v, "trainer_printed": printed.get(k)}
        rec["yaml_vs_trainer"] = yd
        # launcher provenance sidecars: allocated nodes incl. any spare
        prov_files = []
        for fn in sorted(os.listdir(RUNS)):
            if fn.startswith(tag + ".") and fn.endswith("provenance.txt"):
                t = open(RUNS + "/" + fn, errors="replace").read()
                mm = re.search(r"nodes=(\d+) \(\+(\d+) spare\)", t)
                mn = re.search(r"nodes=(\d+) \(no spare", t)
                prov_files.append({"file": RUNS + "/" + fn,
                                   "nodes": int(mm.group(1)) if mm else (int(mn.group(1)) if mn else None),
                                   "spare_nodes": int(mm.group(2)) if mm else (0 if mn else None)})
        rec["provenance_sidecars"] = prov_files
        rec["sources"] = {"trainer_log": logpath, "rendered_yaml": ypath,
                          "pbs_stdout": [s["stdout"]["path"] for s in seg_out],
                          "job_ids": [s["job_source"] for s in seg_out]}
        configs[label] = rec
    return configs


def fmt_sched(r):
    if r["scheduler"] == "CosineAnnealingWarmRestarts":
        return "SGDR: T₀ %d, T_mult %d, min %s" % (r["scheduler_T_0"], r["scheduler_T_mult"], dec(r["scheduler_min_lr"]))
    if r["scheduler"] == "CosineAnnealingLR":
        return "cosine: T_max %d, min %s" % (r["scheduler_T_max"], dec(r["scheduler_min_lr"]))
    if r["scheduler"] == "ReduceLROnPlateau":
        return "reduce-on-plateau: factor %s, patience %d, no min printed" % (dec(r["scheduler_factor"]), r["scheduler_patience"])
    return r["scheduler"]


def dec(x):
    s = ("%.10f" % x).rstrip("0").rstrip(".")
    return s


def main():
    configs = build()
    arms = rollout_arms()
    k56, k56_dirs = k56_arms()
    scr = screens()
    prot = protocols()
    # acronyms per training run, from the rollout records
    for lab in configs:
        names = sorted(set(r["name"] for r in arms.values() if r.get("run_label") == lab))
        configs[lab]["rollout_arms"] = names
        configs[lab]["n_rollout_arms_5yr"] = len(names)
        configs[lab]["rollout_epochs_5yr"] = sorted(set(r["ckpt_epoch"] for r in arms.values()
                                                        if r.get("run_label") == lab and r.get("ckpt_epoch") is not None))
        configs[lab]["n_rollout_epochs_5yr"] = len(configs[lab]["rollout_epochs_5yr"])
    sf2 = RUNS + "/g4d_sf2_2026-10-19"
    sf2_files = sorted(os.listdir(sf2))
    sf2_val = None
    for line in open(sf2 + "/submit_g4d_sf2.sh", errors="replace"):
        mm = re.search(r"SCALE_FACTOR=(\d+)", line)
        if mm:
            sf2_val = int(mm.group(1))
            break
    not_run = {
        "G4D-sf2": {"dir": sf2, "files": sf2_files, "scale_factor": sf2_val,
                    "run_dir_exists": os.path.isdir(EXP + "/g4d_sf2"),
                    "note": "prepared 2026-10-19, never submitted (CHANGELOG 2026-10-19; no run dir, no log)"},
    }
    kn = open(KNOW, errors="replace").read()
    mm = re.search(r"Cancelled before running:\*\* the A-based Stage-1 arms (\d+) \(T-anneal\), (\d+)\s*\(`anneal_dryair` from A\) and (\d+) \(T-d8\), on (\d{4}-\d\d-\d\d)", kn)
    if mm:
        not_run["cancelled_stage1"] = {"T-anneal": int(mm.group(1)), "anneal_dryair": int(mm.group(2)),
                                       "T-d8": int(mm.group(3)), "date": mm.group(4),
                                       "source": "SECONDARY: " + KNOW + " section 1 notes"}
    out = {"_about": "configs.json: receipt for 2026-11-02_makani_configurations.md, written by extract_configs.py",
           "runs": configs, "rollout_arms": arms, "k56_arms": k56, "k56_dirs": k56_dirs,
           "screens": scr, "protocols": prot, "not_run": not_run}
    # flat summary counts used in the prose
    out["counts"] = {
        "training_runs": len(configs),
        "rollout_arm_rows_5yr": len(arms),
        "rollout_arm_names_5yr": len(set(r["name"] for r in arms.values())),
        "rollout_arms_unlabelled": sorted(k for k, r in arms.items() if r.get("run_label") is None),
        "rollout_arms_no_weights": sorted(k for k, r in arms.items() if not r.get("weights")),
        "rollout_arms_version_rule_checked": sum(1 for r in arms.values() if "version_plus_one_is_epoch" in r),
        "rollout_arms_version_rule_holds": sum(1 for r in arms.values() if r.get("version_plus_one_is_epoch")),
        "screen_jobs": len(scr),
        "screen_labels": sum(len(v) for v in scr.values()),
        "unverified_job_ids": sum(len(r["jobs_unverified"]) for r in configs.values()),
    }
    with open(os.path.join(HERE, "configs.json"), "w") as f:
        json.dump(out, f, indent=1, sort_keys=True, default=str)
    for lab, r in configs.items():
        print("%-13s jobs=%s ep=%d upd=%d h=%.2f nh=%.1f med=%.0f gb=%d nodes=%d amp=%s ms=%d nf=%d ema=%s dry=%s verified=%s"
              % (lab, r["jobs"], r["epochs_trained"], r["updates"], r["training_h"], r["node_hours"],
                 r["median_epoch_time_s"], r["global_batch"], r["nodes"], r["amp_mode"], r["multistep_count"],
                 r["n_future"], r["ema_decay"], r["conserve_dry_air"], [s["job_verified"] for s in r["segments"]]))
    print("CONFIGS_OK runs=%d arms=%d" % (len(configs), len(arms)))
    if "--render" in sys.argv:
        render(out)
        render_arms(out)


def arm_family(name):
    """FSE23 -> FSE<e>, G4E15fix -> G4E<e>fix, B22fp32 -> B<e>fp32."""
    mm = re.match(r"^((?:G\d)?[A-Z]+)(\d\d)(.*)$", name)
    return (mm.group(1) + "<e>" + mm.group(3)) if mm else name


def ranges(nums):
    nums = sorted(set(nums))
    out, start, prev = [], None, None
    for n in nums:
        if start is None:
            start = prev = n
        elif n == prev + 1:
            prev = n
        else:
            out.append((start, prev))
            start = prev = n
    if start is not None:
        out.append((start, prev))
    return ", ".join(("e%d" % a) if a == b else ("e%d–e%d" % (a, b)) for a, b in out)


def render_arms(out):
    """Table 2 rows: one row per arm family (5-yr protocol arms), from the rollout records."""
    fam = {}
    for a, r in out["rollout_arms"].items():
        k = (r.get("run_label"), arm_family(r["name"]), r.get("weights") or "?", r.get("feedback_dtype"),
             r.get("dry_air_fix"), r.get("flags") or "", r.get("force_positive"))
        fam.setdefault(k, []).append((a, r))
    print()
    for k in sorted(fam, key=lambda x: ([d[0] for d in RUN_DEFS].index(x[0]) if x[0] in [d[0] for d in RUN_DEFS] else 99, x[1])):
        rows = fam[k]
        eps = [r.get("ckpt_epoch") for _, r in rows if r.get("ckpt_epoch") is not None]
        jobs = sorted(set(r.get("protocol_job") for _, r in rows))
        files = sorted(set(r.get("ckpt_file") or "(member log only)" for _, r in rows))
        print("| %s | %s | %d | %s | %s | %s | dry_air_fix=%s %s | %s | %s |" % (
            k[0], k[1], len(rows), ranges(eps), k[2], (k[3] or "?").replace("torch.", ""), k[4], k[5],
            ", ".join(files[:3]) + (" …" if len(files) > 3 else ""), ", ".join(jobs)))


def render(out):
    """Print the markdown rows of Tables 1a/1b/1c (pasted into the doc verbatim)."""
    C = out["runs"]
    print()
    for lab, r in C.items():
        jobs = " + ".join(str(j) + (" UNVERIFIED" if j in r["jobs_unverified"] else "") for j in r["jobs"])
        print("| %s | `%s` | %s | | %d; soil %s | %d / %d | %s | %s | %s | |" % (
            lab, r["run_tag"], jobs, r["output_channels"], "yes" if r["soil_outputs"] else "no",
            r["multistep_count"], r["n_future"],
            "trained in" if r["conserve_dry_air"] else ("soil-moisture fix trained in" if r["conserve_soil_moisture"] else "none"),
            r["amp_mode"], dec(r["ema_decay"]) if r["ema_decay"] else "none"))
    print()
    for lab, r in C.items():
        print("| %s | %s, %s, %s | %s | %s | %d | %d | %d | %d | %d | %d x %d | %d | %d | %.1f | %.1f | %d | %d | |" % (
            lab, r["optimizer_type"], dec(r["optimizer_beta2"]), r["optimizer_max_grad_norm"], dec(r["lr"]),
            fmt_sched(r), r["lr_warmup_steps"], r["local_batch"], r["gpus"], r["nodes"], r["global_batch"],
            r["h_parallel_size"], r["w_parallel_size"], r["epochs_trained"], r["updates"], r["training_h"],
            r["node_hours"], int(round(r["median_epoch_time_s"])), r["scale_factor"]))
    print()
    for lab, r in C.items():
        for s in r["segments"]:
            so = s["stdout"]
            print("| %s | %s | %d–%d | %d | %d | %.2f | %s | %s | %s |" % (
                lab, s["job"], s["epoch_first"], s["epoch_last"], s["nodes"], s["max_epochs_printed"],
                s["training_s"], "epoch-time sum" if s["training_s_basis"] != "Total training time line" else "Total training time",
                so.get("killed_walltime_elapsed_s", "not recorded"), s["git_hash"]))


if __name__ == "__main__":
    main()
