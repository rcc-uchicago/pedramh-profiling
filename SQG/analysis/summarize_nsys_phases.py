"""summarize_nsys_phases.py -- host syncs, device-to-host copies and launches per optimizer step, phase by phase,
from the SQLite export of an nsys capture (profile_overlap.py --nsys under SQG_NVTX=1; profile_phases_polaris.pbs).

  python summarize_nsys_phases.py <capture.sqlite> --out <json> [--step-regex '^step$']

Every CUDA API call is attributed to the stack of harness NVTX ranges (domain 0, torch.cuda.nvtx) open on its thread
when it started, e.g. main:step/advance/dispatch/solver. Its GPU work (kernels, memcpys, memsets, linked by
correlationId) goes with it. Syncs and DtoH copies are also attributed to the innermost range of any OTHER domain open
at that moment: for JAX that is XLA's per-op `Thunk:#...hlo_op=...#` annotation, i.e. the op that asked for the sync.
Everything is divided by the number of ranges matching --step-regex.

Classes of API call (name without its _v<N>/_ptsz suffix):
  sync         cu/cuda{Stream,Event,Ctx,Device,Thread}Synchronize -- the host thread blocks until the GPU catches up
  stream_wait  cu/cudaStreamWaitEvent -- a GPU-side dependency between streams (e.g. the JAX -> torch dlpack hand-off);
               the host does not block
  poll         cu/cudaEventQuery, cu/cudaStreamQuery -- non-blocking
  launch       *LaunchKernel*, *GraphLaunch*; "slow" = longer than --slow-launch-us on the host (launch-queue back-
               pressure: the host is paced by the GPU without any sync call)
  memcpy       *Memcpy* -- host time inside the call; a copy to pageable host memory returns only when it is done
Stdlib only (sqlite3): runs inside the job, or on a login node. Prints PHASES_OK.
"""
import argparse
import collections
import json
import re
import sqlite3

SYNC = re.compile(r"^cu(da)?(Stream|Event|Ctx|Device|Thread)Synchronize$")
STREAM_WAIT = re.compile(r"^cu(da)?StreamWaitEvent$")
POLL = re.compile(r"^cu(da)?(Event|Stream)Query$")
LAUNCH = re.compile(r"LaunchKernel|GraphLaunch|LaunchCooperativeKernel")
MEMCPY = re.compile(r"[Mm]emcpy")
# nsys's own enum ids (ENUM_CUDA_MEMCPY_OPER / ENUM_CUDA_MEM_KIND), used when the tables are missing
COPY_KIND = {1: "htod", 2: "dtoh", 8: "dtod"}
MEM_KIND = {0: "Pageable", 1: "Pinned", 2: "Device", 3: "Array", 4: "Managed", 5: "Device Static",
            6: "Managed Static", 7: "Unknown"}


def api_name(s):
    """cuMemcpyDtoHAsync_v2 -> cuMemcpyDtoHAsync, cudaLaunchKernel_v7000 -> cudaLaunchKernel."""
    return re.sub(r"(_v\d+)?(_pt[sd]z)?$", "", s)


def short_issuer(s):
    """XLA's `Thunk:#name=while/body/cond[...],hlo_op=conditional.277.0#` -> `conditional.277.0 (while/body/cond)`."""
    op = re.search(r"hlo_op=([^#,]+)", s)
    name = re.search(r"name=([^#,\[]+)", s)
    if op:
        return f"{op.group(1)} ({name.group(1)})" if name else op.group(1)
    return s[:80]


def sweep(ranges, events):
    """Open-range stacks of ONE thread and domain class.

    ranges: [(start, end, name)], properly nested (push/pop); events: [(time, key)].
    Returns ({key: (outer, ..., inner)}, [(path, duration)] for every range).
    """
    ranges = sorted(ranges, key=lambda r: (r[0], -r[1]))
    stack, paths, rpaths, i = [], {}, [], 0

    def push(r):
        while stack and stack[-1][1] <= r[0]:
            stack.pop()
        stack.append(r)
        rpaths.append((tuple(x[2] for x in stack), r[1] - r[0]))

    for t, key in sorted(events):
        while i < len(ranges) and ranges[i][0] <= t:
            push(ranges[i])
            i += 1
        while stack and stack[-1][1] <= t:
            stack.pop()
        paths[key] = tuple(x[2] for x in stack)
    for r in ranges[i:]:
        push(r)
    return paths, rpaths


def table(cur, name):
    return cur.execute("select count(*) from sqlite_master where type='table' and name=?", (name,)).fetchone()[0] > 0


def enum(cur, name, default):
    if not table(cur, name):
        return default
    return {i: lab for i, _, lab in cur.execute(f"select id, name, label from {name}")}


def new_phase():
    return {"wall_ns": 0, "ranges": 0, "api_calls": 0, "launches": 0, "slow_launches": 0, "launch_ns": 0,
            "syncs": collections.Counter(), "sync_ns": 0, "stream_waits": 0, "polls": 0, "memcpy_api_ns": 0,
            "dtoh": 0, "dtoh_bytes": 0, "dtoh_dst": collections.Counter(), "htod": 0, "htod_bytes": 0,
            "htod_src": collections.Counter(), "dtod": 0, "dtod_bytes": 0, "gpu_ns": 0,
            "sync_issuers": collections.Counter(), "dtoh_issuers": collections.Counter()}


def summarize(path, step_regex=r"^step$", domain=0, slow_launch_us=100.0):
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    cur = db.cursor()
    strings = dict(cur.execute("select id, value from StringIds"))
    copy_label = enum(cur, "ENUM_CUDA_MEMCPY_OPER", None)
    mem_kind = enum(cur, "ENUM_CUDA_MEM_KIND", MEM_KIND)
    if copy_label is None:
        copy_kind = COPY_KIND
    else:                                               # map by label, not by id: the ids are nsys's, not CUPTI's
        by_label = {"Host-to-Device": "htod", "Device-to-Host": "dtoh", "Device-to-Device": "dtod"}
        copy_kind = {i: by_label[lab] for i, lab in copy_label.items() if lab in by_label}

    # ---- NVTX push/pop ranges, per thread, split into the harness domain and everything else ----
    harness, other = collections.defaultdict(list), collections.defaultdict(list)
    if table(cur, "NVTX_EVENTS"):
        for start, end, text, text_id, tid, dom in cur.execute(
                "select start, end, text, textId, globalTid, domainId from NVTX_EVENTS "
                "where eventType = 59 and end is not null"):
            name = text if text is not None else strings.get(text_id, "?")
            (harness if (dom or 0) == domain else other)[tid].append((start, end, name))
    step_re = re.compile(step_regex)
    step_ranges = [(tid, r) for tid, rs in harness.items() for r in rs if step_re.search(r[2])]
    n_steps = len(step_ranges)
    if n_steps == 0:
        raise SystemExit(f"ERROR NO_STEP_RANGES: no NVTX range matches {step_regex!r} in domain {domain}")

    # thread labels: the thread(s) holding the step ranges are "main"; other threads with harness ranges are
    # thread1, thread2, ... in order of their first range; the rest share one bucket
    main_tids = sorted({tid for tid, _ in step_ranges})
    labels = {tid: ("main" if len(main_tids) == 1 else f"main{k}") for k, tid in enumerate(main_tids)}
    for k, tid in enumerate(sorted((t for t in harness if t not in labels), key=lambda t: min(r[0] for r in harness[t]))):
        labels[tid] = f"thread{k + 1}"

    # ---- CUDA API calls ----
    api = {}                                             # correlationId -> (tid, start, end, name)
    by_tid = collections.defaultdict(list)
    for cid, start, end, name_id, tid in cur.execute(
            "select correlationId, start, end, nameId, globalTid from CUPTI_ACTIVITY_KIND_RUNTIME"):
        api[cid] = (tid, start, end, api_name(strings.get(name_id, "?")))
        by_tid[tid].append((start, cid))

    phase_of, issuer_of = {}, {}
    phases = collections.defaultdict(new_phase)
    for tid, events in by_tid.items():
        label = labels.get(tid)
        hp, rp = sweep(harness.get(tid, []), events)
        op, _ = sweep(other.get(tid, []), events)
        for cid in hp:
            p = hp[cid]
            phase_of[cid] = f"{label}:{'/'.join(p)}" if label and p else \
                f"{label}:(no range)" if label else "other threads:(no range)"
            issuer_of[cid] = short_issuer(op[cid][-1]) if op.get(cid) else "-"
        if label:
            for p, dur in rp:
                ph = phases[f"{label}:{'/'.join(p)}"]
                ph["wall_ns"] += dur
                ph["ranges"] += 1

    slow_ns = slow_launch_us * 1e3
    api_totals = collections.Counter()
    for cid, (tid, start, end, name) in api.items():
        ph = phases[phase_of[cid]]
        dur = end - start
        ph["api_calls"] += 1
        api_totals[name] += 1
        if SYNC.search(name):
            ph["syncs"][name] += 1
            ph["sync_ns"] += dur
            ph["sync_issuers"][issuer_of[cid]] += 1
        elif STREAM_WAIT.search(name):
            ph["stream_waits"] += 1
        elif POLL.search(name):
            ph["polls"] += 1
        elif LAUNCH.search(name):
            ph["launches"] += 1
            ph["launch_ns"] += dur
            ph["slow_launches"] += dur > slow_ns
        elif MEMCPY.search(name):
            ph["memcpy_api_ns"] += dur

    # ---- GPU activities, attributed through the API call that issued them ----
    unmatched = 0
    if table(cur, "CUPTI_ACTIVITY_KIND_MEMCPY"):
        for cid, start, end, nbytes, kind, src, dst in cur.execute(
                "select correlationId, start, end, bytes, copyKind, srcKind, dstKind from CUPTI_ACTIVITY_KIND_MEMCPY"):
            if cid not in api:
                unmatched += 1
                continue
            ph = phases[phase_of[cid]]
            ph["gpu_ns"] += end - start
            k = copy_kind.get(kind)
            if k == "dtoh":
                ph["dtoh"] += 1
                ph["dtoh_bytes"] += nbytes
                ph["dtoh_dst"][mem_kind.get(dst, str(dst))] += 1
                ph["dtoh_issuers"][issuer_of[cid]] += 1
            elif k == "htod":
                ph["htod"] += 1
                ph["htod_bytes"] += nbytes
                ph["htod_src"][mem_kind.get(src, str(src))] += 1
            elif k == "dtod":
                ph["dtod"] += 1
                ph["dtod_bytes"] += nbytes
    for t in ("CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_MEMSET"):
        if table(cur, t):
            for cid, start, end in cur.execute(f"select correlationId, start, end from {t}"):
                if cid in api:
                    phases[phase_of[cid]]["gpu_ns"] += end - start
                else:
                    unmatched += 1

    # ---- per optimizer step ----
    s = n_steps

    def per(x, nd=3):
        return round(x / s, nd)

    def ms(ns):
        return round(ns / 1e6 / s, 4)

    out_phases = {}
    for key in sorted(phases):
        p = phases[key]
        if not p["api_calls"] and not p["ranges"]:
            continue
        out_phases[key] = {
            "wall_ms": ms(p["wall_ns"]), "ranges": per(p["ranges"]), "api_calls": per(p["api_calls"]),
            "launches": per(p["launches"]), "slow_launches": per(p["slow_launches"]), "launch_ms": ms(p["launch_ns"]),
            "syncs": {k: per(v) for k, v in p["syncs"].most_common()}, "sync_ms": ms(p["sync_ns"]),
            "stream_waits": per(p["stream_waits"]), "polls": per(p["polls"]), "memcpy_api_ms": ms(p["memcpy_api_ns"]),
            "dtoh": per(p["dtoh"]), "dtoh_bytes": per(p["dtoh_bytes"], 1),
            "dtoh_dst": {k: per(v) for k, v in p["dtoh_dst"].items()},
            "htod": per(p["htod"]), "htod_bytes": per(p["htod_bytes"], 1),
            "htod_src": {k: per(v) for k, v in p["htod_src"].items()},
            "dtod": per(p["dtod"]), "dtod_bytes": per(p["dtod_bytes"], 1), "gpu_ms": ms(p["gpu_ns"]),
            "sync_issuers": {k: per(v) for k, v in p["sync_issuers"].most_common(5)},
            "dtoh_issuers": {k: per(v) for k, v in p["dtoh_issuers"].most_common(5)}}
    tot = {k: sum(p[k] for p in phases.values()) for k in
           ("api_calls", "launches", "slow_launches", "sync_ns", "stream_waits", "memcpy_api_ns", "dtoh", "dtoh_bytes",
            "htod", "dtod", "dtod_bytes", "gpu_ns")}
    n_sync = sum(sum(p["syncs"].values()) for p in phases.values())
    step_ns = [r[1] - r[0] for _, r in step_ranges]
    return {
        "sqlite": path, "step_regex": step_regex, "domain": domain, "steps": n_steps,
        "step_wall_ms_mean": round(sum(step_ns) / s / 1e6, 3), "slow_launch_us": slow_launch_us,
        "threads": {v: str(k) for k, v in labels.items()},
        "per_step": {"syncs": per(n_sync), "sync_ms": ms(tot["sync_ns"]), "dtoh": per(tot["dtoh"]),
                     "dtoh_bytes": per(tot["dtoh_bytes"], 1), "htod": per(tot["htod"]), "dtod": per(tot["dtod"]),
                     "dtod_bytes": per(tot["dtod_bytes"], 1), "stream_waits": per(tot["stream_waits"]),
                     "launches": per(tot["launches"]), "slow_launches": per(tot["slow_launches"]),
                     "memcpy_api_ms": ms(tot["memcpy_api_ns"]), "api_calls": per(tot["api_calls"]),
                     "gpu_ms_sum": ms(tot["gpu_ns"])},
        "api_calls_per_step": {k: per(v) for k, v in api_totals.most_common()},
        "unmatched_gpu_activities": unmatched,
        "phases": out_phases}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sqlite")
    ap.add_argument("--out", required=True)
    ap.add_argument("--step-regex", default=r"^step$", help="NVTX ranges that delimit one optimizer step")
    ap.add_argument("--domain", type=int, default=0, help="NVTX domain id of the harness ranges (torch: 0)")
    ap.add_argument("--slow-launch-us", type=float, default=100.0)
    a = ap.parse_args()
    r = summarize(a.sqlite, a.step_regex, a.domain, a.slow_launch_us)
    with open(a.out, "w") as fh:
        json.dump(r, fh, indent=1)
    t = r["per_step"]
    print(f"{r['steps']} steps of {r['step_wall_ms_mean']} ms: per step {t['syncs']} syncs ({t['sync_ms']} ms blocked), "
          f"{t['dtoh']} DtoH ({t['dtoh_bytes']:.0f} B), {t['stream_waits']} stream waits, {t['launches']} launches "
          f"({t['slow_launches']} slow), {t['dtod']} DtoD")
    step_ms = r["step_wall_ms_mean"]
    rows = [(k, p) for k, p in r["phases"].items()
            if sum(p["syncs"].values()) or p["dtoh"] or p["slow_launches"] or p["wall_ms"] >= 0.01 * step_ms]
    for k, p in sorted(rows, key=lambda kp: -kp[1]["wall_ms"])[:12]:
        iss = next(iter(p["sync_issuers"] or p["dtoh_issuers"] or {"-": 0}))
        print(f"  {k[:48]:48s} {p['wall_ms']:9.2f} ms  sync {sum(p['syncs'].values()):7.1f} ({p['sync_ms']:8.2f} ms)  "
              f"DtoH {p['dtoh']:6.1f}  launch {p['launches']:7.1f} (slow {p['slow_launches']:6.1f})  {iss[:40]}")
    print(f"PHASES_OK {a.out}")


if __name__ == "__main__":
    main()
