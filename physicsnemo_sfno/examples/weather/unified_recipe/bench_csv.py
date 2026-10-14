# ⚠ VENDOR DIVERGENCE — this file belongs to pedramh-profiling, NOT to upstream
# physicsnemo (subtree of NVIDIA/physicsnemo @ a8eedb65, imported 94f9e4dd). It is a NEW
# file, so a `git subtree pull` will not conflict here; the conflict surface is the five
# small delimited blocks in train.py that call it (labeled 1/4 .. 4b/4: import, factory,
# minibatch, epoch, validation). If a pull ever drops one, polaris/bench_csv_test.py
# fails loudly (it greps train.py for the wiring).
"""Env-gated CSV tee for unified_recipe/train.py's four logged metrics.

Why: train.py logs exactly four metrics (loss per minibatch; Learning Rate, GB/s,
Validation error per epoch) through LaunchLogger into an *offline MLflow file store* —
readable only through the mlflow client, which is nothing this repo's tooling greps.
This tee ALSO writes them to a plain CSV. MLflow is untouched.

Usage:  PHYSICSNEMO_BENCH_CSV=<path>  — unset (the default) means from_env() returns
None and every hook in train.py is a no-op; behavior is then byte-identical to upstream.
Prefer an absolute path: train.py runs under hydra with job.chdir, so a relative path
lands in the hydra run dir (which may be what you want — it keeps metrics beside
./checkpoints).

Schema (FROZEN — CLAUDE.md #10: once fixed, columns never drift; a consumer greps them):

    timestamp,epoch,step,loss,lr,gb_per_s,valid_error,n_gpus,git_sha,run_name

Three row kinds, one schema; a metric that does not apply at that cadence is EMPTY:
  minibatch row : timestamp,epoch,step,loss,,,,n_gpus,git_sha,run_name
  epoch row     : timestamp,epoch,,,lr,gb_per_s,,n_gpus,git_sha,run_name
  validation row: timestamp,epoch,,,,,valid_error,n_gpus,git_sha,run_name
This is deliberately NOT the 19-column S2S_BENCH schema — train.py measures almost none
of those columns, and empty columns pretending to be a bench harness would be worse than
a small honest schema. gb_per_s here is train.py's own H2D-bytes/wall metric (recomputed
microseconds after the value MLflow gets — equal to ~6 significant figures), not a
step-time measurement.

Perf contract: the tee must not add a per-step GPU sync that upstream does not have.
log_minibatch accumulates the loss as a live CUDA tensor and only syncs when it string-
formats every mini_batch_log_freq=100 steps (launch.py:164-177). So minibatch() buffers
loss.clone() (async; the clone matters — under CUDA graphs the loss can be a static
buffer, and N unclone'd references would all read the LAST step's value) and float()s
the buffer only on a flush: every FLUSH_EVERY=100 rows, at every epoch()/validation(),
and best-effort at exit. A kill can therefore lose at most the last <100 minibatch rows.

Rank policy: from_env() returns None on every rank but 0. A CSV appended by 4 ranks is
a corrupted CSV.
"""
import atexit
import csv
import os
import subprocess
import time

SCHEMA = [
    "timestamp", "epoch", "step", "loss", "lr",
    "gb_per_s", "valid_error", "n_gpus", "git_sha", "run_name",
]
FLUSH_EVERY = 100  # == LaunchLogger's mini_batch_log_freq: no sync cadence upstream lacks
ENV_VAR = "PHYSICSNEMO_BENCH_CSV"


def _git_sha():
    """Short sha of the checkout this file runs from; '' if git is unavailable."""
    try:
        # stdout/stderr=PIPE (not capture_output=) so this also runs under the login
        # node's python 3.6, where the test executes.
        out = subprocess.run(
            ["git", "-C", os.path.dirname(os.path.abspath(__file__)),
             "rev-parse", "--short", "HEAD"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            universal_newlines=True, timeout=10,
        )
        return out.stdout.strip() if out.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


class BenchCSVTee:
    """Appends metric rows to one CSV file. Construct via from_env()."""

    @classmethod
    def from_env(cls, rank, n_gpus, run_name):
        """The only supported constructor path: None unless $PHYSICSNEMO_BENCH_CSV is
        set AND this is rank 0 — so call sites can be a bare `if bench_csv:`."""
        path = os.environ.get(ENV_VAR, "").strip()
        if not path or rank != 0:
            return None
        return cls(path, n_gpus=n_gpus, run_name=run_name)

    def __init__(self, path, n_gpus, run_name):
        self.path = path
        self.n_gpus = int(n_gpus)
        self.run_name = str(run_name)
        self.git_sha = _git_sha()
        self._buf = []  # (timestamp, epoch, step, loss-tensor-or-float)
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        if os.path.exists(path) and os.path.getsize(path) > 0:
            # Appending under a different schema silently corrupts every downstream
            # parse — refuse loudly instead (CLAUDE.md #10). Point the env var at a
            # fresh file; never edit the schema to match an old one.
            with open(path, newline="") as f:
                have = f.readline().rstrip("\r\n").split(",")
            if have != SCHEMA:
                print(f"ERROR PHYSICSNEMO_BENCH_CSV_SCHEMA_MISMATCH: {path}")
                print(f"  file header : {have}")
                print(f"  frozen schema: {SCHEMA}")
                raise RuntimeError(f"schema mismatch in {path}")
        else:
            with open(path, "a", newline="") as f:
                csv.writer(f).writerow(SCHEMA)
        print(f"{ENV_VAR} tee -> {path}", flush=True)
        atexit.register(self._flush_best_effort)

    # -- hooks (train.py calls these; all no-ops happen at from_env via None) --------

    def minibatch(self, epoch, step, loss):
        """Buffer one training-loss sample. No GPU sync here: clone if it is a tensor
        (async, and immune to CUDA-graph static-buffer aliasing), convert at flush."""
        clone = getattr(loss, "clone", None)
        if callable(clone):
            loss = clone()
        self._buf.append((time.time(), int(epoch), int(step), loss))
        if len(self._buf) >= FLUSH_EVERY:
            self._flush()

    def epoch(self, epoch, lr, gb_per_s):
        self._flush()
        self._write([self._row(time.time(), epoch=epoch, lr=lr, gb_per_s=gb_per_s)])

    def validation(self, epoch, valid_error):
        self._flush()
        self._write([self._row(time.time(), epoch=epoch, valid_error=valid_error)])

    # -- internals --------------------------------------------------------------------

    def _row(self, ts, epoch, step="", loss="", lr="", gb_per_s="", valid_error=""):
        iso = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(ts))
        fmt = lambda v: v if v == "" else repr(float(v))  # noqa: E731 — shortest round-trip
        return [iso, int(epoch), step, fmt(loss), fmt(lr), fmt(gb_per_s),
                fmt(valid_error), self.n_gpus, self.git_sha, self.run_name]

    def _flush(self):
        if not self._buf:
            return
        rows = [self._row(ts, epoch=ep, step=st, loss=ls) for ts, ep, st, ls in self._buf]
        self._buf = []
        self._write(rows)

    def _write(self, rows):
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerows(rows)

    def _flush_best_effort(self):
        # At interpreter shutdown CUDA may already be torn down, so float(tensor) can
        # throw — losing <100 tail rows then is acceptable; corrupting shutdown is not.
        try:
            self._flush()
        except Exception:
            pass
