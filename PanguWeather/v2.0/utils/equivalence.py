"""DESIGN §4.1 equivalence-baseline recorder for PanguWeather.

Plan item 18. Emits the JSON record that
`physicsnemo_ai_rossby/polaris/compare_baselines.py` already knows how to compare, so
only the *capture* side is new — the comparison logic is reused unchanged.

Why this lives here rather than in a standalone harness: a harness that re-implements
the training step would measure something subtly different from what `train.py`
computes, and `compare_baselines.py`'s own header warns that comparing two captures
which "measure different things" produces "a number that looks valid and means
nothing". The real step loop already computes `train_batch_loss`, `batch_grad_norm`
and `batch_grad_max` per step; this only records them.

torch is imported lazily so the record/validation logic is testable on a login node,
where importing torch can hang or core-dump (CLAUDE.md #3).
"""
import hashlib
import json
import math
import os

# Exactly `compare_baselines.MUST_MATCH`. If these disagree between two records the
# comparison aborts rather than producing a meaningless number — so a record missing
# any of them is not a baseline, it is a trap.
REQUIRED_CONFIG = ("seed", "steps", "world_size", "n_params", "batch_size",
                   "amp_dtype", "config_yaml_sha256", "mode")


def config_sha256(path):
    """Hash the *rendered* config — two runs are only comparable at identical config."""
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def config_sha256_from_env(var="S2S_YAML"):
    """Hash the rendered config named by `$var`, or raise.

    Deliberately fatal rather than defaulting to `"unknown"`: `config_yaml_sha256` is a
    `MUST_MATCH` field, so two records both carrying `"unknown"` would compare *clean*
    across two genuinely different configs — the exact apples-to-oranges comparison the
    field exists to prevent.
    """
    path = os.environ.get(var)
    if not path or not os.path.exists(path):
        raise ValueError("ERROR EQUIV_NO_CONFIG $%s unset or missing (%r) — two records "
                         "are only comparable at identical config, so a baseline "
                         "without a config hash is not a baseline" % (var, path))
    return config_sha256(path)


def effective_seed():
    """The seed actually applied, recovered from `PYTHONHASHSEED`.

    `train.py`'s `seed_torch()` sets `os.environ['PYTHONHASHSEED'] = str(seed)` as its
    first act (train.py:3886), and the effective per-rank seed
    (`global_seed * world_size + rank`) never reaches `params` — so this is the only
    in-process record of what was actually seeded.
    """
    raw = os.environ.get("PYTHONHASHSEED")
    if raw is None or not raw.lstrip("-").isdigit():
        raise ValueError("ERROR EQUIV_NO_SEED PYTHONHASHSEED=%r — seed_torch() sets it; "
                         "if it is unset the run was not deterministically seeded" % raw)
    return int(raw)


def tensor_stats(t):
    """{shape, mean, std, min, max} — summary only, never tensors (DESIGN §7).

    Duck-typed rather than importing torch: this module must stay importable on a
    login node, where importing torch can hang or core-dump (CLAUDE.md #3). `.detach()`
    already severs autograd, so `torch.no_grad()` would add nothing here.
    """
    f = t.detach().float()
    return {"shape": list(t.shape),
            "mean": f.mean().item(),
            "std": f.std().item() if f.numel() > 1 else 0.0,
            "min": f.min().item(),
            "max": f.max().item()}


def validate_record(rec):
    """Raise ValueError unless `rec` is actually comparable. Called before writing.

    The subtle one is the **stable key set**. `compare_baselines` iterates
    `sorted(set(baseline_step) & set(candidate_step))` — the INTERSECTION — so a record
    that silently drops a metric still compares clean on whatever remains and prints
    PASS. A baseline that can pass by omission is worse than no baseline.
    """
    missing = [k for k in REQUIRED_CONFIG if rec.get(k) is None]
    if missing:
        raise ValueError("ERROR EQUIV_RECORD_INCOMPLETE missing=%s" % ",".join(missing))
    traj = rec.get("loss_trajectory")
    if not traj:
        raise ValueError("ERROR EQUIV_RECORD_EMPTY_TRAJECTORY")
    if len(traj) != rec["steps"]:
        raise ValueError("ERROR EQUIV_TRAJECTORY_LENGTH %d != steps %d"
                         % (len(traj), rec["steps"]))
    keys = set(traj[0])
    if not keys:
        raise ValueError("ERROR EQUIV_STEP_HAS_NO_METRICS")
    for i, step in enumerate(traj):
        if set(step) != keys:
            raise ValueError("ERROR EQUIV_UNSTABLE_KEYS step %d has %s, step 0 has %s "
                             "— compare_baselines intersects key sets, so a dropped "
                             "metric would silently pass"
                             % (i, sorted(set(step)), sorted(keys)))
        for k, v in step.items():
            if not isinstance(v, float) or not math.isfinite(v):
                raise ValueError("ERROR EQUIV_NON_FINITE step %d %s=%r" % (i, k, v))
    for grp, st in (rec.get("forward_output_stats") or {}).items():
        for k in ("shape", "mean", "std", "min", "max"):
            if k not in st:
                raise ValueError("ERROR EQUIV_STATS_INCOMPLETE %s missing %s" % (grp, k))
    return rec


class EquivalenceRecorder:
    """Off unless `PANGU_EQUIV_JSON` names an output path.

    Deliberately inert by default: this rides inside the real training step, and an
    always-on recorder is instrumentation drift waiting to happen (CLAUDE.md #10).
    """

    def __init__(self, path=None):
        self.path = path or os.environ.get("PANGU_EQUIV_JSON") or None
        self.enabled = bool(self.path)
        self.trajectory = []
        self.output_stats = {}

    def record_step(self, **metrics):
        if not self.enabled:
            return
        self.trajectory.append({k: float(v) for k, v in sorted(metrics.items())})

    def record_output(self, group, tensor):
        if not self.enabled:
            return
        self.output_stats[group] = tensor_stats(tensor)

    def finalize(self, **config):
        """Validate then write. Returns the path, or None when disabled."""
        if not self.enabled:
            return None
        rec = dict(config)
        rec["steps"] = len(self.trajectory)
        rec["loss_trajectory"] = self.trajectory
        rec["forward_output_stats"] = self.output_stats
        validate_record(rec)
        d = os.path.dirname(self.path)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(self.path, "w") as fh:
            fh.write(json.dumps(rec, indent=2) + "\n")
        return self.path
