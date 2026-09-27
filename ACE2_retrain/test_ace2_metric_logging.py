"""Tests for the ACE2 metric-capture hooks in `ace2_telemetry.py`.

WHAT IS BEING PINNED, AND WHY IT IS WORTH A TEST
------------------------------------------------
fme hands every per-channel validation metric to `wandb.log` and nowhere else,
and `config_polaris.yaml` sets `log_to_wandb: false` -- so on Polaris that
payload was discarded on every epoch ever run. The fix has two halves, and the
half that can fail SILENTLY is key parity:

`DiskMetricLogger._extract_serializable` drops anything `json.dumps` refuses, at
DEBUG level. `trainer.py:509-512` builds its `batch_*` metrics from a bare
`dist.reduce_mean(...)` -- torch tensors -- so without `_coerce_scalars` those
keys vanish from `metrics.jsonl` with no message and no error. That is exactly
the class of defect this project keeps finding (a metric computed, routed
somewhere unreadable, and never noticed), so it gets a test rather than a
comment.

No torch import: the tensors are duck-typed fakes, which is also the reason
`_coerce_scalars` is duck-typed. Runs anywhere, including a login node.

    python3 ACE2_retrain/test_ace2_metric_logging.py     -> ACE2_METRIC_LOG_OK
"""

import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ace2_telemetry as tel  # noqa: E402

_FAILURES = []


def check(name, cond, detail=""):
    if cond:
        return
    _FAILURES.append(f"{name}: {detail}" if detail else name)


class FakeTensor:
    """Duck-types the parts of `torch.Tensor` that `_coerce_scalars` touches."""

    def __init__(self, values):
        self._values = list(values)

    def numel(self):
        return len(self._values)

    def item(self):
        if len(self._values) != 1:
            raise ValueError("only one element tensors can be converted to scalars")
        return self._values[0]


class FakeArray:
    """numpy-shaped: `.size` is an int attribute, not a method."""

    def __init__(self, values):
        self._values = list(values)
        self.size = len(self._values)

    def item(self):
        if len(self._values) != 1:
            raise ValueError("can only convert an array of size 1 to a scalar")
        return self._values[0]


class FakeImage:
    """Stands in for a wandb.Image -- genuinely not serializable, must be dropped."""


def test_coerce_unwraps_scalar_tensors():
    out = tel._coerce_scalars({"batch_loss": FakeTensor([0.25])})
    check(
        "scalar tensor -> float",
        isinstance(out["batch_loss"], float) and out["batch_loss"] == 0.25,
        repr(out["batch_loss"]),
    )


def test_coerce_unwraps_numpy_scalars():
    out = tel._coerce_scalars({"lr": FakeArray([3e-4])})
    check("numpy scalar -> float", out["lr"] == 3e-4, repr(out["lr"]))


def test_coerce_leaves_non_scalars_alone():
    """A multi-element tensor has no scalar reading; it must survive untouched so
    the drop report can NAME it rather than a coercion error hiding it."""
    multi = FakeTensor([1.0, 2.0])
    image = FakeImage()
    out = tel._coerce_scalars({"series": multi, "image-error/T": image})
    check("multi-element tensor passes through", out["series"] is multi)
    check("image passes through", out["image-error/T"] is image)


def test_coerce_preserves_plain_values_and_keys():
    payload = {"epoch": 3, "val/mean/loss": 0.19579, "name": "ace2", "none": None}
    out = tel._coerce_scalars(payload)
    check("plain values unchanged", out == payload, repr(out))
    check("no key added or lost", set(out) == set(payload))


def test_split_matches_disk_metric_logger():
    """`_split_serializable` must agree with fme's own `_extract_serializable`,
    or the "dropped" list is a guess about a file we did not inspect."""
    payload = {"a": 1.0, "b": "x", "c": True, "d": FakeImage(), "e": [1, 2]}
    kept, dropped = tel._split_serializable(payload)
    expected_kept = {}
    for k, v in payload.items():
        if isinstance(v, (bool, int, float, str)):
            expected_kept[k] = v
        else:
            try:
                json.dumps(v)
            except (TypeError, ValueError, OverflowError):
                continue
            expected_kept[k] = v
    check("kept set matches fme's rule", kept == expected_kept, repr(kept))
    check("dropped names the image", dropped == ["d"], repr(dropped))
    check("kept + dropped == input", set(kept) | set(dropped) == set(payload))


def test_tensor_metric_survives_the_round_trip():
    """THE REGRESSION. Without coercion a tensor-valued `batch_*` is dropped from
    metrics.jsonl silently; with it, the key set is preserved."""
    raw = {"epoch": 1, "batch_loss": FakeTensor([0.5])}
    _, dropped_raw = tel._split_serializable(raw)
    check("without coercion the tensor IS dropped", dropped_raw == ["batch_loss"],
          repr(dropped_raw))
    kept, dropped = tel._split_serializable(tel._coerce_scalars(raw))
    check("with coercion nothing is dropped", dropped == [], repr(dropped))
    check("value survives intact", kept["batch_loss"] == 0.5, repr(kept))


class _Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.lines = []

    def emit(self, record):
        self.lines.append(record.getMessage())


def _echo(payload, step=100):
    handler = _Capture()
    root = logging.getLogger()
    old_level = root.level
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    try:
        tel._echo_epoch_metrics(payload, step)
    finally:
        root.removeHandler(handler)
        root.setLevel(old_level)
    return handler.lines


def test_echo_is_silent_off_epoch_boundaries():
    """The per-step call at trainer.py:571 fires every log_train_every_n_batches;
    echoing it would bury the epoch summary in thousands of lines."""
    lines = _echo({"batch_loss": 0.5, "lr": 3e-4})
    check("no epoch key -> silent", lines == [], repr(lines))


def test_echo_prints_every_scalar_key_sorted():
    payload = {
        "epoch": 3,
        "val/mean/loss": 0.19579,
        "val/mean/weighted_rmse/air_temperature_0": 1.25,
        "lr": 3e-4,
    }
    lines = _echo(payload)
    body = [ln for ln in lines if ln.startswith("    ")]
    check("header present", lines[0].startswith("ACE2_EPOCH_METRICS"), repr(lines[:1]))
    check("header counts keys", "keys=4 logged=4 dropped=0" in lines[0], lines[0])
    check("one line per key", len(body) == 4, repr(body))
    keys = [ln.strip().split(":")[0] for ln in body]
    check("keys sorted", keys == sorted(keys), repr(keys))
    check("per-channel metric present",
          any("weighted_rmse/air_temperature_0" in ln for ln in body), repr(body))


def test_echo_names_what_it_dropped():
    lines = _echo({"epoch": 1, "val/mean/loss": 0.2, "image-error/T": FakeImage()})
    check("header counts the drop", "keys=3 logged=2 dropped=1" in lines[0], lines[0])
    tail = [ln for ln in lines if "NOT LOGGED" in ln]
    check("drop line present", len(tail) == 1, repr(lines))
    check("drop line names the key", "image-error/T" in tail[0], tail[0])


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        try:
            t()
        except Exception as exc:  # noqa: BLE001
            _FAILURES.append(f"{t.__name__} raised {type(exc).__name__}: {exc}")
    if _FAILURES:
        for f in _FAILURES:
            print(f"ERROR {f}")
        print(f"ERROR ACE2_METRIC_LOG_FAILED {len(_FAILURES)} of {len(tests)}")
        return 1
    print(f"ACE2_METRIC_LOG_OK {len(tests)} tests")
    return 0


if __name__ == "__main__":
    sys.exit(main())
