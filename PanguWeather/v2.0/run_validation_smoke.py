#!/usr/bin/env python
"""Validation-memory probe driver for PanguWeather train.py (additive; train.py untouched).

WHY THIS EXISTS
    Every prior PanguWeather smoke gated on PANGU_BENCH (nee S2S_BENCH), whose
    _bench_finalize() ends in dist.barrier(); sys.exit(0) UPSTREAM of the epoch loop's
    validate_one_epoch() call (train.py:969). So no smoke in this repo had ever executed
    validation — which is exactly where two risks live:
      1. a predicted first-validation OOM on the 40 GB A100s. validate_one_epoch runs
         BEFORE the first checkpoint save (train.py:969 vs :1001), so a validation OOM
         means: train ~2 h, die, requeue, repeat — zero progress, nothing checkpointed;
      2. the UNVERIFIED utils/metrics.py fix for it (the three climatology tensors are
         now held on CPU; update() gathers the [batch] slice on CPU and ships ~MBs to
         the GPU per batch instead of keeping ~8-13 GiB resident). Nothing had ever
         executed that code before this smoke.

WHAT IT DOES
    torchrun launches THIS file instead of train.py, forwarding train.py's own CLI args
    untouched. Before executing train.py we patch utils.metrics.create_metrics_aggregator_new.
    That interception is sound because:
      * train.py binds the name at exec time ("from utils.metrics import
        create_metrics_aggregator_new", train.py:53), and we patch the module attribute
        BEFORE runpy executes train.py, so train.py imports the wrapped function;
      * its only train.py call site is at the top of validate_one_epoch (train.py:2036,
        verified sole call site) — i.e. the call IS "validation started";
      * the factory receives the Trainer instance and returns the MetricsAggregator,
        whose compute() is called exactly once per validation (train.py:2473), AFTER the
        full batch loop and the metrics all_reduce — i.e. compute() IS "validation's
        GPU work finished".
    The hook therefore:
      * on create: synchronizes, records torch.cuda.max_memory_allocated() (= the peak
        over init + the training epoch), resets the peak counters, and prints
        PANGU_VAL_SMOKE_PRE; then prints PANGU_VAL_SMOKE_CLIM_DEVICE with the device of
        the climatology tensors (MUST be cpu — that is the metrics.py fix under test);
      * on compute: synchronizes, reads the validation-window peak, and prints the
        PANGU_VAL_SMOKE_MEM token with allocated + reserved peaks per rank.
    Then runpy.run_path(train.py, run_name="__main__") executes train.py byte-identically
    to `python train.py ...`: the main block runs in its own namespace, and module
    globals like world_rank resolve there, not here.

    Resetting the peak counters mid-run is safe here: the only other reader of
    torch.cuda.max_memory_allocated in this tree is _bench_finalize (train.py:1296),
    which cannot run because this driver refuses to start with any bench env set —
    the bench harness exits before validation, which would defeat the entire probe.

PASS/FAIL is decided by the submitting PBS gate
(HPC_scripts/polaris_val_e3sm_sfno_alldata_smoke.pbs). It requires one
PANGU_VAL_SMOKE_MEM token per rank; the token is only ever printed from inside
compute(), so a run that never reaches validation cannot pass.

KNOWN LIMITS (measure, don't infer, past these):
    * The validation-window peak is read at compute() time. Anything allocated on the
      GPU after compute() but inside validate_one_epoch is not attributed to the window
      — code inspection says that region is CPU plotting + O(few-float) reductions, and
      the plotting is force-disabled for sigma-level configs anyway (train.py:4281-4286).
    * A DataLoader worker started with the 'spawn' method would re-import __main__ and
      double-execute; PanguWeather's loaders fork (Linux default) and the only spawn
      user (plot_in_separate_process) is unreachable here (long_validation False,
      diagnostic_spectra force-disabled for sigma levels).
"""

import os
import runpy
import sys

# Refuse to coexist with the bench harness: BENCH=1 breaks out of the train loop and
# sys.exit(0)s before validate_one_epoch — the probe would measure nothing and a
# careless gate might mistake the silence for health. The legacy S2S_* names are also
# rejected (train.py itself now SystemExits on them, but failing here is clearer).
_BENCH_ENV = [k for k in ("PANGU_BENCH", "PANGU_BENCH_WARMUP", "PANGU_BENCH_STEPS",
                          "PANGU_BENCH_CSV", "PANGU_NVTX",
                          "S2S_BENCH", "S2S_BENCH_WARMUP", "S2S_BENCH_STEPS",
                          "S2S_BENCH_CSV", "S2S_NVTX") if os.environ.get(k)]
if _BENCH_ENV:
    print("ERROR PANGU_VAL_SMOKE_BENCH_ENV: %s set. The bench harness exits before "
          "validation runs; unset it — this probe exists precisely to reach validation."
          % ", ".join(_BENCH_ENV))
    sys.exit(2)

import torch  # noqa: E402  (after the cheap env guard on purpose)

if not torch.cuda.is_available():
    print("ERROR PANGU_VAL_SMOKE_NO_CUDA: this is a GPU-memory probe; run it on a "
          "compute node under torchrun.")
    sys.exit(3)

# Same module object train.py will bind from ("from utils.metrics import ...").
import utils.metrics as _metrics_mod  # noqa: E402

_RANK = int(os.environ.get("RANK", "0"))
_GIB = 1024 ** 3
_TRAIN_PY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train.py")

_orig_create = _metrics_mod.create_metrics_aggregator_new
_pre = {"peak": None}


def _hooked_create(trainer, *args, **kwargs):
    """Validation-start hook: snapshot the pre-validation peak, then instrument compute()."""
    torch.cuda.synchronize()
    pre_peak = torch.cuda.max_memory_allocated()
    pre_reserved = torch.cuda.max_memory_reserved()
    resident = torch.cuda.memory_allocated()
    total = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
    torch.cuda.reset_peak_memory_stats()
    _pre["peak"] = pre_peak
    print("PANGU_VAL_SMOKE_PRE rank=%d train_peak_gib=%.3f train_peak_reserved_gib=%.3f "
          "resident_at_val_start_gib=%.3f device_total_gib=%.3f"
          % (_RANK, pre_peak / _GIB, pre_reserved / _GIB, resident / _GIB, total / _GIB),
          flush=True)

    agg = _orig_create(trainer, *args, **kwargs)

    # The fix under test: utils/metrics.py now keeps these on CPU. Print, don't assert —
    # the PBS gate hard-fails on 'cuda' here so the failure is visible in one grep.
    devs = ",".join("%s=%s" % (name, getattr(agg, name).device)
                    for name in ("clim_surface", "clim_upper_air", "clim_diagnostic")
                    if getattr(agg, name, None) is not None)
    print("PANGU_VAL_SMOKE_CLIM_DEVICE rank=%d %s" % (_RANK, devs), flush=True)

    _orig_compute = agg.compute

    def _hooked_compute(*a, **k):
        results = _orig_compute(*a, **k)
        torch.cuda.synchronize()
        val_peak = torch.cuda.max_memory_allocated()
        val_reserved = torch.cuda.max_memory_reserved()
        overall = max(_pre["peak"], val_peak)
        finite = all(torch.isfinite(v).all().item() for v in results.values())
        # This token is the PASS currency: it can only be printed after this rank has
        # finished its entire validation batch loop and the metrics all_reduce.
        print("PANGU_VAL_SMOKE_MEM rank=%d val_peak_gib=%.3f val_peak_reserved_gib=%.3f "
              "train_peak_gib=%.3f overall_peak_gib=%.3f metrics_finite=%s"
              % (_RANK, val_peak / _GIB, val_reserved / _GIB,
                 _pre["peak"] / _GIB, overall / _GIB, finite),
              flush=True)
        return results

    agg.compute = _hooked_compute
    return agg


_metrics_mod.create_metrics_aggregator_new = _hooked_create

print("PANGU_VAL_SMOKE_DRIVER rank=%d hooking utils.metrics.create_metrics_aggregator_new; "
      "executing %s with args: %s" % (_RANK, _TRAIN_PY, " ".join(sys.argv[1:])), flush=True)

# Hand argv to train.py's argparse exactly as if it had been launched directly.
sys.argv = [_TRAIN_PY] + sys.argv[1:]
runpy.run_path(_TRAIN_PY, run_name="__main__")
