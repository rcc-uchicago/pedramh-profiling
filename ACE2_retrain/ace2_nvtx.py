"""NVTX instrumentation for ACE2, injected without editing the vendored tree.

WHY THIS EXISTS
fme ships no profiling hooks at all -- no cudaProfilerApi, no torch.profiler, no
NVTX in the SFNO lat-lon path. Two consequences the profiling work kept hitting:
`midway_bench_nsys.sh` cannot use the house
`--capture-range=cudaProfilerApi` flags (nothing would be captured), and
`parse_nsys.py` produces no phase breakdown, so every attribution so far --
including "copies come from the stacker" -- is inferred from kernel names plus
code reading rather than measured.

RANGE NAMES ARE A CROSS-PROJECT CONTRACT (CLAUDE.md #10). The shared names are
reused verbatim, never renamed:

    step_{N}      one training step, the outer range
    forward_loss  the network forward + loss
    backward      gradient computation
    optimizer     the weight update
    ema           the EMA update (name borrowed from PanguWeather, same meaning)

ACE2-SPECIFIC ADDITIONS follow the precedent that SI added `preprocess` and
Pangu added `ema` -- new phases get new names, existing phases never get
renamed. These four exist to settle the copy attribution:

    stack         Stacker._stack_levels -- gathers ~43 named tensors into one
    unstack       stacker.unstack -- splits the output back into ~50 tensors
    normalize     normalizer._normalize
    denormalize   normalizer._denormalize

DELIBERATE GAP: there is no `data_prep` range. The shared contract has one, but
fme's loader wait happens in `for batch in epoch_data:` inside the trainer, and
wrapping the iterator from outside is more invasive than it is worth. In the
timeline the loader wait is the gap BETWEEN consecutive step_{N} ranges. Do not
add a differently-named range for it -- add `data_prep` properly, or leave it.

READ BEFORE USING THESE NUMBERS: an NVTX range bounds CPU time. CUDA is async,
so a range around a launch site does NOT contain the GPU time of the kernels it
launches -- `stack` measures 0.07 ms of launch cost while its kernels run later.
To attribute GPU time to a range, join CUPTI_ACTIVITY_KIND_RUNTIME to
CUPTI_ACTIVITY_KIND_KERNEL on correlationId and find the range enclosing the
LAUNCH. Note also that backward-pass kernels are launched from autograd worker
threads, so they fall outside any range pushed on the main thread.

KNOBS (per-project by convention, ACE2_* like PANGU_*/SI_*):
    ACE2_NVTX=1          master gate; no effect when unset
    ACE2_NVTX_WARMUP     steps to skip before cudaProfilerStart (default 20)
    ACE2_NVTX_STEPS      steps to capture before cudaProfilerStop (default 80)

The cudaProfilerStart/Stop pair is what makes
`nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop` work, so
the capture window is exactly the steady-state steps -- no warmup, no NCCL init,
and no need for the hand-derived --delay/--duration this project has been using.
"""

import os

import torch

ENABLED = os.environ.get("ACE2_NVTX", "0") == "1"
WARMUP = int(os.environ.get("ACE2_NVTX_WARMUP", "20"))
STEPS = int(os.environ.get("ACE2_NVTX_STEPS", "80"))

_state = {"step": 0, "profiling": False}


def _range(name):
    """Wrap a callable in an NVTX push/pop pair."""

    def decorate(fn):
        def wrapper(*args, **kwargs):
            torch.cuda.nvtx.range_push(name)
            try:
                return fn(*args, **kwargs)
            finally:
                torch.cuda.nvtx.range_pop()

        return wrapper

    return decorate


def install():
    if not ENABLED:
        return []
    applied = []

    import fme.core.normalizer as _norm
    import fme.core.optimization as _opt
    import fme.core.stacker as _stacker
    import fme.core.step.single_module as _step
    from fme.ace.stepper.single_module import TrainStepper

    # --- shared contract names -------------------------------------------
    # Map onto what fme ACTUALLY does, not onto method names. With
    # use_gradient_accumulation: false (our config) `accumulate_loss` only does
    # `_accumulated_loss += loss` -- the real backward runs inside
    # `step_weights`. Wrapping those two by name gave backward=0.12 ms and
    # optimizer=263 ms in job 53533290, i.e. `optimizer` was silently swallowing
    # the backward pass. Under CLAUDE.md #10 that is worse than no range at all:
    # the name would mean something different here than in every other project.
    _opt.Optimization._backward = _range("backward")(_opt.Optimization._backward)
    _opt.Optimization._step_weights = _range("optimizer")(
        _opt.Optimization._step_weights
    )
    _step.SingleModuleStep.step = _range("forward_loss")(_step.SingleModuleStep.step)
    applied += ["backward", "optimizer", "forward_loss"]

    # --- ACE2-specific: the copy attribution ------------------------------
    _stacker.Stacker._stack_levels = _range("stack")(_stacker.Stacker._stack_levels)
    # NO `unstack` range: fme's training path never calls stacker.unstack (grep
    # finds no callers outside its own definition). Job 53533290 recorded zero
    # events for it. The "dict -> tensor -> dict round trip" this file originally
    # set out to measure is a stack-only path in training.
    _norm._normalize = _range("normalize")(_norm._normalize)
    _norm._denormalize = _range("denormalize")(_norm._denormalize)
    applied += ["stack", "normalize", "denormalize"]

    # --- outer step_{N} + the cudaProfilerApi capture window --------------
    _orig_train_on_batch = TrainStepper.train_on_batch

    def train_on_batch(self, *args, **kwargs):
        n = _state["step"]
        if n == WARMUP and not _state["profiling"]:
            torch.cuda.cudart().cudaProfilerStart()
            _state["profiling"] = True
            print(f"ACE2_NVTX capture started at step {n}", flush=True)
        torch.cuda.nvtx.range_push(f"step_{n}")
        try:
            return _orig_train_on_batch(self, *args, **kwargs)
        finally:
            torch.cuda.nvtx.range_pop()
            _state["step"] = n + 1
            if _state["profiling"] and _state["step"] >= WARMUP + STEPS:
                torch.cuda.cudart().cudaProfilerStop()
                _state["profiling"] = False
                print(f"ACE2_NVTX capture stopped at step {_state['step']}", flush=True)

    TrainStepper.train_on_batch = train_on_batch
    applied.append("step_{N}+cudaProfilerApi")

    print(
        f"ACE2_NVTX enabled: warmup={WARMUP} steps={STEPS} ranges={applied}",
        flush=True,
    )
    return applied


if __name__ == "__main__":
    import runpy

    install()
    # argv is already (config, --override ...) as fme.ace.train expects.
    runpy.run_module("fme.ace.train", run_name="__main__")
