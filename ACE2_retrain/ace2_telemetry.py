"""Per-epoch step telemetry for ACE2, injected without editing the vendored tree.

WHY THIS EXISTS
---------------
`PROFILING_PLAN.md` §5 records that "ACE2 still has no bench CSV and no
equivalence baseline, unlike S2S/SI/PanguWeather".  Every other harness in this
repo emits one `epoch_telemetry.csv` row per epoch; ACE2 emitted nothing, so a
Polaris scaling row could only have been read off wall-clock -- which cannot
separate a comms cost from a loader cost, and the loader is the leading suspect
here (one 2.4 TB NetCDF, `lfs getstripe` = 1 OST, read by every rank).

`gpu_busy_frac` is the number this file exists to produce.

HOW IT ATTACHES
---------------
Same shape as `ace2_nvtx.py`: monkeypatch on import, then `runpy` the real
entrypoint.  Nothing under `ace_exp/` is edited.

    python ace2_telemetry.py <config.yaml> --override key=value ...

is a drop-in replacement for `python -m fme.ace.train <config.yaml> ...`.

WHERE THE STEP WINDOW OPENS AND CLOSES -- read before comparing to another harness
----------------------------------------------------------------------------------
The window is a CROSS-PROJECT CONTRACT (CLAUDE.md #10), not a local choice.
ai-rossby closes its window after `ema.update`; PanguWeather closes it after
`scheduler.step()` and before its per-iteration diagnostics.  fme's training loop
(`fme/core/generics/trainer.py:546-556`) is::

    for batch in epoch_data:
        stepped = self.stepper.train_on_batch(batch, self.optimization)   # <- open
        self._end_of_batch_callback()
        self._ema(model=self.stepper.modules)
        self.optimization.step_scheduler(is_iteration=True)               # <- close
        ...  bookkeeping, aggregation, wandb

so opening at `train_on_batch` and closing at the per-iteration `step_scheduler`
spans *forward + backward + optimizer + EMA + scheduler* -- the same work as
ai-rossby's window, and it excludes the metric aggregation the other two also
exclude.  On a 456 M-parameter model the EMA sweep is real work (~2.7 GB of
elementwise traffic per step); leaving it outside the window would book it as
"loader idle" and understate `gpu_busy_frac`.

THREE DISCRIMINATIONS THAT MATTER
---------------------------------
1. `train_on_batch` is called from THREE places, only one of which is a training
   step: the loop above, `_log_first_batch_metrics` (trainer.py:502), and the
   post-epoch train-evaluation pass (trainer.py:591).  The latter two pass
   `NullOptimization`.  Timing them would mix no-grad forward-only batches into
   the step distribution.  Hence the `isinstance(..., NullOptimization)` test --
   not a call counter, which would break the moment the config changes
   `train_evaluation_samples`.
2. `epoch_end` fires from `GriddedData.alternate_shuffle` (trainer.py:583), the
   single production call site, which sits between the training loop and the
   train-evaluation pass.  Closing the epoch at the END of `train_one_epoch`
   instead would put that evaluation pass inside `epoch_wall_s` and deflate
   `gpu_busy_frac` by a config-dependent amount -- i.e. an ACE2 row would stop
   meaning what an ai-rossby row means while still looking comparable.
3. `epoch_end` performs an all-reduce (peak memory across ranks), so it must run
   on every rank.  `alternate_shuffle` does.

KNOBS (per-project by convention, ACE2_* like PANGU_*/SI_*)
-----------------------------------------------------------
    ACE2_EPOCH_TELEMETRY=1            master gate; unset => training path is
                                      byte-identical and nothing is written
    ACE2_EPOCH_TELEMETRY_CSV=<path>   default: epoch_telemetry.csv in cwd
    ACE2_EPOCH_TELEMETRY_EPOCHS=<n>   record only the first n epochs (0 = all)
    ACE2_RUN_NAME=<name>              row identity; the parser selects on it
    ACE2_N_LOADERS=<n>                recorded, not applied -- the launcher owns
                                      the value and fme does not expose it on the
                                      objects reachable from the trainer
    ACE2_MEM_LOG=1                    additionally print one ACE2_MEM line per
                                      epoch with the PER-EPOCH peak (reset each
                                      epoch), for the batch-size search
    ACE2_FR_DUMP=<path prefix>        write this rank's NCCL flight-recorder
                                      buffer to <prefix><rank>.pickle at epoch
                                      end. ⚠ Needed because the recorder dumps
                                      only ON TIMEOUT by default -- a run that
                                      SUCCEEDS produces no artifact, and the
                                      per-collective sizes are exactly what
                                      settles whether ACE2 is exposed to the
                                      Polaris tree-corruption defect. Read with
                                      polaris/read_nccl_trace.py.

COST
----
Two `cudaEvent`s per step and one all-reduce per epoch, no `torch.cuda.synchronize`
in the step path -- see `epoch_telemetry.py`, which is a verbatim copy of the
PanguWeather/ai-rossby module and is drift-guarded by `epoch_telemetry_test.py`.
"""

import os
import sys

_ENABLED = os.environ.get("ACE2_EPOCH_TELEMETRY") == "1"
_MEM_LOG = os.environ.get("ACE2_MEM_LOG") == "1"
_FR_DUMP = os.environ.get("ACE2_FR_DUMP", "")


def _dump_flight_recorder(rank: int) -> None:
    """Write this rank's NCCL flight-recorder buffer to <prefix><rank>.pickle.

    ⚠ This exists because `TORCH_NCCL_DUMP_ON_TIMEOUT=1` dumps only when the
    watchdog fires. A run that SUCCEEDS leaves nothing on disk -- and the
    per-collective sizes in that buffer are what settle whether ACE2's largest
    all-reduce lands in the 25 MiB..1000 MiB gap where a Polaris TREE all-reduce
    silently returns partially reduced data.

    torch has moved this symbol around (and renamed the buffer-size env var)
    between 2.8 and 2.10, so both spellings are tried and the one that worked is
    printed. A failure here must never take the run down: it is a diagnostic.
    """
    import torch

    c10d = torch._C._distributed_c10d
    for name in ("_dump_nccl_trace", "_dump_nccl_trace_json"):
        fn = getattr(c10d, name, None)
        if fn is None:
            continue
        try:
            blob = fn()
        except Exception as exc:  # noqa: BLE001
            print(f"ACE2_FR_DUMP {name} failed: {type(exc).__name__}: {exc}", flush=True)
            continue
        path = f"{_FR_DUMP}{rank}.pickle"
        mode, payload = ("wb", blob) if isinstance(blob, bytes) else ("w", str(blob))
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, mode) as fh:
            fh.write(payload)
        print("ACE2_FR_DUMP rank=%d via=%s bytes=%d -> %s"
              % (rank, name, len(payload), path), flush=True)
        return
    print("ACE2_FR_DUMP UNAVAILABLE: no _dump_nccl_trace* on this torch build "
          "(%s). Confirm the symbol name with help() on a compute node."
          % torch.__version__, flush=True)

# Module-local state.  `tel` is built lazily: EpochTelemetry needs the world size
# and the local batch size, neither of which exists until fme has initialized
# distributed and built the loaders.  `pending_epoch` carries the epoch number
# from `train_one_epoch` to the `subset_loader` call that actually opens the
# timed window -- see the epoch_start hook for why the two are not the same place.
_state = {"tel": None, "trainer": None, "banner": False, "pending_epoch": None}


def _banner(trainer) -> None:
    """One line per RANK, printed by the trainer process, once.

    This is what the scaling parser reads `world_size` and `steps_per_epoch` off.
    It is deliberately a `print`, not `logging.info`: fme routes logging to
    rank 0 only, and a banner that only rank 0 emits cannot support the
    `ranks_reporting` guard (a rank that dies before training leaves world_size
    correct and ranks_reporting wrong -- that is the whole point of having two).

    fme itself prints no world-size line anywhere (`grep` over ace_exp/fme finds
    only "DONE ---- rank N"), so there is no upstream banner to prefer.
    """
    if _state["banner"]:
        return
    import torch

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        world = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
    else:
        world, rank = 1, 0
    local_batch = int(trainer.train_data.batch_size)
    try:
        steps_per_epoch = len(trainer.train_data.loader)
    except TypeError:  # a sampler without __len__ -- report it rather than guess
        steps_per_epoch = -1
    print(
        "ACE2_BANNER steps_per_epoch=%d world_size=%d rank=%d local_batch=%d "
        "global_batch=%d device=cuda:%s torch=%s"
        % (
            steps_per_epoch,
            world,
            rank,
            local_batch,
            local_batch * world,
            os.environ.get("LOCAL_RANK", "?"),
            torch.__version__,
        ),
        flush=True,
    )
    _state["banner"] = True


def _get_tel(trainer):
    if _state["tel"] is not None:
        return _state["tel"]

    import torch

    from epoch_telemetry import EpochTelemetry

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        world = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
    else:
        world, rank = 1, 0

    # bf16 whenever AMP is on: Optimization.autocast pins dtype=torch.bfloat16
    # (fme/core/optimization.py) rather than taking it from the config.
    amp = "bf16" if getattr(trainer.optimization, "gscaler", None) is not None else "fp32"

    _state["tel"] = EpochTelemetry(
        prefix="ACE2",
        harness="ace2",
        run_name=os.environ.get("ACE2_RUN_NAME", "ace2"),
        n_gpus=world,
        batch_per_gpu=int(trainer.train_data.batch_size),
        amp_dtype=amp,
        n_loaders=os.environ.get("ACE2_N_LOADERS", "?"),
        rank=rank,
        repo_dir=os.path.dirname(os.path.abspath(__file__)),
    )
    return _state["tel"]


def install():
    """Patch fme in place.  Returns the list of hooks applied (empty when off)."""
    if not _ENABLED:
        return []

    import torch

    from fme.ace.data_loading.gridded_data import GriddedData
    from fme.ace.stepper.single_module import TrainStepper
    from fme.core.generics.trainer import Trainer
    from fme.core.optimization import NullOptimization, Optimization

    applied = []

    # --- epoch open --------------------------------------------------------
    # ⚠ NOT at the top of `train_one_epoch`, and this is the whole reason the
    # hook is split in two.  On the first epoch of a run fme calls
    # `_log_first_batch_metrics()` (trainer.py:526-528) BEFORE the training loop:
    # it does `next(iter(self.train_data.loader))`, i.e. it spins up dataloader
    # workers and pulls one batch off a 2.4 TB NetCDF, then runs a forward pass.
    # That is tens of seconds of one-off cost with no timed steps in it, and a
    # 60-step arm is ~30 s of steps -- so starting the wall clock above it could
    # roughly HALVE `gpu_busy_frac`, the one number this harness exists to
    # produce.  `subset_loader` (trainer.py:535) is the next thing to run, so the
    # window opens there instead: after the probe, before the training loop's own
    # first batch fetch.  That first fetch stays INSIDE the window, which is what
    # makes the figure comparable to ai-rossby's (whose epoch_start also sits
    # ahead of its first batch).
    _orig_train_one_epoch = Trainer.train_one_epoch

    def train_one_epoch(self):
        _banner(self)
        _get_tel(self)
        _state["trainer"] = self
        _state["pending_epoch"] = self._epochs_trained + 1
        if _MEM_LOG and torch.cuda.is_available():
            # PER-EPOCH peak, unlike the CSV's run-to-date column. §1f of the
            # handoff: makani's memory model was fitted twice and refuted twice,
            # so ACE2's largest local batch is found by MEASURING one arm per
            # value. That needs a per-epoch number, and the CSV column
            # deliberately does not reset (its accumulate semantic is contracted).
            torch.cuda.reset_peak_memory_stats()
        try:
            return _orig_train_one_epoch(self)
        finally:
            _state["pending_epoch"] = None

    Trainer.train_one_epoch = train_one_epoch

    _orig_subset_loader = GriddedData.subset_loader

    def subset_loader(self, *args, **kwargs):
        # Fires on the FIRST subset_loader of the epoch only. fme calls it a
        # second time at trainer.py:587 to build the train-evaluation pass, by
        # which point epoch_end has already run; re-opening the epoch there would
        # leave a window that nothing ever closes.
        pending = _state["pending_epoch"]
        tel = _state["tel"]
        if pending is not None and tel is not None:
            _state["pending_epoch"] = None
            tel.epoch_start(pending)
        return _orig_subset_loader(self, *args, **kwargs)

    GriddedData.subset_loader = subset_loader
    applied.append("epoch_start")

    # --- epoch close -------------------------------------------------------
    _orig_alternate_shuffle = GriddedData.alternate_shuffle

    def alternate_shuffle(self):
        trainer = _state["trainer"]
        tel = _state["tel"]
        if tel is not None and trainer is not None:
            lr = float("nan")
            try:
                lr = trainer.optimization.learning_rate
            except AttributeError:
                pass
            # fme applies EMA on every step from epoch 1 (trainer.py:550), with
            # no warmup gate -- unlike PanguWeather, which skips the sweep until
            # `ema_warmup_epochs`. Recording 1 here is a fact about fme, and it
            # is precisely the difference the column exists to make visible.
            tel.epoch_end(lr=lr, ema_active=trainer._ema is not None)
            if _FR_DUMP:
                # After epoch_end, so the buffer contains a full epoch of steady
                # -state collectives rather than only DDP's setup broadcast.
                _dump_flight_recorder(tel.rank)
            if _MEM_LOG and torch.cuda.is_available():
                free_b, total_b = torch.cuda.mem_get_info()
                print(
                    "ACE2_MEM epoch=%s alloc_peak_gib=%.3f reserved_peak_gib=%.3f "
                    "device_total_gib=%.2f alloc_retries=%d"
                    % (
                        trainer._epochs_trained + 1,
                        torch.cuda.max_memory_allocated() / 1024**3,
                        torch.cuda.max_memory_reserved() / 1024**3,
                        total_b / 1024**3,
                        torch.cuda.memory_stats().get("num_alloc_retries", 0),
                    ),
                    flush=True,
                )
        return _orig_alternate_shuffle(self)

    GriddedData.alternate_shuffle = alternate_shuffle
    applied.append("epoch_end")

    # --- step open ---------------------------------------------------------
    _orig_train_on_batch = TrainStepper.train_on_batch

    def train_on_batch(self, batch, optimization, *args, **kwargs):
        tel = _state["tel"]
        if tel is not None and not isinstance(optimization, NullOptimization):
            tel.step_start()
        return _orig_train_on_batch(self, batch, optimization, *args, **kwargs)

    TrainStepper.train_on_batch = train_on_batch
    applied.append("step_start")

    # --- step close --------------------------------------------------------
    _orig_step_scheduler = Optimization.step_scheduler

    def step_scheduler(self, valid_loss=None, is_iteration=False):
        out = _orig_step_scheduler(self, valid_loss=valid_loss, is_iteration=is_iteration)
        tel = _state["tel"]
        if tel is not None and is_iteration:
            tel.step_end()
        return out

    Optimization.step_scheduler = step_scheduler
    applied.append("step_end")

    print("ACE2_TELEMETRY enabled: hooks=%s csv=%s"
          % (applied, os.environ.get("ACE2_EPOCH_TELEMETRY_CSV", "epoch_telemetry.csv")),
          flush=True)
    return applied


if __name__ == "__main__":
    import runpy

    # `epoch_telemetry.py` sits beside this file, not on the venv's path.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    install()
    # argv is already (config, --override ...) as fme.ace.train expects.
    runpy.run_module("fme.ace.train", run_name="__main__")
