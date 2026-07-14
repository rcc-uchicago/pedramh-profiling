# S2S to PyTorch Lightning Port -- Port Record (Phases 0--6, PORT_COMPLETE)

> Status: PORT_COMPLETE on branch ``lightning-port`` (green floor ``9261fc5``).
> Sections 1--6 are the Phase-5 reconciliation audit; Sections 7--10 (appended at
> PORT_COMPLETE) add the Phase-6 docstring/reference-integrity pass, the
> consolidated per-phase smoke-id table, launch instructions, and the final
> certification.

This document is the Phase-5 reconciliation of the S2S to PyTorch Lightning port
against the sibling SI template. It is an **evidence-based audit**, not a
refactor: Phases 0--4 already built every Lightning component to mirror SI's
shape, and this phase verifies, classifies, and documents the residual
differences. The audit was performed by reading the ported files against the
SI template files and recording exact ``file:line`` evidence.

The reconciliation goal of the port is: *one shared codebase whose only material
difference is the model definition* -- S2S's ``PanguModel_Plasim`` versus SI's
DiT-family models. This document's conclusion is that the goal is met, modulo an
enumerated set of necessary-S2S infrastructure differences that cannot converge
without rewriting S2S's science, config, or data layers (out of scope under the
port's "reuse, do not rewrite" mandate).

## Reference paths

* S2S repo root (ported tree): ``/project/pedramh/shared/S2S``
* SI template root: ``/project/pedramh/shared/anthonyz`` (referred to below as
  ``$SI_DIR``)
* HDF5 dataset (both partitions, no staging step):
  ``/project/pedramh/h5data/h5data``
* Lightning-port env (LPORT_ENV): ``/project/pedramh/shared/S2S/v2.0/venv``
  (torch 2.6.0+cu124, lightning 2.5.0.post0, Python 3.11.11)
* ``PYTHONPATH`` for every ported script: ``v2.0:.`` -- ``v2.0/`` resolves
  ``from utils...`` / ``from networks...``; the repo root resolves the ported
  ``data`` / ``modules`` / ``common`` packages.

## 1. Per-phase summary (as landed)

The port restructured S2S in place (option 2: recreate SI's layout inside the
S2S repo, deleting/moving nothing under ``v2.0/``). Each phase smoke-tested on a
real GPU before committing on branch ``lightning-port``.

| Phase | Commit | What landed | SI counterpart (as landed) |
|---|---|---|---|
| 0 | (parent ``2a7ad1c``) | Branch + scaffold: created ``modules/`` (``models/``, ``layers/``), ``configs/``, ``common/`` (placeholder ``README.md`` each; no ``__init__.py``, mirroring SI's PEP-420 namespace packages). No deletions under ``v2.0/``. | Mirrors ``$SI_DIR`` package layout (``modules/`` ``data/`` ``configs/`` ``common/``). |
| 1 | ``ab021a1`` | ``data/datamodule.py::ClimateDataModule(L.LightningDataModule)`` wrapping S2S's HDF5 loaders. ``.gitignore`` carve-out (``data/*`` + ``!data/*.py``) so DataModule sources track while data assets stay ignored. | ``$SI_DIR/data/datamodule.py::ClimateDataModule`` (same class name + public API). |
| 2 | (Phase-2 commits incl. ``4354351``) | ``modules/train_module.py::TrainModule(L.LightningModule)`` wrapping ``PanguModel_Plasim`` + the ``utils.losses`` losses; ports ``cal_loss`` into ``training_step`` (7-tuple VAE forward, CRPS+KL), reproduces the dead-module freeze, adds ``configure_optimizers`` / ``validation_step``. | ``$SI_DIR/modules/train_module.py::TrainModule`` (same class name + public method API). |
| 3 | ``1ce0c35``, ``a2b98c4``, ``c812adc`` | Root ``train.py`` + ``bench.py``; ``common/bench_callback.py::BenchCallback`` + ``common/set_epoch_callback.py::SetEpochCallback``; ``configs/test_midway.yaml``. Gauntlet fix ``a2b98c4`` deferred DataLoader/sampler construction to ``ClimateDataModule.setup`` (group up per-rank inside ``fit``). | ``$SI_DIR/{train.py,bench.py,common/bench_callback.py}``. (SI has no ``SetEpochCallback`` -- see classification B.) |
| 4 | ``8e97798``, ``540e680`` | Root ``val.py`` (single-device ``trainer.validate``) + the netCDF inference path in ``TrainModule`` (``predict`` / ``save_predictions`` / ``predict_step``); ``ClimateDataModule.predict_dataloader`` routes ``get_infer_data``. | ``$SI_DIR/val.py`` (same single-device ``devices=1`` / ``strategy="auto"`` shape). |

The per-phase smoke job ids are recorded in the agent's project memory and in the
commit messages; the consolidated smoke-id table is the PORT_COMPLETE step, not
this audit. The Phase-5 certification smoke id is cited in Section 6.

## 2. Component-to-SI mapping (as landed)

Every ported component has a same-named SI counterpart. The ported file
docstrings each open with a "mirrors the SI template at ``$SI_DIR/...``"
sentence, so the mapping is self-documenting in the source.

| Ported file | SI counterpart | Shape mirrored |
|---|---|---|
| ``train.py`` | ``$SI_DIR/train.py`` | argparse -> config -> callbacks (two ``ModelCheckpoint`` + ``LearningRateMonitor`` + EMA) -> ``L.Trainer`` -> ``trainer.fit(model, datamodule)``. |
| ``bench.py`` | ``$SI_DIR/bench.py`` | bench overrides + explicit ``DDPStrategy`` (bucket / bf16-compress) + ``BenchCallback`` + ``torch.compile(model.model)`` -> ``trainer.fit``. |
| ``val.py`` | ``$SI_DIR/val.py`` | force ``devices=1`` / ``strategy="auto"`` -> ``trainer.validate(model, datamodule)``. |
| ``data/datamodule.py`` | ``$SI_DIR/data/datamodule.py`` | ``ClimateDataModule(L.LightningDataModule)``: ``__init__`` builds the dataset (normalizer); ``train_dataloader`` / ``val_dataloader`` / ``test_dataloader`` / ``predict_dataloader``. |
| ``modules/train_module.py`` | ``$SI_DIR/modules/train_module.py`` | ``TrainModule(L.LightningModule)``: ``self.model``, ``self.ddp``, ``training_step`` / ``validation_step`` / ``predict`` / ``save_predictions`` / ``configure_optimizers`` + ``self.save_hyperparameters()``. |
| ``common/bench_callback.py`` | ``$SI_DIR/common/bench_callback.py`` | ``BenchCallback(L.Callback)`` + ``BENCH_WARMUP`` / ``BENCH_STEPS``; sync brackets in ``on_train_batch_start`` / ``_end``; NVTX ``step_N`` / ``backward`` / ``optimizer``; ``cudaProfilerStart/Stop``; rank-0 CSV row. |
| ``common/set_epoch_callback.py`` | (no SI counterpart) | Necessary-S2S; see classification B. |

## 3. Divergence classification

Every ported-vs-SI difference falls into exactly two buckets: (A) the model
definition and its direct consequences, and (B) necessary-S2S science/config/data
infrastructure. There is no residual (C) clean-convergence edit (Section 5
records the candidates that were considered and rejected, with the reason each
would fail review).

### (A) MODEL -- the intended single material difference

These differences all flow from holding ``PanguModel_Plasim`` as ``self.model``
instead of SI's DiT-family models. They are the reason the port exists and are
*intended*.

* **The architecture.** ``networks.pangu.PanguModel_Plasim`` (held as
  ``self.model`` in ``modules/train_module.py`` -- import at line 59, build in
  ``_get_model`` at line 377) versus SI's ``DiT`` / ``SI_DiT`` / ``SI_X`` /
  ``FM`` models (``$SI_DIR/modules/train_module.py`` lines 68--88, branched on
  ``model_name`` with a ``self.scheduler`` from a diffusion interpolant).
* **The 7-tuple VAE forward.** S2S's model returns
  ``(output_surface, output_upper_air, output_diagnostic, mu, sigma, mu2,
  sigma2)``, consumed positionally in ``TrainModule.training_step``
  (``modules/train_module.py`` lines 688--694, total-loss assembly documented at
  lines 639--645). SI's model returns a single output tensor
  (``$SI_DIR/modules/train_module.py`` ``forward`` at line 109,
  ``training_step`` at line 123).
* **The losses.** S2S combines latitude-weighted CRPS surface/upper-air/
  diagnostic losses with a VAE KL term
  (``utils.losses.Latitude_weighted_CRPSLoss`` +
  ``utils.losses.Kl_divergence_gaussians``, instantiated in
  ``modules/train_module.py::_setup_loss_fun`` at line 391; KL applied at line
  717). SI uses flow-matching losses + a spectral loss
  (``train/spectral_loss`` logged at ``$SI_DIR/modules/train_module.py`` line
  174) and an RMSE metric suite in ``log_losses`` (``$SI_DIR/.../train_module.py``
  lines 382--419: ``val/t2m_*`` / ``val/pr_6h_*`` / ``val/z500_*`` / ``val/u250_*``
  / ``val/t850_*`` / ``val/q850_*``) that has no S2S counterpart.
* **Ensemble CRPS tiling.** ``to_ensemble_batch`` (module-level function at
  ``modules/train_module.py`` lines 76--102; applied in input prep at lines 558
  and 631) tiles each sample ``num_ensemble_members`` times so the CRPS loss can
  score an ensemble per sample. SI has no ensemble tiling (its losses are
  per-sample deterministic/flow-matching).
* **Batch arity.** S2S unpacks the 3-/6-tuple batch produced by
  ``GetDataset`` (consumed in ``training_step`` input prep,
  ``modules/train_module.py`` around line 678 ``pre_ens_b = batch_rows //
  self.num_ensemble_members``). SI unpacks an 8-tuple that carries a calendar
  tensor (``$SI_DIR/modules/train_module.py`` line 126:
  ``surface_t, upper_air_t, diagnostic_t, surface_t1, upper_air_t1,
  diagnostic_t1, varying_boundary_data, calendar``).
* **No calendar forcing / no ``_ModelWithScalar``.** SI routes per-step
  calendar info through a ``_ModelWithScalar`` wrapper
  (``$SI_DIR/modules/train_module.py`` lines 14--26, ``c_scalar`` rebinding in
  ``training_step`` lines 147--167). PanguModel_Plasim takes its forcing as
  explicit ``constant_boundary`` / ``varying_boundary`` tensors, so the port has
  no such wrapper.
* **netCDF prediction output.** The port writes per-sample netCDF in
  ``TrainModule.save_predictions`` (``modules/train_module.py`` line 1071, using
  the ``cf_xarray`` accessor imported at line 52) -- a port of
  ``v2.0/inference.py::Stepper.save_prediction``. SI's ``save_predictions``
  writes ``.pt`` dicts (``$SI_DIR/modules/train_module.py`` line 193).
* **Model-internal NVTX ranges.** ``networks/pangu.py`` carries its own
  ``_NVTX``-gated ranges ``vae_encoder1`` / ``vae_encoder2`` /
  ``vae_encoder2_bwd`` (``v2.0/networks/pangu.py`` line 74 gate; ranges at lines
  534, 543, 90/101). These live inside the model and travel with it; SI's DiT
  has no equivalent.

SI's ``modules/ae_module.py`` (standalone autoencoder/downscaler trainer),
``modules/combined_module.py`` (evaluation-only combined module),
``common/loss.py`` (``latitude_weighted_rmse`` + the DiT spectral/flow losses),
and ``common/plotting.py`` (``plot_result`` / ``plot_spectrum``) are all
model-specific to SI's DiT pipeline and have **no S2S counterpart**, so they
are correctly absent from the port rather than a missed convergence. (See
Section 5 for why pulling them in would be dead code.)

### (B) NECESSARY-S2S -- infrastructure that cannot converge without a rewrite

These differences are not the model, but they cannot be made identical to SI
without rewriting S2S's config system, data backend, or DDP/instrumentation
contracts -- all explicitly out of scope under "reuse, do not rewrite". Each is
load-bearing for S2S's correctness or its benchmark instrumentation.

* **YParams sectioned config.** The port feeds S2S's flat, attribute-style
  ``utils.YParams.YParams`` (loaded from a sectioned YAML) to both modules,
  rather than SI's nested ``model:`` / ``data:`` / ``training:`` dict from
  ``common.utils.get_yaml`` (``yaml.safe_load``). Evidence: ``train.py`` import
  at line 60 (``from utils.YParams import YParams``), load at line 255
  (``params = YParams(cfg, args.config)``), and the flat-key resolution in
  ``process_args`` at lines 205--244 (e.g.
  ``params["max_epochs"] if "max_epochs" in params else 1``) versus SI's
  ``config['training']`` / ``config['data']`` / ``config['model']`` dict access
  (``$SI_DIR/train.py`` ``process_args`` lines 46--64). Converging would mean
  rewriting every S2S config and the loader that reads them. The port keeps the
  flat config and mirrors only the *flow* (argparse -> config -> Trainer).
* **Lightning-2.5 EMA ``WeightAveraging`` import guard.** SI imports
  ``WeightAveraging`` unconditionally (``$SI_DIR/train.py`` line 16,
  ``EMAWeightAveraging`` at lines 21--27) because its env is newer Lightning.
  The LPORT_ENV is Lightning 2.5.0.post0, which lacks ``WeightAveraging``, so the
  port guards the import and only defines ``EMAWeightAveraging`` when the base
  class is present (``train.py`` lines 72--137; ``_build_ema_callback`` raises a
  clear error if EMA is requested without the base class). This is an
  env-version difference; convergence is the Phase-5 env-unification concern, not
  a code defect. EMA is off by default in ``configs/test_midway.yaml``.
* **S2S owns its DistributedSampler (``use_distributed_sampler=False`` +
  ``SetEpochCallback``).** S2S's
  ``utils.data_loader_multifiles.get_data_loader`` builds its own sampler
  (reproduced per-rank in ``data/datamodule.py::setup`` at lines 206--209:
  ``DistributedSampler`` under a live process group, else ``RandomSampler``), so
  the Trainer is built with ``use_distributed_sampler=False`` (``train.py`` line
  331; ``bench.py`` line 211; ``val.py`` line 158) to avoid a second injected
  sampler. Because Lightning then stops calling ``set_epoch`` (it only does so
  for samplers it injects), ``common/set_epoch_callback.py::SetEpochCallback``
  restores the per-epoch ``set_epoch`` (added to the callback list at
  ``train.py`` lines 63, 298; the DataModule retains the sampler as
  ``_train_sampler``). SI lets Lightning inject and drive its sampler, so it
  needs neither flag nor callback. This pair is the cost of reusing S2S's loader
  verbatim.
* **DDP invariants: ``DDPStrategy(find_unused_parameters=False,
  static_graph=True)`` + the dead-module freeze.** S2S requires
  ``static_graph=True`` (the manual loop ran ``DistributedDataParallel(...,
  find_unused_parameters=False, static_graph=True)``), which is only safe because
  two dead modules (``layer_perturbation2`` at ``v2.0/networks/pangu.py`` line
  363 with its forward call commented out, and ``layer_purturbation_e2`` at line
  408, never called) are frozen. The port wires the explicit strategy at
  ``train.py`` line 305 and ``bench.py`` lines 183--193, and reproduces the
  freeze in ``modules/train_module.py::_get_model`` at lines 385--388
  (``_dead_modules = {"layer_perturbation2", "layer_purturbation_e2"}`` ->
  ``mod.requires_grad_(False)``). SI's ``bench.py`` builds a ``DDPStrategy``
  with only ``bucket_cap_mb`` / ``gradient_as_bucket_view`` /
  ``ddp_comm_hook`` (``$SI_DIR/bench.py`` lines 148--156) and does **not** set
  ``find_unused_parameters`` / ``static_graph`` -- the S2S port adds those two
  invariants alongside the shared bucket/bf16 knobs.
* **The ``S2S_BENCH`` / ``S2S_NVTX`` / ``S2S_AMP_DTYPE`` env namespace + the
  CSV ``amp_dtype`` / ``ddp_find_unused`` columns.** The port preserves S2S's
  benchmark env-var namespace rather than renaming to SI's ``SI_*``:
  ``common/bench_callback.py`` reads ``S2S_BENCH_WARMUP`` / ``S2S_BENCH_STEPS`` /
  ``S2S_BENCH_CSV`` / ``S2S_NVTX`` (lines 43--46 region) versus SI's
  ``SI_BENCH_*`` / ``SI_NVTX`` (``$SI_DIR/common/bench_callback.py`` lines
  43--46). The in-step NVTX gate in the module is ``S2S_NVTX``
  (``modules/train_module.py`` line 73) versus SI's ``SI_NVTX``
  (``$SI_DIR/modules/train_module.py`` line 11); ``bench.py`` likewise reads
  ``S2S_DDP_*`` / ``S2S_PRECISION`` / ``TORCH_COMPILE_MODE`` (lines 73--81, 128)
  versus SI's ``SI_DDP_*`` / ``SI_PRECISION`` / ``SI_COMPILE_MODE``
  (``$SI_DIR/bench.py`` lines 60--67, 100). The CSV row carries two extra
  columns SI's row lacks -- ``amp_dtype`` (sourced from ``S2S_AMP_DTYPE``,
  default ``fp16``) and ``ddp_find_unused`` -- at
  ``common/bench_callback.py`` lines 300 and 310--311; the remaining columns
  (``timestamp`` / ``git_sha`` / ``config_sha16`` / ``run_num`` / ``n_gpus`` /
  ``batch_per_gpu`` / ``precision`` / ``step_*`` / ``samples_per_s*`` /
  ``data_idle_frac`` / ``peak_mem_gb_max_rank`` / ``n_steps_counted``) are
  identical to SI's (``$SI_DIR/common/bench_callback.py`` lines 194--211).
  Keeping the S2S namespace + columns preserves comparability with S2S's
  pre-port bench history (the CLAUDE.md hard constraint to *preserve* S2S's bench
  instrumentation); renaming would orphan that history.
* **HDF5 ``GetDataset`` / ``get_data_loader`` data backend.** The port wraps
  S2S's HDF5-backed dataset (``data/datamodule.py`` imports
  ``GetDataset`` / ``get_data_loader`` / ``get_infer_data`` from
  ``utils.data_loader_multifiles`` at lines 30--34; builds ``GetDataset`` at
  lines 142 and 151; routes ``get_infer_data`` through ``predict_dataloader`` at
  line 320) versus SI's ``data.amip_new.get_data_loader``
  (``$SI_DIR/data/datamodule.py`` line 2). The data backends are different
  filesystems and formats; reusing S2S's is the whole point.

## 4. Conclusion

**The only material difference between the ported S2S tree and the SI template
is the model definition** -- ``PanguModel_Plasim`` (with its 7-tuple VAE forward,
CRPS+KL losses, ensemble-CRPS tiling, calendar-free explicit-boundary forcing,
and netCDF prediction output) in place of SI's DiT-family models -- **modulo
the enumerated set (B) of necessary-S2S infrastructure differences** (YParams
sectioned config; the Lightning-2.5 EMA import guard; S2S owning its
DistributedSampler via ``use_distributed_sampler=False`` + ``SetEpochCallback``;
the ``DDPStrategy(find_unused_parameters=False, static_graph=True)`` + dead-module
freeze; the ``S2S_*`` bench env namespace + ``amp_dtype`` / ``ddp_find_unused``
CSV columns; and the HDF5 ``GetDataset`` data backend). Each (B) difference is
load-bearing and cannot converge without rewriting S2S's science, config, or data
layers -- explicitly out of scope under the port's "reuse, do not rewrite"
mandate. Every other ported component (entry-point flow, DataModule scaffolding,
LightningModule public API, bench harness, callbacks) mirrors SI's shape.

## 5. Known minor non-convergences (deliberate)

The following SI features were deliberately **not** pulled into the port. Each
would fail a code-review gauntlet for the reason given; none represents a missed
convergence.

* **SI's ``modules/ae_module.py`` / ``modules/combined_module.py`` /
  ``common/loss.py`` / ``common/plotting.py``.** These implement SI's DiT
  autoencoder/downscaler trainer, its evaluation-only combined module, its
  flow-matching/spectral/RMSE losses, and its DiT result/spectrum plots. None has
  an S2S analogue (S2S has no autoencoder stage, no DiT, and a CRPS+KL loss
  family). Porting them would add unreachable, SI-model-specific code. They are
  correctly absent.
* **SI's ``save_yaml``-to-run-dir config dump.** SI writes the resolved
  config to ``<run_dir>/config.yml`` (``$SI_DIR/train.py`` line 86,
  ``save_yaml`` defined in ``$SI_DIR/common/utils.py`` lines 14--16 using
  ``yaml.dump``). The port instead logs the config *path* (``train.py`` line 254
  "Loading config: ..." and the run-dir at line 274 "Logging to: ..."). A YParams
  dump would use ``ruamel.yaml`` and would not even reproduce SI's
  ``yaml.dump`` byte shape, so it is not true convergence; logging the path is
  low-value and sufficient for provenance (the bench CSV already records a
  ``config_sha16``). Not pulled.
* **SI's ``common/utils.py`` tensor helpers + ``load_partial_weights`` /
  ``val_num_inferences``.** SI's ``common/utils.py`` provides
  ``assemble_input`` / ``disassemble_input`` / ``assemble_forcing`` /
  ``disassemble_forcing`` / ``fix_state_dict`` / ``load_vanilla_weights_for_subpixel``
  (``$SI_DIR/common/utils.py`` lines 28--109), which all operate on a single
  packed ``b c h w`` tensor and perform DiT subpixel-unpatch checkpoint surgery;
  PanguModel_Plasim takes separate surface/upper-air/boundary tensors, so these
  are dead for S2S. Likewise ``load_partial_weights``
  (``$SI_DIR/train.py`` lines 29--44) is a ~16-line shape-filtered partial-load
  helper exercised only by SI's ``partial_checkpoint`` config key, which no S2S
  config sets -- it would be dead code in the port. And
  ``val_num_inferences`` (``$SI_DIR/data/datamodule.py`` lines 14, 27) sizes how
  many inference samples SI's val loader loads; wiring it into S2S would be an
  **inference-path behavior change**, not a structural convergence. (S2S's
  validation/inference already routes through ``num_inferences`` via
  ``get_infer_data`` in ``data/datamodule.py::predict_dataloader``.) None of
  these was pulled in.
* **Docstring "concise"-ness.** SI's source is lightly documented (several
  ``__init__`` methods have one-line or lowercase-``args:`` docstrings). The port
  deliberately standardized on full Google-style docstrings
  (``Args:`` / ``Returns:`` / ``Raises:`` / ``Attributes:``) with exact, verified
  cross-references. Rewriting the ported docstrings to match SI's terser style
  would regress the port's documentation deliverable, so it was not done.

## 6. Certification

Phase 5 adds only this document; the tracked port files (``train.py``,
``bench.py``, ``val.py``, ``modules/``, ``data/datamodule.py``, ``common/``,
``configs/``) are byte-identical to the green Phase-4 floor (HEAD ``540e680``),
verified with ``git diff --stat HEAD`` (empty). The Phase-5 commit is therefore a
**certification re-run** of the established end-to-end entry-point smoke
(``_scratch/smoke_phase3.sh`` on ``pedramh-gpu``: real ``train.py`` train path
with finite per-step losses and ``use_distributed_sampler=False`` confirmed, plus
the ``bench.py`` path emitting a CSV data row), not a debug loop. ``val.py`` is
unchanged since its green Phase-4 smoke, so ``smoke_phase3`` alone certifies the
Phase-5 floor.

* Certification smoke job: **51311092** (``_scratch/smoke_phase3_51311092.out``)
* Result: ``TRAIN_PATH_OK`` + ``BENCH_PATH_OK`` + ``SMOKE_OK``

## 7. Phase 6 -- docstring / reference-integrity pass (as landed)

Phase 6 is verification-only: it adds no executable behavior. It audited every
ported docstring for Google-style completeness and verified every cross-reference
resolves to a real symbol, then resolved the documentation findings the gauntlet
surfaced. Three commits, each gated on a real GPU smoke:

* ``cb77e3c`` -- docstring + reference-integrity audit. The core ported tree
  (``train.py`` / ``val.py`` / ``bench.py`` / ``common/`` / ``data/datamodule.py`` /
  ``modules/train_module.py``) was already clean -- every ``v2.0`` cross-reference
  verified exact (``PanguModel_Plasim`` 7/5/4-tuple forward arity; dead modules
  ``layer_perturbation2`` @ ``pangu.py:363`` / ``layer_purturbation_e2`` @ ``:408``;
  ``Stepper.save_prediction`` @ ``inference.py:231``). The only Check-1 gaps filled
  were the smoke harnesses + the ``verify_bench.py`` analysis utility.
  Behavior-neutral (executable AST byte-identical, docstring-only). Gate smoke
  **51311252** (2-step ``fit``, ``SMOKE_OK``).
* ``15c2812`` -- resolved the drift-auditor's documentation findings: 3 of the 5
  Phase-0 scaffold READMEs (``modules/`` / ``modules/models/`` / ``common/``)
  rewritten from forward-tense scaffold to as-landed state (the model is
  **reused in place**, not copied into ``modules/models/``; the config system is
  settled-YParams), plus an AST-neutral ``smoke_datamodule.py`` docstring fix.
  Gate smoke **51312978** (datamodule, ``SMOKE_OK``).
* ``9261fc5`` -- finished the doc pass: the remaining 2 scaffold READMEs
  (``modules/layers/`` / ``configs/``) brought into the same as-landed form,
  resolving a same-tree contradiction the prior partial pass had left. Pure
  Markdown; executable tree byte-identical to ``15c2812``
  (``git diff 15c2812 -- ':!*.md'`` empty). Gate smoke **51313651** (datamodule
  tree-health, ``SMOKE_OK``).

All 5 scaffold READMEs are now mutually consistent; no forward-tense /
open-decision drift remains anywhere in the ported subtree.

## 8. Consolidated per-phase smoke-id table (deferred from Section 1)

| Phase | Landing commit(s) | Gate smoke job id(s) | Result |
|---|---|---|---|
| 0 scaffold | ``a4388b5`` (+ ``8b4bb19`` / ``2a7ad1c``) | CPU import (build node; no Slurm job) | ``import lightning, torch`` OK (2.5.0.post0 / 2.6.0+cu124) |
| 1 DataModule | ``ab021a1`` (+ ``f43e8c7`` gauntlet) | 51309741, **51309754** | DataModule instantiate + one batch, pedramh-gpu |
| 2 TrainModule | ``28ee6bc`` (+ ``e4a7930`` / ``4354351`` gauntlet) | **51310303** | 2-step ``fit``, gpu:1, finite CRPS+KL losses |
| 3 entry points | ``1ce0c35`` (+ ``a2b98c4`` P0 / ``c812adc`` gauntlet) | 51310553 -> 51310608 / 51310609 / **51310616** | train + bench path, ``use_distributed_sampler=False`` confirmed |
| 4 inference | ``8e97798`` (+ ``540e680`` gauntlet) | **51310825** | ``val.py`` ``trainer.validate`` + netCDF inference path, 1x H100 |
| 5 reconcile | ``7e81363`` | **51311092** | ``TRAIN_PATH_OK`` + ``BENCH_PATH_OK`` + ``SMOKE_OK`` |
| 6 docstring/refs | ``cb77e3c`` -> ``15c2812`` -> ``9261fc5`` | **51311252**, 51312978, 51313651 | 2-step ``fit`` / datamodule, all ``SMOKE_OK`` |

Every gate smoke ran as a nested ``sbatch`` on ``--partition=pedramh-gpu
--account=pi-pedramh`` and was read (``.out`` ``SMOKE_OK`` + ``.err`` clean +
``sacct`` ``COMPLETED 0:0``) before its commit landed. No phase committed on an
unverified or red smoke. Each phase diff additionally passed the review gauntlet
(``s2s-code-reviewer`` + ``drift-auditor``, adjudicated by
``s2s-code-reviewer-critic``) before the green floor advanced.

## 9. How to launch the Lightning port

All entry points require ``PYTHONPATH=v2.0:.`` and the LPORT_ENV
(``module load python/miniforge-25.3.0; mamba activate
/project/pedramh/shared/S2S/v2.0/venv; module load cuda/12.6``).

* **Train (DDP, 4 GPU):**
  ``PYTHONPATH=v2.0:. python train.py --yaml_config configs/test_midway.yaml
  --config S2S --devices 0 1 2 3``
  -> ``L.Trainer(strategy=DDPStrategy(find_unused_parameters=False,
  static_graph=True), use_distributed_sampler=False,
  ...).fit(TrainModule, ClimateDataModule)``.
* **Validate / inference (single device):**
  ``PYTHONPATH=v2.0:. python val.py --yaml_config configs/test_midway.yaml
  --config S2S`` -> ``trainer.validate`` + the netCDF prediction path.
* **Benchmark (throughput, S2S_BENCH):**
  ``S2S_BENCH=1 S2S_BENCH_WARMUP=20 S2S_BENCH_STEPS=80 S2S_BENCH_CSV=bench.csv
  PYTHONPATH=v2.0:. python bench.py --yaml_config configs/test_midway.yaml
  --config S2S --devices 0 1 2 3`` -> ``BenchCallback`` writes a rank-0 CSV row;
  ``S2S_NVTX=1`` adds NVTX + ``cudaProfilerStart/Stop`` around the measured window.

The smoke harnesses (``smoke_datamodule.py``, ``smoke_train_module.py``) and their
sbatch wrappers (``midway_smoke_datamodule.sh``, ``midway_smoke_train_module.sh``)
pin ``v2.0/config/test.yaml`` for a fast 1--2-step single-GPU check.

## 10. PORT_COMPLETE certification

Phases 0--6 are landed on branch ``lightning-port``; the SI-mirrored structure
is in place; **the only material difference from SI is the model definition**
(``PanguModel_Plasim``), modulo the enumerated set (B) of necessary-S2S
infrastructure (Sections 3--4). Every phase smoke is green (Section 8), the
docstring/reference pass is complete (Section 7), and Phase 6's final gauntlet
returned APPROVE + REVIEW STANDS with all 5 scaffold READMEs reconciled.

Green floor at PORT_COMPLETE: ``9261fc5``. The branch is left for operator review
-- not pushed, not merged.
