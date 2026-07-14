# Code Readability Guide

Repo-specific guide for docstrings, module headers, and inline comments in
pedramh-profiling. Not a generic Python style guide and does not override
[`CLAUDE.md`](../CLAUDE.md) or [`DESIGN.md`](../DESIGN.md).

## Use this guide when

Creating a new module under `s2s/v2.0/` (model / losses / loaders / bench), in the
Lightning port (`s2s-lightning/modules/`, `data/`, `common/`), or in `si/`; adding
or updating functions in the model, loss, data, or bench paths; writing an HPC
launch script; or doing a readability pass on existing code.

Before readability edits, read: the touched module, the relevant section of
[`DESIGN.md`](../DESIGN.md) (§2 the three models, §3 the pipeline, §4 the
equivalence gate + invariants, §5 the optimization ladder), the matching
[`CLAUDE.md`](../CLAUDE.md) "Things NOT to do" rule for any invariant the code
touches, and the model's bench doc if you're in the bench path
(`s2s/v2.0/bench_report.md`, `si/bench_midway_notes.md`,
`s2s-lightning/LIGHTNING_PORT.md`). **A docstring citing `DESIGN.md §4.3` (an
invariant) or `LIGHTNING_PORT.md` (a mapped decision) is usually more valuable than
one re-deriving the math** — those docs are authoritative.

## Repo standard

- Every Python file opens with a 2–5 line module docstring naming its role in the
  pipeline (data → `PanguModel_Plasim` → VAE ensembles → CRPS+KL loss → DDP step)
  **and which harness it belongs to** (canonical S2S / the Lightning port / SI), since
  `s2s/v2.0/` is imported by two harnesses.
- Every top-level function has a docstring; the loss, model-forward, data-loader,
  and bench/instrumentation cores hit the depth target (below).
- Private helpers get short docstrings when they resolve tensor shapes, cast
  precision (AMP), place tensors on a device, encode a `DESIGN.md §4.3` invariant,
  or bracket an NVTX range.
- Inline comments answer *why*, not *what*.
- **Shortcuts require a justification comment** citing the authoritative doc:
  AMP/precision casts, `torch.compile` regions/graph-break boundaries, DDP
  `static_graph` assumptions + the dead-module freeze, the VAE reparameterization
  noise draw, `normalize`↔`inverse` and the predict-delta add-back, the
  `os.path.isfile` checkpoint guard, and NVTX range placement. Cite `DESIGN.md §X`
  or the `CLAUDE.md` rule that makes the shortcut safe — don't leave it bare.

## Module docstrings

Opening `"""` and continuation lines flush to column 0; do not indent the body.

```python
"""Latitude-weighted CRPS + KL loss for the S2S VAE ensemble (canonical S2S).

Shared code under s2s/v2.0/ — imported by BOTH the canonical torchrun harness and
the Lightning port, so changes here must serve both (CLAUDE.md #5). Implements the
primary loss per DESIGN.md §3: skill-minus-spread CRPS normalized over
num_ensemble_members and cos-latitude weighted, plus the Gaussian KL term. The
sign/normalization/weighting here are §4.3 invariants — the equivalence gate exists
to protect them; do not "simplify" the math for speed.
"""
```

## Function docstrings

A top-level function docstring should give a maintainer everything needed to
understand its role without reading the body:

- What it does and where it sits in the pipeline / which harness.
- Which invariant or formula it implements (cite `DESIGN.md §4.3`, or the mapped
  decision in `LIGHTNING_PORT.md`).
- Any non-obvious tensor shapes, the AMP/precision requirement, and the DDP/autograd
  policy.
- Output invariants a caller can rely on (finite loss, differentiable w.r.t. …).

**Depth target for non-trivial functions:** explain *why* each arg exists (shape
constraint, its role in the CRPS/KL/VAE math, the invariant it participates in), the
precision requirement if not the default, and the autograd/DDP flow. Use
sub-sections (`Shapes:`, `Precision:`, `Autograd:`, `DDP:`) when those are complex
enough that a reader would otherwise need to read the body.

Include `Args`/`Returns` only when they add content beyond the type hints. "A tensor
used by the model" is a signal to omit the section, not to keep boilerplate.

One-liner when it's enough:

```python
def _cos_lat_weights(lat: Tensor) -> Tensor:
    """Return per-latitude cos-weights normalized to mean 1 (DESIGN.md §4.3)."""
```

Depth target — the primary loss; `Args` entries explain the mathematical role, not the type:

```python
def weightedCRPS(
    ensemble: Tensor, target: Tensor, lat_weights: Tensor,
    *, num_ensemble_members: int,
) -> Tensor:
    """Latitude-weighted CRPS over the ensemble dimension (DESIGN.md §3, §4.3).

    CRPS = E|X - y| - 0.5 E|X - X'|, estimated from the ensemble and cos-latitude
    weighted. The pairwise spread term is normalized over `num_ensemble_members`;
    the skill-minus-spread SIGN is a §4.3 invariant guarded by the equivalence gate.

    Shapes: `ensemble` is [B, E, C, H, W] (E = num_ensemble_members); `target` is
        [B, C, H, W]; `lat_weights` is [H] broadcast over W.
    Precision: runs under the harness AMP dtype (S2S_AMP_DTYPE); the pairwise term
        is order-sensitive, so a vectorized rewrite must pass the §4 bf16 gate.
    Autograd: flows through `ensemble` (the model output), not `target`/`lat_weights`.

    Args:
        num_ensemble_members: E; the spread normalization denominator — changing it
            changes the score, so it is fixed by the config, never inferred here.

    Returns:
        Scalar loss, finite and differentiable w.r.t. the model output.
    """
```

Anti-pattern (do not continue if touching files that contain it):

```python
def _to_ensemble_batch(x: Tensor, e: int) -> Tensor:
    """Reshape to ensemble batch.
    Args:
        x: A tensor used by the model.
        e: A value used by the helper.
    """
```

Rewrite as a one-liner:

```python
def _to_ensemble_batch(x: Tensor, e: int) -> Tensor:
    """Tile [B, ...] to [B*E, ...] so the E ensemble members forward in one batch."""
```

## Inline comments

Every inline comment answers *why*, not *what*. A comment that restates the code
adds noise; a comment that explains a numerical assumption, an invariant, or an
engineering reason not visible from the code is worth keeping. Cite the section in
`DESIGN.md`/`CLAUDE.md` rather than pasting it.

Good uses:

```python
# static_graph=True: the graph is fixed across steps (no conditional branches in
# forward) and the dead VAE-decoder modules are frozen, so DDP can skip the
# find-unused-parameters trace. Breaking either assumption silently corrupts DDP.
# DESIGN.md §4.3 / CLAUDE.md #5.
model = DDP(model, find_unused_parameters=False, static_graph=True)
```

```python
# Guard the restore: production runs crashed here on FileNotFoundError BEFORE any
# kernel launched, which looked like "missing kernels" in nsys. Only restore if the
# checkpoint actually exists. DESIGN.md §4.3 (checkpoint guard invariant).
if os.path.isfile(checkpoint_path):
    restore_checkpoint(...)
```

```python
# Fix the reparam noise for the equivalence baseline: the VAE draw consumes RNG, and
# torch.compile/FlexAttention can reorder RNG kernel selection, so a CORRECT
# optimization would otherwise fail the §4 gate. Seed a dedicated generator here.
eps = torch.randn(mu.shape, generator=self._baseline_gen, device=mu.device)
```

```python
# NVTX boundary must stay exactly here (after H2D, before forward) — parse_nsys.py
# and the v2.0 baseline key on this range name/placement; renaming or moving it
# breaks comparability. DESIGN.md §3 (instrumentation is part of the contract).
if NVTX: nvtx.range_push("data_prep")
```

Avoid:

- Restating what the code says (`# increment step` above `step += 1`).
- Narrating simple assignments, reshapes, or `.to(device)` calls that carry no invariant.
- *What* without *why* — the point is "why is it safe / necessary / this exact value".
- Pasting `DESIGN.md`/`LIGHTNING_PORT.md` prose verbatim; cite the section and move on.

## Config / params docstrings

Configs are YAML-driven via `YParams`; variables are split into groups (upper-air /
surface / diagnostic / land / ocean / constant-boundary / varying-boundary). When a
function consumes a non-obvious params field, name the group and the invariant, not
the type:

```python
# params.num_ensemble_members — E in the CRPS spread normalization (DESIGN.md §4.3);
#   distinct from batch_size. params.predict_difference — if set, the model outputs a
#   delta that is add-backed to the input before inverse-normalization (§4.3);
#   the normalize<->inverse symmetry depends on doing both or neither.
```

## Smoke / equivalence tests

The port smokes signal success with a printed `SMOKE_OK` token (the commit gate keys
on it, not the exit code) — say so in the module docstring. For a §4 equivalence
check, the test docstring must state the tolerance and metric it enforces (e.g.
"max |a−b|/(|b|+1e-8) ≤ 1e-2 on the bf16 path, VAE noise fixed") so a reader knows
what "pass" means. Inner helpers/closures in a test get a one-liner; the enclosing
test already supplies context.

## Comments don't replace the docs

Docstrings and comments help someone read the code. They don't replace
[`DESIGN.md`](../DESIGN.md) (authoritative for what/why + the §4 equivalence gate +
the §4.3 invariants), [`CLAUDE.md`](../CLAUDE.md) (conventions, the "Things NOT to
do" list, the cluster-facts SSOT), [`CHANGELOG.md`](../CHANGELOG.md) (progress +
decisions + failed approaches), or the bench docs (`bench_report.md`,
`si/bench_midway_notes.md`, `LIGHTNING_PORT.md` — measured evidence + mapped port
decisions). A docstring citation (`DESIGN.md §4.3`) is a pointer; the full
justification lives in the authoritative source.

## Pattern and review

Module docstring first (role + harness); then concise top-level function docstrings
(depth target for the loss / model-forward / data / bench cores); then inline
comments only around non-obvious logic (precision, autograd/DDP, the §4.3
invariants, NVTX/instrumentation). Check comments still match the code after
finishing.

Before finishing, verify: can a new contributor tell what each file is for — and
which harness it serves — from the first few lines? Does every function touching a
§4.3 invariant (CRPS sign/normalization, lat-weighting, VAE/KL, normalize↔inverse,
predict-delta, the checkpoint guard) cite it? Are precision (AMP dtype), DDP
`static_graph`, and the NVTX contract called out where they matter? Any comments
redundant, stale, or too vague?

## Don't

- Long tutorial-style docstrings on every helper.
- `DESIGN.md`/`LIGHTNING_PORT.md` prose verbatim in docstrings — cite and move on.
- Duplicate the same explanation across module docstring, function docstring, inline
  comment, and CHANGELOG unless each serves a different audience.
- Skip docstrings for private helpers that encode meaningful behavior (invariant
  normalizations, ensemble/shape permutations, precision casts, NVTX boundaries).
- Shortcuts (AMP casts, `torch.compile` regions, `static_graph`, the reparam draw,
  manual device placement) without a comment pointing to the `DESIGN.md`/`CLAUDE.md`
  section that justifies them.
- Manual `.to(device)` / hand-rolled autocast in the Lightning port — the reviewer
  flags it; if you must, comment why it doesn't violate the automatic-optimization
  invariant (CLAUDE.md / `LIGHTNING_PORT.md`).
```
