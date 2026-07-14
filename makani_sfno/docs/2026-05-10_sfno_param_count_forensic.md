# SFNO parameter-count forensic: GB4-prod vs GB8 group-clone vs SFNO-5410

Date: 2026-05-10

## TL;DR

The reported **106.9 M** parameter count for SFNO-5410 and our **56.5 M** count
for the AI-RES Makani SFNO **describe the same model in different counting
units**. When you sum *real* degrees of freedom for both checkpoints, the
totals match within **0.017 %** (≈18 K params out of 106.9 M). There is no
~50 M capacity gap — there is a `complex64` numel convention mismatch.

| Model | `sum(p.numel())` | Real DoF | Notes |
|---|---:|---:|---|
| OWN-GB4-prod (Makani)        | **56,545,538** | 106,877,186 | 12 weights stored as `complex64` |
| OWN-GB8 group-clone (Makani) | **56,545,538** | 106,877,186 | identical arch to GB4 |
| SFNO-5410 (group `sfno_plasim`) | **106,895,104** | 106,895,104 | same 12 weights stored as real `[…, 2]` |

Diff to real DoF: 106,895,104 − 106,877,186 = **17,918 (= 0.017 %)**, all
explained by three small topology differences (filter bias, inner-skip bias,
decoder big-skip routing).

## 1. Source artefacts

| Component | GB4-prod | GB8 group-clone | SFNO-5410 |
|---|---|---|---|
| Config | `src/sfno_training/config/plasim_sim52_zgplev_full.yaml` (live YAML now reads `batch_size: 32`, historical) | `src/sfno_training/config/plasim_sim52_zgplev_group_clone.yaml` | `…/artifacts/derecho_blocking/sfno5410_blocking_epoch48_20260509/config/SFNO_PLASIM_H5_DERECHO_5410_deterministic.yaml` |
| Effective training config | `/scratch/.../sfno_zgplev_full/plasim_sim52_zgplev_full/0/config.json` (per-rank batch=1 → global=4) | `/scratch/.../sfno_zgplev_group_clone/.../0/config.json` (per-rank batch=2 → global=8) | `…/sfno5410_blocking_epoch48_20260509/config/hyperparams.yaml` (`global_batch_size: 32`, `world_size: 4`) |
| Checkpoint | `…/sfno_zgplev_full/plasim_sim52_zgplev_full/0/training_checkpoints/best_ckpt_mp0.tar` (1.71 GB, epoch 50, iters 1,819,900) | `…/sfno_zgplev_group_clone/plasim_sim52_zgplev_group_clone/0/training_checkpoints/best_ckpt_mp0.tar` (1.71 GB, epoch 50, iters 909,950) | `…/sfno5410_blocking_epoch48_20260509/checkpoints/ckpt_epoch_48.tar` (1.28 GB, epoch 48, iters 219,024) |
| Net class | `makani.models.networks.sfnonet.SphericalFourierNeuralOperatorNet` (`nettype: "SFNO"`) | same | `networks.modulus_sfno.sfnonet.SphericalFourierNeuralOperatorNet_v2` (`nettype: "sfno_plasim"`) |
| Spectral-conv module | `makani/models/common/spectral_convolution.py:31` `SpectralConv` — weight is `nn.Parameter(complex64)` of shape `[1, 256, 256, 64]` | same | `…/modulus_sfno/s2convolutions.py:49` `SpectralConvS2` — weight is `nn.Parameter(float32)` of shape `[256, 256, 64, 2]` |

> **The exact storage line that produces the counting mismatch:**
> - Makani (`spectral_convolution.py:90-94`): `init = scale * torch.randn(*weight_shape, dtype=torch.complex64); self.weight = nn.Parameter(init)` — `numel()` returns # of complex entries (= ½ the real DoF).
> - Group (`s2convolutions.py:150`): `self.weight = nn.Parameter(scale * torch.randn(*weight_shape, 2))` — a real tensor with a trailing length-2 axis; `numel()` returns # real entries (= the real DoF).

## 2. Checkpoint composition (verifying 106.9 M is weights only)

The 106.9 M reported in the group `MANIFEST.md` is the count of `model_state`
tensors only — it does **not** double-count from optimizer or EMA. I summed
`numel()` over each ckpt's actual state dictionaries (loaded to CPU, with
`torch.load`).

| Bucket | OWN-GB4-prod | OWN-GB8-clone | SFNO-5410 |
|---|---:|---:|---:|
| `model_state` numel | 56,545,538 | 56,545,538 | 106,895,104 |
| `ema_state` numel | 56,545,538 | 56,545,538 | *(none)* |
| `optimizer_state_dict` numel | 113,091,204 | 113,091,204 | 213,790,359 |
| ckpt bytes | 1,710,240,431 (1.71 GB) | 1,710,240,559 (1.71 GB) | 1,282,930,938 (1.28 GB) |

The pre-EMA AI-RES ckpt (`sfno_zgplev_full.pre-ema-20260504/.../best_ckpt_mp0.tar`)
is **1,282,694,703 bytes** — within 240 KB of the 5410 ckpt. The 0.43 GB
inflation in the post-2026-05-04 AI-RES ckpts is the EMA shadow weights, not
extra model capacity.

## 3. Module-level breakdown

`block.filter (spectral)` rows are the FNO spectral-convolution weights. The
group's per-entry numel is exactly 2× Makani's (because of real-vs-complex
storage); the *real DoF* columns match within rounding.

| Module class | OWN-GB4 numel | OWN-GB4 real DoF | SFNO-5410 numel = real DoF | Notes |
|---|---:|---:|---:|---|
| `block.filter (spectral)`  | 50,331,648 | **100,663,296** | **100,666,368** | 12 dhconv weights; 5410 has 12× 256 bias terms Makani lacks → +3,072 |
| `block.mlp`                | 3,154,944 | 3,154,944 | 3,154,944 | identical |
| `block.norm` (InstanceNorm)| 12,288 | 12,288 | 12,288 | identical |
| `block.inner_skip` (5410 only) | — | — | 789,504 | 12× `Conv2d(256,256,1)` with bias |
| `block.outer_skip` (Makani only) | 786,432 | 786,432 | — | 12× `Conv2d(256,256,1)` without bias → −3,072 |
| `encoder`                  | 80,640 | 80,640 | 80,640 | identical; `Conv2d(58,256,1,bias=True) + Conv2d(256,256,1,bias=False)` |
| `decoder` (+`residual_transform`) | 82,434 | 82,434 | 94,208 | different big_skip routing (see below) → +11,774 |
| `pos_embed`                | 2,097,152 | 2,097,152 | 2,097,152 | identical learned `[1,256,64,128]` map |
| **TOTAL**                  | **56,545,538** | **106,877,186** | **106,895,104** | gap = 17,918 (0.017 %) |

Decoder topology difference (the +11,774):
- Group `sfno_plasim`: concatenates the big_skip residual into the channel
  axis before the first decoder conv → `Conv2d(256+58, 256, 1, bias=True)` (80,640)
  + `Conv2d(256, 53, 1, bias=False)` (13,568) = 94,208.
- Makani SFNO: keeps the first decoder conv at `Conv2d(256, 256, 1, bias=True)`
  (65,792) and adds the big_skip residual through a separate
  `residual_transform = Conv2d(58, 53, 1, bias=False)` (3,074); plus
  `Conv2d(256, 53, 1, bias=False)` (13,568) = 82,434.

Both achieve the same "concat input into output projection" effect, just
plumbed differently.

## 4. Architecture-knob side-by-side

| Knob | GB4-prod | GB8 group-clone | SFNO-5410 | Affects param count? |
|---|---|---|---|---|
| `nettype` | `SFNO` (Makani) | `SFNO` (Makani) | `sfno_plasim` (group `SphericalFourierNeuralOperatorNet_v2`) | Yes (storage convention) |
| `embed_dim` | 256 | 256 | 256 | — |
| `num_layers` (FNO blocks) | 12 | 12 | 12 | — |
| `encoder_layers` | 1 | 1 | 1 | — |
| `mlp_ratio` | 2.0 | 2.0 | 2.0 | — |
| `spectral_layers` | 3 | 3 | 3 | — (this is `SpectralAttention*` depth; **unused** with `filter_type: linear`) |
| `num_blocks` | absent (Makani **kwargs swallow) | absent | 16 | — (only consulted by `SpectralAttention*` path; **unused** with `filter_type: linear`) |
| `filter_type` | `linear` | `linear` | `linear` | — |
| `operator_type` | `dhconv` | `dhconv` | `dhconv` | weight depends on `modes_lat` only |
| `hard_thresholding_fraction` | 1.0 | 1.0 | 1.0 | sets `modes_lat=64, modes_lon=65` |
| `scale_factor` | 1 | 1 | 1 | — |
| `img_shape` | 64×128 | 64×128 | 64×128 | sets pos_embed size + modes |
| `model_grid_type` | legendre-gauss | legendre-gauss | legendre-gauss (lat list matches Gauss-Legendre roots) | — |
| `normalization_layer` | instance_norm | instance_norm | instance_norm | — |
| `big_skip` | True | True | True | adds `residual_transform`/concat |
| `rank` | 1.0 | 1.0 | 1.0 | — (only relevant when `factorization≠None`) |
| `separable` | False | False | False | — |
| `factorization` | absent (None) | absent (None) | None | — (None routes group code to `ComplexDense` w/o tensorly; routes Makani to dense complex64) |
| `complex_network` / `use_complex_kernels` / `sparsity_threshold` / `complex_activation` | absent | absent | True / True / 0.0 / real | **inert** with `filter_type=linear` (swallowed in Makani **kwargs; only consulted in `SpectralAttention*` path of group) |
| `pos_embed` | `direct` (learned) | `direct` (learned) | `True` (learned) | + 2,097,152 |
| `n_history` / `n_future` | 0 / 0 | 0 / 0 | upstream uses 1-step iterative training (no autoregressive bptt) | — |
| `input_noise` | off | mode `perturb` σ=0.05 on 52 state channels | `epsilon_factor: 0.05` in data loader (≡ Gaussian σ=0.05 on normalized state inputs) | — (no learnable params) |
| In-channels | 58 | 58 | 58 | — |
| Out-channels | 53 | 53 | 53 | — |

Every knob that *could* change parameter count is identical. The only delta
in real DoF is from (a) absent biases on Makani's `outer_skip` / spectral
filter, and (b) the decoder big_skip plumbing.

## 5. Which knobs explain the apparent gap, quantitatively

The naive gap is **106,895,104 − 56,545,538 = 50,349,566 numel** (group >
Makani). It is explained almost entirely by one variable:

| Source | Contribution | Share of naive gap |
|---|---:|---:|
| 12 spectral-conv weights stored as real `[..., 2]` instead of `complex64` | 100,666,368 − 50,331,648 = **+50,334,720** | **99.97 %** |
| 5410 decoder concat-big_skip (94,208) vs Makani decoder + residual_transform (82,434) | **+11,774** | 0.023 % |
| 5410 spectral-filter biases (12 × 256) | **+3,072** | 0.006 % |
| 5410 inner_skip biases (12 × 256) | **+3,072** | 0.006 % |
| **Sum** | **+50,352,638** | ≈ naive gap (rounding) |

Sensitivity formulas to corroborate that the model is at the steep part of
the parameter curve for the spectral term:
- Per spectral-conv weight (real DoF): `2 × embed_dim² × modes_lat = 2 × 256² × 64 = 8,388,608`. Scales **quadratically** with `embed_dim` and **linearly** with `modes_lat = h × hard_thresholding_fraction`.
- Per MLP weight: `2 × embed_dim × (embed_dim × mlp_ratio) = 2 × 256 × 512 = 262,144` per block. Scales linearly in `mlp_ratio` and quadratically in `embed_dim`.
- Across all 12 blocks the spectral term contributes ≈ 100.7 M of the 106.9 M real DoF: **96 % of the parameter budget lives in the spectral convolutions**, not MLPs or norms.

So if someone wanted to make the model *actually* larger, the leverage is
`embed_dim` (quadratic) and `modes_lat` (linear via grid/scale_factor or
hard_thresholding_fraction>1). The knobs that 5410 sets *differently* from
us (`num_blocks=16`, `sparsity_threshold`, `use_complex_kernels`,
`complex_network`, `factorization`, `sync_norm`) are all inert under
`filter_type=linear` with `operator_type=dhconv` — they would only matter
under the `non-linear` (SpectralAttention) filter path that none of the
three runs use.

## 6. Task comparability

> **Correction (2026-05-10, second revision):** earlier revisions of this
> section made two wrong claims in turn — first that "ours is pressure-coord,
> theirs is sigma-coord," then that "only `zg` differs, ours plev vs theirs
> sigma." Both are wrong. **`zg` is on the same 10 pressure levels
> (200..1000 hPa) in *both* tracks**, and the rest of the upper-air block
> (`ta/ua/va/hus`) is on the same 10 sigma coefficients in both tracks.
> There is no vertical-coordinate difference between the two pipelines.
> Verified from (a) the group data loader `GEOPOTENTIAL_VARIABLES = {'zg',
> 'geopotential_height'}` short-circuit at
> `utils/data_loader_multifiles.py:74,129-141,564`, (b) the 5410 H5 sample
> file `/scratch/.../sim52/h5/sigma_data/11_0000.h5` which stores `zg` as
> `zg_<P_in_Pa>` for `P ∈ {5000, 10000, 15000, 20000, 25000, 30000, 40000,
> 50000, 60000, 70000, 85000, 92500, 100000}` (pressure, not sigma), and
> (c) the 5410 yaml `levels: [20000, 25000, 30000, 40000, 50000, 60000,
> 70000, 85000, 92500, 100000]` (Pa) = `[200, 250, 300, 400, 500, 600,
> 700, 850, 925, 1000]` (hPa), matching our `ZG_PLEV_HPA` exactly.

| Axis | GB4-prod / GB8-clone (own track v10) | SFNO-5410 (group) | Comparable? |
|---|---|---|---|
| Input variables | `pl, tas, ta×10, ua×10, va×10, hus×10, zg×10` + forcing `lsm, sg, z0, sst, rsdt, sic` = 52 state + 6 forcing = **58 in** | `pl, tas, ta×10, ua×10, va×10, hus×10, zg×10` + boundary `lsm, sg, z0, sst, rsdt, sic` = **58 in** | Same per-variable bouquet; **58 → 58** input/output channel counts match (verified in `encoder.fwd.0.weight` / `encoder.0.weight` shape `[256, 58, 1, 1]`) |
| Vertical basis (per variable) | `ta/ua/va/hus`: **sigma** (10 levels `[0.0383, …, 0.9833]`); `zg`: **pressure** (200, 250, 300, 400, 500, 600, 700, 850, 925, 1000 hPa, in gpm); `pl`: surface `ln(p_s_Pa)`; `tas`: 2 m T | `ta/ua/va/hus`: **sigma** (10 levels `[0.0383, …, 0.9833]`); `zg`: **pressure** (same 10 levels 200..1000 hPa, in gpm; the H5 stores `zg` only on plev, the data loader hard-routes zg to the plev branch via `GEOPOTENTIAL_VARIABLES`); `pl = ln(p_s)`; `tas` surface | **Both tracks: same coordinate for every variable.** ta/ua/va/hus on identical sigma; zg on identical pressure levels; pl/tas/pr_6h on the same surface fields. |
| Diagnostic | `pr_6h` (rate × 6h, mass/6h) | `pr_6h` (rate × 6h, mass/6h) | Same convention per memory note `project_5410_eval_track` |
| Horizontal resolution | 64×128 Gauss-Legendre | 64×128 Gauss-Legendre (lat list literally matches Gauss-Legendre roots) | Same |
| Normalisation | per-channel z-score (`global_means.npy`, `global_stds.npy`) computed by `plasim_makani_packager` | per-channel mean/std from `data_12-132_mean_sigma.nc` (sigma-data stats over yrs 12–132) | Same scheme (z-score), different stats (different data slice + sigma) — not strictly the same normalisation |
| Train years | 12–111 (100 yrs) | 12–111 (100 yrs) | Same |
| Validation | 1 year (year 11) | 1 year (year 11) | Same |
| Data source | sim52 (PlaSim), preprocessed to pressure-level h5 via `plasim_makani_packager` | sim52 (PlaSim) sigma-level h5 from `…/sim52/h5/sigma_data` (group preprocessing) | Same upstream simulation, **different preprocessing** (sigma vs pressure interpolation) |
| Loss | `l2` squared=True, constant channel weights | `raw_l2`, same channel weighting | Observationally equivalent |
| Schedule | LR 1e-4, AdamW, 50 epochs, CosineAnnealing+5ep linear warmup, max_grad_norm = (GB4: 32; GB8-clone: 0/off matching group) | LR 1e-6 (continuing from prior `load_exp_dir`, `resuming: True`), AdamW, 50 epochs, LinearWarmupCosineAnnealing, no grad clip | **Different**: 5410 epoch-48 ckpt is a *continuation* of an earlier 5410 series at very low LR (1e-6); ours train from scratch at LR 1e-4. The 5410 ckpt has trained on more compute upstream. |
| Global batch | 4 (GB4-prod) / 8 (GB8-clone) | 32 (`world_size: 4`, per-rank batch 8) | Different |
| Iters at logged epoch | 1,819,900 @ ep50 (GB4); 909,950 @ ep50 (GB8) | 219,024 @ ep48 | Same *samples-per-epoch* (≈ 145 k = 1 year of 6-hourly fields) — gradient updates differ by global-batch factor |
| Train-data perturbation | GB4: off; GB8-clone: white σ=0.05 on state | `epsilon_factor: 0.05` (Gaussian σ=0.05 on normalized state inputs in `data_loader_multifiles.py:869`) | GB8-clone matches; GB4-prod does not |

**Net**: the three runs share architecture and channel count, but the
vertical-level basis (sigma vs pressure) and preprocessing pipeline differ.
5410 also benefits from prior training (it is being continued at LR 1e-6
from an earlier state), so its epoch-48 ckpt has more cumulative compute
than ours.

## 7. Conclusions

**(a) Why does the parameter count differ so much?**
It doesn't, really. The 50 M numel gap is **99.97 % a counting convention**:
Makani stores spherical-spectral-conv weights as `nn.Parameter(complex64)`
(numel = number of complex entries), while the group's `modulus_sfno`
stores the same weight as `nn.Parameter(float32, shape=[..., 2])` (numel =
2 × number of complex entries). Both encode the same `embed_dim² ×
modes_lat` complex coefficients per FNO layer. The remaining ~18 K real-DoF
gap (0.017 %) is three bias terms and decoder big_skip plumbing.

**(b) Is 106.9 M real model capacity or a counting artefact?**
**Counting artefact.** The model contains 106,895,104 real degrees of
freedom; the Makani build of the same architecture contains 106,877,186
real DoF. Both have ~50.3 M complex spectral coefficients (= 100.66 M real
DoF) which are 96 % of the budget. The 56.5 M figure for our GB4/GB8 runs
is what PyTorch returns from `sum(p.numel() for p in model.parameters())`
when complex tensors are present — it does not represent half the capacity.

**(c) Is the 5410 advantage due to capacity, recipe, data, or apples-to-oranges?**
Not capacity, and **not vertical coordinate either**. Every state and
diagnostic channel lives on the same coordinate in both tracks (sigma for
ta/ua/va/hus on identical coefficients; the same 10 pressure levels
200..1000 hPa for zg; surface for pl/tas/pr_6h). Candidates that remain,
in order of likely impact:

1. **Prior training compute.** The 5410 epoch-48 ckpt is *resumed* from
   `load_exp_dir: /glade/work/marchakitus/PLASIM/PanguWeather/v2.0/results`
   at LR 1e-6 (`resuming: True`, `lr: 1e-6`) — it inherits weights that
   already saw substantial training before this epoch counter started.
   Our GB4/GB8 train from scratch. This is the strongest single suspect.
2. **Forcing preprocessing.** Our `rsdt` is `astronomical` (computed from
   solar geometry by `boundary_astro/sim52`); 5410's `rsdt` comes from
   `bias_data_dir: /glade/derecho/scratch/awikner/PLASIM/data/2100_year_sims_rerun/sim52/bias`
   (group's own pipeline, not audited here). Same applies to `sst` /
   `sic`. Same channel names, possibly different numerics. Worth a
   sanity check (compare mean/std/spatial maps of `rsdt`, `sst`, `sic`
   between the two boundary stacks) before attributing anything to
   recipe.
3. **Input noise σ=0.05.** The 5410 recipe and the GB8 group-clone both
   inject white noise on state channels each step; GB4-prod doesn't.
4. **Global batch / LR.** 5410 trains at GB=32; GB4-prod at GB=4 (lr
   scaled from a prior baseline); GB8-clone at GB=8 with `lr=1e-4` and
   no grad clip. Different optimisation trajectories.
5. **Normalisation stats.** Both tracks z-score, but ours from the
   packaged training years' stats, theirs from `data_12-132_mean_sigma.nc`
   (years 12-132 sigma file). Slight numerical differences.

The capacity hypothesis is wrong. The vertical-coord hypothesis is also
wrong (both prior versions of this section were incorrect on that point —
see the §6 correction header). Apples-to-oranges enters most strongly
via items 1, 2.

**(d) Single next experiment that best tests whether PlaSim is capacity-limited.**

> **Bump `embed_dim` from 256 → 384 on the GB8 group-clone recipe, keep
> everything else identical, train to 50 epochs from scratch, compare
> validation curves head-to-head with the current GB8 group-clone.**

Rationale:
- Param count scales as `embed_dim²` in both spectral conv and MLP — the
  steep direction. 384 ≈ 2.25× real DoF (≈ 240 M), 512 ≈ 4× (≈ 430 M).
- Keeps everything else (data, recipe, schedule, batch) constant, so a
  win is unambiguous capacity, not recipe.
- If 384 doesn't help on the same recipe, capacity isn't the bottleneck —
  the gap to 5410 is recipe/data/compute, not parameter count.
- Cheaper test than doubling depth, which scales linearly and also
  changes the optimisation dynamics more.

Given that vertical coordinate is **not** a difference between the two
tracks (both run zg on plev and ta/ua/va/hus on sigma on identical
levels), the two cheap controls that should run before any embed_dim
sweep are:

1. **Compare forcing numerics.** Load one matched `(year, day, hour)`
   step from our `boundary_astro/sim52` adaptor and from the 5410
   `…/sim52/bias` source, plot/diff `sst`, `rsdt`, `sic`. If they're
   numerically close, item 2 in (c) is ruled out and recipe / prior
   compute remain.
2. **Train GB8 group-clone for additional epochs** (or warm-start from
   the v9 sigma ckpt that we already have on disk) to isolate the
   "5410 was resumed at LR 1e-6 from an earlier run" effect from
   everything else.

Run these before the embed_dim sweep — they're far cheaper, and the
embed_dim conclusion is much harder to interpret if forcing numerics
and training-compute history aren't already matched.

## 7b. Vertical-coordinate audit (per channel)

Channel order (verified from `data['coords']['channel_state']` in
`/scratch/11114/zhixingliu/SFNO_Climate_Emulator/data/makani/sim52_zgplev_full/metadata/data.json`,
and identical for GB8 group-clone which reads the same dataset):

| Idx | Channel | Variable | Vertical basis | Source (own track) | 5410 has it on… | Same as 5410? |
|---:|---|---|---|---|---|---|
| 0 | `pl` | log surface pressure | **surface** (2D) | `MOST.YYYY.nc[pl]`, attrs `standard_name: log_surface_pressure`, mean ≈ 11.47, std ≈ 0.11 | surface, `pl = ln(p_s)` | ✅ same physical quantity |
| 1 | `tas` | 2 m air temperature | **surface** (2D) | `MOST.YYYY.nc[tas]`, K | surface | ✅ |
| 2-11 | `ta1..ta10` | air temperature | **sigma** σ=0.0383..0.9833 | `MOST.YYYY.nc[ta]` (dims `time, lev, lat, lon`), lev[0]=TOA → `ta1`, lev[9]=surface → `ta10` | sigma, same `sigma_levels: [0.0383..0.9833]` | ✅ identical coordinate |
| 12-21 | `ua1..ua10` | zonal wind | **sigma** σ=0.0383..0.9833 | `MOST.YYYY.nc[ua]` | sigma | ✅ identical |
| 22-31 | `va1..va10` | meridional wind | **sigma** σ=0.0383..0.9833 | `MOST.YYYY.nc[va]` | sigma | ✅ identical |
| 32-41 | `hus1..hus10` | specific humidity | **sigma** σ=0.0383..0.9833 | `MOST.YYYY.nc[hus]` | sigma | ✅ identical |
| 42-51 | `zg200, zg250, zg300, zg400, zg500, zg600, zg700, zg850, zg925, zg1000` | geopotential height | **pressure** (hPa, gpm) | `MOST.YYYY.nc[zg_plev]` on coord `lev_2`, selected by value from `[50, 100, 150, 200, ..., 1000]` | **pressure** (same 10 levels 200..1000 hPa in gpm). The 5410 H5 stores `zg_<P_in_Pa>` for `P ∈ {5000..100000}`; the data loader's `GEOPOTENTIAL_VARIABLES = {'zg', 'geopotential_height'}` short-circuit hard-routes zg to the plev branch even when `use_sigma_levels=True`. The 5410 yaml selects pressure levels `[20000, 25000, 30000, 40000, 50000, 60000, 70000, 85000, 92500, 100000] Pa` = `[200..1000] hPa`. | ✅ same 10 levels |
| 52 (diag) | `pr_6h` | precipitation 6 h accum | **surface** | `MOST.YYYY.nc[pr_6h]` (kg m⁻² per 6 h) | surface, `pr_6h = rate × 6h` | ✅ same convention |
| forcing 0-2 | `lsm, sg, z0` | land/surface masks + roughness | **surface, static** | `MOST.YYYY.nc[{lsm,sg,z0}]` | same | ✅ |
| forcing 3 | `sst` | sea surface temperature | **surface, varying** | `boundary_astro/sim52` adaptor; NaN over land filled with 271.35 K | from `…/sim52/bias` | structurally same, byte-equality not verified |
| forcing 4 | `rsdt` | TOA downward solar | **surface, varying** | `boundary_astro/sim52`, method=`astronomical` (computed from solar geometry) | from `…/sim52/bias` (method not audited here) | structurally same, numerical equality not verified |
| forcing 5 | `sic` | sea ice concentration | **surface, varying** | `boundary_astro/sim52`, clipped to [0,1] | from `…/sim52/bias` | structurally same |

**Summary of the only real differences between own track and 5410:**

| Channels affected | What differs |
|---|---|
| Vertical coordinate | **No difference.** Both tracks: sigma for ta/ua/va/hus on the same 10 sigma coefficients; pressure for zg on the same 10 pressure levels (200..1000 hPa); surface for pl/tas/pr_6h. |
| Normalisation stats source | Both use z-score; ours computed from the packaged training years, theirs from `data_12-132_mean_sigma.nc` (years 12-132). Same scheme, slightly different stats. |
| `sst, rsdt, sic` forcing | Same channels and same forcing roles, but our boundary adaptor (`boundary_astro` with astronomical rsdt) is provably a different source path from their `…/sim52/bias`. **Numerical equality not verified — worth checking before attributing skill differences to recipe.** |
| `ta/ua/va/hus, pl, tas, lsm, sg, z0, zg, pr_6h` | **None — same physical quantity on the same coordinate** (modulo float32/float64 representation, modulo the normalisation-stats item above). |

## 8. Reproducibility

The numbers above were produced by loading each ckpt with
`torch.load(..., map_location='cpu', weights_only=False)` from the
AI-RES `.venv` (torch 2.11.0+cu130) and iterating `model_state` /
`ema_state` / `optimizer_state_dict` to compute `numel` and a
real-DoF count (`2*n if torch.is_complex(t) else n`). Module-class
buckets are keyed by name prefix (`encoder`, `decoder`,
`residual_transform`, `blocks.*.filter|mlp|norm|inner_skip|outer_skip`,
`pos_embed`).
