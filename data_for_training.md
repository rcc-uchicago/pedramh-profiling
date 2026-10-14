# Does the E3SM data prep actually break training?


## RISK

| Risk | Is it a real training issue? | Confidence |
|---|---|---|
| **R2**<br>frozen ocean<br>forcing | **No; optimization is unaffected.** Pangu and makani receive SST and ICE as prescribed boundary inputs, and a boundary condition that repeats annually remains a valid boundary condition: the network learns the atmospheric response to it correctly. The consequence is interpretive, not numerical. For PhysicsNeMo, which forecasts these fields, the repetition renders them an exact function of day-of-year, so the model attains high apparent skill on two channels carrying no forecastable information — contaminating the metric rather than the optimization. Secondarily, a by-year train/validation split shares identical ocean forcing across both partitions, leaving validation less independent than it appears, though the atmospheric state still differs. | **Conditional, and binary.** Were the fixed-SST control design intended, there is no issue whatever; were it a defect, every model trained on this archive is trained on incorrect forcing. Not a spectrum but a switch, and only jesswan can throw it. |
| **R3**<br>TSOI fill/stats<br>mismatch | **Yes, but bounded to a single channel.** The regression remains well-posed — the target sits at +1.22σ ± 0.19σ and the network can fit it — and the inverse transform is unbiased, so the physical Kelvin output is correct whenever the normalized prediction is. The sole harm is gradient allocation: a channel's contribution to the loss scales with its signal variance, so `TSOI_10CM` draws roughly 26× less optimization pressure than a correctly scaled channel and is forecast less accurately than the data supports. No other channel is perturbed, and no instability arises. | **High** on the mechanism, **low** on the consequence. Whether the degradation is observable depends entirely on whether soil temperature is evaluated at all — it is 1 of 108 channels, so a genuine defect may have no noticed effect. |
| **R4**<br>SST 270 fill | **Probably not.** The field is input-only in Pangu and therefore carries no loss term. Normalization is an affine, invertible map and the land mask is static (verified bit-identical across every sampled file), so the compression is recoverable by a single first-layer weight — and unlike R1 the gradient for learning that weight is attenuated only ~11×, which is readily trainable. The residual concern is numerical rather than statistical: at bf16 a 0.09σ signal offset to −0.77σ retains order-30 distinct representable levels, though bf16-mixed training preserves fp32 master weights and softens even that. | **Medium-low.** Best characterized as a conditioning inefficiency rather than a defect. The earlier framing that "Pangu gets both SST and TSOI wrong" overstated this: the concrete harm is R3's, not R4's. |
| **R5**<br>16 zero-variance<br>cloud channels | **No; they are inert.** Channels with σ = 0 pass through BatchNorm to exactly zero (`eps` precludes division by zero), their targets are zero, and their contribution to both the error norm and the target norm is nil. No NaN pathway exists: Pangu consumes `std_corr.nc`, in which those 16 zeros are already replaced by 1.0, and excludes the cloud variables regardless. The cost is confined to resources — some 19 of 157 predicted channels and ~12% of the store representing a constant, with the corresponding FLOPs expended at every step. | **High.** This is the *only* item that falls squarely within a performance mandate: removing these channels alters nothing the model computes, since they are constant, while reclaiming ~12% of store and compute. |
| **R6**<br>sigma levels<br>labelled isobaric | **No training impact whatever.** The network learns whichever vertical coordinate the data is presented on. SFNO is a spectral architecture with no physics-informed component presuming isobaric surfaces, and makani's selection of the lowest ten levels is well-defined under either interpretation, so training is internally self-consistent. The defect is semantic: makani labels the channels `T850` and `Z500`, asserting isobaric surfaces the data does not represent, which biases any comparison against isobaric references (ERA5 Z500, standard climatologies) over elevated terrain. | **High.** The failure surfaces at evaluation, never at optimization. |

---

## Where each fix belongs

The distinction matters, because it determines what must be settled **before** the conversion and
what can follow it.

| Risk | Fix lives in | Gates the conversion? |
|---|---|---|
| **R2** | **Upstream** — the E3SM run itself, or nothing if intended. | **Yes** — if it is a defect, the archive is regenerated and anything converted from it is wasted. |
| **R3** | **Config or stats** (Pangu) — align `mask_fill['TSOI_10CM']` with the stats, or recompute the stats under the 270 fill. No conversion involved. | **No.** |
| **R4** | Same as R3, and likely not worth fixing. | **No.** |
| **R5** | **Converter** — drop the constant channels. | **Optional**, but cheapest to take at conversion time. |
| **R6** | **Channel labels** (makani) + docs. | **No.** |


## Caveats on the whole assessment

- **No empirical training run demonstrates any of this.** Every verdict rests on static analysis
  of the training path plus measurements of the archive. No skill degradation has been observed;
  it has been *predicted* from arithmetic.
- **R1's batch-σ ≈ global-σ step is inferred.** The amplitude factor uses measured global σ;
  `momentum=None` makes the running estimate converge to global, but this was not run end-to-end.
- **R3 and R4 are conditioning arguments.** Normalization is affine and invertible, so a
  sufficiently expressive first layer can in principle absorb both. The claims are about
  *optimization pressure and numerical headroom*, not about information being destroyed.
  Read them as "miscalibrated normalization", never "corrupted data".
- **R2's verdict is not a judgement about the data**, which is internally consistent either way.
  It is a statement that the question is unanswerable from the archive alone.
