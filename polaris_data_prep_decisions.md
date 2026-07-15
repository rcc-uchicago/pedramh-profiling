# Review of the E3SM data preparation, and five questions it raises

Three pipelines — PanguWeather, makani, and PhysicsNeMo — prepare the same E3SM archive for
three models. This note reports what each one actually uses, where they disagree, and the five
questions that need a decision before the full conversion is run. Every number below was
measured from the archive.

Supporting detail, the state of the conversion scripts, and how to verify a store are in
`polaris_data_prep_handoff_prompt.md`. This note is the review.

---

## The archive

`E3SMv3_SSP245AMIP_CTL_SST0051_REST0101/h5/plev_data/` holds 51,100 files, one per timestep,
covering 2015–2049 at six-hourly resolution (1460 per year, no leap years). Each file contains
162 fields on a 180×360 grid: eight upper-air variables at eighteen levels, plus eighteen
surface fields. Every field is single-precision. The structure is entirely regular.

Two properties of the data matter for what follows, and both were measured rather than assumed:

**Sea-surface temperature is in degrees Celsius**, ranging from −1.80 to 32.21, and is undefined
over land (37.4% of the globe). The minimum is the freezing point of seawater.

**Soil temperature at ten centimetres is in Kelvin**, averaging 268 over land, and is undefined
over the ocean (61.4% of the globe).

The archive mixes unit conventions: the ocean surface is in Celsius, every other temperature is
in Kelvin. Field names do not indicate which.

---

## What each pipeline uses

Read directly from the three configurations. Roles: *prognostic* means the model forecasts the
field and is scored on it; *prescribed* means the model is given the field at every step and
never predicts it; *diagnostic* means it is predicted but not fed back.

| field | levels | PanguWeather | PhysicsNeMo | makani | fill where undefined |
|---|---|---|---|---|---|
| T, U, V, Z3, RELHUM | 18 | prognostic | prognostic | prognostic | — |
| CLDICE, CLDLIQ, CLOUD | 18 | **not used** | prognostic | **not used** | — |
| PS, TREFHT | 1 | prognostic | prognostic | prognostic | — |
| PSL, TMQ, U10, RHREFHT | 1 | prognostic | prognostic | **not used** | — |
| FSNT, FSNTOA | 1 | diagnostic | prognostic | **not used** | — |
| PRECT | 1 | diagnostic | prognostic | diagnostic | — |
| SOILWATER_10CM | 1 | prognostic | prognostic | **not used** | 0 / 0 / — |
| TSOI_10CM | 1 | prognostic | prognostic | **not used** | **270 / 0** / — |
| SST | 1 | prescribed | **prognostic** | prescribed | **270 / −1.8 / −1.8** |
| ICE | 1 | prescribed | **prognostic** | prescribed | 0 / 0 / 0 |
| sol_in | 1 | prescribed | prescribed | prescribed | — |
| TOPO, PFTDATA_MASK | 1 | prescribed | prescribed | prescribed | 0 / 0 / 0 |
| PCT_GLACIER | 1 | prescribed | prescribed | prescribed | 0 / 0 / 0 |
| PCT_NATVEG | 1 | prescribed | prescribed | **not used** | 0 / 0 / — |

**Coverage:** PanguWeather uses 108 of the 162 fields, PhysicsNeMo uses all 162, makani uses 59
and keeps only ten of the eighteen levels.

The three pipelines agree on the majority of the archive. They disagree in four places, marked
in bold: the cloud variables, the role of the ocean surface fields, and two fill values.

---

## Finding: there is no shared convention for missing data

The archive marks ocean-only fields as undefined over land and land-only fields as undefined
over the ocean. Nothing can train on undefined values, and none of the data pipelines mask them,
so each converter substitutes a constant. Each chose its own, independently.

Where they agree — writing zero for topography, ice fraction, glacier and vegetation percentages
— the agreement is incidental. Those are non-negative quantities for which zero genuinely means
"none". Where the physical meaning of zero is not "none", they diverge.

No pipeline uses the alternative, which is to exclude the undefined cells from the loss. This is
worth noting because all three already supply the land-sea mask to the model as an input field,
so the network is told where the mask is. The substituted constant therefore does not need to
carry that information. Its only requirement is to stay inside the field's own distribution so
that normalization is not dominated by the land-ocean step.

Measured against that requirement:

- Filling the ocean surface with −1.8 (makani, PhysicsNeMo) places it at the physical minimum of
  the real data. The temperature signal survives normalization.
- Filling the ocean surface with 270 (PanguWeather) places it roughly eight times outside the
  range of a Celsius field. After normalization the real temperature variation spans about 0.25
  standard deviations while the land-ocean step spans about 2. The channel becomes a coastline
  map with the temperature compressed into it.
- Filling soil temperature with 270 (PanguWeather) sits in the middle of the real Kelvin
  distribution and is appropriate.
- Filling soil temperature with 0 (PhysicsNeMo) is absolute zero over 61% of the globe.

Each pipeline handles one of these two fields well and the other poorly, in mirror image.

---

## The five questions

**1. Should the PhysicsNeMo model forecast the ocean surface?**

It currently treats sea-surface temperature and sea ice as prognostic: it predicts them and is
scored on them. PanguWeather and makani both prescribe them.

The experiment settles this. `SSP245AMIP` denotes an Atmospheric Model Intercomparison Project
run, a design in which the ocean surface is imposed and the atmosphere responds to it. Those two
fields are inputs to the simulation, not outputs of it. A model asked to forecast them is being
scored on reproducing a boundary condition. It will train and the loss will fall; two of its 157
output channels will not be forecasting anything.

This one is ours rather than jesswan's — the list is in our own converter, not inherited from
upstream — and the change is a single line. It has been left alone because it alters what the
model learns.

**2. Is PanguWeather's ocean-surface fill of 270 intended?**

See the finding above: it places a Celsius field eight times outside its own range and reduces
the temperature signal to roughly a tenth of the land-ocean step.

If it is changed, the shipped statistics must be recomputed with it. `normalize_mean.npz` was
computed on data filled this way and is self-consistent with it: its ocean-surface mean of
109.963 is 37.4% of the globe at 270 plus 62.6% at 14.70, which is 110.06. That file is not
wrong; it encodes this choice. (It has previously been recorded as evidence of a Kelvin/Celsius
unit error. It is not — that reading is refuted by the arithmetic above.)

**3. Is PhysicsNeMo's soil-temperature fill of zero intended?**

The same issue in the opposite pipeline: absolute zero written into a Kelvin field over most of
the planet. PanguWeather's 270 is the sensible value here.

**4. Should PhysicsNeMo train on the cloud variables?**

Nineteen of its 157 predicted channels contain no information. The cloud ice, cloud liquid and
cloud fraction fields between roughly 5 and 200 hectopascals are either identically zero across
the whole archive or hold values near 1e-26. This is not a property of the sample examined: the
shipped statistics record a standard deviation of exactly zero for sixteen of them across all
thirty-five years.

PanguWeather and makani exclude the cloud variables entirely; in PanguWeather's configuration
they are commented out, so the exclusion was deliberate. PhysicsNeMo is alone in taking them, at
a cost of roughly 12% of a one-terabyte store and nineteen output channels spent predicting a
constant.

A prior question has not been asked: are these fields zero in the original model output, or were
they lost in the conversion from netCDF to HDF5? Negligible condensate in the upper stratosphere
is physically plausible, but plausible and verified are different. The raw source that would
settle it (`Gridded_EAM_Lev_Subset/`, 1.9 TB) has never been opened, and the conversion code
that produced the archive is not in this repository — it is in
`/eagle/.../jesswan/PanguWeather/data_utils/`, adapted from FourCastNet.

**5. Is it intended that the three models train on different data?**

108, 162 and 59 fields respectively, with makani additionally at ten of eighteen levels. Some of
this is clearly deliberate — makani packs into a contract inherited from another model. But the
three are sometimes described as the same experiment through different frameworks, and they are
not: they do not see the same fields, the same levels, or the same physics for the fields they
share. This does not make any of them wrong. It does mean their skill is not directly comparable,
and any report placing their numbers side by side should say so.

---

## Recommendation

Question 1 is ours and costs one line; it can be answered now. Questions 2, 3 and 4 belong to
jesswan and are best taken together, since all three are the same underlying question: what to
write where the data is missing, and whether that choice keeps the field inside its own
distribution. Question 5 is for whoever writes the cross-model comparison.

None of them should wait for the conversion. A terabyte written under the current choices is a
terabyte that encodes them.
