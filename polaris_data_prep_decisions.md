# Decisions needed on the E3SM data preparation

Five questions need answering before the full data conversion runs. None of them can be settled
by a test, because none of them are bugs — they are choices about what the models should learn,
and DESIGN.md §1 says the science is frozen unless someone decides otherwise. Each one below
comes with the measurement that raises it, so the decision can be made on numbers rather than
on what a variable is named.

Scope is the S2S family: **PanguWeather**, **makani**, and **PhysicsNeMo**.

The measurements themselves, the archive layout, and the state of the conversion scripts are in
**`polaris_data_prep_handoff_prompt.md`**. This file is only the open questions.

---

## Should the PhysicsNeMo model forecast sea-surface temperature and sea ice?

Right now it does, and it is alone in that.

The three pipelines disagree about what kind of thing sea-surface temperature is. PanguWeather
lists it under `varying_boundary_variables` and makani lists it under forcing — both meaning
"the model is handed this at every step and never has to predict it". PhysicsNeMo's converter
puts it in the predicted set, which means the model forecasts it and is scored on how well it
does. Sea ice gets exactly the same treatment in all three.

The dataset settles the argument. It is `E3SMv3_SSP245AMIP` — an Atmospheric Model
Intercomparison Project run, which is a class of experiment defined by prescribing the ocean
surface and letting the atmosphere respond to it. Sea-surface temperature and sea ice are the
experiment's inputs. Asking the model to forecast them is asking it to predict something that
was imposed rather than computed. It will train, the loss will fall, and two of the hundred and
fifty-seven output channels will be learning something that is not a forecast.

**This one is ours to fix, not jesswan's.** The list lives in our own converter
(`physicsnemo_sfno/polaris/e3sm_h5_to_seqzarr.py`), it is not inherited from upstream, and the
change is a single line — moving sea-surface temperature and sea ice into the prescribed set.
That would leave seven prescribed channels and a hundred and fifty-five predicted ones, and all
three pipelines would finally agree.

The only reason I have not already made the change is that it alters what the model learns, and
that is a decision rather than a repair.

---

## Is PanguWeather's sea-surface temperature fill of 270 intended?

The E3SM archive leaves sea-surface temperature undefined over land, and something has to be put
there before a model can train on it. PanguWeather's configuration fills those cells with 270.

Measured from the archive, sea-surface temperature is in **degrees Celsius**, and its real values
run from **−1.80 to 32.21** — the minimum being exactly the freezing point of seawater. So the
fill sits roughly eight times outside the range of the actual data, across the 37% of the globe
that is land.

The consequence is measurable rather than theoretical. After normalization, the genuine
sea-surface temperature signal spans about a quarter of a standard deviation, while the step
between ocean and land spans just over two. The channel has become, in effect, a map of where
the coastlines are, with the temperature information compressed into the noise. The model can
still see where the ocean is — but it can barely see how warm it is.

For comparison, makani and PhysicsNeMo both fill the same cells with −1.8, the physical minimum,
which keeps the fill inside the distribution and leaves the real signal intact.

A related point worth raising at the same time: the shipped statistics file
(`normalize_mean.npz`) was computed on data filled this way, so it is entirely self-consistent
with the choice. Its sea-surface temperature mean of about 110 is not evidence of a unit error —
it is arithmetic: 37% of the globe at 270 plus 63% at an average of 14.7 gives 110.06, against
the file's 109.963. If the fill changes, that statistics file has to be recomputed with it.

---

## Is PhysicsNeMo's soil-temperature fill of zero intended?

This is the same problem as the previous one, running in the opposite direction.

Soil temperature at ten centimetres is undefined over the ocean, which is about 61% of the globe.
PhysicsNeMo's converter fills those cells with 0.0. Measured, the field is in **Kelvin**, with a
land average around 268. A fill of zero is therefore absolute zero, imposed over most of the
planet, in a field whose real values never go near it.

PanguWeather fills the same field with 270, which is sensible — it is Kelvin, it sits right in
the middle of the real distribution, and it leaves the land signal readable.

So each pipeline gets one of these two fields right and the other wrong, in mirror image.
PanguWeather is right about soil temperature and wrong about sea-surface temperature; PhysicsNeMo
is the reverse.

---

## Should PhysicsNeMo train on the cloud variables at all?

Nineteen of its hundred and fifty-seven predicted channels contain **no information**. The cloud
ice, cloud liquid and cloud fraction fields between roughly 5 and 200 hectopascals are either
identically zero across the entire archive, or hold floating-point values around 1e-26, which is
zero for every practical purpose. That is not a property of the sample we looked at — the shipped
statistics file records a standard deviation of exactly zero for the same sixteen of them, across
all thirty-five years.

PanguWeather and makani both exclude the cloud variables entirely. In PanguWeather's
configuration they are commented out on the line that lists the upper-air fields, so this was a
deliberate choice someone already made. PhysicsNeMo is the only pipeline that takes them, and it
asks the model to predict fifty-four cloud channels, nineteen of which are constant.

The cost is roughly 12% of a one-terabyte store, plus nineteen output channels of the network
spent on predicting a constant.

There is a prior question hiding underneath this one, and nobody has asked it: **are those fields
zero in the original model output, or were they lost somewhere in the conversion from E3SM to
HDF5?** Condensate really is negligible in the upper stratosphere, so zero is physically
plausible — but "physically plausible" and "verified" are different things, and the raw source
that would settle it has never been opened.

---

## Is it intended that the three models train on different data?

They read very different subsets of the same archive. PanguWeather reads 108 of the 162 fields,
PhysicsNeMo reads all 162, and makani reads 59 — and makani additionally keeps only ten of the
eighteen vertical levels.

Some of that is clearly deliberate: makani is packing into a fixed contract inherited from a
different model, and PanguWeather's exclusion of clouds is commented in place. But the three are
sometimes described as though they are the same experiment run through different frameworks, and
they are not. They do not see the same fields, the same levels, or — per the questions above —
the same physics for the fields they share.

That does not make any of them wrong. It does mean that comparing their skill is not a
like-for-like comparison, and any report that puts their numbers side by side needs to say so.

---

## Suggested order

The first question is ours and costs one line, so it can be answered immediately. The middle
three belong to jesswan and are best taken together, since they are all the same underlying
question — what to write where the data is missing, and whether the choice keeps the field
inside its own distribution. The last one is a framing question for whoever writes the
cross-model comparison.

What none of them can wait for is the full conversion. A terabyte written under the current
choices is a terabyte that encodes them.
