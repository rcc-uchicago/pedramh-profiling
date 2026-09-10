# The `n_future` experiment, in plain English

Companion to `2026-09-10_nfuture_ladder_prereg.md`, which is the formal
pre-registration. This file is the same plan written as prose, for anyone who
wants to understand what is being done and why without decoding the jargon.

---

## What we are trying to find out

We want to train with `n_future = 3` or `4` — meaning that during training the
model predicts, eats its own prediction, predicts again, four or five times
over, and gets corrected on all of them at once. The hope is that practising on
its own mistakes stops the long rollout from exploding.

The trouble is that we already have evidence pointing the other way. Going from
0 to 1 made the model **more accurate everywhere** but moved its instability
**not at all**. And ACE2 — which trains with the same two steps we do — runs
five years without falling over. So the honest starting position is *"this
probably will not work,"* and the experiment has to give that view a fair chance
of winning.

## Why we cannot just run it and look

Every number we have is a single measurement:

```
   base   ────────●
   C1     ──────●
                 ↑
        Is that gap real, or is it just how wide the dot is?
```

Nobody has run the same thing twice here, so we do not know how wide the dot is.
If we train at `n_future=3`, get a slightly better number and declare victory,
we would have no way of knowing whether we had measured anything.

So the first thing we do is not the experiment. It is measuring the width of the
dot.

## Step 1 — how noisy is our ruler? (no training)

Take the model we already have and roll it forward from three different starting
dates. Same model, same everything; only the start date changes:

```
   from Jan 1   ──────────●
   from Apr 1   ───────●
   from Jul 1   ─────────────●
                └────────────┘
                   this spread is sigma_0 -- the width of the dot
```

Then a second way, which is free because the files already exist. The long
training run saved a checkpoint every 20 epochs, so twelve slightly-different
versions of essentially the same model are on disk. Roll three of them forward
from the *same* date:

```
   epoch 203  ────────●
   epoch 223  ─────────●
   epoch 243  ───────●
```

That reads how much two near-identical models disagree.

None of this needs training — about six forward rollouts, minutes of GPU each.
And it can kill the plan cheaply: **if the spread is as big as the improvement
we hope for, then no `n_future` experiment will ever be readable and we should
do something else.** That is a good outcome for two cheap jobs.

## Step 2 — does the cheap test predict the expensive one?

Training at `n_future=3` properly takes hours. We want a short version — about
one epoch instead of twenty-four. But a short test is only useful if it tells
the truth about the long one.

We can check, because we measured a real answer: C1 (`n_future=1`, 24 epochs,
8 hours) came out **3-4.7 % better than base at every lead past the first**.

So run the *cheap* version of C1 — one epoch, three times with different seeds —
and see whether it finds the same thing:

```
   the real C1   (24 epochs, 8 h)  ──────────►  -4.7 %
   cheap C1 #1   (1 epoch, 40 min) ──────────►  -?
   cheap C1 #2                     ──────────►  -?
   cheap C1 #3                     ──────────►  -?
                                                 ↑
                    if these land near -4.7 %, the shortcut is trustworthy
                    if they do not, the shortcut is useless and we stop here
```

This is the step that keeps us honest. It also gives us three independent
confirmations of the C1 result, which currently rests on one run.

## Step 3 — then, and only then, the ladder

```
   n_future = 1  (control)   ● ● ●
   n_future = 3              ● ● ●
   n_future = 4              ● ● ●   <- only if 3 actually wins
```

One important detail: deeper training eats more memory, which forces a smaller
batch. If batch shrank as depth grew we could never tell which caused the
change. So **every arm runs at the same batch size of 8**, including the
`n_future=1` control. Depth is then the only thing moving.

## What we measure, and the arithmetic

Not *"what step does it explode"* — that is one dramatic event, it needs a run
to failure, and an arm that survives our cutoff only tells us "longer than we
watched," which cannot be ranked.

Instead we measure a slope. Soil moisture dries out steadily, so fit a straight
line to its global average over a short rollout:

    m(t) = m_0 + s * t

On the current model, over 56 steps:

    s   = -0.00958 per step
    m_0 = 8.11

which over a 500-step run gives

    500 * (-0.00958) = -4.79      ->   -4.79 / 8.11 = -59 %

The model would lose 59 % of its soil moisture. That is what a fix has to
shrink — and `s` is measurable from a **56-step** rollout, a couple of minutes,
rather than a 500-step one. It is a smooth number rather than a cliff, so it can
be averaged and compared across seeds.

```
soil
moisture
   8.1 ●
       │ ●
       │   ●  ●            slope s = -0.00958/step
       │        ●  ●
   7.6 │              ●    <- measured out to step 56
       │
       │  . . . . . . . . . . . . . . . extrapolated
   3.3 │                                            ● step 500
       └──────────────────────────────────────────────
```

A good arm bends that line flatter. A useless arm leaves it alone.

## Will `n_future = 3` or `4` even fit in memory?

Memory grows with both depth and batch. From our measurements:

    GiB  ~=  10.31 + 2.12 * (n+1) * b

with `n` = `n_future` and `b` = samples per GPU. The card holds 39.49 GiB, so

    b  <=  (39.49 - 10.31) / (2.12 * (n+1))  =  13.76 / (n+1)

    n = 3   ->   b <= 3.44   ->   3 per GPU  (global batch 12)
    n = 4   ->   b <= 2.75   ->   2 per GPU  (global batch  8)

Both fit. **But that formula has only ever been checked at `n_future` 0 and 1.**
Using it at 4 is extrapolation, and extrapolated memory formulas are how you get
six crashed jobs. So the first thing we run is a one-epoch probe at depth 3 and
4 just to watch the memory number.

## How we decide, agreed in advance

Call `sigma_0` the width of the dot from Step 1.

- improvement bigger than **2 sigma_0**, all three seeds agreeing on direction
  -> it works, escalate to `n_future = 4`
- between **1 and 2 sigma_0**, or seeds disagreeing -> call it nothing, do not
  escalate
- under **1 sigma_0** -> stop; depth is not the lever, and the effort goes to
  the corrector instead

And one rule fixed now, because it is the trap we already fell into once:
**an arm that comes back more accurate but no more stable counts as a failure,
not a success.** That is exactly what C1 did, and it must not look like a win a
second time.

## What it costs

Thirteen jobs: two for the memory probe, two for the noise floor, three for the
proxy check, three for `n_future=3`, three for `n_future=4`. All but the last
three are one-node, hour-long `debug` jobs that run one at a time — six to eight
hours of wall clock with no input needed. The `n_future=4` arm needs a longer
slot and will be surfaced before it is taken.
