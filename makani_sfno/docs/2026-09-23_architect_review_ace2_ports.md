# Architect review — the ACE2 ports plan (`polaris_makani_ace2_ports_handoff.md`)

*2026-09-23. Reviewer session: "code review architect". Target session: `makani_ports_from_ace2`.*
*Goal being served: lower RMSE / higher ACC on the makani E3SM model without changing what is
being measured, on evidence.* Every line-number below was read from the installed venv
(`$MEMBER_ROOT/conda-envs/sfno-venv/.../makani`) or the fork on `feat/multinode-ddp-port` today.

## Verdict

The order **F → B → D → C → A → E** is right. **Port F as written in §6a would train a
silently wrong model**, and ports B and C can be collapsed into one config-only change that
also removes the `PRECT`-deletion trap. Three blockers, two simplifications, one experiment
design. Nothing here needs a node-hour to confirm; it was read from the code.

## 1. 🔴 BLOCKER — port F: the trainer builds channel lists as contiguous PREFIXES

`src/sfno_training/trainer/plasim_trainer.py:151-155`:

```python
n_state  = params.get("n_state_channels", 52)
n_target = n_state + params.get("n_diagnostic_channels", 1)
in_channels  = list(range(n_state))
out_channels = list(range(n_target))
```

In `e3sm_alldata_full.yaml`, **`SOILWATER_10CM` is index 8 and `TSOI_10CM` is index 9**;
`RELHUM_l16`/`RELHUM_l17` are 98/99, `PRECT` is 100. So `n_state_channels: 98` drops
**`RELHUM_l16` and `RELHUM_l17`**, keeps both soil channels, and raises no error.

It compounds, because three different consumers each build their own channel map:

| consumer | source of its channel indices | under a naive "98 + 99 names" edit |
|---|---|---|
| data read (`plasim_forcing_dataset.py:308,320`) | fork's `range(n_state)` | reads channels 0..97 → wrong two dropped |
| data normalization (`data_loader_multifiles.py:125-133`, `out_bias = bias[:, out_channels]`) | same prefix | `PRECT` (target slot 98) is normalized by **`RELHUM_l16`'s** mean/std |
| loss scale (`loss.py:101,106`) and ACC climatology (`data_helpers.py:113`) | `params.out_channels`, set **by name** in `parse_dataset_metada.py:47-60` from `channel_names` vs the pack's `data.json` | the *correct* 99 indices |

Result: data, normalization and loss disagree on which physical channel sits in each slot.
Loud only if the widths differ; silent when they agree — which is exactly the "99 names +
98" edit the handoff proposes.

**Fix (small, and it is the stock mechanism).** Stock makani already resolves
`params.channel_names` **by name** against `metadata/data.json` and hands back
non-contiguous `params.in_channels == params.out_channels`. Make the fork honour it:

```python
out_channels = list(params.out_channels)                 # name-resolved by stock
assert params.channel_names[-1] == <diagnostic name>     # :320 assumes diag LAST
in_channels  = out_channels[:-1]                         # state only
assert len(in_channels) == params.n_state_channels
```

Then the config carries the decision as **names** — `channel_names` = the 99 kept names,
`PRECT` last, `n_state_channels: 98` — and the pack, the six `.npy` stats files and
`k56_metrics.h5` stay untouched (they get indexed, as §6a already argues).

**The gate.** `polaris_sfno_alldata_full.pbs:172-202` asserts `channel_names ==
conv.TARGET_CHANNELS` and *will* fail loudly — good. ⚠ But **the production launcher the
handoff names, `polaris_makani_multinode_scaling.pbs`, has no such gate** (it only checks
`metadata/data.json` exists, line 275). The loud failure §6a relies on does not exist on the
path you would actually launch. Teach the gate rather than weaken it: `channel_names` must be
an **ordered subsequence** of `TARGET_CHANNELS`, the diagnostic must be last,
`n_state_channels == len(channel_names) - 1`, and print the dropped names. Put the same block
in the scaling launcher.

**Test that must ship with F** (CPU-only, on the smoke pack, PASS = `PORT_F_CHANNELS_OK`):
1. `inp_state[:, k]` equals raw `/fields_state[:, channel_names.index(name)]` after
   normalization, for `RELHUM_l17`, `Z3_l17`, `PS` (channels on both sides of the gap);
2. `tar[:, -1]` is `PRECT` normalized by `global_stds[100]`, not by index 98;
3. `LossHandler.scale` equals `global_stds[out_channels]` and has length 99;
4. the names `SOILWATER_10CM`, `TSOI_10CM` appear nowhere in the model's channel list.

## 2. 🔴 BLOCKER — port C via `channel_weights: "auto"` is a silent no-op on E3SM names

`utils/losses/base_loss.py:46-57`: `auto` matches lowercase ERA5 names (`u10m`, `t2m`,
`z500`…). **Every E3SM name is uppercase**, so all 101 (or 99) fall into the `else` branch
and get `0.01` — a uniform vector, i.e. `constant` scaled. The arm would train, converge to
the same model, and "prove" weights don't matter. `new auto` and `custom` are the same.

⇒ Port C can only be an **explicit list**, which is what makes §3 possible.

## 3. 🟢 SIMPLIFICATION — collapse B and C into one config-only, cap-safe port

`loss.py:160-164` accepts an explicit per-loss weight list — ⚠ **nested**, `[[w0, …]]`:
the assert reads `chw.shape[1]`, so a flat list raises `IndexError`. With a list,
`temp_diff_normalization` can stay `False`, which sidesteps **both** traps the probe found:

* no physical-unit `1e-4` clamp ⇒ `PRECT` keeps its correct ≈1.04, not 8.3e-4;
* the cap is explicit and dimensionless: `w_c = clip(σ_c/δ_c, 1, W_max) / mean(...)`,
  computed offline from the **full-split** `δ_c` (port B step 1's converter pass), on the
  **99-channel order**, and written into the yaml where a reviewer can read it.

No edit to the venv's makani, nothing to fork, and the realised weight vector is a config
artifact rather than something to print from a smoke. `W_max`: ACE2's hand-picked span is
40×; start one arm at **30** and report the vector. Trap 4 (`LOAD_LOSS=0`) applies — the
running-stats buffers are channel-shaped.

This is also the *right* answer to the `Z3_l17`-gets-9518× problem: the cap bounds it at
`W_max` while the science question (should it be prognostic at all) goes to jesswan
unchanged. ⚠ It is still a loss change ⇒ **sign-off before it is quoted**, but the
argument to bring her is now a 99-number table, not a mechanism.

## 4. 🟡 Port F cost — prove the surgical transfer before spending 46 node-hours

Only the encoder's first layer (input rows) and the decoder's last layer (output rows) are
per-channel; the trunk is `embed_dim`-sized and unchanged. Dropping rows 8, 9 from both
and warm-starting is a slice, not surgery. **Proof, one short arm:** the 99-channel model's
first validation must land near the base's single-step loss re-taken on the 99 subset —
not near initialisation. If it does, F becomes a few-epoch fine-tune and every later arm
(B/C list, EMA) is cheap; if not, run from scratch and say so. Do the from-scratch arm
*eventually* for a clean claim, not first.

## 5. 🟡 Port D — enable EMA in the F arm; it is not a confound

EMA is a shadow; it does not alter the trajectory. Enabling it in the F arm yields **raw and
EMA checkpoints from one run**, a free within-run A/B that does not break §7's
single-variable rule. Size `decay` from a window in epochs: at 1368 updates/epoch, a
≈2.5-epoch window is `1 − 1/(2.5·1368) ≈ 0.9997`, not the copied `0.999` (0.7 epoch).
Assert `save_checkpoint: "legacy"` in the log as the handoff says.

## 6. ⚠ Adversarial read of F itself — what it removes, and the fallback to pre-register

The measured defect is the **fill region** (62–72 % of the globe constant by
construction), i.e. a masking problem; dropping the channels is the *simplest* remedy, not
the only one. The alternative that keeps land memory: mask the loss over the fill region
and reset fill cells to their constant in the feedback path (preprocessor). Bigger code
change, smaller science change. F is decided, so **run F — but pre-register the check that
tells the two apart**: on the 99-channel model vs the 101-baseline restricted to the same
99, compare **`PRECT`, `TREFHT`, `RHREFHT`, `TMQ` over LAND cells** at the scorecard leads
and at 336 h. If land precipitation/temperature get *worse* while the global median gets
better, the soil-moisture feedback was doing work, and the masked variant is the next arm.
That is the experiment F actually is; the 99-channel median alone cannot see it.

The 99-channel rebaseline (`k56_metrics.h5` re-take, no re-run) goes **before** any new
number exists — the handoff already says this; keep it.

## 7. Order I would set

1. F code fix (§1) + `PORT_F_CHANNELS_OK` test + gate on **both** launchers.
2. 99-channel rebaseline from `k56_metrics.h5` (medians *and* the land-only §6 panels).
3. Surgical-transfer proof arm (§4), EMA on (§5).
4. F arm proper (fine-tune if 3 passes, else from scratch), EMA on.
5. Long rollout to divergence on the F model — the causal test for the ~500-step blow-up.
   Port A only if `PRECT`/`RHREFHT`/humidity are seen going negative *in that rollout*.
6. B+C as one explicit-list arm (§3), `W_max = 30`, after jesswan sees the table.
7. E and the `Z3_l17` prognostic question → jesswan, together.

Protocol unchanged: fixed leads, RMSE **and** ACC, 1.92 % bar, pre-registered read-out,
`validation loss` never used for long-lead claims.
