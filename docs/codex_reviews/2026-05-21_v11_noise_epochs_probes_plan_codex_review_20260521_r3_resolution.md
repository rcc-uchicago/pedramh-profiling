# Round 3 resolution

## Applied
- P2-a CKPT preflight scope — restructured both eval blocks to use `export` + `test -s` + `scripts/submit_eval.sh` so the preflight sees the same CKPT value
- P2-b noise σ history — narrowed "never tested above 0.05" to "never on the active lr8e4/minlr1e5 branch"; cited the retired-lr1p13e3 noise0p075 historical config
