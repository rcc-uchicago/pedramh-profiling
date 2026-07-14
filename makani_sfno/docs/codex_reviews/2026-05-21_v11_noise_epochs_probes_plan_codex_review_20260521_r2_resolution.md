# Round 2 resolution

## Applied
- P2-a warm-start rationale — corrected to "GB=8 lr=1e-4" templates, dropped wrong "re-warm to 8e-4" claim
- P2-b pr_6h ACC — downgraded from "comparable" to "secondary diagnostic only" per render_eval_figures.py:28,278
- P2-c CKPT existence preflight — added `test -s "$CKPT"` before each eval invocation in §5
- P2-d login-node convention — added "run from login1/login2/login3" note per submit_beta1_chains.sh:7
- Suggested edit — added §4.5 "Artifacts to create" with exact filenames + config keys for both YAMLs + SLURMs
