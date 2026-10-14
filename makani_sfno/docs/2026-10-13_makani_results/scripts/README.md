# Scripts behind `../2026-10-13_makani_results.md`

Stdlib-only Python 3.6 (runs on a Polaris login node; reads text artifacts only). Paths default to this directory;
override with `MAKANI_RESULTS_WORK=<dir>`.

1. `parse_epochs.py <training log> [keys...]` -> per-epoch TSV (`epochs/*.tsv`; `epochs/all.json` merges them,
   with F-scratch and F-depth4 joined across their resume boundary).
2. `score_recompute.py` -> `arms.json`, `pairs.json` (recomputes score A per rollout from
   `protocol_scores/*_channels.csv`, gates from `*_budget.txt`, census from `*_census.csv`).
3. `protocol_survival.json` / `A_screens.json`: tallies of `readout.log` headers and `screen_summary.csv` rows
   (produced inline during the analysis; the parsing rule is in the write-up's design section).
4. `gen_tables.py` -> `tables.md` (every table in the write-up). `make_figs.py <outdir>` -> the two SVGs.
   `check_svg.py <svg...>` -> geometric layout check (text overlap, canvas overflow, line-through-label).
5. `make_fig3.py <outdir>` -> `fig3_candidates.svg` and `table6.md` (the checkpoint choice, Result 14).
6. `make_fig4.py <outdir>` -> `fig4_precision_cross.svg`, `table7.md` (Result 15; caches its inputs in `cross_cells.json`).
7. `build_doc.py` splices `tables.md`, `table6.md` and `table7.md` into `results_template.md` -> `../2026-10-13_makani_results.md`.
