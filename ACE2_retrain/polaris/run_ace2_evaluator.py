"""The missing command-line entrypoint for fme's inference EVALUATOR.

WHY THIS FILE EXISTS
--------------------
`fme.ace.inference.evaluator` has a `main()` (evaluator.py:288) but nothing
wires it to a CLI: there is no `evaluator/__main__.py`, and `pyproject.toml` has
no `[project.scripts]` / `entry_points` block at all. `python -m
fme.ace.inference` resolves to `fme/ace/inference/__main__.py`, which imports
`from .inference import main` — the standalone INFERENCE path, not the
evaluator.

That distinction is load-bearing here. The evaluator is the entry point whose
`loader:` / `aggregator:` schema matches `config_polaris.yaml`'s `inference:`
block, and the only one that computes the climate scorecard (time-mean bias,
zonal and seasonal means, paired spectra, per-lead RMSE). The standalone
inference path wants `initial_condition:` + `forcing_loader:` instead, and its
aggregator has no `log_histograms`, so under `dacite.Config(strict=True)` the
repo's own config cannot load into it.

Deliberately a verbatim structural copy of `fme/ace/train/__main__.py` — same
parser, same `Distributed.context()` wrapper — so it inherits `--override` and
the distributed teardown rather than inventing either.

    python run_ace2_evaluator.py <config.yaml> --override key=value ...
"""

from fme.ace.inference.evaluator import main
from fme.core.cli import get_parser
from fme.core.distributed.distributed import Distributed

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    with Distributed.context():
        main(args.yaml_config, override_dotlist=args.override)
