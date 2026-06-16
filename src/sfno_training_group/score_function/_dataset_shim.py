"""Dataset-like adapter for `SphericalFourierNeuralOperatorNet_v2.__init__`.

The group SFNO_v2 constructor (``networks/modulus_sfno/sfnonet.py:740-755``)
reads three attrs from the ``dataset`` arg to compute channel counts:
    in_chans  = len(dataset.variable_list_in) + len(dataset.constant_boundary_variables)
    out_chans = len(dataset.variable_list_out)

The shim provides those plus a couple of others (varying_boundary_variables,
surface_variables, diagnostic_variables) that downstream code (and our wrapper)
consults for channel-count bookkeeping. We do NOT instantiate the full
``GetDataset`` (which would require the whole training-data pipeline).

Channel-list construction follows ``GetDataset._get_variable_list`` in
``utils/data_loader_multifiles.py:632-643``:

    For each upper_air var:
        if use_sigma_levels and var not in ('zg','geopotential_height'):
            keys += [f"{var}_{sigma_value}" for sigma_value in sigma_levels]
        else:
            keys += [f"{var}_{int(level)}.0" for level in levels]
    variable_list_out = upper_air_keys + surface_variables + diagnostic_variables
    variable_list_in  = upper_air_keys + surface_variables + varying_boundary_variables
"""

from __future__ import annotations

from typing import Iterable


class DatasetShim:
    """Minimal dataset-like adapter sufficient for SFNO_v2 construction."""

    def __init__(
        self,
        upper_air_variables: Iterable[str],
        surface_variables: Iterable[str],
        diagnostic_variables: Iterable[str],
        varying_boundary_variables: Iterable[str],
        constant_boundary_variables: Iterable[str],
        sigma_levels: Iterable[float],
        levels: Iterable[float],
        use_sigma_levels: bool,
    ) -> None:
        self.upper_air_variables = list(upper_air_variables)
        self.surface_variables = list(surface_variables)
        self.diagnostic_variables = list(diagnostic_variables)
        self.varying_boundary_variables = list(varying_boundary_variables)
        self.constant_boundary_variables = list(constant_boundary_variables)
        self.sigma_levels = list(sigma_levels)
        self.levels = list(levels)
        self.use_sigma_levels = bool(use_sigma_levels)

        # Build keys following _get_variable_list (line 632-643).
        self._upper_air_keys: list[str] = []
        for var in self.upper_air_variables:
            if self.use_sigma_levels and var not in ("zg", "geopotential_height"):
                for lev in self.sigma_levels:
                    self._upper_air_keys.append(f"{var}_{lev}")
            else:
                for lev in self.levels:
                    self._upper_air_keys.append(f"{var}_{int(lev)}.0")

        self.variable_list_out = (
            self._upper_air_keys + self.surface_variables + self.diagnostic_variables
        )
        self.variable_list_in = (
            self._upper_air_keys + self.surface_variables + self.varying_boundary_variables
        )

    @property
    def upper_air_keys(self) -> list[str]:
        return list(self._upper_air_keys)

    @property
    def in_chans(self) -> int:
        return len(self.variable_list_in) + len(self.constant_boundary_variables)

    @property
    def out_chans(self) -> int:
        return len(self.variable_list_out)
