"""sfno_eval — scoring + climatology for the PlaSim SFNO emulator.

Modules:
  - ``metrics`` — lat-weighted RMSE, ACC, bias maps. No earth2studio
    runtime dep (round 1 fix 3).
  - ``climatology`` — build per-(month, day, hour_quarter) climatology
    from the 100 training files (sim52_full/train/), Welford-style.
  - ``nc_io`` — xarray helpers for our NetCDF format.
"""

__all__: list[str] = []
