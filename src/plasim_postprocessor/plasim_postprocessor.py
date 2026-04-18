#!/usr/bin/env python3
"""
plasim_postprocessor.py — Unified one-pass post-processor for PlaSim output.

Replaces burn.sh + Postprocessor class from postprocess_data.py for experiments
using postprocessor_version=3.0.

Runs once per particle inside the existing run_one_particle_one_step() parallelism.
Produces the T42 netCDF for scoring/PFS and optionally the Pangu-regridded input.

Usage:
    python3 plasim_postprocessor.py \
        --config /path/to/EXP15_postproc.yaml \
        --input plasim_output \
        --plasim_output /path/to/plasim_out.step_K.particle_I.nc \
        [--pangu_output /path/to/panguplasim_in.step_K.particle_I.nc]
"""

import argparse
import logging
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class PlasimPostprocessor:
    """One-pass post-processor for a single particle's PlaSim output.

    Stages:
      1. burn7   : PlaSim binary -> sigma-level netCDF
      2. Z500    : add geopotential height at 500 hPa (burn7 VTYPE=P or NCL)
      3. Precip  : accumulate precipitation (optional)
      4. Regrid  : regrid to Pangu grid (optional, when forecast_method=Pangu-Plasim)
    """

    def __init__(self, config_path: str):
        """Load YAML config and resolve all tool paths."""
        with open(config_path) as f:
            raw = yaml.safe_load(f)

        # Support both top-level and nested under 'postprocessing:' key
        cfg = raw.get("postprocessing", raw)

        # --- Variable lists (burn7 accepts variable name strings) ---
        self.upper_air_variables = cfg.get("upper_air_variables", [])
        self.surface_variables = cfg.get("surface_variables", [])
        self.land_variables = cfg.get("land_variables", [])
        self.ocean_variables = cfg.get("ocean_variables", [])

        # --- Z500 ---
        self.zg_source = cfg.get("zg_source", "burn7")
        self.pressure_levels = cfg.get("pressure_levels", [500])

        # --- Precipitation ---
        self.accumulate_precip = cfg.get("accumulate_precip", False)
        self.precip_accumulation_hours = cfg.get("precip_accumulation_hours", [6, 24])

        # --- Output targets ---
        outputs = cfg.get("outputs", {})
        pangu_cfg = outputs.get("pangu", {})
        self.pangu_enabled = pangu_cfg.get("enabled", False)
        self.pangu_grid_file = pangu_cfg.get("grid_file", "")
        self.pangu_variables = pangu_cfg.get("variables", [])

        # --- Tool paths ---
        self.burn7_wrapper = cfg.get("burn7_wrapper", "")
        self.cdo_path = cfg.get("cdo_path", "cdo")
        self.ncap2_path = cfg.get("ncap2_path", "ncap2")
        self.sg_filepath = cfg.get("sg_filepath", "")
        self.ncl_script_path = cfg.get("ncl_script_path", "")

        # Warn on deprecated fields that were used by old postprocessor
        for deprecated in ("upper_air_codes", "surface_codes", "land_codes", "ocean_codes"):
            if deprecated in cfg:
                logger.warning(
                    "Config field %r is not used by plasim_postprocessor.py "
                    "(burn7 uses variable names, not numeric codes). Field ignored.",
                    deprecated,
                )

    # -------------------------------------------------------------------------
    # Namelist helpers
    # -------------------------------------------------------------------------

    def _sigma_variables(self) -> list[str]:
        """Collect all variables for the sigma-level burn7 namelist.

        Precip-accumulation sentinels (variables matching pr_*h) are excluded;
        the base 'pr' variable is added instead if accumulation is requested.
        """
        _precip_pattern = re.compile(r"^pr_\d+h$")
        vars_out = []
        has_precip_sentinel = False

        all_vars = (
            list(self.upper_air_variables)
            + list(self.surface_variables)
            + list(self.land_variables)
            + list(self.ocean_variables)
        )

        for var in all_vars:
            if var == "zg" and self.zg_source == "burn7":
                continue  # zg handled separately via VTYPE=P
            if _precip_pattern.match(var):
                has_precip_sentinel = True
                continue  # skip accumulated-precip sentinel
            vars_out.append(var)

        if (self.accumulate_precip or has_precip_sentinel) and "pr" not in vars_out:
            vars_out.append("pr")  # add base precipitation for accumulation

        return vars_out

    def _write_sigma_namelist(self, tmp_dir: str) -> str:
        """Write sigma-level burn7 namelist; return its path."""
        nl_path = os.path.join(tmp_dir, f"burn_sigma_{os.getpid()}.nl")
        vars_str = ",".join(self._sigma_variables())
        with open(nl_path, "w") as f:
            f.write(f"code={vars_str},\n")
            f.write("MODLEV = 10,9,8,7,6,5,4,3,2,1,0\n")
            f.write("vtype=sigma,htype=g,mean=0,netcdf=1\n")
        return nl_path

    def _write_z500_namelist(self, tmp_dir: str) -> str:
        """Write pressure-level burn7 namelist for Z500; return its path."""
        nl_path = os.path.join(tmp_dir, f"burn_z500_{os.getpid()}.nl")
        hpa_str = ",".join(str(p) for p in self.pressure_levels)
        with open(nl_path, "w") as f:
            f.write("code=zg,\n")
            f.write(f"hpa={hpa_str}\n")
            f.write("vtype=p,htype=g,mean=0,netcdf=1\n")
        return nl_path

    def _run_burn7_wrapper(self, namelist: str, input_file: str, output_file: str) -> None:
        """Invoke the cluster burn7 wrapper script."""
        if not self.burn7_wrapper:
            raise ValueError("burn7_wrapper is not set in config")
        result = subprocess.run(
            [self.burn7_wrapper, namelist, input_file, output_file],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"burn7 failed (rc={result.returncode})\n"
                f"  namelist : {namelist}\n"
                f"  input    : {input_file}\n"
                f"  stdout   : {result.stdout.strip()}\n"
                f"  stderr   : {result.stderr.strip()}"
            )

    # -------------------------------------------------------------------------
    # Stage 1: burn7 sigma-level extraction
    # -------------------------------------------------------------------------

    def run_burn7(self, input_file: str, output_file: str) -> str:
        """Convert raw PlaSim binary -> netCDF on sigma levels.

        Generates the burn7 namelist from YAML config variables and runs
        the cluster burn7 wrapper (which handles module loads and LD_LIBRARY_PATH).

        Returns:
            output_file path (same as argument, written in place).
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            nl_path = self._write_sigma_namelist(tmp_dir)
            vars_preview = ",".join(self._sigma_variables())
            logger.info("burn7 (sigma): code=%s -> %s", vars_preview, output_file)
            self._run_burn7_wrapper(nl_path, input_file, output_file)
        return output_file

    # -------------------------------------------------------------------------
    # Stage 2: Z500 computation
    # -------------------------------------------------------------------------

    def compute_z500(self, plasim_output: str, input_file: str) -> str:
        """Compute geopotential height at pressure levels and merge into plasim_output.

        Routes to burn7 VTYPE=P (zg_source='burn7') or NCL hydro() (zg_source='ncl').

        Args:
            plasim_output: Path to the sigma-level netCDF produced by run_burn7().
            input_file: Path to the original PlaSim binary (needed for burn7 VTYPE=P).

        Returns:
            plasim_output path (Z500 merged in-place).
        """
        if self.zg_source == "burn7":
            return self._compute_z500_burn7(plasim_output, input_file)
        elif self.zg_source == "ncl":
            return self._compute_z500_ncl(plasim_output, input_file)
        else:
            raise ValueError(f"Unknown zg_source: {self.zg_source!r}. Use 'burn7' or 'ncl'.")

    def _compute_z500_burn7(self, plasim_output: str, input_file: str) -> str:
        """Compute Z500 via burn7 VTYPE=P and CDO-merge into plasim_output."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            nl_path = self._write_z500_namelist(tmp_dir)
            z500_file = os.path.join(tmp_dir, f"zg_plev_{os.getpid()}.nc")
            logger.info("burn7 (VTYPE=P): zg at %s hPa -> %s", self.pressure_levels, z500_file)
            self._run_burn7_wrapper(nl_path, input_file, z500_file)
            # CDO merge: add Z500 into the main sigma-level output
            merged = plasim_output + ".tmp_z500merge.nc"
            self._cdo(["merge", plasim_output, z500_file, merged])
            os.replace(merged, plasim_output)
        return plasim_output

    def _compute_z500_ncl(self, plasim_output: str, input_file: str) -> str:
        """Compute Z500 via NCL hydro() for migration-equivalence testing."""
        if not self.ncl_script_path:
            raise ValueError("ncl_script_path must be set in config when zg_source='ncl'")
        if not self.sg_filepath:
            raise ValueError("sg_filepath must be set in config when zg_source='ncl'")

        with tempfile.TemporaryDirectory() as tmp_dir:
            # First extract sigma-level zg from the burn7 output (already present)
            # then run NCL hydro() for pressure-level interpolation
            ncl_output = os.path.join(tmp_dir, f"zg_ncl_{os.getpid()}.nc")
            ncl_cmd = [
                "ncl",
                f"input_file={plasim_output!r}",
                f"output_file={ncl_output!r}",
                f"sg_filepath={self.sg_filepath!r}",
                self.ncl_script_path,
            ]
            result = subprocess.run(ncl_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(
                    f"NCL Z500 computation failed (rc={result.returncode})\n"
                    f"  stderr: {result.stderr.strip()}"
                )
            # Merge NCL output into the main file
            merged = plasim_output + ".tmp_ncl_merge.nc"
            self._cdo(["merge", plasim_output, ncl_output, merged])
            os.replace(merged, plasim_output)
        return plasim_output

    # -------------------------------------------------------------------------
    # Stage 3: Precipitation accumulation
    # -------------------------------------------------------------------------

    def accumulate_precipitation(self, plasim_output: str) -> str:
        """Compute accumulated precipitation via CDO runmean.

        Computes running accumulations for each configured window (e.g. 6h, 24h)
        and appends them to the output file.  Only runs if accumulate_precip=True.

        Returns:
            plasim_output path (updated in place, or unchanged if skipped).
        """
        if not self.accumulate_precip:
            return plasim_output

        with tempfile.TemporaryDirectory() as tmp_dir:
            accum_files = []

            for hours in self.precip_accumulation_hours:
                accum_file = os.path.join(tmp_dir, f"pr_{hours}h_{os.getpid()}.nc")
                # Select base precipitation, compute running sum, rename variable
                pr_tmp = os.path.join(tmp_dir, f"pr_base_{hours}h.nc")
                self._cdo(["selname,pr", plasim_output, pr_tmp])
                self._cdo([f"runsum,{hours}", pr_tmp, accum_file])
                # Rename variable to pr_{hours}h
                renamed = os.path.join(tmp_dir, f"pr_{hours}h_renamed.nc")
                self._cdo([f"chname,pr,pr_{hours}h", accum_file, renamed])
                accum_files.append(renamed)

            if accum_files:
                merged = plasim_output + ".tmp_precip_merge.nc"
                self._cdo(["merge", plasim_output] + accum_files + [merged])
                os.replace(merged, plasim_output)

        return plasim_output

    # -------------------------------------------------------------------------
    # Stage 4: Pangu regridding (optional)
    # -------------------------------------------------------------------------

    def regrid_to_pangu(self, plasim_output: str, pangu_output: str) -> str:
        """Regrid T42 -> Pangu grid (0.25°) via CDO remapbil.

        Selects only the Pangu-required variables and writes a separate file.
        Only called when outputs.pangu.enabled=true.

        Returns:
            pangu_output path.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Select Pangu variables
            selected = os.path.join(tmp_dir, f"pangu_vars_{os.getpid()}.nc")
            if self.pangu_variables:
                vars_str = ",".join(self.pangu_variables)
                self._cdo([f"selname,{vars_str}", plasim_output, selected])
            else:
                selected = plasim_output  # use all variables

            # Regrid to Pangu grid
            if not self.pangu_grid_file:
                raise ValueError("pangu_grid_file must be set in config when outputs.pangu.enabled=true")
            logger.info("Regridding to Pangu grid: %s -> %s", selected, pangu_output)
            self._cdo([f"remapbil,{self.pangu_grid_file}", selected, pangu_output])

        return pangu_output

    # -------------------------------------------------------------------------
    # Orchestrator
    # -------------------------------------------------------------------------

    def run(
        self,
        input_file: str,
        plasim_output: str,
        pangu_output: str | None = None,
    ) -> dict:
        """Full pipeline: burn7 -> Z500 -> precip -> (optional) regrid.

        Args:
            input_file:     Path to raw PlaSim binary (plasim_output from QDMC).
            plasim_output:  Destination path for T42 netCDF output.
            pangu_output:   Destination path for Pangu-grid netCDF (or None to skip).

        Returns:
            dict with keys 'plasim' and 'pangu' (None if not produced).
        """
        # Stage 1: burn7 sigma-level extraction
        self.run_burn7(input_file, plasim_output)

        # Stage 2: Z500 computation
        self.compute_z500(plasim_output, input_file)

        # Stage 3: Precipitation accumulation
        self.accumulate_precipitation(plasim_output)

        # Stage 4: Pangu regridding
        result = {"plasim": plasim_output, "pangu": None}
        if pangu_output is not None and self.pangu_enabled:
            self.regrid_to_pangu(plasim_output, pangu_output)
            result["pangu"] = pangu_output
        elif pangu_output is not None and not self.pangu_enabled:
            logger.debug("pangu_output provided but outputs.pangu.enabled=false; skipping regrid")

        logger.info("Post-processing complete: plasim=%s pangu=%s", plasim_output, result["pangu"])
        return result

    # -------------------------------------------------------------------------
    # CDO helper
    # -------------------------------------------------------------------------

    def _cdo(self, args: list[str]) -> None:
        """Run a CDO command; args is a list of CDO operator+args and files."""
        cmd = [self.cdo_path, "-s", "-O"] + args
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"CDO failed (rc={result.returncode}): {' '.join(cmd)}\n"
                f"  stderr: {result.stderr.strip()}"
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Unified PlaSim post-processor (postprocessor_version=3.0)"
    )
    p.add_argument(
        "--config",
        required=True,
        help="Path to postprocessing YAML config (PATH_POSTPROC_CONFIG)",
    )
    p.add_argument(
        "--input",
        required=True,
        help="Path to raw PlaSim binary output (plasim_output file)",
    )
    p.add_argument(
        "--plasim_output",
        required=True,
        help="Destination path for T42 sigma-level netCDF",
    )
    p.add_argument(
        "--pangu_output",
        default=None,
        help="Destination path for Pangu-grid netCDF (optional)",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    postprocessor = PlasimPostprocessor(args.config)
    postprocessor.run(
        input_file=args.input,
        plasim_output=args.plasim_output,
        pangu_output=args.pangu_output,
    )


if __name__ == "__main__":
    main()
