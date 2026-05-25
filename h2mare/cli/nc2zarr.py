"""
nc2zarr — standalone NetCDF/GRIB → Zarr converter.

Converts downloaded raw files for one or more variable keys into processed
Zarr stores without running the full download pipeline. Use this when you
want to re-process existing downloads, convert files placed manually in the
downloads directory, or recover from a failed conversion step.

Examples
--------
Convert SST downloads from the default downloads directory (DOWNLOADS_DIR/.env):

    uv run h2mare convert -v sst

Convert multiple variables in one call:

    uv run h2mare convert -v sst -v ssh -v mld

Convert files from a custom input directory:

    uv run h2mare convert -v sst --in-dir /data/raw/CMEMS_SST

Notes
-----
- Raw files must match the regex ``pattern`` defined for the variable in
  ``config.yaml``.
- Zarr stores are written to ``STORE_ROOT/<local_folder>/`` (from ``.env``).
- Provenance sidecars (``*_prov.json``) are written alongside each Zarr
  only when a ``h2mare_manifest.json`` exists in the input directory (created
  automatically by CMEMSDownloader after a download run).
"""

from pathlib import Path
from typing import List, Optional

import typer
from loguru import logger

from h2mare.config import get_settings
from h2mare.format_converters.netcdf2zarr import Netcdf2Zarr

app = typer.Typer(
    help="Convert downloaded NetCDF/GRIB files to Zarr without re-downloading."
)


@app.command()
def convert(
    var_keys: List[str] = typer.Option(
        ...,
        "--vars",
        "-v",
        help=(
            "Variable key to convert (repeat for multiple: -v sst -v ssh). "
            "Must match a key defined in config.yaml — "
            "e.g. sst, ssh, mld, chl, fsle."
        ),
    ),
    input_root: Optional[Path] = typer.Option(
        None,
        "--in-dir",
        help=(
            "Root directory that contains the downloaded raw files. "
            "The variable's local_folder is appended automatically. "
            "Defaults to DOWNLOADS_DIR from .env."
        ),
    ),
) -> None:
    """Convert downloaded raw NetCDF/GRIB files to Zarr for one or more variables."""

    base_dir = input_root if input_root is not None else get_settings().DOWNLOADS_DIR

    log_path = get_settings().LOGS_DIR / f"{Path(__file__).stem}.log"
    logger.add(log_path, level="INFO")

    for var in var_keys:
        var_config = get_settings().app_config.variables.get(var)
        if var_config is None:
            logger.error(f"Unknown variable key '{var}' — skipping. Check config.yaml.")
            continue

        in_dir = base_dir / var_config.local_folder
        logger.info(f"Converting '{var}' from {in_dir}")
        Netcdf2Zarr(var, download_root=in_dir).run()


if __name__ == "__main__":
    app()
