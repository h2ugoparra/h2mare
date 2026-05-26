"""
h2mare run — download and convert pipeline.

Downloads raw data from configured providers and converts it to Zarr.
When no dates are given the pipeline infers what is missing from the
local store and downloads only the gap.

Examples
--------
    # First-time download — dates must be explicit
    uv run h2mare run -v sst --start-date 2021-01-01 --end-date 2021-12-31

    # Update existing store (dates inferred automatically)
    uv run h2mare run -v sst

    # Multiple variables at once
    uv run h2mare run -v seapodym -v mld -v o2 -v chl

    # Download only, skip Zarr conversion
    uv run h2mare run -v sst --no-convert

    # Skip the compile step after conversion
    uv run h2mare run -v sst --no-compile

    # Validate configuration without downloading
    uv run h2mare run -v sst --dry-run

    # Process all variables in config.yaml
    uv run h2mare run
"""

import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd
import typer
from loguru import logger

from h2mare.config import get_settings
from h2mare.downloader.registry import DOWNLOADER_REGISTRY
from h2mare.pipeline_manager import PipelineManager

app = typer.Typer()


def run(
    vars: Optional[List[str]] = typer.Option(
        None,
        "--vars",
        "-v",
        help=(
            "Variable key(s) to process (repeat for multiple: -v sst -v ssh). "
            "Defaults to all keys in config.yaml."
        ),
    ),
    start_date: Optional[str] = typer.Option(
        None,
        "--start-date",
        help="Start date (YYYY-MM-DD). Must be paired with --end-date.",
    ),
    end_date: Optional[str] = typer.Option(
        None,
        "--end-date",
        help="End date (YYYY-MM-DD). Must be paired with --start-date.",
    ),
    store_path: Optional[Path] = typer.Option(
        None,
        "--store-path",
        help="Override the Zarr store root (defaults to STORE_ROOT from .env).",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        is_flag=True,
        help="Plan the download and log tasks without fetching any data.",
    ),
    no_convert: bool = typer.Option(
        False,
        "--no-convert",
        is_flag=True,
        help="Download raw files but skip Zarr conversion and compile.",
    ),
    no_compile: bool = typer.Option(
        False,
        "--no-compile",
        is_flag=True,
        help="Skip the compile step (h2ds dataset merge) after Zarr conversion.",
    ),
    no_parquet: bool = typer.Option(
        False,
        "--no-parquet",
        is_flag=True,
        help=(
            "Skip the Zarr → Parquet conversion step after compilation. "
            "Implied automatically by --no-compile and --no-convert."
        ),
    ),
    no_backup: bool = typer.Option(
        False,
        "--no-backup",
        is_flag=True,
        help="Skip both backup steps: zarr files are not copied to the local store and Parquet is not copied to the remote store.",
    ),
    no_zarr_backup: bool = typer.Option(
        False,
        "--no-zarr-backup",
        is_flag=True,
        help="Skip copying compiled zarr files to the local backup store.",
    ),
    no_parquet_backup: bool = typer.Option(
        False,
        "--no-parquet-backup",
        is_flag=True,
        help="Skip copying the Parquet output to the remote store.",
    ),
    zarr_backup_dir: Optional[Path] = typer.Option(
        None,
        "--zarr-backup-dir",
        help="Override destination directory for the zarr backup.",
    ),
    parquet_backup_dir: Optional[Path] = typer.Option(
        None,
        "--parquet-backup-dir",
        help="Override destination for the Parquet backup.",
    ),
) -> None:
    """Download and convert climate/ocean data for one or more variable keys."""

    log_path = get_settings().LOGS_DIR / "h2mare.log"
    logger.add(log_path, level="INFO")
    logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)

    # Validate date arguments
    if bool(start_date) ^ bool(end_date):
        typer.echo(
            "Error: --start-date and --end-date must be provided together.", err=True
        )
        raise typer.Exit(code=1)

    if start_date and end_date:
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)
        if start_ts >= end_ts:
            typer.echo(
                f"Error: --start-date ({start_date}) must be before --end-date ({end_date}).",
                err=True,
            )
            raise typer.Exit(code=1)

    # Validate variable keys
    available = set(get_settings().app_config.variables.keys())
    selected = list(vars) if vars else list(available)
    unknown = set(selected) - available
    if unknown:
        typer.echo(
            f"Error: unknown variable key(s): {', '.join(sorted(unknown))}. "
            f"Available: {', '.join(sorted(available))}.",
            err=True,
        )
        raise typer.Exit(code=1)

    store_root = store_path or get_settings().STORE_ROOT
    if store_root is None:
        typer.echo(
            "Error: STORE_ROOT is not set. Define it in .env or pass --store-path.",
            err=True,
        )
        raise typer.Exit(code=1)

    success = PipelineManager(
        app_config=get_settings().app_config,
        registry=DOWNLOADER_REGISTRY,
        store_root=store_root,
        dry_run=dry_run,
        start_date=pd.Timestamp(start_date) if start_date else None,
        end_date=pd.Timestamp(end_date) if end_date else None,
        no_convert=no_convert,
        no_compile=no_compile,
        no_parquet=no_parquet,
        no_zarr_backup=no_backup or no_zarr_backup,
        no_parquet_backup=no_backup or no_parquet_backup,
        zarr_backup_dir=zarr_backup_dir,
        parquet_backup_dir=parquet_backup_dir,
    ).run(variables=selected)
    if not success:
        raise typer.Exit(code=1)


app.command()(run)
