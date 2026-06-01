"""
h2mare compile — merge per-variable Zarr stores into a unified h2ds dataset.

Reads the individual per-variable Zarr stores and interpolates them to a
common 0.25° daily grid, writing the result as the h2ds compiled dataset.
When no dates are given the step infers what is missing from the local store.

Examples
--------
    # Compile all available variables (dates inferred from store)
    uv run h2mare compile

    # Compile specific variables over a date range
    uv run h2mare compile -v sst -v ssh -v mld --start-date 2024-01-01 --end-date 2024-12-31

    # Compile with a custom store path
    uv run h2mare compile --store-path D:/GlobalData
"""

from pathlib import Path
from typing import List, Optional

import pandas as pd
import typer

from h2mare.config import get_settings
from h2mare.utils.logging import add_file_logger

app = typer.Typer()


def compile(
    vars: Optional[List[str]] = typer.Option(
        None,
        "--vars",
        "-v",
        help=(
            "Variable key(s) to compile (repeat for multiple: -v sst -v ssh). "
            "Defaults to all available keys."
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
    no_zarr_backup: bool = typer.Option(
        False,
        "--no-zarr-backup",
        is_flag=True,
        help="Skip copying compiled zarr files to the local backup store after writing.",
    ),
    zarr_backup_dir: Optional[Path] = typer.Option(
        None,
        "--zarr-backup-dir",
        help="Override destination directory for the zarr backup (defaults to local_store_root from settings).",
    ),
) -> None:
    """Merge per-variable Zarr stores into the unified h2ds compiled dataset."""

    log_path = get_settings().LOGS_DIR / "h2mare.log"
    add_file_logger(log_path)

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

    if vars:
        available = set(get_settings().app_config.variables.keys())
        unknown = set(vars) - available
        if unknown:
            typer.echo(
                f"Error: unknown variable key(s): {', '.join(sorted(unknown))}. "
                f"Available: {', '.join(sorted(available))}.",
                err=True,
            )
            raise typer.Exit(code=1)

    from h2mare.processing.compiler import Compiler

    Compiler(remote_store_root=store_path or get_settings().STORE_ROOT).run(
        start_date=start_date,
        end_date=end_date,
        var_keys=list(vars) if vars else None,
        no_zarr_backup=no_zarr_backup,
        zarr_backup_dir=zarr_backup_dir,
    )


app.command()(compile)
