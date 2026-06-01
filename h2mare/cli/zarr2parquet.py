"""
h2mare parquet — convert compiled Zarr stores to Hive-partitioned Parquet.

Reads one or more variable Zarr stores (default: h2ds) and writes them as
month-partitioned Parquet files.  When no dates are given the command runs in
incremental mode: it appends genuinely new trailing dates and, mirroring the
compile step, backfills any var_key whose column lags behind its source
coverage (e.g. written as NaN while a faster variable advanced), merging the
caught-up data into the affected partitions.  Repeated runs only process what
is new.

Examples
--------
Convert the compiled h2ds store (dates inferred automatically):

    uv run h2mare parquet

Convert a specific date range:

    uv run h2mare parquet --start-date 1998-01-01 --end-date 1998-12-31

Convert a non-default variable:

    uv run h2mare parquet -v sst -v ssh

Add new variable columns to an existing h2ds Parquet store:

    uv run h2mare parquet --add-var thetao
    uv run h2mare parquet --add-var thetao --add-var o2

Write to a custom output directory:

    uv run h2mare parquet --out-dir D:/parquet_store

Read from a custom Zarr store root:

    uv run h2mare parquet --store-path D:/GlobalData
"""

from pathlib import Path
from typing import List, Optional

import typer
from loguru import logger

from h2mare.config import get_settings

app = typer.Typer(help="Convert compiled Zarr stores to Hive-partitioned Parquet.")


@app.command()
def parquet(
    var_keys: Optional[List[str]] = typer.Option(
        None,
        "--vars",
        "-v",
        help=(
            "Variable key(s) to convert (repeat for multiple: -v h2ds -v sst). "
            "Defaults to 'h2ds' when omitted."
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
    out_dir: Optional[Path] = typer.Option(
        None,
        "--out-dir",
        help=(
            "Root directory for Parquet output. "
            "Each variable is written to a <out-dir>/<var-key> sub-directory. "
            "Defaults to get_settings().PARQUET_DIR."
        ),
    ),
    store_path: Optional[Path] = typer.Option(
        None,
        "--store-path",
        help="Override the Zarr store root (defaults to STORE_ROOT from .env).",
    ),
    depth: Optional[float] = typer.Option(
        None,
        "--depth",
        help=(
            "Depth level in metres to select for variables with a depth dimension "
            "(e.g. thetao, o2). The nearest available level is chosen. "
            "Required for depth-aware variables; ignored for surface-only ones."
        ),
    ),
    add_var_keys: Optional[List[str]] = typer.Option(
        None,
        "--add-var",
        help=(
            "Variable key(s) whose compiled columns should be merged into the existing "
            "h2ds Parquet store (repeat for multiple: --add-var thetao --add-var o2). "
            "Reads variables_to_compile from config.yaml and JOINs them into every "
            "existing partition. Cannot be combined with -v."
        ),
    ),
    no_parquet_backup: bool = typer.Option(
        False,
        "--no-parquet-backup",
        is_flag=True,
        help="Skip copying the Parquet output to the remote store.",
    ),
    parquet_backup_dir: Optional[Path] = typer.Option(
        None,
        "--parquet-backup-dir",
        help="Override destination for the Parquet backup (defaults to STORE_ROOT/parquet).",
    ),
) -> None:
    """Convert compiled Zarr stores to Hive-partitioned Parquet for one or more variable keys."""

    log_path = get_settings().LOGS_DIR / "h2mare.log"
    logger.add(log_path, level="INFO")

    # ---- Validate date arguments ----
    if bool(start_date) ^ bool(end_date):
        typer.echo(
            "Error: --start-date and --end-date must be provided together.", err=True
        )
        raise typer.Exit(code=1)

    if start_date and end_date:
        import pandas as pd

        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)
        if start_ts >= end_ts:
            typer.echo(
                f"Error: --start-date ({start_date}) must be before --end-date ({end_date}).",
                err=True,
            )
            raise typer.Exit(code=1)

    if add_var_keys and var_keys:
        typer.echo("Error: --add-var and -v cannot be used together.", err=True)
        raise typer.Exit(code=1)

    available = set(get_settings().app_config.variables.keys())

    # ---- Resolve output root ----
    parquet_base = out_dir or get_settings().PARQUET_DIR

    from h2mare.format_converters.zarr2parquet import Zarr2Parquet

    # ---- add-var mode: merge new columns into existing h2ds Parquet ----
    if add_var_keys:
        unknown = set(add_var_keys) - available
        if unknown:
            typer.echo(
                f"Error: unknown variable key(s): {', '.join(sorted(unknown))}. "
                f"Available: {', '.join(sorted(available))}.",
                err=True,
            )
            raise typer.Exit(code=1)

        variables: list[str] = []
        for key in add_var_keys:
            v2c = get_settings().app_config.variables[key].variables_to_compile
            if not v2c:
                typer.echo(
                    f"Error: '{key}' has no variables_to_compile defined in config.yaml.",
                    err=True,
                )
                raise typer.Exit(code=1)
            variables.extend(v2c)

        logger.info(f"add-var: merging {variables} into h2ds Parquet under {parquet_base}")
        try:
            converter = Zarr2Parquet(
                var_key="h2ds",
                parquet_root=parquet_base,
                store_root=store_path,
            )
            converter.run(
                start_date=start_date,
                end_date=end_date,
                variables=variables,
            )
            if not no_parquet_backup:
                converter.sync_data(remote_root=parquet_backup_dir)
        except ValueError as e:
            logger.error(f"add-var failed: {e}")
        return

    # ---- Standard mode: convert one or more var_keys ----
    keys = list(var_keys) if var_keys else ["h2ds"]
    unknown = set(keys) - available
    if unknown:
        typer.echo(
            f"Error: unknown variable key(s): {', '.join(sorted(unknown))}. "
            f"Available: {', '.join(sorted(available))}.",
            err=True,
        )
        raise typer.Exit(code=1)

    for key in keys:
        try:
            converter = Zarr2Parquet(
                var_key=key,
                parquet_root=parquet_base,
                store_root=store_path,
            )
            converter.run(start_date=start_date, end_date=end_date, depth=depth)
            if not no_parquet_backup:
                converter.sync_data(remote_root=parquet_backup_dir)
        except ValueError as e:
            logger.error(f"Skipping '{key}': {e}")
            continue
