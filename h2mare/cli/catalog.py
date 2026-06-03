"""
h2mare catalog — inspect ZarrCatalog metadata for a variable.

Shows coverage, file count, variables, and per-dataset breakdown from the
local Parquet index without opening any Zarr files.

Examples
--------
    # Summary for SST
    uv run h2mare catalog sst

    # Summary for all configured variables
    uv run h2mare catalog --all

    # Show individual catalog rows
    uv run h2mare catalog sst --rows
"""

from typing import Optional

import pandas as pd
import typer

from h2mare.config import get_settings

app = typer.Typer()


def _print_catalog(var_key: str, show_rows: bool) -> None:
    from h2mare.storage.zarr_catalog import ZarrCatalog

    try:
        cat = ZarrCatalog(var_key)
    except Exception as e:
        typer.echo(f"  [{var_key}] Could not load catalog: {e}", err=True)
        return

    df = cat.df
    summary = cat.summary()
    cov = summary.get("time_coverage")

    typer.echo(f"\nZarrCatalog — {var_key.upper()}")
    typer.echo(f"  Files      : {summary['num_files']}")

    if cov and cov != "No data":
        typer.echo(f"  Coverage   : {cov.start.date()} → {cov.end.date()}")
    else:
        typer.echo("  Coverage   : No data")

    variables = summary.get("variables") or set()
    typer.echo(f"  Variables  : {', '.join(sorted(variables)) if variables else '—'}")
    typer.echo(f"  Timesteps  : {summary.get('total_timesteps', '—')}")
    typer.echo(f"  Store      : {summary['store_root']}")
    typer.echo(f"  Catalog    : {summary['catalog_path']}")
    last = summary.get("last_scanned")
    last_str = (
        last.strftime("%Y-%m-%d %H:%M:%S")
        if last is not None and pd.notna(last)
        else "—"
    )
    typer.echo(f"  Scanned    : {last_str}")

    if not df.empty and "dataset" in df.columns:
        typer.echo("\n  Dataset breakdown:")
        for dataset, group in df.groupby("dataset", sort=True):
            start = group["start_date"].min()
            end = group["end_date"].max()
            n_ts = (
                group["num_timesteps"].sum()
                if "num_timesteps" in group.columns
                else "—"
            )
            typer.echo(f"    {dataset}")
            typer.echo(f"      {start.date()} → {end.date()}  ({n_ts} timesteps)")

    if show_rows and not df.empty:
        cols = [
            c
            for c in ["filename", "dataset", "start_date", "end_date", "num_timesteps"]
            if c in df.columns
        ]
        typer.echo(f"\n  Rows:\n{df[cols].to_string(index=False)}")


def catalog(
    var_key: Optional[str] = typer.Argument(
        None,
        help="Variable key to inspect (e.g. sst, ssh). Omit with --all to show every variable.",
    ),
    all_vars: bool = typer.Option(
        False,
        "--all",
        "-a",
        is_flag=True,
        help="Show catalog summary for all variables configured in config.yaml.",
    ),
    show_rows: bool = typer.Option(
        False,
        "--rows",
        "-r",
        is_flag=True,
        help="Print individual catalog rows (filename, dataset, dates, timesteps).",
    ),
) -> None:
    """Inspect ZarrCatalog metadata: coverage, file count, and per-dataset breakdown."""

    if not var_key and not all_vars:
        typer.echo("Provide a variable key or use --all.", err=True)
        raise typer.Exit(code=1)

    keys = list(get_settings().app_config.variables.keys()) if all_vars else [var_key]

    for key in keys:
        if key not in get_settings().app_config.variables:
            typer.echo(
                f"Unknown variable key '{key}'. Available: {', '.join(get_settings().app_config.variables)}.",
                err=True,
            )
            continue
        _print_catalog(key, show_rows)


app.command()(catalog)
