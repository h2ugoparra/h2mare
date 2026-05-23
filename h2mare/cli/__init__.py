"""H2GIS command-line interface."""

import typer

from h2mare.cli.catalog import catalog
from h2mare.cli.compile import compile
from h2mare.cli.main import run
from h2mare.cli.nc2zarr import convert
from h2mare.cli.zarr2parquet import parquet

app = typer.Typer(
    name="h2mare",
    help="Climate and ocean data pipeline — download, convert, and inspect.",
    no_args_is_help=True,
)

app.command("run", help="Download and convert data for one or more variable keys.")(run)
app.command(
    "convert", help="Convert downloaded NetCDF/GRIB files to Zarr (no download)."
)(convert)
app.command("catalog", help="Inspect ZarrCatalog metadata for a variable.")(catalog)
app.command(
    "compile",
    help="Merge per-variable Zarr stores into the unified h2ds compiled dataset.",
)(compile)
app.command(
    "parquet",
    help="Convert compiled Zarr stores to Hive-partitioned Parquet.",
)(parquet)
