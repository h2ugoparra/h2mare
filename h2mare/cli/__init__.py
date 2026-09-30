"""H2GIS command-line interface."""

import sys
import warnings

import typer
from loguru import logger

from h2mare.cli.audit import audit
from h2mare.cli.catalog import catalog
from h2mare.cli.compile import compile
from h2mare.cli.main import run
from h2mare.cli.nc2zarr import convert
from h2mare.cli.parquet2zarr import parquet2zarr
from h2mare.cli.zarr2parquet import parquet
from h2mare.config import get_settings
from h2mare.utils.logging import configure_logging

app = typer.Typer(
    name="h2mare",
    help="Climate and ocean data pipeline — download, convert, and inspect.",
    no_args_is_help=True,
)


def _use_utf8_console() -> None:
    """Switch stdout/stderr to UTF-8 so command output can carry non-ASCII.

    Windows hands Python a cp1252 console, which raises UnicodeEncodeError on
    the arrows and dashes the summaries print — killing the command mid-output.
    Reconfigure in place so sinks already bound to these streams follow along.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        encoding = (getattr(stream, "encoding", "") or "").lower()
        if reconfigure is not None and encoding.replace("-", "") != "utf8":
            reconfigure(encoding="utf-8", errors="replace")


def _silence_known_benign_warnings() -> None:
    """Filter warnings that are noise for every h2mare command, by message.

    Installed here rather than at module import: importing h2mare as a library
    must leave the interpreter's filters untouched, which
    ``tests/test_warning_filters.py`` enforces after six modules once installed
    blanket ignores at import and swallowed the AVISO missing-credentials
    warning. Each entry is pinned to an exact message so a *new* warning still
    reaches the console. ``pyproject.toml``'s ``filterwarnings`` covers the same
    ground under pytest; keep the two lists in step.
    """
    from zarr.errors import ZarrUserWarning

    # zarr 3.x warns that consolidated metadata sits outside the Zarr v3 spec.
    # It is raised from `consolidate_metadata`, so it is a write-path warning
    # only: every `to_zarr` triggers it (xarray consolidates by default) and no
    # read ever does. h2mare cannot be bitten by what it warns about, because
    # every `open_zarr` call site passes `consolidated=False` and so never reads
    # the consolidated block. Appends re-consolidate, so it cannot go stale for
    # an outside reader either. Drops out if xarray stops consolidating by
    # default or zarr folds this into the spec.
    warnings.filterwarnings(
        "ignore",
        message="Consolidated metadata is currently not part in the Zarr format 3 specification",
        category=ZarrUserWarning,
    )

    # dask's reduction `divide` (numpy_compat.divide, the numpy-2.x fallback)
    # computes `total / n` for var/std. A 3x3 rolling std whose window lies
    # wholly on land has n == 0, so it evaluates 0/0 -> NaN and numpy flags the
    # invalid operation. NaN is the correct answer there, and `clip_land_data`
    # makes such windows certain on every run over the configured bbox, so the
    # warning fires whatever the data looks like and reports nothing about it.
    # Real holes in a store are what `h2mare audit` is for, not this.
    #
    # Pinned to dask's module so h2mare's own 0/0 or x/0 still surfaces: the
    # `_std` chain is lazy (see process_sst), so it is raised at compute time,
    # far from the expression, and a local errstate around the call site would
    # not catch it. `mean` is unaffected — dask guards that one already.
    warnings.filterwarnings(
        "ignore",
        message="invalid value encountered in divide",
        category=RuntimeWarning,
        module=r"dask\.array\.numpy_compat",
    )


@app.callback()
def _configure() -> None:
    """
    Configure the console and logging once for every h2mare command, and say
    which project, config and store root the command resolved.
    """
    _use_utf8_console()
    _silence_known_benign_warnings()
    configure_logging()
    logger.info(get_settings().describe_resolution())


app.command("run", help="Download and convert data for one or more variable keys.")(run)
app.command(
    "convert", help="Convert downloaded NetCDF/GRIB files to Zarr (no download)."
)(convert)
app.command("catalog", help="Inspect ZarrCatalog metadata for a variable.")(catalog)
app.command(
    "audit",
    help="Report days missing from the middle of a store's own time span.",
)(audit)
app.command(
    "compile",
    help="Merge per-variable Zarr stores into the unified h2ds compiled dataset.",
)(compile)
app.command(
    "parquet",
    help="Convert compiled Zarr stores to Hive-partitioned Parquet.",
)(parquet)
app.command(
    "parquet2zarr",
    help="Rebuild per-period Zarr files from a Parquet store.",
)(parquet2zarr)
