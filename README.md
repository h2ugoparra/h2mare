# H2MARE - Geospatial Processing for Climate and Ocean Data

![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)
[![PyPI](https://img.shields.io/pypi/v/h2mare)](https://pypi.org/project/h2mare/)
[![Docs](https://img.shields.io/badge/docs-github%20pages-blue)](https://h2ugoparra.github.io/h2mare/)

A Python pipeline for downloading and preprocessing multi-source oceanographic and atmospheric data into analysis-ready formats. H2MARE streamlines the acquisition and harmonization of data from major climate and ocean observation services, optimized for large-scale spatiotemporal analysis.

## Features

- **Multi-source data integration**: Download and process data from CMEMS, AVISO, and ERA5.
- **Format conversion**: Automated conversion from NetCDF/GRIB to optimized Zarr and Parquet formats.
- **Data compilation**: Regrid and interpolate multi-resolution datasets to a common grid.
- **Point and geometry extraction**: Extract time series for specific locations or spatial features.

## Data Sources

H2MARE supports the following data providers. API keys and authentication are required for each.

- **[CMEMS](https://marine.copernicus.eu/)** — Copernicus Marine Service: satellite and in-situ ocean observations
- **[AVISO](https://www.aviso.altimetry.fr/en/home.html)** — Archiving, Validation and Interpretation of Satellite Oceanographic data
- **[CDS-ERA5](https://cds.climate.copernicus.eu/)** — ERA5 hourly atmospheric reanalysis (1940–present)

Refer to each provider's documentation for authentication setup before use.

## Installation

### Prerequisites

- Python >= 3.11
- [uv](https://docs.astral.sh/uv/) — fast Python package and project manager

### Install from PyPI

```bash
pip install h2mare
# or
uv add h2mare
```

That installs the pipeline: download, convert, compile, extract, Parquet. The
plotting helpers (`h2mare.utils.plot`, `ParquetIndexer.plot`) need the `viz`
extra, which adds cartopy, matplotlib and plotly:

```bash
pip install "h2mare[viz]"
# or
uv add "h2mare[viz]"
```

### Install from source

```bash
git clone https://github.com/h2ugoparra/h2mare.git
cd h2mare
uv sync
```

## Configuration

H2MARE requires two configuration files in your working directory before first use.

### 1. `config.yaml`

Defines variables, dataset IDs, bounding boxes, and processing parameters.

Start from [`config.example.yaml`](https://github.com/h2ugoparra/h2mare/blob/main/config.example.yaml) — copy it to `config.yaml` and edit. It holds four entries chosen to cover the shapes you are likely to need: a plain 2-D daily variable, a 3-D one with depth levels, an hourly packed ERA5 one, and the `h2ds` entry the compile step writes to.

The repository's own [`config.yaml`](https://github.com/h2ugoparra/h2mare/blob/main/config.yaml) configures 16 variables and is a **reference**, not a starting point — useful for seeing how a particular source is wired up, too heavy to copy wholesale. Every field is documented in the [configuration reference](https://h2ugoparra.github.io/h2mare/configuration/).

### 2. `.env`

```env
# Path to external or large-capacity storage for processed Zarr files
STORE_ROOT=/path/to/your/storage

# AVISO credentials (required for FSLE, Eddies)
AVISO_USERNAME=your_username
AVISO_PASSWORD=your_password
AVISO_FTP_SERVER=ftp-access.aviso.altimetry.fr
```

> **AVISO credentials cross the network in cleartext.** The AVISO server offers plain FTP only — it refuses `AUTH TLS` ("500 AUTH not understood", checked 2026-09-01) — so there is no FTPS option to enable. Use a password unique to AVISO, and prefer running downloads from a trusted network.

CMEMS credentials are configured via the `copernicusmarine` client. ERA5 / CDS credentials are configured via the `cdsapi` client. See the [CDS documentation](https://cds.climate.copernicus.eu/how-to-api) for setup.

> **Note:** Both files must be present in the directory where you run `h2mare`. You can also set the `H2MARE_ROOT` environment variable to point to a different directory containing them.

## Quick Start

```bash
# Download and process a single variable for a specific date range
uv run h2mare run -v sst --start-date 2021-01-01 --end-date 2021-12-31

# Multiple variables at once
uv run h2mare run -v seapodym -v mld -v o2 -v chl

# Infer missing dates from the existing store and download what's new
uv run h2mare run -v sst

# Download only (skip Zarr conversion)
uv run h2mare run -v sst --no-convert

# Validate configuration without downloading
uv run h2mare run -v sst --dry-run

# Process all configured variables
uv run h2mare run

# Check the stores for days that went missing without failing anything
uv run h2mare audit --all
```

Coverage elsewhere in the pipeline is a frontier — a start date and an end date — so a year holding
January 1st and December 31st reports itself complete however much is missing in between. `audit`
reads the time axes and reports the holes. See [the CLI docs](https://h2ugoparra.github.io/h2mare/cli/)
for the value check and for `known_gaps`, which excludes days the provider never published.

## Development

```bash
# Run the full test suite
uv run pytest tests/

# Run a single test file
uv run pytest tests/test_zarr_catalog.py -v

# Lint and format
uv run ruff check h2mare/
uv run ruff format h2mare/
```

## Built with

| Library | Role |
|---------|------|
| [xarray](https://xarray.dev/) | N-dimensional labelled arrays and NetCDF/Zarr I/O |
| [zarr](https://zarr.dev/) | Chunked, compressed array storage |
| [dask](https://www.dask.org/) | Parallel and out-of-core computation |
| [polars](https://pola.rs/) | Fast DataFrame engine for Parquet I/O |
| [duckdb](https://duckdb.org/) | In-process SQL for Parquet overlap resolution and scanning |
| [geopandas](https://geopandas.org/) | Geometry-based spatial extraction |
| [plotly](https://plotly.com/python/) | Interactive time-series and spatial visualizations |
| [copernicusmarine](https://pypi.org/project/copernicusmarine/) | CMEMS dataset access |
| [cdsapi](https://pypi.org/project/cdsapi/) | ERA5 / CDS dataset access |

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests on [GitHub](https://github.com/h2ugoparra/h2mare.git).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This project was developed under the framework of [COSTA project](https://costaproject.org/en/). This project relies on data from Copernicus Marine Service, AVISO, Copernicus Climate Data Store, and NOAA NCEI. We gratefully acknowledge these organizations for providing open access to their datasets.
