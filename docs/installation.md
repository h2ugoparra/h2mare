# Installation

## Prerequisites

- Python >= 3.11
- [uv](https://docs.astral.sh/uv/) — fast Python package and project manager
- Sufficient disk space for downloaded datasets (varies by region and time range)
- API credentials for the data providers you intend to use

## Install from PyPI

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

## Install from source

```bash
git clone https://github.com/h2ugoparra/h2mare.git
cd h2mare
uv sync
```

Include development dependencies (pytest, ruff, and the `viz` extra the plotting tests need):

```bash
uv sync --dev
```

## Verify installation

```bash
h2mare --help
```

Expected output lists the available commands: `run`, `compile`, `convert`, `catalog`

## First-time setup

H2MARE requires two files in your working directory before running any command.

### `config.yaml`

Defines variables, dataset IDs, bounding boxes, and processing parameters. Start from the example, which holds four entries covering the shapes you are likely to need — a plain 2-D daily variable, a 3-D one with depth levels, an hourly packed ERA5 one, and the `h2ds` entry the compile step writes to:

```bash
curl -o config.yaml https://raw.githubusercontent.com/h2ugoparra/h2mare/main/config.example.yaml
```

The repository's own `config.yaml` configures 16 variables. It is a **reference** — useful for seeing how a particular source is wired up, too heavy to copy wholesale. Every field is documented in the [configuration reference](configuration.md).

### `.env`

Create a `.env` file with at minimum:

```env
STORE_ROOT=/path/to/your/storage
```

See [Configuration](configuration.md#env) for the full list of variables and credentials.

### Where to place these files

By default, h2mare searches for `config.yaml` by walking up from your current working directory. As long as you run `h2mare` from inside your project tree, no extra configuration is needed.

If auto-detection fails — for example, you run `h2mare` from an unrelated directory, or a script elsewhere imports h2mare — set `H2MARE_ROOT` to the directory containing your `config.yaml` and `.env`:

```env
H2MARE_ROOT=/path/to/your/h2mare/project
```

Without it, h2mare falls back to `~/.h2mare`, which holds no `config.yaml`, and commands will fail.

## Data storage layout

```
$PROJECT_ROOT/
├── data/raw/
│   └── downloads/<local_folder>/    # raw NetCDF / GRIB files from downloaders
├── data/interim/                    # temporary scratch files (checkpoints, tmp Zarr)
├── data/processed/
│   ├── zarr/<local_folder>/         # per-variable Zarr stores (fallback when STORE_ROOT is not set)
│   ├── parquet/                     # Hive-partitioned Parquet store
│   └── metadata/                    # ZarrCatalog Parquet indices
└── logs/                            # pipeline log files

$STORE_ROOT/<local_folder>/          # per-variable Zarr stores (when STORE_ROOT is set)
```

`local_folder` is defined per variable in `config.yaml` (e.g. `CMEMS_SST`, `CMEMS_SSH`). When `STORE_ROOT` is set, Zarr output goes there; otherwise it falls back to `data/processed/zarr/`.

These directories are created on first write, not at import — so a project root only grows the branches it actually uses. Call `get_settings().ensure_directories()` to scaffold the whole tree up front.
