# Downloaders

Three downloader classes share a common interface. All inherit from `BaseDownloader` and are selected automatically by `PipelineManager` based on the `source` field in `config.yaml` (`downloader/registry.py`, `DOWNLOADER_REGISTRY`).

## Constructor

```python
CMEMSDownloader(          # same for AVISODownloader, CDSDownloader
    var_key,
    *,
    app_config=None,
    store_root=None,
    download_root=None,
)
```

| Parameter | Default | Description |
|---|---|---|
| `var_key` | — | Variable key; must exist in `config.yaml` |
| `app_config` | settings | Configuration to read `var_key` from |
| `store_root` | `STORE_ROOT` | Root of the Zarr stores, used to infer what is missing and to check the REP/NRT boundary against this variable's own store. A var_key's own `store_root` in config wins; see [Where a variable's store lives](../configuration.md#where-a-variables-store-lives) |
| `download_root` | `DOWNLOADS_DIR` | Root for the raw files |

---

## CMEMSDownloader

Downloads data from the Copernicus Marine Service using the `copernicusmarine` Python client.

```python
from h2mare.downloader.cmems_downloader import CMEMSDownloader

dl = CMEMSDownloader("sst")
dl.run(start_date="2024-01-01", end_date="2024-12-31")
```

Automatically switches from the reprocessed (`rep`) dataset to the near-real-time (`nrt`) dataset at the appropriate boundary date. The boundary is fetched from the CMEMS catalogue on each run.

**Used for:** `sst`, `ssh`, `mld`, `thetao`, `chl`, `seapodym`, `o2`

---

## AVISODownloader

Downloads files from AVISO via FTP. Credentials are read from `.env` (`AVISO_USERNAME`, `AVISO_PASSWORD`, `AVISO_FTP_SERVER`). The server offers plain FTP only, so they cross the network in cleartext: use a password unique to AVISO.

```python
from h2mare.downloader.aviso_downloader import AVISODownloader

dl = AVISODownloader("fsle")
dl.run(start_date="2024-01-01", end_date="2024-12-31")
```

**Used for:** `fsle`, `eddies`

---

## CDSDownloader

Downloads ERA5 data from the Copernicus Climate Data Store using `cdsapi`.

```python
from h2mare.downloader.cds_downloader import CDSDownloader

dl = CDSDownloader("atm-instante")
dl.run(start_date="2024-01-01", end_date="2024-12-31")
```

**Used for:** `atm-instante`, `atm-accum-avg`, `radiation`, `waves`

---

## `run()`

```python
dl.run(
    start_date=None,
    end_date=None,
    output_dir=None,
    dry_run=False,
    time_split=FilePeriod.MONTH,   # CMEMSDownloader, CDSDownloader
    # parallel=True,               # AVISODownloader instead of time_split
    # max_workers=2,
)
```

| Parameter | Description |
|---|---|
| `start_date`, `end_date` | Range to download. Inferred when `None`; see [Date inference](#date-inference) |
| `output_dir` | Where the raw files go. Defaults to the downloader's own directory under `download_root` |
| `dry_run` | Plan and log the downloads without fetching anything |
| `time_split` | CMEMS/CDS: split the request into monthly or yearly downloads |
| `parallel`, `max_workers` | AVISO: download files over this many FTP connections at once |

Returns `True` when downloads ran, `False` when there was nothing to fetch or on a dry run.

## Common interface

| Method | Classes | Description |
|---|---|---|
| `run(...)` | all | Download data for the given date range (above) |
| `get_rep_availability()` | CMEMS, AVISO | The `DateRange` the reprocessed (REP) product covers, from the CMEMS catalogue or the FTP listing. Cached after the first call |

### Date inference

When `start_date` / `end_date` are `None`, the range comes from
`h2mare.storage.coverage.resolve_date_range(var_key, start, end)`, which reads the
existing Zarr store's coverage through `ZarrCatalog` and returns only the gap after
its last date. When nothing new has been published yet the run logs that the
variable is up to date. A requested range outside the product's coverage is logged
as a warning instead.
