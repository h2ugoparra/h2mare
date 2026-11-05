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

How it transfers:

- Only `.nc` files in the dataset folder are considered; other files sharing it (AVISO's report PDF) are ignored. A NetCDF file whose name yields no date is skipped with a warning rather than downloaded.
- Each file is written to a `<name>.part` sidecar and renamed only once its size matches the server's, so a dropped connection never leaves a truncated `.nc` for the convert step to read.
- A `.part` left by a dropped transfer — this run's or a killed one — is **resumed** with FTP `REST` instead of refetched (the eddy trajectory files take hours on a slow link). Without a remote size the transfer starts over.
- FTP connections time out after 120 s of silence (`FTP_TIMEOUT_S`), so a dead connection fails and is retried rather than hanging the run.
- REP and NRT files are fetched into separate staging directories. The download plan (also on `dry_run`) logs the requested range and each file with the dates it covers.

The downloader holds an FTP control connection; `close()` releases it. `PipelineManager` does this for you, and every downloader is a context manager (`with AVISODownloader("fsle") as dl: ...`).

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
| `get_nrt_availability()` | CMEMS, AVISO | The same for the near-real-time (NRT) product, or `None` when the entry has no `dataset_id_nrt` |
| `close()` | all | Release any connection the downloader holds (the AVISO FTP session; a no-op for the others) |

`h2mare.downloader` also exports the module-level CMEMS helpers the class uses: `cmems_download_subset(dataset_id, start, end, output_dir, *, variables=None, bbox=None, depth_range=None)` (`copernicusmarine.subset`), `cmems_download_original(dataset_id, start, end, output_dir)` (`copernicusmarine.get`, full files) and `generate_copernicus_patterns(start, end)` (the `get` filename filters for a date range).

### Date inference

When `start_date` / `end_date` are `None`, the range comes from
`h2mare.storage.coverage.resolve_date_range(var_key, start, end)`, which reads the
existing Zarr store's coverage through `ZarrCatalog` and returns only the gap after
its last date. When nothing new has been published yet the run logs that the
variable is up to date. A requested range outside the product's coverage is logged
as a warning instead.
