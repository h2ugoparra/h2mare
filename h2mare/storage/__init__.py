from .coverage import get_store_coverage, split_time_range
from .parquet_catalog import ParquetCatalog
from .parquet_helpers import (
    aggregate_by_space_time,
    aggregate_by_time,
    aggregate_by_time_stats,
    polars_float64_to_float32,
)
from .parquet_indexer import ParquetIndexer
from .parquet_store import ParquetStore, iter_store_parquet_files
from .recovery import recover_parquet_store, recover_zarr_store
from .storage import write_append_zarr
from .var_routing import catalog_for_var, compiled_var_key, coverage_for_var
from .xarray_helpers import (
    chunk_dataset,
    convert360_180,
    ds_float64_to_float32,
    rename_dims,
    rename_source_vars,
    unified_time_chunk,
    xr_float64_to_float32,
)
from .zarr_catalog import ZarrCatalog
from .zarr_index import ZarrIndex
from .zarr_reader import ZarrReader
from .zarr_scanner import ZarrDirectoryScanner

__all__ = [
    "ZarrCatalog",
    "ZarrIndex",
    "ZarrReader",
    "ZarrDirectoryScanner",
    "ParquetIndexer",
    "ParquetStore",
    "ParquetCatalog",
    "catalog_for_var",
    "compiled_var_key",
    "coverage_for_var",
    "get_store_coverage",
    "split_time_range",
    "aggregate_by_space_time",
    "aggregate_by_time",
    "aggregate_by_time_stats",
    "polars_float64_to_float32",
    "chunk_dataset",
    "rename_dims",
    "rename_source_vars",
    "unified_time_chunk",
    "xr_float64_to_float32",
    "ds_float64_to_float32",  # backward-compatible alias
    "convert360_180",
    "write_append_zarr",
    "iter_store_parquet_files",
    "recover_zarr_store",
    "recover_parquet_store",
]
