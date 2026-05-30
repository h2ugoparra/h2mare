"""
Classes representing Data models for spatial and variable configurations.
"""

from typing import Optional

import msgspec


class KeyVarConfigEntry(msgspec.Struct):
    """Configuration for a single Key variable/dataset."""

    # Subdirectory under STORE_ROOT for this variable's Zarr files.
    local_folder: str
    # Variable names to extract from source files.
    variables: str | list[str]
    # Reprocessed (multiyear) dataset identifier.
    dataset_id_rep: str
    # Provider: "cmems", "aviso", or "cds".
    source: str
    # Regex for extracting dates from raw filenames.
    pattern: str
    # Near-real-time dataset identifier. Omit for reanalysis-only products.
    dataset_id_nrt: Optional[str] = None
    # Whether to spatially subset on download (default True).
    subset: Optional[bool] = True
    # Set True for CDS/ERA5 accumulated or averaged variables whose GRIB files
    # have a 2-D time×step coordinate grid instead of a flat time axis
    # (e.g. atm-accum-avg, radiation). Triggers a preprocess step that merges
    # the two dimensions and trims overlapping timestamps at month edges.
    merge_time_step: bool = False
    # Set True when the filename pattern has two capture groups encoding a
    # (start, end) date range (e.g. CMEMS/CDS: "2021-01-01-2021-01-31.nc").
    # Set False (default) when the pattern yields a single date
    # (e.g. AVISO FSLE: "_20210115_").
    filename_date_range: bool = False
    # Bounding box [xmin, ymin, xmax, ymax] for spatial subsetting.
    bbox: Optional[tuple[float, float, float, float]] = None
    # Depth range [min_depth, max_depth] for 3-D variables (e.g. o2, thetao).
    depth_range: Optional[tuple[float, float]] = None
    # Filename of the static source file at the configured output resolution.
    # Used by compile-only variables such as bathy.
    data_file: Optional[str] = None
    # High-resolution static source file. Used by bathy when extracting at
    # full native resolution (e.g. from SHP geometries).
    data_file_hires: Optional[str] = None
    # Set True for trajectory-format datasets (e.g. eddies) that require
    # spatial binning before they can be stored as a gridded Zarr.
    # The standard open_mfdataset pipeline is bypassed entirely.
    trajectory_format: bool = False
    # Set True for variables whose Zarr store uses lon/lat coordinate names that
    # must be renamed to x/y before rioxarray clip (e.g. AVISO fsle, eddies).
    rename_lonlat: bool = False
    # Depth levels (metres) to extract when slicing a 3-D variable during
    # Extractor runs. Each level becomes a separate output column
    # (e.g. [0, 100, 500] → o2_0, o2_100, o2_500). None = no depth slicing.
    extract_depth_slices: Optional[list[int]] = None
    # Depth levels (metres) to select when compiling a 3-D variable into h2ds.
    # Each level becomes a separate output variable (e.g. [0, 100, 500, 1000]
    # → o2_0, o2_100, o2_500, o2_1000). None = no depth slicing in compiler.
    compile_depth_slices: Optional[list[int]] = None

    def __post_init__(self):
        if self.bbox is not None:
            lon_min, lat_min, lon_max, lat_max = self.bbox
            if not (-180 <= lon_min <= 180 and -180 <= lon_max <= 180):
                raise ValueError("Longitude must be between -180 and 180")
            if not (-90 <= lat_min <= 90 and -90 <= lat_max <= 90):
                raise ValueError("Latitude must be between -90 and 90")
            if lon_min >= lon_max:
                raise ValueError("lon_min must be less than lon_max")
            if lat_min >= lat_max:
                raise ValueError("lat_min must be less than lat_max")

        if self.depth_range is not None:
            if self.depth_range[0] >= self.depth_range[1]:
                raise ValueError("depth_min must be less than depth_max")


class SecretsConfig(msgspec.Struct):
    """External service credentials."""

    aviso_ftp_server: Optional[str] = None
    aviso_username: Optional[str] = None
    aviso_password: Optional[str] = None


# VariablesConfig is now a plain dict — kept as a type alias for compatibility
VariablesConfig = dict[str, KeyVarConfigEntry]

# Variables that are derived/computed inside the pipeline and never downloaded
# from an external source. Excluded from download loops and catalog date-range
# inference; each has its own dedicated processing path in the Compiler.
SYSTEM_VAR_KEYS: frozenset[str] = frozenset({"h2ds", "bathy", "moon"})


class AppConfig(msgspec.Struct):
    """Complete application configuration."""

    variables: VariablesConfig
    secrets: SecretsConfig
