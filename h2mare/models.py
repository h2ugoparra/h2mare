"""
Classes representing Data models for spatial and variable configurations.
"""

from typing import Optional

import msgspec


class KeyVarConfigEntry(msgspec.Struct):
    """Configuration for a single Key variable/dataset."""

    local_folder: str
    variables: str | list[str]
    dataset_id_rep: str
    source: str
    pattern: str
    dataset_id_nrt: Optional[str] = None
    subset: Optional[bool] = True
    merge_time_step: bool = False
    bbox: Optional[tuple[float, float, float, float]] = None
    depth_range: Optional[tuple[float, float]] = None
    data_file: Optional[str] = None
    data_file_hires: Optional[str] = None

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
