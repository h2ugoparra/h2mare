"""
h2mare - Geospatial Processing for Climate and Ocean Data

Main components:
- config: Project paths and settings
- models: Data models for configuration
"""

from importlib.metadata import version

__version__ = version("h2mare")

from .config import get_settings
from .models import SYSTEM_VAR_KEYS, AppConfig, KeyVarConfigEntry, VariablesConfig
from .types import (
    BBox,
    DateLike,
    DateRange,
    DownloadTask,
    FTPDownloadTask,
    TimeResolution,
)
from .validators import validate_time_resolution, validate_var_key, validate_var_keys

__all__ = [
    "get_settings",
    "AppConfig",
    "VariablesConfig",
    "KeyVarConfigEntry",
    "SYSTEM_VAR_KEYS",
    "DateLike",
    "DateRange",
    "BBox",
    "TimeResolution",
    "DownloadTask",
    "FTPDownloadTask",
    "validate_var_key",
    "validate_var_keys",
    "validate_time_resolution",
]
