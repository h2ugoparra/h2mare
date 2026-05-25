"""
h2mare - Geospatial Processing for Climate and Ocean Data

Main components:
- config: Project paths and settings
- models: Data models for configuration
"""

__version__ = "0.1.0"

from .config import get_settings, settings
from .models import AppConfig, KeyVarConfigEntry, VariablesConfig, SYSTEM_VAR_KEYS
from .types import BBox, DateLike, DateRange, DownloadTask, TimeResolution
from .validators import validate_time_resolution, validate_var_key, validate_var_keys

__all__ = [
    "get_settings",
    "settings",
    "AppConfig",
    "VariablesConfig",
    "KeyVarConfigEntry",
    "SYSTEM_VAR_KEYS",
    "DateLike",
    "DateRange",
    "BBox",
    "TimeResolution",
    "DownloadTask",
    "validate_var_key",
    "validate_var_keys",
    "validate_time_resolution",
]
