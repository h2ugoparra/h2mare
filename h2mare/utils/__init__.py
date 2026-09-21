from .datetime_utils import (
    more_than_one_year,
    normalize_date,
    normalize_dates,
    to_datetime,
)
from .files_io import safe_move_files, safe_rmtree
from .labels import create_filename_label, create_label_from_dataset
from .logging import log_time
from .paths import (
    resolve_download_path,
    resolve_store_path,
    static_layer_path,
    store_root_for,
)
from .spatial import GridBuilder, clip_land_data, haversine_min_distance_kdtree

# ``.plot`` is deliberately NOT re-exported here. Importing it eagerly made every
# ``h2mare.utils.*`` import pull in matplotlib/cartopy/geopandas, and — because
# plot needs ZarrCatalog while storage.zarr_catalog imports utils.labels — it
# forced a cycle that plot.py had to work around with a function-level import.
# Import the module directly instead: ``from h2mare.utils.plot import plot_maps``.

__all__ = [
    "log_time",
    "resolve_store_path",
    "resolve_download_path",
    "store_root_for",
    "static_layer_path",
    "create_filename_label",
    "create_label_from_dataset",
    "safe_move_files",
    "safe_rmtree",
    "GridBuilder",
    "haversine_min_distance_kdtree",
    "clip_land_data",
    "normalize_date",
    "normalize_dates",
    "to_datetime",
    "more_than_one_year",
]
