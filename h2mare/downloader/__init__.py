from h2mare.utils.date_range import resolve_date_range

from .aviso_downloader import AVISODownloader
from .base import BaseDownloader
from .cds_downloader import CDSDownloader
from .cmems_downloader import CMEMSDownloader, generate_copernicus_patterns
from .cmems_downloader import download_original as cmems_download_original
from .cmems_downloader import download_subset as cmems_download_subset
from .registry import DOWNLOADER_REGISTRY

__all__ = [
    "BaseDownloader",
    "CMEMSDownloader",
    "cmems_download_subset",
    "cmems_download_original",
    "generate_copernicus_patterns",
    "CDSDownloader",
    "AVISODownloader",
    "resolve_date_range",
    "DOWNLOADER_REGISTRY",
]
