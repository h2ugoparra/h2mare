"""Maps source keys to their downloader classes."""

from __future__ import annotations

from h2mare.downloader.aviso_downloader import AVISODownloader
from h2mare.downloader.cds_downloader import CDSDownloader
from h2mare.downloader.cmems_downloader import CMEMSDownloader

DOWNLOADER_REGISTRY: dict[str, type] = {
    "cmems": CMEMSDownloader,
    "aviso": AVISODownloader,
    "cds": CDSDownloader,
}
