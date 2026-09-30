"""Abstract base class shared by all downloaders."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd
from loguru import logger
from tenacity import Retrying, stop_after_attempt, wait_exponential

from h2mare.config import AppConfig, get_settings
from h2mare.types import DateRange
from h2mare.utils.files_io import prune_empty_dirs
from h2mare.utils.paths import resolve_store_path
from h2mare.validators import validate_var_key


class BaseDownloader(ABC):
    """
    Common setup for all downloaders.

    Handles var_key validation, directory resolution, and logging so each
    concrete downloader only needs to implement ``run()``.

    Args:
        var_key: Variable key from ``app_config.variables``.
        app_config: Application configuration; loaded from settings if None.
        store_root: Root for processed zarr files; defaults to STORE_ROOT.
        download_root: Root for raw downloads; defaults to DOWNLOADS_DIR.
    """

    def __init__(
        self,
        var_key: str,
        *,
        app_config: Optional[AppConfig] = None,
        store_root: Optional[Path] = None,
        download_root: Optional[Path] = None,
    ) -> None:
        self.app_config = app_config or get_settings().app_config
        self.var_key = validate_var_key(var_key, self.app_config)
        self.var_config = self.app_config.variables[self.var_key]

        self.store_root = resolve_store_path(self.var_config, store_root)
        download_dir = download_root or get_settings().DOWNLOADS_DIR
        self.download_dir = download_dir / self.var_config.local_folder
        self.download_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Initialized {type(self).__name__} for '{var_key}' "
            f"(store={self.store_root}, download={self.download_dir})"
        )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(\n"
            f"  var_key={self.var_key},\n"
            f"  store={self.store_root},\n"
            f"  download={self.download_dir}\n"
            f")"
        )

    def _warn_if_rep_updated(self, api_rep_end: pd.Timestamp) -> None:
        """
        Warn when the rep dataset's end date (from the API/FTP) has advanced
        beyond the end date currently recorded in the local catalog.

        This signals that new reprocessed data has been published and will be
        downloaded on the next pipeline run.
        """
        from h2mare.storage.zarr_catalog import ZarrCatalog

        try:
            # The downloader's own config and store, not the ambient ones: this
            # is a check about the archive this instance writes to. Resolving it
            # through get_settings() instead read whichever store the machine
            # had deployed, and a var_key the deployed config did not define
            # raised out of the constructor into the debug branch below — the
            # check disabling itself, silently, on a machine pointed elsewhere.
            df = ZarrCatalog(
                self.var_key,
                app_config=self.app_config,
                store_root=self.store_root,
                auto_refresh=False,
            ).df
        except FileNotFoundError:
            return
        except Exception as e:
            logger.debug(f"Could not check rep catalog for '{self.var_key}': {e}")
            return

        if df.empty or "dataset" not in df.columns:
            return

        rep_rows = df[df["dataset"] == self.var_config.dataset_id_rep]
        if rep_rows.empty:
            return

        catalog_rep_end = pd.to_datetime(rep_rows["end_date"].max()).normalize()
        api_rep_end = pd.to_datetime(api_rep_end).normalize()

        has_nrt = getattr(self.var_config, "dataset_id_nrt", None) is not None
        if api_rep_end > catalog_rep_end and has_nrt:
            logger.warning(
                f"{self.var_key.upper()}: REP dataset end date advanced "
                f"from {catalog_rep_end.date()} (catalog) "
                f"to {api_rep_end.date()} (API) — "
                "new reprocessed data is available."
            )

    def _log_nothing_to_download(
        self,
        requested: DateRange,
        rep_avail: DateRange,
        nrt_avail: Optional[DateRange],
    ) -> None:
        """
        Explain why a resolved range produced no download tasks.

        Past the provider's last published day this is the ordinary "nothing new
        yet" case, which is what an inferred range on a current store hits.
        Anywhere else (before the product begins, or in a gap between REP and
        NRT) the request cannot be served at all, and "up to date" hid that.
        """
        published_end = (
            max(rep_avail.end, nrt_avail.end) if nrt_avail else rep_avail.end
        )
        if requested.start > published_end:
            logger.info(
                f"'{self.var_key}' is already up to date — skipping "
                f"(nothing published after {published_end.date()})."
            )
            return

        coverage = f"REP {rep_avail}" + (f", NRT {nrt_avail}" if nrt_avail else "")
        logger.warning(
            f"'{self.var_key}': requested {requested} is outside the published "
            f"coverage ({coverage}) — nothing to download."
        )

    def _retry_call(
        self,
        fn: Callable[..., Any],
        *args: Any,
        max_attempts: int = 3,
        wait_min: float = 30,
        wait_max: float = 300,
        **kwargs: Any,
    ) -> Any:
        """Execute fn(*args, **kwargs) with exponential-backoff retry on any exception.

        Logs a warning before each sleep so the user can see what failed and when
        the next attempt will run. After max_attempts, re-raises the last exception.
        """
        label = f"{type(self).__name__}[{self.var_key}]"

        def _before_sleep(retry_state) -> None:
            exc = retry_state.outcome.exception()
            wait = retry_state.next_action.sleep
            logger.warning(
                f"{label}: attempt {retry_state.attempt_number} failed "
                f"({type(exc).__name__}: {exc}). Retrying in {wait:.0f}s."
            )

        for attempt in Retrying(
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential(multiplier=1, min=wait_min, max=wait_max),
            before_sleep=_before_sleep,
            reraise=True,
        ):
            with attempt:
                return fn(*args, **kwargs)

    def _cleanup_empty_download_dir(self) -> None:
        """
        Remove the per-variable download directory if it holds no files after
        a run. Empty subfolders (e.g. eddies' rep/nrt staging dirs) are pruned
        first so they cannot keep the directory alive.
        """
        prune_empty_dirs(self.download_dir)
        if self.download_dir.exists() and not any(self.download_dir.iterdir()):
            self.download_dir.rmdir()

    def close(self) -> None:
        """Release any connection the downloader holds open.

        A no-op for the HTTP-based sources, which open a connection per request.
        AVISO keeps one FTP control connection alive for the whole run and
        overrides this.
        """

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    @abstractmethod
    def run(self, *args, **kwargs) -> bool:
        """Execute the download. Returns True if files were downloaded, False otherwise."""
        ...
