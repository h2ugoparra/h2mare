"""
Configuration management for h2mare project
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

import msgspec
import yaml
from dotenv import load_dotenv
from loguru import logger

from h2mare.models import AppConfig, KeyVarConfigEntry


class Settings:
    """Application settings and paths."""

    def __init__(self):

        # project root: directory containing h2mare's config.yaml
        self.BASE_DIR, self._project_mode = self._find_project_root()

        # Load .env file
        self._load_dotenv()

        # === Data Directories ===
        self.DATA_DIR = self.BASE_DIR / "data"

        # Raw Data (immutable)
        self.RAW_DIR = self.DATA_DIR / "raw"
        self.DOWNLOADS_DIR = self.RAW_DIR / "downloads"

        # Interim data (processing stages)
        self.INTERIM_DIR = self.DATA_DIR / "interim"

        # Processed data (final outputs)
        self.PROCESSED_DIR = self.DATA_DIR / "processed"
        self.ZARR_DIR = self.PROCESSED_DIR / "zarr"
        self.PARQUET_DIR = self.PROCESSED_DIR / "parquet"
        self.METADATA_DIR = self.PROCESSED_DIR / "metadata"

        # Logs
        self.LOGS_DIR = self.BASE_DIR / "logs"

        # External Storage (where all data lives)
        self.STORE_ROOT = self._get_store_dir()
        # Whether STORE_ROOT was set by --store-path rather than read from .env.
        # A variable may name its own root in config.yaml, and that root beats
        # the configured STORE_ROOT — but not an operator who explicitly asked
        # this run to go somewhere else. See utils.paths.store_root_for.
        self._store_root_overridden = False

        # Ceiling on every worker pool, for a machine smaller than the one a
        # site's default was measured on. None leaves each site's own default
        # (and the host's CPU count) in charge — see
        # utils.parallel.resolve_n_workers.
        self.MAX_WORKERS = self._get_max_workers()

        # No directories are created here: Settings() runs on any import, and
        # BASE_DIR may be a directory that merely contains a config.yaml. Writers
        # mkdir their own parents.

        # Application config (lazy loaded)
        self._app_config: Optional[AppConfig] = None
        self._global_attrs = None
        self._variable_attrs = None
        self._native_attr_overrides = None

    def _find_project_root(self) -> tuple[Path, bool]:
        """Find the h2mare project root and whether we are in project mode.

        Returns (root_path, is_project_mode). is_project_mode is True only when
        an h2mare config.yaml is found, meaning h2mare is being run as a project
        rather than used as a library dependency inside another project.

        Search order:
        1. H2MARE_ROOT env var (explicit override) → project mode.
        2. Walk up from cwd() looking for config.yaml only.
        3. Walk up from __file__ looking for config.yaml only (editable installs).
        4. Fallback: ~/.h2mare → library mode, no directories created.
        """
        if root_env := os.getenv("H2MARE_ROOT"):
            return Path(root_env).resolve(), True

        for start in (Path.cwd(), Path(__file__).resolve().parent):
            current = start.resolve()
            while current != current.parent:
                if (current / "config.yaml").exists():
                    return current, True
                current = current.parent

        return Path.home() / ".h2mare", False

    def _load_dotenv(self):
        """Load environment variables from .env file."""
        env_file = self.BASE_DIR / ".env"
        if env_file.exists():
            load_dotenv(env_file)

    def _get_store_dir(self) -> Path | None:
        """Get external storage directory from environment."""
        if store_dir := os.getenv("STORE_ROOT"):
            return Path(store_dir).resolve()
        return None

    @staticmethod
    def _get_max_workers() -> Optional[int]:
        """
        Ceiling on worker pools from ``H2MARE_MAX_WORKERS``, if it is usable.

        A malformed value is warned about and ignored rather than raised:
        ``Settings()`` runs on any import, so a typo in a tuning knob would
        otherwise stop every command, including the ones that start no pool.
        Ignoring it leaves each site at its own default, which is the behaviour
        of not setting it at all.
        """
        raw = os.getenv("H2MARE_MAX_WORKERS")
        if not raw:
            return None

        try:
            value = int(raw)
        except ValueError:
            logger.warning(
                f"H2MARE_MAX_WORKERS={raw!r} is not a whole number — ignoring it. "
                f"Worker pools keep their own defaults."
            )
            return None

        if value < 1:
            logger.warning(
                f"H2MARE_MAX_WORKERS={value} is below 1 — ignoring it. Worker "
                f"pools keep their own defaults."
            )
            return None

        return value

    def override_store_root(self, store_root: Path) -> None:
        """
        Point ``STORE_ROOT`` somewhere else for the rest of this process.

        Backs the ``--store-path`` flag. Every store location in h2mare is
        resolved from this one value through ``resolve_store_path``, including
        places nothing threads an argument to — the per-variable catalogs the
        compiler opens, ``cds.get_previous_dates_da``, the eddies processor. So
        the override is applied at the source rather than passed down: a flag
        handed step by step reaches only the steps someone remembered to change,
        which is exactly how ``--store-path`` came to relocate the download and
        Parquet steps while convert and compile stayed on the configured root.

        Call once, from a CLI entry point, before any work begins. The value is
        a *root* holding one subdirectory per variable (``local_folder``), the
        same shape ``STORE_ROOT`` has in ``.env`` — not a single store's path.

        It also outranks a per-variable ``store_root`` from ``config.yaml``:
        relocating a run is something an operator asks for deliberately, and a
        flag that moved only the variables which had not opted out would be a
        partial relocation nobody could reason about. That is why the override
        is recorded as such rather than just written into ``STORE_ROOT`` — the
        resolver has to tell "the configured root" apart from "the root this
        run was told to use".
        """
        resolved = Path(store_root).resolve()
        if self.STORE_ROOT != resolved:
            logger.info(f"Store root overridden: {resolved}")
        self.STORE_ROOT = resolved
        self._store_root_overridden = True

    @property
    def store_root_overridden(self) -> bool:
        """True when ``STORE_ROOT`` came from ``--store-path``, not from ``.env``."""
        return self._store_root_overridden

    @property
    def CLIMATOLOGY_DIR(self) -> Path | None:
        if self.STORE_ROOT is None:
            return None
        return self.STORE_ROOT / "Climatology"

    def ensure_directories(self):
        """Scaffold the project directory tree under BASE_DIR. Opt-in, not automatic."""
        dirs = [
            self.DOWNLOADS_DIR,
            self.INTERIM_DIR,
            self.ZARR_DIR,
            self.PARQUET_DIR,
            self.METADATA_DIR,
            self.LOGS_DIR,
        ]

        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)

    def load_app_config(self, config_path: Optional[Path] = None) -> AppConfig:
        """
        Load application configuration from YAML.

        Args:
            config_path: Path to config.yaml. If None, uses BASE_DIR/config.yaml

        Returns:
            Validated AppConfig instance
        """
        if self._app_config is not None:
            return self._app_config

        if config_path is None:
            config_path = self.BASE_DIR / "config.yaml"

        if not config_path.exists():
            raise FileNotFoundError(
                f"config.yaml not found at {config_path}\n"
                f"Expected location: {self.BASE_DIR / 'config.yaml'}"
            )

        # Load YAML
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f) or {}

        # Warn on unrecognised top-level keys — catches typos like "varibles:"
        _KNOWN_KEYS = {
            "variables",
            "global_attrs",
            "variable_attrs",
            "native_attr_overrides",
        }
        unknown = set(config_dict) - _KNOWN_KEYS
        if unknown:
            logger.warning(
                f"config.yaml contains unrecognised top-level key(s): "
                f"{sorted(unknown)} — these will be ignored. "
                f"Expected keys: {sorted(_KNOWN_KEYS)}."
            )

        # Warn on unrecognised per-variable keys — catches "store_roots:" or
        # "local_foldr:". msgspec structs here do not set forbid_unknown_fields,
        # so an unknown key is dropped in silence and the variable keeps the
        # default the author meant to override. Warn rather than raise: the
        # top-level check above warns too, and a config that has been loading
        # for months should not start failing over a key that was already
        # being ignored.
        _VAR_FIELDS = set(KeyVarConfigEntry.__struct_fields__)
        for var_key, entry in (config_dict.get("variables") or {}).items():
            if not isinstance(entry, dict):
                continue
            unknown_fields = set(entry) - _VAR_FIELDS
            if unknown_fields:
                logger.warning(
                    f"config.yaml variable '{var_key}' has unrecognised key(s): "
                    f"{sorted(unknown_fields)} — these will be ignored."
                )

        # Extract global and variable metadata (not part of AppConfig)
        self._global_attrs = config_dict.get("global_attrs", {})
        self._variable_attrs = config_dict.get("variable_attrs", {})
        self._native_attr_overrides = config_dict.get("native_attr_overrides", {})

        # Add secrets from environment
        secrets_dict = {
            "aviso_ftp_server": os.getenv("AVISO_FTP_SERVER"),
            "aviso_username": os.getenv("AVISO_USERNAME"),
            "aviso_password": os.getenv("AVISO_PASSWORD"),
        }
        config_dict["secrets"] = secrets_dict

        # Warn early if AVISO variables are configured but credentials are absent
        aviso_vars = [
            k
            for k, v in config_dict.get("variables", {}).items()
            if isinstance(v, dict) and v.get("source") == "aviso"
        ]
        if aviso_vars:
            missing = [k for k, v in secrets_dict.items() if v is None]
            if missing:
                import warnings

                warnings.warn(
                    f"AVISO variables {aviso_vars} configured but env secrets missing: {missing}",
                    RuntimeWarning,
                    stacklevel=2,
                )

        # `subset` only affects CMEMS downloads (subset() vs get() API choice);
        # it is ignored for every other source. Flag it when set elsewhere so a
        # misplaced `subset:` is not silently no-op'd.
        misplaced_subset = [
            k
            for k, v in config_dict.get("variables", {}).items()
            if isinstance(v, dict) and v.get("source") != "cmems" and "subset" in v
        ]
        if misplaced_subset:
            logger.warning(
                f"`subset` is set on non-CMEMS variable(s) {sorted(misplaced_subset)} "
                "but only applies to CMEMS downloads — it will be ignored there."
            )

        self._app_config = msgspec.convert(config_dict, AppConfig)
        return self._app_config

    def get_available_var_keys(self) -> list[str]:
        """
        Get list of available variable keys from config.
        """
        if self._app_config is None:
            self._app_config = self.load_app_config()
        return list(self._app_config.variables.keys())

    def get_var_info(self, var_name: str) -> dict:
        """Get variable attributes from yaml. Returns {} if var_name is not in config."""
        if self._variable_attrs is None:
            self.load_app_config()
        return (self._variable_attrs or {}).get(var_name, {})

    @property
    def global_attrs(self) -> dict:
        if self._global_attrs is None:
            self.load_app_config()
        return self._global_attrs or {}

    @property
    def variable_attrs(self) -> dict:
        if self._variable_attrs is None:
            self.load_app_config()
        return self._variable_attrs or {}

    @property
    def native_attr_overrides(self) -> dict:
        """
        Per-var_key attribute deltas for the native stores, ``{var_key: {var: attrs}}``.

        Empty for most variables: a native store usually publishes exactly what
        h2ds does. It is the hourly CDS stores that differ, holding ERA5's own
        units and cadence while h2ds holds the converted daily reduction.
        """
        if self._native_attr_overrides is None:
            self.load_app_config()
        return self._native_attr_overrides or {}

    @property
    def app_config(self) -> AppConfig:
        """Lazy-loaded application config."""
        if self._app_config is None:
            self._app_config = self.load_app_config()
        return self._app_config


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the application-wide Settings instance (cached after first call).

    Tests can reset the cache with ``get_settings.cache_clear()`` before
    monkeypatching environment variables to obtain a fresh instance.
    """
    return Settings()
