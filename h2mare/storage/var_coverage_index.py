"""Per-variable end-date tracking for h2ds compilation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger


class VarCoverageIndex:
    """
    Persists the last compiled end date per source variable inside h2ds.

    Stored as a JSON file: ``{var_key: "YYYY-MM-DD", ...}``.  Updated after
    every chunk write so that a mid-run crash leaves the index in a consistent
    (partially-advanced) state.

    When no entry exists for a variable the caller should fall back to the
    source catalog start date, triggering a full compile for that variable.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._data: dict[str, str] = self._load()

    # ------------------------------------------------------------------ IO

    def _load(self) -> dict[str, str]:
        if not self._path.exists():
            return {}
        try:
            return json.loads(self._path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning(f"Could not read {self._path}: {exc} — starting fresh")
            return {}

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(self._data, indent=2, sort_keys=True), encoding="utf-8"
        )

    # ------------------------------------------------------------------ API

    def get_end(self, var_key: str) -> Optional[pd.Timestamp]:
        raw = self._data.get(var_key)
        return pd.Timestamp(raw) if raw else None

    def update(self, var_key: str, end: pd.Timestamp) -> None:
        """Advance the recorded end date for *var_key*; never goes backwards."""
        current = self.get_end(var_key)
        if current is None or end > current:
            self._data[var_key] = end.strftime("%Y-%m-%d")
