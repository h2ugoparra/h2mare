"""Shared logging configuration for h2mare.

The console (default loguru stderr handler) keeps the full
``{name}:{function}:{line}`` location for interactive debugging. The persistent
``h2mare.log`` file uses :func:`add_file_logger` instead, which drops the
function and line — only the module name, level, and message are recorded.
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

# File format mirrors loguru's default but omits the ``{function}:{line}``
# location component that the terminal handler still shows.
LOG_FILE_FORMAT = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name} - {message}"


def add_file_logger(log_path: str | Path, level: str = "INFO") -> int:
    """
    Attach the persistent file sink with the project's file log format.

    Args:
        log_path: Destination log file.
        level: Minimum level to record. Defaults to ``"INFO"``.

    Returns:
        The loguru handler id (so callers may remove it if needed).
    """
    return logger.add(log_path, level=level, format=LOG_FILE_FORMAT)
