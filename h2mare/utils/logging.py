"""Shared logging configuration for h2mare.

A single rolling file — ``LOGS_DIR/h2mare.log`` — captures every run. The
console keeps loguru's default ``{name}:{function}:{line}`` location for
interactive debugging; the file records only time, level, and message.

Call :func:`configure_logging` once per process. The CLI does this in a
top-level Typer callback so every command logs identically. The call is
idempotent.
"""

from __future__ import annotations

import logging as _stdlib_logging
import os
import time
from pathlib import Path
from typing import Optional

from loguru import logger

# File format keeps only time, level, and message. The source location
# ({name}:{function}:{line}) stays on the console handler only.
LOG_FILE_FORMAT = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {message}"

# Third-party loggers that flood the logs at INFO.
_NOISY_LOGGERS = ("urllib3.connectionpool",)

_configured = False


def add_file_logger(
    log_path: str | Path,
    level: str = "INFO",
    *,
    rotation: str = "20 MB",
    retention: str = "180 days",
) -> int:
    """
    Attach the persistent file sink with the project's file log format.

    Size-based rotation (not daily) suits an infrequently-scheduled pipeline:
    one rolling file, rotated only once it grows large, old segments compressed
    and pruned so disk stays bounded. ``enqueue=True`` makes the sink safe under
    the multiprocessing used during eddies/AVISO processing.

    Args:
        log_path: Destination log file.
        level: Minimum level to record. Defaults to ``"INFO"``.
        rotation: When to roll the current file (loguru rotation spec).
        retention: How long to keep rotated files.

    Returns:
        The loguru handler id (so callers may remove it if needed).
    """
    return logger.add(
        log_path,
        level=level,
        format=LOG_FILE_FORMAT,
        rotation=rotation,
        retention=retention,
        compression="zip",
        enqueue=True,
        backtrace=True,
        diagnose=False,
    )


def configure_logging(
    level: Optional[str] = None, *, log_dir: Optional[Path] = None
) -> None:
    """
    Configure h2mare logging once per process (idempotent).

    Leaves loguru's console handler in place (so the terminal keeps the full
    ``function:line`` location) and adds the rolling ``h2mare.log`` file sink.
    The file level defaults to the ``H2MARE_LOG_LEVEL`` env var, else ``INFO``.

    Args:
        level: Override the file log level. Falls back to ``H2MARE_LOG_LEVEL``
            then ``"INFO"``.
        log_dir: Override the log directory. Defaults to ``settings.LOGS_DIR``.
    """
    global _configured
    if _configured:
        return

    # Lazy import: avoids a config import at module load and any import cycle.
    from h2mare.config import get_settings

    level = level or os.getenv("H2MARE_LOG_LEVEL", "INFO")
    log_dir = log_dir or get_settings().LOGS_DIR
    log_dir.mkdir(parents=True, exist_ok=True)

    add_file_logger(log_dir / "h2mare.log", level=level)

    for name in _NOISY_LOGGERS:
        _stdlib_logging.getLogger(name).setLevel(_stdlib_logging.ERROR)

    _configured = True


def log_time(func):
    """Decorator to log execution time of a function."""

    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start_time
        if elapsed < 60:
            logger.info(
                f"Function '{func.__name__}' took {elapsed:.4f} secs to complete."
            )
        else:
            minutes = int(elapsed // 60)
            seconds = elapsed % 60
            logger.info(
                f"Function '{func.__name__}' took {minutes} min {seconds:.1f} secs to complete."
            )
        return result

    return wrapper
