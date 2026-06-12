"""Shared logging configuration for h2mare.

A single rolling file — ``LOGS_DIR/pipeline.log`` — captures every run,
including third-party stdlib logging (cdsapi request status,
copernicusmarine), which is intercepted into loguru. The console keeps
loguru's default ``{name}:{function}:{line}`` location for interactive
debugging; the file records only time, level, and message. When stderr is
not a real terminal (Task Scheduler wrapper, redirected output), the console
sink is dropped — otherwise the redirection would duplicate every line, and
a second writer appending to the sink's file can clobber it on Windows.

Call :func:`configure_logging` once per process. The CLI does this in a
top-level Typer callback so every command logs identically. The call is
idempotent.
"""

from __future__ import annotations

import logging as _stdlib_logging
import os
import sys
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


class _InterceptHandler(_stdlib_logging.Handler):
    """
    Route stdlib logging records into loguru, so third-party libraries that
    use ``logging`` (cdsapi, copernicusmarine, …) land in the same sinks —
    and therefore the same file — as h2mare's own messages.
    """

    def emit(self, record: _stdlib_logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Walk back to the caller outside the stdlib logging machinery so
        # loguru reports the real origin, not this handler.
        frame, depth = _stdlib_logging.currentframe(), 2
        while frame and frame.f_code.co_filename == _stdlib_logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


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

    Adds the rolling ``pipeline.log`` file sink and intercepts stdlib logging
    into loguru so third-party output shares the file. On a real terminal,
    loguru's console handler stays (full ``function:line`` location); when
    stderr is redirected it is removed — the redirection would otherwise
    duplicate every line and risk clobbering the sink's file with a second
    writer. The file level defaults to ``H2MARE_LOG_LEVEL``, else ``INFO``.

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

    try:
        add_file_logger(log_dir / "pipeline.log", level=level)
        file_sink_ok = True
    except OSError as e:
        # E.g. the file is held open by another writer (an old-style wrapper
        # still redirecting into it, or a locking viewer). Don't kill the run
        # over logging — fall back to console only.
        file_sink_ok = False
        file_sink_err = e

    # Drop loguru's default console handler only under redirection AND with a
    # working file sink — when stderr is piped into a file, the console copy
    # would duplicate every line; but if the file sink failed, the redirected
    # console stream is the only record left, so keep it.
    if file_sink_ok and not sys.stderr.isatty():
        try:
            logger.remove(0)  # loguru's default stderr handler
        except ValueError:
            pass

    if not file_sink_ok:
        logger.warning(
            f"pipeline.log unavailable ({file_sink_err}) — logging to console only."
        )

    # Route stdlib logging through loguru (replacing any handlers libraries
    # installed on the root logger) so it reaches the same file sink.
    _stdlib_logging.basicConfig(
        handlers=[_InterceptHandler()], level=_stdlib_logging.INFO, force=True
    )
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
