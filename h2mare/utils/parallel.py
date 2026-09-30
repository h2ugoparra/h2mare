"""
Sizing for the **process** pools: BOA front detection and the eddy rasterisation.

Each of those sites proposes its own default, measured there and deliberately
different: 10 for front detection, whose workers each read one lat×lon slab per
day out of a staged Zarr, against 4 for the eddy rasterisation, whose workers
each hold a period's observations and where profiling found 8 no faster than 4.
What they share is the ceiling — neither wants more processes than the machine
has cores, and an operator moving to a smaller box may want fewer still without
editing every entry in config.yaml.

Not for the thread pools (the AVISO FTP downloads, ``parquet2csv``, geometry
extraction). Those wait on the network or the filesystem, so more threads than
cores is the point rather than oversubscription, and capping them by CPU count
would slow the downloads down for no gain.
"""

from __future__ import annotations

import os

from loguru import logger

from h2mare.config import get_settings


def resolve_n_workers(requested: int | None, default: int, label: str = "") -> int:
    """
    Pool size for one site: what was asked for, capped by the machine.

    Args:
        requested: Pool size from config or the caller. None uses *default*.
        default: The site's own measured sweet spot.
        label: Identity for the log line — a var_key, or ``var_key/layer``.

    Returns:
        The number of workers to start: at least 1, and never more than the
        host's CPU count, nor than ``H2MARE_MAX_WORKERS`` where that is set.
    """
    wanted = requested or default

    limits = [("the host's CPU count", os.cpu_count() or 1)]
    if (ceiling := get_settings().MAX_WORKERS) is not None:
        limits.append(("H2MARE_MAX_WORKERS", ceiling))

    reason, allowed = min(limits, key=lambda limit: limit[1])
    if wanted <= allowed:
        return wanted

    tag = f"[{label}] " if label else ""
    logger.info(
        f"{tag}Pool capped to {allowed} worker(s), from {wanted}: {reason}. "
        f"These are spawn pools, so a worker the machine cannot run still "
        f"re-imports h2mare and is sent its own copy of every task."
    )
    return allowed
