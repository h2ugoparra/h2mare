"""
Classes representing Data models for spatial and variable configurations.
"""

from enum import Enum
from pathlib import PurePosixPath, PureWindowsPath
from typing import Optional

import msgspec


class TimeStep(str, Enum):
    """
    Native cadence of a variable's stored Zarr — how far apart its time steps are.

    Distinct from :class:`h2mare.types.FilePeriod`, which is about the storage
    layout rather than the data: a store can be hourly and still be written one
    file per year.
    """

    DAILY = "daily"
    HOURLY = "hourly"


class StoreDtype(str, Enum):
    """
    On-disk encoding for a variable's Zarr.

    ``FLOAT32`` (the default) writes what the pipeline has always written and is
    byte-identical to it. ``INT16`` stores scale/offset-packed integers instead,
    matching the ~16-bit packing ERA5's GRIB already uses — so it discards
    quantisation noise rather than signal, at roughly two thirds the size.

    Opt-in per variable: a store keeps whatever encoding it was created with,
    and appends inherit it.
    """

    FLOAT32 = "float32"
    INT16 = "int16"


class DerivedOp(str, Enum):
    """Operations a ``derived_vars`` entry can apply; see ``processing/derived.py``."""

    # Standard deviation over a square lon/lat window centred on each cell.
    ROLLING_STD = "rolling_std"
    # 0.5 * (u**2 + v**2) — kinetic energy per unit mass of a velocity pair.
    KINETIC_ENERGY = "kinetic_energy"


# How many source variables each operation reads.
_DERIVED_OP_ARITY = {DerivedOp.ROLLING_STD: 1, DerivedOp.KINETIC_ENERGY: 2}


class DerivedVarSpec(msgspec.Struct, forbid_unknown_fields=True):
    """
    One variable computed at convert time from others in the same dataset.

    Unknown fields are refused so a misspelt ``window`` fails at load instead
    of silently falling back to the default.
    """

    op: DerivedOp
    # Variable name(s) read, as they stand after the var_key's processor ran
    # (``sst``, not ``analysed_sst``). One name for rolling_std, [u, v] for
    # kinetic_energy.
    source: str | list[str]
    # rolling_std only: side of the square window, in cells. Odd, so the
    # window centres on its cell.
    window: Optional[int] = None
    # Depths (metres) to compute at, instead of the sources' whole depth axis.
    # Each level is matched to the nearest source depth and written as a 2-D
    # variable <name>_<level> — the name depth_levels would give it — so only
    # these levels are read, computed and stored. None keeps the depth axis.
    depth: Optional[list[int]] = None

    @property
    def sources(self) -> list[str]:
        return [self.source] if isinstance(self.source, str) else list(self.source)

    def output_names(self, name: str) -> list[str]:
        """Variables this entry writes: ``name``, or one per ``depth`` level."""
        if self.depth is None:
            return [name]
        return depth_column_names({name: self.depth})

    def __post_init__(self):
        if self.depth is not None:
            check_depth_levels("depth", self.depth)
        arity = _DERIVED_OP_ARITY[self.op]
        if len(self.sources) != arity:
            raise ValueError(
                f"{self.op.value} reads {arity} source variable(s); got {self.source!r}"
            )
        if self.op is DerivedOp.ROLLING_STD:
            if self.window is None:
                self.window = 3
            if self.window < 1 or self.window % 2 == 0:
                raise ValueError(
                    f"rolling_std window must be an odd number of cells >= 1; "
                    f"got {self.window}"
                )
        elif self.window is not None:
            raise ValueError(f"{self.op.value} takes no window; got {self.window}")


def step_freq(var_config) -> str:
    """
    Pandas frequency alias matching a variable's cadence — ``"h"`` or ``"D"``.

    Used by the gap checks so they compare a store against a calendar at its own
    resolution: a daily grid cannot see a missing hour, and an hourly grid built
    for a daily store would report 23 phantom gaps per day.

    Takes any object with a ``time_step`` attribute and defaults to daily, so
    stand-in configs and entries predating the field keep the old behaviour.
    """
    return "h" if getattr(var_config, "time_step", None) is TimeStep.HOURLY else "D"


def check_depth_levels(key: str, levels: list[int]) -> None:
    """Refuse an empty, negative, non-integer or duplicated level list."""
    # bool is an int subclass; msgspec already rejects it from config, but a
    # list built in Python reaches here unchecked.
    if not all(
        isinstance(level, int) and not isinstance(level, bool) for level in levels
    ):
        raise ValueError(f"{key} levels must be whole metres; got {levels}")
    if not levels:
        raise ValueError(f"{key} has an empty level list; omit the entry instead")
    if any(level < 0 for level in levels):
        raise ValueError(f"{key} levels must be depths in metres >= 0; got {levels}")
    if len(set(levels)) != len(levels):
        raise ValueError(f"{key} has duplicate levels: {levels}")


def _validate_depth_keys(
    new_key: str,
    new: Optional[dict[str, list[int]]],
    old_key: str,
    old: Optional[list[int]],
) -> None:
    """Shape checks for one depth key and its older single-variable form."""
    if new is not None and old is not None:
        raise ValueError(
            f"{new_key} and {old_key} are both set; keep {new_key} only "
            f"({old_key}: [...] is the same as {new_key}: {{<var_key>: [...]}})"
        )
    if new is not None:
        if not new:
            raise ValueError(f"{new_key} is empty; omit the key instead")
        for var, levels in new.items():
            check_depth_levels(f"{new_key}.{var}", levels)
    if old is not None:
        check_depth_levels(old_key, old)


def depth_levels_for(
    var_key: str, var_config, purpose: str = "compile"
) -> dict[str, list[int]]:
    """
    Depth levels per store variable, for ``"compile"`` or ``"extract"``.

    The single reader of the four depth keys, so that compile and extraction
    cannot disagree on what the older list forms mean. Returns ``{}`` for a
    variable without depth levels.

    Extraction starts from the compile levels and replaces, per variable,
    whatever ``extract_depth_levels`` lists — narrowing one field must not drop
    the others.

    Takes any object with the attributes (``getattr``), like :func:`step_freq`,
    so stand-in configs predating these fields resolve to their old meaning.
    """
    if purpose not in ("compile", "extract"):
        raise ValueError(f"purpose must be 'compile' or 'extract'; got {purpose!r}")

    def _resolve(new_attr: str, old_attr: str) -> dict[str, list[int]]:
        new = getattr(var_config, new_attr, None)
        if new is not None:
            return {var: list(levels) for var, levels in new.items()}
        old = getattr(var_config, old_attr, None)
        return {var_key: list(old)} if old is not None else {}

    levels = _resolve("depth_levels", "compile_depth_slices")
    if purpose == "extract":
        levels.update(_resolve("extract_depth_levels", "extract_depth_slices"))
    return levels


def depth_column_names(levels: dict[str, list[int]]) -> list[str]:
    """Output columns for resolved depth levels: ``<variable>_<level>``."""
    return [
        f"{var}_{level}" for var, var_levels in levels.items() for level in var_levels
    ]


class KeyVarConfigEntry(msgspec.Struct):
    """Configuration for a single Key variable/dataset."""

    # Subdirectory under STORE_ROOT for this variable's Zarr files.
    local_folder: str
    # Variable names to extract from source files.
    source_vars: str | list[str]
    # Reprocessed (multiyear) dataset identifier.
    dataset_id_rep: str
    # Provider: "cmems", "aviso", or "cds".
    source: str
    # Whether this variable's raw NetCDF/GRIB files are archived into the store
    # (and kept) after conversion, or deleted per-period. Required and explicit:
    # True keeps raw files, False deletes them.
    archive_raw: bool
    # Near-real-time dataset identifier. Omit for reanalysis-only products.
    dataset_id_nrt: Optional[str] = None
    # CMEMS only. Chooses the copernicusmarine download API: True (default)
    # downloads via subset() (spatial/variable subset honoring bbox/source_vars);
    # False downloads full original files via get(). Ignored for non-CMEMS
    # sources (aviso, cds, …), which do not consult this field.
    subset: Optional[bool] = True
    # Set True for CDS/ERA5 accumulated or averaged variables whose GRIB files
    # have a 2-D time×step coordinate grid instead of a flat time axis
    # (e.g. atm-accum-avg, radiation). Triggers a preprocess step that merges
    # the two dimensions and trims overlapping timestamps at month edges.
    merge_time_step: bool = False
    # Regex matched (via re.search) against each raw filename to extract the
    # date component(s); the capture groups must agree with filename_date_range:
    #   - filename_date_range=True  -> exactly 2 groups, each a full parseable
    #     date (start, end); the file expands to that daily range.
    #   - filename_date_range=False -> the groups are joined with "-" and parsed
    #     as one date (e.g. (\d{8}) -> "20210115";
    #     (\d{4})(\d{2})(\d{2}) -> "2021-01-15").
    # None for derived/system variables (bathy, moon, h2ds) that are never
    # matched against downloaded filenames.
    pattern: Optional[str] = None
    # Set True when the filename pattern has two capture groups encoding a
    # (start, end) date range (e.g. CMEMS/CDS: "2021-01-01-2021-01-31.nc").
    # Set False (default) when the pattern yields a single date
    # (e.g. AVISO FSLE: "_20210115_").
    filename_date_range: bool = False
    # Regex matched (via re.search) against each raw filename; only matching
    # files are converted. Use when a download directory holds files the
    # pipeline must not read — AVISO ships META3.2 eddy trajectories as
    # long/short/untracked variants alongside each other, and only the long
    # ones belong in the store (the untracked files have no `track` variable at
    # all). None (default) converts every file the date pattern matches.
    raw_include: Optional[str] = None
    # Bounding box [xmin, ymin, xmax, ymax] for spatial subsetting.
    bbox: Optional[tuple[float, float, float, float]] = None
    # Depth range [min_depth, max_depth] for 3-D variables (e.g. o2, thetao).
    depth_range: Optional[tuple[float, float]] = None
    # Filename of the static source file at the configured output resolution.
    # Used by compile-only variables such as bathy.
    data_file: Optional[str] = None
    # High-resolution static source file. Used by bathy when extracting at
    # full native resolution (e.g. from SHP geometries).
    data_file_hires: Optional[str] = None
    # Set True for trajectory-format datasets (e.g. eddies) that require
    # spatial binning before they can be stored as a gridded Zarr.
    # The standard open_mfdataset pipeline is bypassed entirely.
    trajectory_format: bool = False
    # Set True for variables whose Zarr store uses lon/lat coordinate names that
    # must be renamed to x/y before rioxarray clip (e.g. AVISO fsle, eddies).
    rename_lonlat: bool = False
    # Discrete depths (metres) each 3-D variable is published at, keyed by the
    # variable's name *in the store* (e.g. {thetao: [0, 50, 100], uo: [0]}).
    # Each level becomes a column named <variable>_<level>; store variables
    # not listed pass through, which is what lets one store mix 2-D and 3-D
    # fields. Levels are matched to the store's axis by nearest depth.
    # Distinct from depth_range, the continuous band that is downloaded.
    # Used by compile, and the default for extraction. Resolve it through
    # depth_levels_for rather than reading it directly.
    depth_levels: Optional[dict[str, list[int]]] = None
    # Extraction-only override, same shape as depth_levels. Merged per
    # variable: a variable listed here replaces its depth_levels entry, the
    # others keep theirs.
    extract_depth_levels: Optional[dict[str, list[int]]] = None
    # Older single-variable forms of the two keys above, still accepted. A list
    # here means {var_key: [...]}, i.e. the store's variable shares the
    # var_key's name (o2, thetao). Not combinable with their newer key.
    extract_depth_slices: Optional[list[int]] = None
    compile_depth_slices: Optional[list[int]] = None
    # Exact variable names as they appear in the compiled h2ds Zarr for this
    # var_key. Used to select only these columns when adding a variable to an
    # existing Parquet store (--add-var). None means not yet declared.
    compiled_vars: Optional[list[str]] = None
    # Cadence of this variable's stored Zarr. DAILY (the default, and what every
    # existing store is) means one step per calendar day. HOURLY keeps the
    # source's sub-daily axis instead of aggregating it away at convert time —
    # for ERA5 that preserves the native resolution the raw GRIB already has.
    #
    # Read paths normalize a DAILY store's stamps to midnight (sources often
    # publish at 12:00); doing that to an HOURLY store would collapse 24 steps
    # onto one timestamp, so the normalization is skipped for it.
    time_step: TimeStep = TimeStep.DAILY
    # On-disk encoding. Default writes exactly what it always has; INT16 packs
    # to scale/offset integers (see StoreDtype). Only consulted when a store is
    # first created — an existing store keeps its own encoding through appends.
    store_dtype: StoreDtype = StoreDtype.FLOAT32
    # Days the provider never published, which therefore cannot be downloaded,
    # converted or backfilled. Each entry is either "YYYY-MM-DD" or a closed
    # interval "YYYY-MM-DD/YYYY-MM-DD".
    #
    # Needed because a source that ships one file per day produces an *axis*
    # hole when it skips one, which is otherwise indistinguishable from data
    # the pipeline lost — AVISO simply has no fsle file for 2025-06-02, and its
    # remote listing jumps 20250601 → 20250603. Without somewhere to record
    # that, the gap checks would report it on every run forever, and a check
    # that cries wolf is one people stop reading.
    #
    # Only for gaps confirmed absent at the source. Anything else is a defect
    # and belongs fixed, not listed.
    known_gaps: Optional[list[str]] = None
    # Root holding this variable's ``local_folder``, for stores that should not
    # live under STORE_ROOT — one drive for the hourly ERA5 stores, another for
    # the CMEMS dailies. None (the default, and what every shipped variable
    # uses) falls back to STORE_ROOT, so a config that declares none resolves
    # exactly as it always has.
    #
    # Must be absolute: ``resolve_store_path`` calls ``.resolve()`` on the
    # joined path, so a relative value would silently resolve against whatever
    # directory the process happened to start in.
    #
    # Outranked by ``--store-path``, which relocates a whole run on purpose.
    # See ``h2mare.utils.paths.store_root_for`` for the full precedence.
    store_root: Optional[str] = None
    # Variables computed at convert time and written to the native store beside
    # the ones downloaded, keyed by output name — e.g.
    # {gke: {op: kinetic_energy, source: [ugos, vgos]}}. Applied after the
    # var_key's registered processor, in declaration order, so an entry may
    # read an earlier one. A variable derived from a 3-D source keeps its depth
    # axis and needs its own depth_levels entry like any other.
    derived_vars: Optional[dict[str, DerivedVarSpec]] = None

    def __post_init__(self):
        if self.bbox is not None:
            lon_min, lat_min, lon_max, lat_max = self.bbox
            if not (-180 <= lon_min <= 180 and -180 <= lon_max <= 180):
                raise ValueError("Longitude must be between -180 and 180")
            if not (-90 <= lat_min <= 90 and -90 <= lat_max <= 90):
                raise ValueError("Latitude must be between -90 and 90")
            if lon_min >= lon_max:
                raise ValueError("lon_min must be less than lon_max")
            if lat_min >= lat_max:
                raise ValueError("lat_min must be less than lat_max")

        if self.depth_range is not None:
            if self.depth_range[0] >= self.depth_range[1]:
                raise ValueError("depth_min must be less than depth_max")

        _validate_depth_keys(
            "depth_levels",
            self.depth_levels,
            "compile_depth_slices",
            self.compile_depth_slices,
        )
        _validate_depth_keys(
            "extract_depth_levels",
            self.extract_depth_levels,
            "extract_depth_slices",
            self.extract_depth_slices,
        )

        # Range-mode parsing unpacks exactly two capture groups (start, end);
        # fail fast at config load rather than deep in the convert step with a
        # cryptic "not enough values to unpack".
        if self.filename_date_range:
            if self.pattern is None:
                raise ValueError(
                    "filename_date_range=True requires a `pattern` with 2 capture "
                    "groups (start, end)"
                )
            import re

            ngroups = re.compile(self.pattern).groups
            if ngroups != 2:
                raise ValueError(
                    "filename_date_range=True requires `pattern` to have exactly 2 "
                    f"capture groups (start, end); got {ngroups} in {self.pattern!r}"
                )

        # A relative store_root would resolve against the process cwd, so the
        # same config would point somewhere different depending on where the
        # command was run from. Fail at config load rather than write a store
        # somewhere nobody meant.
        #
        # Checked against both flavours rather than the running platform's:
        # "/data/store" is not absolute to PureWindowsPath (no drive) and
        # "D:\\data" is not absolute to PurePosixPath, so using plain Path here
        # would reject a config merely for having been written on the other OS.
        # What must be caught is a *relative* path, which neither accepts.
        if self.store_root is not None and not (
            PurePosixPath(self.store_root).is_absolute()
            or PureWindowsPath(self.store_root).is_absolute()
        ):
            raise ValueError(
                f"store_root must be an absolute path; got {self.store_root!r}. "
                "It is the root *above* local_folder — the same shape STORE_ROOT "
                "has in .env, not a single store's directory."
            )


class SecretsConfig(msgspec.Struct):
    """External service credentials."""

    aviso_ftp_server: Optional[str] = None
    aviso_username: Optional[str] = None
    aviso_password: Optional[str] = None


# VariablesConfig is now a plain dict — kept as a type alias for compatibility
VariablesConfig = dict[str, KeyVarConfigEntry]

# Variables that are derived/computed inside the pipeline and never downloaded
# from an external source. Excluded from download loops and catalog date-range
# inference; each has its own dedicated processing path in the Compiler.
SYSTEM_VAR_KEYS: frozenset[str] = frozenset({"h2ds", "bathy", "moon"})


def _check_derived_names(var_key: str, var_config: KeyVarConfigEntry) -> None:
    """
    Refuse derived_vars that would clash with each other or with depth_levels.

    Both write ``<variable>_<level>`` names, so ``ke: {depth: [0]}`` beside
    ``depth_levels: {ke: [0]}`` would describe one column twice — and the
    latter would fail only at compile, because the 2-D ``ke_0`` leaves no
    ``ke`` in the store to slice.
    """
    if not var_config.derived_vars:
        return

    written: dict[str, str] = {}
    for name, spec in var_config.derived_vars.items():
        for out in spec.output_names(name):
            if out in written:
                raise ValueError(
                    f"'{var_key}': derived_vars.{name} and "
                    f"derived_vars.{written[out]} both write '{out}'"
                )
            written[out] = name

    for purpose in ("compile", "extract"):
        levels = depth_levels_for(var_key, var_config, purpose)
        for var in levels:
            spec = var_config.derived_vars.get(var)
            if spec is not None and spec.depth is not None:
                raise ValueError(
                    f"'{var_key}': derived_vars.{var} is computed at depth "
                    f"{spec.depth} and stored without a depth axis, so depth "
                    f"levels cannot slice it. Drop '{var}' from the depth "
                    f"levels, or drop its `depth` to store every level."
                )
        clash = sorted(set(written) & set(depth_column_names(levels)))
        if clash:
            raise ValueError(
                f"'{var_key}': {clash} are written both by derived_vars and "
                f"by depth levels; keep one."
            )


class AppConfig(msgspec.Struct):
    """Complete application configuration."""

    variables: VariablesConfig
    secrets: SecretsConfig

    def __post_init__(self):
        for var_key, var_config in self.variables.items():
            _check_derived_names(var_key, var_config)

        # compiled_vars is written by hand and read by Parquet, routing and the
        # CF checks, so a depth column compile produces but compiled_vars omits
        # would be silently left out of all of them. The reverse slip matters as
        # much: a sliced variable listed bare (thetao) names a column h2ds never
        # holds. Only checked where compiled_vars is declared at all.
        for var_key, var_config in self.variables.items():
            declared = var_config.compiled_vars
            levels = depth_levels_for(var_key, var_config)
            if declared is None or not levels:
                continue

            missing = [c for c in depth_column_names(levels) if c not in declared]
            unsliced = [v for v in levels if v in declared]
            if not (missing or unsliced):
                continue

            # The list to write: each bare name replaced in place by its level
            # columns, then anything still missing, so the order the author
            # chose survives.
            suggested: list[str] = []
            for name in declared:
                if name in unsliced:
                    suggested.extend(depth_column_names({name: levels[name]}))
                else:
                    suggested.append(name)
            suggested = list(dict.fromkeys([*suggested, *missing]))

            problems = []
            if unsliced:
                problems.append(
                    f"{unsliced} are sliced by depth, so h2ds never holds a "
                    f"column by that name"
                )
            if missing:
                problems.append(f"the depth columns {missing} are not listed")
            raise ValueError(
                f"'{var_key}': compiled_vars must name the columns compile "
                f"writes, but {'; and '.join(problems)}. Replace it with: "
                f"compiled_vars: [{', '.join(suggested)}]"
            )
