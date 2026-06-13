# parquet2csv

`parquet2csv` exports a date-filtered slice of the Parquet store to CSV files, one file per day, month, or year.

```python
from h2mare.format_converters.parquet2csv import parquet2csv

parquet2csv(
    parquet_root="data/processed/parquet",
    csv_root="data/processed/csv",
    start_date="2021-01-01",
    end_date="2021-12-31",
    freq="month",
)
```

---

## Parameters

```python
parquet2csv(
    parquet_root,         # path to Parquet store (file or directory)
    csv_root,             # output directory for CSV files
    start_date,
    end_date,
    freq="day",           # "day" | "month" | "year"
    n_workers=8,
)
```

| Parameter | Default | Description |
|---|---|---|
| `parquet_root` | — | Path to the Parquet store (file or Hive-partitioned directory) |
| `csv_root` | — | Root output directory; year subdirectories are created automatically |
| `start_date` | — | Start of export period (`str`, e.g. `"2021-01-01"`) |
| `end_date` | — | End of export period (`str`, e.g. `"2021-12-31"`) |
| `freq` | `"day"` | Output granularity: `"day"`, `"month"`, or `"year"` |
| `n_workers` | `8` | Number of threads for parallel CSV writes |

Returns the `csv_root` directory that was written.

---

## Output layout

```
csv_root/
└── YYYY/
    ├── YYYY-MM-DD.csv   # day
    ├── YYYY-MM.csv      # month
    └── YYYY.csv         # year
```

Each file contains a `time` column (formatted as `YYYY-MM-DD`) plus one column per variable. Hive partition columns (`year`, `month`) are stripped. Rows where every variable column is empty (null or NaN) are dropped.
