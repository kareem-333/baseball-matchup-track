# Architecture

## Five layers

1. **Sources** — fetch raw data from external systems. Only layer allowed to make network calls.
2. **Storage** — cache raw data locally as Parquet. Swappable for cloud later.
3. **Transforms** — pure functions that shape raw data into analysis-ready tables.
4. **Metrics** — composite scores and models built on transforms.
5. **Presentation** — Streamlit UI. Imports from layers above; contains no logic.

## Data flow

User opens dashboard → `app.py` asks source for data → source checks storage cache → on miss, source fetches from API and writes to storage → `app.py` runs transforms on the raw data → `app.py` calls metrics on transformed data → `app.py` renders results with Streamlit components

## Layer rules

- **Sources** never transform. They return raw API output.
- **Storage** never knows what's in the data. Just bytes keyed by string.
- **Transforms** never do I/O. Pure functions only.
- **Metrics** never do I/O. Pure functions or classes only.
- **Presentation** never computes. It only composes.

## Directory structure

```
baseball-matchup-track/
├── data/
│   ├── raw/          # cached API responses as Parquet, never edited by hand
│   ├── interim/      # intermediate cleaned data, optional
│   └── processed/    # analysis-ready tables produced by transforms
├── sources/          # Layer 1: raw data fetchers
│   ├── base.py       # abstract Source class
│   └── mlb_stats.py  # MLB Stats API source
├── storage/          # Layer 2: where cached data lives
│   ├── base.py       # abstract Storage interface
│   ├── local_parquet.py  # local Parquet implementation
│   └── query.py      # DuckDB query helpers
├── transforms/       # Layer 3: pure functions on dataframes
│   ├── pitchers.py
│   ├── batters.py
│   └── lineups.py
├── metrics/          # Layer 4: composite scores and models
│   ├── mash.py       # Matchup Advantage Score
│   └── miss.py       # Whiff Score
├── dashboard/        # Layer 5: Streamlit UI
│   ├── app.py        # entry point, only Streamlit code
│   ├── pages/        # multi-page Streamlit, if used
│   └── components/   # chart and widget builders
├── tests/
├── notebooks/        # exploratory work, not imported by app
└── archive/          # moved files, kept for reference, not imported
```

## Swapping storage backends

To replace `LocalParquetStorage` with an S3 implementation:
1. Write `storage/s3_parquet.py` implementing the `Storage` abstract interface.
2. In `dashboard/app.py`, change one line: `from storage.s3_parquet import S3ParquetStorage` and update the constructor call.

No other files need to change. This is the guarantee the `Storage` interface provides.

## Key invariants

- `dashboard/app.py` contains zero `requests.get(...)` calls and zero direct file reads.
- All I/O flows through `sources/` and `storage/`.
- `data/raw/`, `data/interim/`, and `data/processed/` are gitignored — derived data is never committed.
- Once data is cached as Parquet, it can be queried with SQL via `storage/query.py` without loading entire files into memory.
