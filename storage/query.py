# Layer 2: DuckDB query helper.
# Lets callers run SQL directly against cached Parquet files without loading entire files into memory.
# DuckDB handles predicate pushdown and file scanning automatically.

import duckdb
import pandas as pd

from storage.base import Storage


def query(sql: str, storage: Storage | None = None) -> pd.DataFrame:
    """Run a DuckDB SQL query that can read Parquet files directly.

    SQL can reference cached data with read_parquet() paths. Example:

        SELECT pitcher_name, AVG(velocity) AS avg_velo
        FROM read_parquet('data/raw/mlb_stats/pitch_data/*.parquet')
        GROUP BY pitcher_name

    The storage parameter is accepted for future use (e.g. resolving logical keys
    to physical paths), but DuckDB can scan files directly via read_parquet().
    """
    conn = duckdb.connect()
    return conn.execute(sql).df()
