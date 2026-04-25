# Layer 2: Local Parquet implementation of the Storage interface.
# Translates logical keys (e.g. 'mlb_stats/schedule/2026-04-24') into file paths under base_dir.
# Must not contain business logic. Swap this file for s3_parquet.py to move to cloud storage.

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from storage.base import Storage


class LocalParquetStorage(Storage):
    def __init__(self, base_dir: str | Path = "data/raw"):
        self._base = Path(base_dir)

    def _path(self, key: str) -> Path:
        return self._base / (key.strip("/") + ".parquet")

    def write(self, key: str, df: pd.DataFrame) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, engine="pyarrow", compression="snappy")

    def read(self, key: str) -> pd.DataFrame:
        return pd.read_parquet(self._path(key), engine="pyarrow")

    def exists(self, key: str) -> bool:
        return self._path(key).exists()

    def age(self, key: str) -> timedelta | None:
        path = self._path(key)
        if not path.exists():
            return None
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        return datetime.now(tz=timezone.utc) - mtime

    def delete(self, key: str) -> None:
        path = self._path(key)
        if path.exists():
            path.unlink()
