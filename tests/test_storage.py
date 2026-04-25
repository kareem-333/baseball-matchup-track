import time

import pandas as pd
import pytest

from storage.local_parquet import LocalParquetStorage
from storage.query import query


@pytest.fixture
def storage(tmp_path):
    return LocalParquetStorage(base_dir=tmp_path)


@pytest.fixture
def sample_df():
    return pd.DataFrame({"pitcher": ["Cole", "Scherzer"], "velocity": [96.5, 93.1]})


def test_write_and_read_roundtrip(storage, sample_df):
    storage.write("test/sample", sample_df)
    result = storage.read("test/sample")
    pd.testing.assert_frame_equal(result, sample_df)


def test_exists_true_after_write(storage, sample_df):
    assert not storage.exists("test/sample")
    storage.write("test/sample", sample_df)
    assert storage.exists("test/sample")


def test_exists_false_before_write(storage):
    assert not storage.exists("nonexistent/key")


def test_age_approximately_zero_after_write(storage, sample_df):
    storage.write("test/age", sample_df)
    age = storage.age("test/age")
    assert age is not None
    assert age.total_seconds() < 5


def test_age_none_when_missing(storage):
    assert storage.age("nonexistent/key") is None


def test_delete_removes_file(storage, sample_df):
    storage.write("test/del", sample_df)
    assert storage.exists("test/del")
    storage.delete("test/del")
    assert not storage.exists("test/del")


def test_delete_nonexistent_is_safe(storage):
    storage.delete("nonexistent/key")  # must not raise


def test_nested_key_creates_directories(storage, sample_df):
    storage.write("mlb_stats/schedule/2026-04-24", sample_df)
    result = storage.read("mlb_stats/schedule/2026-04-24")
    pd.testing.assert_frame_equal(result, sample_df)


def test_query_aggregation(tmp_path):
    df = pd.DataFrame({"pitcher": ["Cole", "Cole", "Scherzer"], "velocity": [96.0, 97.0, 93.0]})
    storage = LocalParquetStorage(base_dir=tmp_path)
    storage.write("pitch_data/test", df)

    parquet_path = str(tmp_path / "pitch_data" / "test.parquet")
    result = query(f"SELECT pitcher, AVG(velocity) AS avg_velo FROM read_parquet('{parquet_path}') GROUP BY pitcher ORDER BY pitcher")

    assert len(result) == 2
    cole_row = result[result["pitcher"] == "Cole"].iloc[0]
    assert abs(cole_row["avg_velo"] - 96.5) < 0.01
