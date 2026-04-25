# Layer 2: Storage interface
# Handles caching of raw data. Must not know what is in the data — just bytes keyed by string.
# Callers own TTL logic; this layer owns read/write/exists/age/delete.

from abc import ABC, abstractmethod
from datetime import timedelta

import pandas as pd


class Storage(ABC):
    @abstractmethod
    def write(self, key: str, df: pd.DataFrame) -> None: ...

    @abstractmethod
    def read(self, key: str) -> pd.DataFrame: ...

    @abstractmethod
    def exists(self, key: str) -> bool: ...

    @abstractmethod
    def age(self, key: str) -> timedelta | None:
        """Return time since last write, or None if key does not exist."""
        ...

    @abstractmethod
    def delete(self, key: str) -> None: ...
