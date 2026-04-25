# Layer 1: Abstract source base class.
# Sources are the only modules allowed to make network calls.
# They know nothing about MASH, MISS, lineups, or Streamlit.

import logging
from abc import ABC, abstractmethod


class Source(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """String identifier used in cache key paths (e.g. 'mlb_stats')."""
        ...

    @abstractmethod
    def fetch(self, endpoint: str, params: dict):
        """Low-level fetch. Subclasses call this internally; callers use typed methods."""
        ...

    @property
    def logger(self) -> logging.Logger:
        return logging.getLogger(f"sources.{self.name}")
