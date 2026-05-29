"""Abstract base class for format converters."""

from __future__ import annotations

from abc import ABC, abstractmethod


# Intentionally minimal — converters have different enough constructors that
# shared __init__ setup would be contrived. The base only enforces the run() contract.
class BaseConverter(ABC):
    @abstractmethod
    def run(self) -> bool: ...
