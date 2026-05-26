"""Abstract base class for format converters."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseConverter(ABC):
    @abstractmethod
    def run(self) -> bool: ...

    def validate(self) -> None:
        pass
