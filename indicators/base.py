"""Base class for technical indicators."""

from abc import ABC, abstractmethod
from typing import Union, Optional
import pandas as pd
import numpy as np


class Indicator(ABC):
    """Abstract base class for all technical indicators.

    Provides common functionality:
    - Warmup period tracking
    - Validity checking
    - Simple caching (optional)

    Subclasses must implement the calculate() method.
    """

    def __init__(self, period: int, name: Optional[str] = None):
        """Initialize indicator.

        Args:
            period: Lookback period for the indicator
            name: Optional name for the indicator
        """
        self.period = period
        self.name = name or self.__class__.__name__
        self._last_input_hash = None
        self._last_result = None

    @abstractmethod
    def calculate(self, *args, **kwargs) -> Union[pd.Series, tuple]:
        """Calculate the indicator values.

        Must be implemented by subclasses.

        Returns:
            pd.Series with indicator values, or tuple of Series for multi-output indicators
        """
        pass

    def is_valid(self, data_length: int) -> bool:
        """Check if there's enough data for valid calculation.

        Args:
            data_length: Length of input data

        Returns:
            True if enough data, False otherwise
        """
        return data_length >= self.period

    def get_warmup_period(self) -> int:
        """Get the minimum number of bars needed before indicator is valid.

        Returns:
            Warmup period in number of bars
        """
        return self.period

    def __call__(self, *args, **kwargs):
        """Allow calling the indicator directly like a function.

        Returns:
            Result of calculate()
        """
        return self.calculate(*args, **kwargs)

    def __repr__(self) -> str:
        return f"{self.name}(period={self.period})"
