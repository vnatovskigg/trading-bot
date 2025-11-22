"""Trend-following indicators."""

import pandas as pd
import numpy as np
from indicators.base import Indicator


class SMA(Indicator):
    """Simple Moving Average.

    Calculates the arithmetic mean of prices over a specified period.
    """

    def calculate(self, data: pd.Series) -> pd.Series:
        """Calculate SMA.

        Args:
            data: Price series (typically close prices)

        Returns:
            Series with SMA values
        """
        return data.rolling(window=self.period).mean()


class EMA(Indicator):
    """Exponential Moving Average.

    Gives more weight to recent prices using exponential smoothing.
    """

    def calculate(self, data: pd.Series) -> pd.Series:
        """Calculate EMA.

        Args:
            data: Price series (typically close prices)

        Returns:
            Series with EMA values
        """
        return data.ewm(span=self.period, adjust=False).mean()


class ADX(Indicator):
    """Average Directional Index.

    Measures trend strength on a scale of 0-100.
    - Below 20: Weak trend (choppy market)
    - 20-25: Emerging trend
    - 25-50: Strong trend
    - Above 50: Very strong trend

    Returns ADX, +DI, and -DI values.
    """

    def __init__(self, period: int = 14):
        """Initialize ADX indicator.

        Args:
            period: Period for smoothing (default 14)
        """
        super().__init__(period)

    def calculate(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate ADX and directional indicators.

        Args:
            high: High prices
            low: Low prices
            close: Close prices

        Returns:
            Tuple of (ADX, +DI, -DI) as Series
        """
        # Calculate True Range
        high_low = high - low
        high_close = np.abs(high - close.shift(1))
        low_close = np.abs(low - close.shift(1))

        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        # Calculate Directional Movement
        high_diff = high.diff()
        low_diff = -low.diff()

        # +DM and -DM
        plus_dm = pd.Series(0.0, index=high.index)
        minus_dm = pd.Series(0.0, index=high.index)

        # +DM when high_diff > low_diff and high_diff > 0
        plus_dm[(high_diff > low_diff) & (high_diff > 0)] = high_diff

        # -DM when low_diff > high_diff and low_diff > 0
        minus_dm[(low_diff > high_diff) & (low_diff > 0)] = low_diff

        # Smooth TR, +DM, -DM using Wilder's smoothing (like EMA with alpha=1/period)
        atr = tr.ewm(alpha=1/self.period, adjust=False).mean()
        plus_dm_smooth = plus_dm.ewm(alpha=1/self.period, adjust=False).mean()
        minus_dm_smooth = minus_dm.ewm(alpha=1/self.period, adjust=False).mean()

        # Calculate +DI and -DI
        plus_di = 100 * (plus_dm_smooth / atr)
        minus_di = 100 * (minus_dm_smooth / atr)

        # Calculate DX
        di_diff = np.abs(plus_di - minus_di)
        di_sum = plus_di + minus_di

        dx = 100 * (di_diff / di_sum)
        dx = dx.replace([np.inf, -np.inf], 0)  # Handle division by zero

        # Calculate ADX (smoothed DX)
        adx = dx.ewm(alpha=1/self.period, adjust=False).mean()

        return adx, plus_di, minus_di

    def get_warmup_period(self) -> int:
        """ADX needs extra warmup due to double smoothing.

        Returns:
            Warmup period (2x period + buffer)
        """
        return self.period * 2 + 10
