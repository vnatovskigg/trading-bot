"""Volatility indicators."""

import pandas as pd
import numpy as np
from indicators.base import Indicator


class ATR(Indicator):
    """Average True Range.

    Measures market volatility by calculating the average of true ranges.
    Higher ATR = higher volatility.

    Uses Wilder's smoothing method (similar to EMA).
    """

    def __init__(self, period: int = 14):
        """Initialize ATR indicator.

        Args:
            period: Period for smoothing (default 14)
        """
        super().__init__(period)

    def calculate(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series
    ) -> pd.Series:
        """Calculate ATR.

        Args:
            high: High prices
            low: Low prices
            close: Close prices

        Returns:
            Series with ATR values
        """
        # Calculate True Range
        # TR = max(high - low, abs(high - prev_close), abs(low - prev_close))
        prev_close = close.shift(1)

        tr1 = high - low
        tr2 = np.abs(high - prev_close)
        tr3 = np.abs(low - prev_close)

        # True range is the max of the three
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Smooth using Wilder's method (EMA with alpha = 1/period)
        atr = tr.ewm(alpha=1/self.period, adjust=False).mean()

        return atr

    def calculate_percent(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series
    ) -> pd.Series:
        """Calculate ATR as percentage of price.

        Useful for comparing volatility across different price levels.

        Args:
            high: High prices
            low: Low prices
            close: Close prices

        Returns:
            Series with ATR as percentage of close price
        """
        atr = self.calculate(high, low, close)
        atr_percent = (atr / close) * 100

        return atr_percent


class RealizedVolatility(Indicator):
    """Realized Volatility (Historical Volatility).

    Calculates volatility based on log returns over a period.
    Commonly annualized for comparison with implied volatility.
    """

    def __init__(self, period: int = 20, annualize: bool = True, periods_per_year: int = 365):
        """Initialize Realized Volatility.

        Args:
            period: Lookback period for calculating std dev
            annualize: Whether to annualize the volatility
            periods_per_year: Periods per year for annualization (365 for daily, 365*24 for hourly)
        """
        super().__init__(period)
        self.annualize = annualize
        self.periods_per_year = periods_per_year

    def calculate(self, data: pd.Series) -> pd.Series:
        """Calculate realized volatility.

        Args:
            data: Price series (typically close prices)

        Returns:
            Series with volatility values (annualized if specified)
        """
        # Calculate log returns
        log_returns = np.log(data / data.shift(1))

        # Calculate rolling standard deviation
        volatility = log_returns.rolling(window=self.period).std()

        # Annualize if requested
        if self.annualize:
            volatility = volatility * np.sqrt(self.periods_per_year)

        return volatility

    def __repr__(self) -> str:
        return f"RealizedVolatility(period={self.period}, annualized={self.annualize})"


class BollingerBandWidth(Indicator):
    """Bollinger Band Width.

    Measures the width of Bollinger Bands as a percentage.
    Low values indicate low volatility (squeeze),
    which often precedes volatility expansion.
    """

    def __init__(self, period: int = 20, std_multiplier: float = 2.0):
        """Initialize BB Width.

        Args:
            period: Period for Bollinger Bands
            std_multiplier: Standard deviation multiplier
        """
        super().__init__(period)
        self.std_multiplier = std_multiplier

    def calculate(self, data: pd.Series) -> pd.Series:
        """Calculate Bollinger Band Width.

        Args:
            data: Price series

        Returns:
            Series with BB Width values (percentage)
        """
        # Calculate Bollinger Bands
        middle = data.rolling(window=self.period).mean()
        std = data.rolling(window=self.period).std()

        upper = middle + (std * self.std_multiplier)
        lower = middle - (std * self.std_multiplier)

        # Width as percentage of middle band
        width = ((upper - lower) / middle) * 100

        return width

    def __repr__(self) -> str:
        return f"BBWidth(period={self.period}, std={self.std_multiplier})"
