"""Mean reversion indicators."""

import pandas as pd
import numpy as np
from indicators.base import Indicator


class RSI(Indicator):
    """Relative Strength Index.

    Oscillator that measures the speed and magnitude of price changes.
    - Below 30: Oversold
    - Above 70: Overbought
    - Around 50: Neutral

    Uses Wilder's smoothing method.
    """

    def __init__(self, period: int = 14):
        """Initialize RSI indicator.

        Args:
            period: Period for RSI calculation (default 14)
        """
        super().__init__(period)

    def calculate(self, data: pd.Series) -> pd.Series:
        """Calculate RSI using Wilder's smoothing.

        Args:
            data: Price series (typically close prices)

        Returns:
            Series with RSI values (0-100)
        """
        # Calculate price changes
        delta = data.diff()

        # Separate gains and losses
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)

        # Wilder's smoothing (EMA with alpha = 1/period)
        avg_gain = gain.ewm(alpha=1/self.period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/self.period, adjust=False).mean()

        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        # Handle edge cases (division by zero)
        rsi = rsi.fillna(50)  # Neutral if no losses

        return rsi


class BollingerBands(Indicator):
    """Bollinger Bands.

    Consists of:
    - Middle band: Simple moving average
    - Upper band: Middle + (std_dev * multiplier)
    - Lower band: Middle - (std_dev * multiplier)

    Price touching upper band suggests overbought,
    touching lower band suggests oversold.
    """

    def __init__(self, period: int = 20, std_multiplier: float = 2.0):
        """Initialize Bollinger Bands.

        Args:
            period: Period for moving average and std dev (default 20)
            std_multiplier: Number of std devs for bands (default 2.0)
        """
        super().__init__(period)
        self.std_multiplier = std_multiplier

    def calculate(self, data: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands.

        Args:
            data: Price series (typically close prices)

        Returns:
            Tuple of (middle, upper, lower) bands as Series
        """
        # Middle band (SMA)
        middle = data.rolling(window=self.period).mean()

        # Standard deviation
        std = data.rolling(window=self.period).std()

        # Upper and lower bands
        upper = middle + (std * self.std_multiplier)
        lower = middle - (std * self.std_multiplier)

        return middle, upper, lower

    def calculate_width(self, data: pd.Series) -> pd.Series:
        """Calculate Bollinger Band Width.

        Useful for detecting volatility squeezes.

        Args:
            data: Price series

        Returns:
            Series with band width percentage
        """
        middle, upper, lower = self.calculate(data)

        # Width as percentage of middle band
        width = ((upper - lower) / middle) * 100

        return width

    def calculate_percent_b(self, data: pd.Series) -> pd.Series:
        """%B indicator - shows where price is relative to bands.

        - Above 1.0: Price above upper band
        - Below 0.0: Price below lower band
        - 0.5: Price at middle band

        Args:
            data: Price series

        Returns:
            Series with %B values
        """
        middle, upper, lower = self.calculate(data)

        percent_b = (data - lower) / (upper - lower)

        return percent_b

    def __repr__(self) -> str:
        return f"BollingerBands(period={self.period}, std={self.std_multiplier})"


class ZScore(Indicator):
    """Z-Score indicator.

    Measures how many standard deviations the current price
    is from the mean.

    - Positive Z-score: Price above mean
    - Negative Z-score: Price below mean
    - |Z| > 2: Potential mean reversion opportunity
    """

    def __init__(self, period: int = 20):
        """Initialize Z-Score indicator.

        Args:
            period: Lookback period for mean and std dev
        """
        super().__init__(period)

    def calculate(self, data: pd.Series) -> pd.Series:
        """Calculate Z-Score.

        Args:
            data: Price series

        Returns:
            Series with Z-Score values
        """
        mean = data.rolling(window=self.period).mean()
        std = data.rolling(window=self.period).std()

        z_score = (data - mean) / std

        # Handle division by zero
        z_score = z_score.replace([np.inf, -np.inf], 0)
        z_score = z_score.fillna(0)

        return z_score
