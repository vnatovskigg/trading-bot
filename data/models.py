"""Data models for market data."""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Optional
import pandas as pd
import numpy as np


@dataclass
class Candle:
    """Represents a single OHLCV candle.

    Attributes:
        timestamp: Opening time of the candle (UTC)
        open: Opening price
        high: Highest price during the period
        low: Lowest price during the period
        close: Closing price
        volume: Trading volume
    """
    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal

    def __post_init__(self):
        """Validate candle data."""
        # Ensure high is highest
        if self.high < self.low:
            raise ValueError(f"High ({self.high}) cannot be less than low ({self.low})")

        if self.high < self.open or self.high < self.close:
            raise ValueError(f"High ({self.high}) must be >= open and close")

        if self.low > self.open or self.low > self.close:
            raise ValueError(f"Low ({self.low}) must be <= open and close")

        if self.volume < 0:
            raise ValueError(f"Volume ({self.volume}) cannot be negative")

    @classmethod
    def from_ccxt(cls, data: list, timestamp_ms: bool = True) -> 'Candle':
        """Create Candle from CCXT format [timestamp, open, high, low, close, volume].

        Args:
            data: List in CCXT format
            timestamp_ms: Whether timestamp is in milliseconds (default: True)

        Returns:
            Candle instance
        """
        timestamp = datetime.fromtimestamp(data[0] / 1000 if timestamp_ms else data[0])

        return cls(
            timestamp=timestamp.replace(tzinfo=None),  # Store as naive UTC
            open=Decimal(str(data[1])),
            high=Decimal(str(data[2])),
            low=Decimal(str(data[3])),
            close=Decimal(str(data[4])),
            volume=Decimal(str(data[5]))
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for database storage."""
        return {
            'timestamp': self.timestamp,
            'open': float(self.open),
            'high': float(self.high),
            'low': float(self.low),
            'close': float(self.close),
            'volume': float(self.volume)
        }


class BarSeries:
    """Wrapper around pandas DataFrame for OHLCV data with validation.

    Provides a clean interface for accessing and manipulating bar data
    with built-in validation and helper methods.

    Attributes:
        df: DataFrame with DatetimeIndex and OHLCV columns
    """

    REQUIRED_COLUMNS = ['open', 'high', 'low', 'close', 'volume']

    def __init__(self, df: pd.DataFrame):
        """Initialize BarSeries with validation.

        Args:
            df: DataFrame with DatetimeIndex and OHLCV columns

        Raises:
            ValueError: If DataFrame doesn't meet requirements
        """
        self.df = df.copy()
        self._validate()

    def _validate(self):
        """Validate the DataFrame structure and data."""
        # Check index
        if not isinstance(self.df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have DatetimeIndex")

        # Check required columns
        missing_cols = set(self.REQUIRED_COLUMNS) - set(self.df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Check sorted
        if not self.df.index.is_monotonic_increasing:
            raise ValueError("DataFrame index must be sorted chronologically")

        # Check for nulls in OHLCV columns
        null_counts = self.df[self.REQUIRED_COLUMNS].isnull().sum()
        if null_counts.any():
            raise ValueError(f"Null values found in columns: {null_counts[null_counts > 0].to_dict()}")

        # Validate OHLC relationships
        if len(self.df) > 0:
            invalid_high = (self.df['high'] < self.df['low']).any()
            if invalid_high:
                raise ValueError("Some candles have high < low")

    def __len__(self) -> int:
        """Return number of candles."""
        return len(self.df)

    def __getitem__(self, key):
        """Allow indexing like a DataFrame."""
        return self.df[key]

    @classmethod
    def from_candles(cls, candles: list[Candle]) -> 'BarSeries':
        """Create BarSeries from list of Candle objects.

        Args:
            candles: List of Candle objects

        Returns:
            BarSeries instance
        """
        if not candles:
            # Return empty BarSeries
            df = pd.DataFrame(columns=cls.REQUIRED_COLUMNS)
            df.index = pd.DatetimeIndex([])
            df.index.name = 'timestamp'
            return cls(df)

        data = []
        for candle in candles:
            data.append({
                'timestamp': candle.timestamp,
                'open': float(candle.open),
                'high': float(candle.high),
                'low': float(candle.low),
                'close': float(candle.close),
                'volume': float(candle.volume)
            })

        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)

        return cls(df)

    def get_closes(self) -> pd.Series:
        """Get closing prices as Series."""
        return self.df['close']

    def get_highs(self) -> pd.Series:
        """Get high prices as Series."""
        return self.df['high']

    def get_lows(self) -> pd.Series:
        """Get low prices as Series."""
        return self.df['low']

    def get_opens(self) -> pd.Series:
        """Get opening prices as Series."""
        return self.df['open']

    def get_volumes(self) -> pd.Series:
        """Get volumes as Series."""
        return self.df['volume']

    def get_latest(self, n: int = 1) -> 'BarSeries':
        """Get the latest N candles.

        Args:
            n: Number of candles to retrieve

        Returns:
            New BarSeries with latest N candles
        """
        return BarSeries(self.df.tail(n))

    def get_range(self, start: Optional[datetime] = None, end: Optional[datetime] = None) -> 'BarSeries':
        """Get candles within a time range.

        Args:
            start: Start datetime (inclusive), None for beginning
            end: End datetime (exclusive), None for end

        Returns:
            New BarSeries with candles in range
        """
        df = self.df

        if start is not None:
            df = df[df.index >= start]

        if end is not None:
            df = df[df.index < end]

        return BarSeries(df) if len(df) > 0 else BarSeries(pd.DataFrame(columns=self.REQUIRED_COLUMNS))

    def resample(self, timeframe: str) -> 'BarSeries':
        """Resample to a different timeframe.

        Args:
            timeframe: Pandas-compatible frequency string (e.g., '1H', '15T', '1D')

        Returns:
            New BarSeries with resampled data
        """
        resampled = self.df.resample(timeframe).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()

        return BarSeries(resampled)

    def to_candles(self) -> list[Candle]:
        """Convert to list of Candle objects.

        Returns:
            List of Candle objects
        """
        candles = []
        for timestamp, row in self.df.iterrows():
            candles.append(Candle(
                timestamp=timestamp,
                open=Decimal(str(row['open'])),
                high=Decimal(str(row['high'])),
                low=Decimal(str(row['low'])),
                close=Decimal(str(row['close'])),
                volume=Decimal(str(row['volume']))
            ))
        return candles

    def check_gaps(self, expected_interval_minutes: int, max_gap_multiplier: float = 2.0) -> list[tuple[datetime, datetime]]:
        """Detect gaps in the data.

        Args:
            expected_interval_minutes: Expected time between candles in minutes
            max_gap_multiplier: Gap is detected if interval > expected * multiplier

        Returns:
            List of (start, end) tuples representing gaps
        """
        if len(self.df) < 2:
            return []

        gaps = []
        expected_delta = pd.Timedelta(minutes=expected_interval_minutes)
        max_delta = expected_delta * max_gap_multiplier

        for i in range(1, len(self.df)):
            actual_delta = self.df.index[i] - self.df.index[i-1]
            if actual_delta > max_delta:
                gaps.append((self.df.index[i-1], self.df.index[i]))

        return gaps

    def summary(self) -> str:
        """Get a summary string of the BarSeries.

        Returns:
            Human-readable summary
        """
        if len(self.df) == 0:
            return "Empty BarSeries"

        return f"""BarSeries Summary:
  Rows: {len(self.df)}
  Start: {self.df.index[0]}
  End: {self.df.index[-1]}
  Duration: {self.df.index[-1] - self.df.index[0]}
  Close Range: {self.df['close'].min():.2f} - {self.df['close'].max():.2f}
  Total Volume: {self.df['volume'].sum():.2f}
"""
