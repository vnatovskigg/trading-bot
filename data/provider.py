"""Unified data provider interface for backtest and live modes."""

from datetime import datetime
from typing import Optional, Dict
import pandas as pd

from data.models import BarSeries
from data.storage import OHLCVStorage


class DataProvider:
    """Unified interface for accessing market data.

    Provides consistent API for both backtesting and live trading modes.
    Enforces no look-ahead bias by only returning fully closed candles.

    Attributes:
        storage: OHLCVStorage instance
        mode: Operating mode ('backtest' or 'live')
    """

    def __init__(self, storage: Optional[OHLCVStorage] = None, mode: str = 'backtest'):
        """Initialize data provider.

        Args:
            storage: OHLCVStorage instance, creates new if None
            mode: Operating mode ('backtest' or 'live')
        """
        self.storage = storage or OHLCVStorage()
        self.mode = mode

        if mode not in ['backtest', 'live']:
            raise ValueError(f"Invalid mode: {mode}. Must be 'backtest' or 'live'")

    def get_bars(
        self,
        symbol: str,
        timeframe: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> BarSeries:
        """Get historical bars for a symbol and timeframe.

        IMPORTANT: This method enforces no look-ahead bias. In backtest mode,
        the 'end' parameter is treated as exclusive - the candle at 'end' time
        is NOT included.

        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe string
            start: Start datetime (inclusive), None for beginning
            end: End datetime (EXCLUSIVE - not included), None for all data
            limit: Maximum number of bars to return

        Returns:
            BarSeries with requested data
        """
        return self.storage.get_candles(
            symbol=symbol,
            timeframe=timeframe,
            start=start,
            end=end,
            limit=limit
        )

    def get_latest_bars(
        self,
        symbol: str,
        timeframe: str,
        count: int
    ) -> BarSeries:
        """Get the most recent N closed bars.

        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe string
            count: Number of bars to retrieve

        Returns:
            BarSeries with latest N bars
        """
        all_bars = self.storage.get_candles(symbol, timeframe)

        if len(all_bars) == 0:
            return all_bars

        return all_bars.get_latest(count)

    def get_aligned_timeframes(
        self,
        symbol: str,
        timeframes: list[str],
        end_timestamp: datetime,
        lookback_bars: int = 500
    ) -> Dict[str, BarSeries]:
        """Get data for multiple timeframes aligned to a specific timestamp.

        Useful for strategies that use multiple timeframes (e.g., trend on 1h,
        mean reversion on 15m).

        Args:
            symbol: Trading pair symbol
            timeframes: List of timeframe strings
            end_timestamp: End time (exclusive) to align all timeframes to
            lookback_bars: Number of bars to fetch for each timeframe

        Returns:
            Dictionary mapping timeframe to BarSeries
        """
        result = {}

        for tf in timeframes:
            bars = self.storage.get_candles(
                symbol=symbol,
                timeframe=tf,
                end=end_timestamp,
                limit=lookback_bars
            )
            result[tf] = bars

        return result

    def validate_completeness(
        self,
        bars: BarSeries,
        expected_interval_minutes: int,
        max_gap_multiplier: float = 2.0
    ) -> tuple[bool, list]:
        """Check if bar data is complete (no significant gaps).

        Args:
            bars: BarSeries to validate
            expected_interval_minutes: Expected minutes between bars
            max_gap_multiplier: Gap threshold multiplier

        Returns:
            Tuple of (is_complete, list_of_gaps)
        """
        gaps = bars.check_gaps(expected_interval_minutes, max_gap_multiplier)
        return len(gaps) == 0, gaps

    def get_data_summary(self) -> pd.DataFrame:
        """Get summary of available data.

        Returns:
            DataFrame with symbol, timeframe, count, date range
        """
        return self.storage.get_summary()

    def get_date_range(
        self,
        symbol: str,
        timeframe: str
    ) -> tuple[Optional[datetime], Optional[datetime]]:
        """Get available date range for a symbol/timeframe.

        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe string

        Returns:
            Tuple of (earliest_timestamp, latest_timestamp)
        """
        earliest = self.storage.get_earliest_timestamp(symbol, timeframe)
        latest = self.storage.get_latest_timestamp(symbol, timeframe)

        return earliest, latest
