"""Base class for trading strategies."""

from abc import ABC, abstractmethod
from typing import Dict, Optional
import pandas as pd

from data.models import BarSeries
from strategies.models import TargetPosition


class Strategy(ABC):
    """Abstract base class for all trading strategies.

    Provides common interface and functionality for strategy implementations.

    Subclasses must implement:
    - on_new_candle(): Process new market data and return target position
    """

    def __init__(self, config: dict, name: Optional[str] = None):
        """Initialize strategy.

        Args:
            config: Strategy configuration parameters
            name: Optional strategy name (defaults to class name)
        """
        self.config = config
        self.name = name or self.__class__.__name__
        self.state = {}  # Internal state tracking

    @abstractmethod
    def on_new_candle(
        self,
        symbol: str,
        timeframe: str,
        bars: BarSeries,
        indicators: Dict[str, pd.Series]
    ) -> TargetPosition:
        """Process new candle and return target position.

        This is called for each new closed candle during backtesting or live trading.

        IMPORTANT: Only use data from 'bars' and 'indicators' - these represent
        historical data up to (but not including) the current bar being processed.
        Never access future data.

        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe of the data
            bars: Historical OHLCV data (BarSeries)
            indicators: Pre-calculated indicators (dict of Series)

        Returns:
            TargetPosition representing desired exposure
        """
        pass

    def reset_state(self):
        """Reset internal state.

        Called before starting a new backtest run or switching modes.
        Subclasses can override to add custom reset logic.
        """
        self.state = {}

    def get_required_indicators(self) -> list[str]:
        """Get list of indicators required by this strategy.

        Returns:
            List of indicator names needed
        """
        return []

    def get_warmup_period(self) -> int:
        """Get minimum bars needed before strategy produces valid signals.

        Returns:
            Number of bars needed for warmup
        """
        # Default: use the largest period from config
        periods = []
        for key, value in self.config.items():
            if 'period' in key and isinstance(value, int):
                periods.append(value)

        return max(periods) if periods else 50

    def __repr__(self) -> str:
        return f"{self.name}(config={self.config})"
