"""Trend following strategy implementation."""

from typing import Dict
import pandas as pd

from data.models import BarSeries
from strategies.base import Strategy
from strategies.models import TargetPosition


class TrendFollowingStrategy(Strategy):
    """Trend following strategy using MA crossover, ADX, and volatility filter.

    Entry Logic:
    - Long: fast_MA > slow_MA AND price > both MAs AND ADX > threshold AND ATR normal
    - Short: fast_MA < slow_MA AND price < both MAs AND ADX > threshold AND ATR normal

    Exit Logic:
    - MA crossover reversal
    - Max holding period exceeded

    Position Management:
    - Single position at a time
    - No position when ADX < threshold (choppy market)
    - No position when ATR extreme (too volatile)
    """

    def __init__(self, config: dict):
        """Initialize trend following strategy.

        Expected config keys:
        - ma_fast: Fast MA period (default: 20)
        - ma_slow: Slow MA period (default: 50)
        - adx_period: ADX calculation period (default: 14)
        - adx_threshold: Minimum ADX for valid trend (default: 25)
        - atr_period: ATR calculation period (default: 14)
        - atr_critical_multiplier: Max ATR multiplier (default: 3.0)
        - max_hold_bars: Maximum bars to hold position (default: 200)
        """
        super().__init__(config, name="TrendFollowing")

        # Parameters
        self.ma_fast_period = config.get('ma_fast', 20)
        self.ma_slow_period = config.get('ma_slow', 50)
        self.adx_threshold = config.get('adx_threshold', 25)
        self.atr_critical_multiplier = config.get('atr_critical_multiplier', 3.0)
        self.max_hold_bars = config.get('max_hold_bars', 200)

        # State
        self.reset_state()

    def reset_state(self):
        """Reset strategy state."""
        self.state = {
            'position': 0.0,  # Current exposure (-1, 0, or +1)
            'entry_price': None,
            'entry_time': None,
            'bars_held': 0
        }

    def on_new_candle(
        self,
        symbol: str,
        timeframe: str,
        bars: BarSeries,
        indicators: Dict[str, pd.Series]
    ) -> TargetPosition:
        """Process new candle and generate target position.

        Args:
            symbol: Trading pair
            timeframe: Timeframe
            bars: Historical bars
            indicators: Calculated indicators

        Returns:
            TargetPosition for this strategy
        """
        # Get current (latest) values
        close = bars.get_closes().iloc[-1]
        ma_fast = indicators['sma_fast'].iloc[-1]
        ma_slow = indicators['sma_slow'].iloc[-1]
        adx = indicators['adx'].iloc[-1]
        atr = indicators['atr'].iloc[-1]

        # Get current timestamp
        timestamp = bars.df.index[-1]

        # Calculate ATR baseline (20-bar average for reference)
        atr_baseline = indicators['atr'].iloc[-20:].mean()

        # Check if indicators are valid (not NaN)
        if pd.isna(ma_fast) or pd.isna(ma_slow) or pd.isna(adx) or pd.isna(atr):
            return self._no_position(symbol, timestamp, "Indicators not ready")

        # Volatility guard - too risky to trade
        if atr > atr_baseline * self.atr_critical_multiplier:
            if self.state['position'] != 0:
                # Exit existing position due to extreme volatility
                return self._exit_position(
                    symbol,
                    timestamp,
                    f"Extreme volatility (ATR={atr:.2f} > {atr_baseline * self.atr_critical_multiplier:.2f})"
                )
            return self._no_position(
                symbol,
                timestamp,
                f"Extreme volatility (ATR={atr:.2f})"
            )

        # Trend strength filter - market too choppy
        if adx < self.adx_threshold:
            if self.state['position'] != 0:
                return self._exit_position(
                    symbol,
                    timestamp,
                    f"Weak trend (ADX={adx:.1f} < {self.adx_threshold})"
                )
            return self._no_position(
                symbol,
                timestamp,
                f"Weak trend (ADX={adx:.1f})"
            )

        # Determine trend direction
        bullish = (ma_fast > ma_slow) and (close > ma_fast) and (close > ma_slow)
        bearish = (ma_fast < ma_slow) and (close < ma_fast) and (close < ma_slow)

        # Update bars held counter
        if self.state['position'] != 0:
            self.state['bars_held'] += 1

        # Check max hold time
        if self.state['bars_held'] >= self.max_hold_bars:
            return self._exit_position(
                symbol,
                timestamp,
                f"Max hold time reached ({self.max_hold_bars} bars)"
            )

        # === POSITION LOGIC ===

        # Currently flat - look for entry
        if self.state['position'] == 0:
            if bullish:
                return self._enter_long(symbol, timestamp, close, adx, ma_fast, ma_slow)
            elif bearish:
                return self._enter_short(symbol, timestamp, close, adx, ma_fast, ma_slow)
            else:
                return self._no_position(symbol, timestamp, "No clear trend")

        # Currently long - check for exit
        elif self.state['position'] > 0:
            # Exit if trend reverses or MA crossover
            if bearish or ma_fast < ma_slow:
                return self._exit_position(
                    symbol,
                    timestamp,
                    "Trend reversal (long exit)"
                )
            # Continue holding
            return self._hold_position(symbol, timestamp, adx, "Holding long")

        # Currently short - check for exit
        elif self.state['position'] < 0:
            # Exit if trend reverses or MA crossover
            if bullish or ma_fast > ma_slow:
                return self._exit_position(
                    symbol,
                    timestamp,
                    "Trend reversal (short exit)"
                )
            # Continue holding
            return self._hold_position(symbol, timestamp, adx, "Holding short")

        # Fallback (shouldn't reach here)
        return self._no_position(symbol, timestamp, "Unknown state")

    def _enter_long(
        self,
        symbol: str,
        timestamp,
        price: float,
        adx: float,
        ma_fast: float,
        ma_slow: float
    ) -> TargetPosition:
        """Enter long position."""
        self.state['position'] = 1.0
        self.state['entry_price'] = price
        self.state['entry_time'] = timestamp
        self.state['bars_held'] = 0

        return TargetPosition(
            symbol=symbol,
            timestamp=timestamp,
            target_exposure=1.0,
            confidence=min(adx / 50.0, 1.0),  # Higher ADX = higher confidence
            strategy_name=self.name,
            timeframe='1h',
            metadata={
                'action': 'ENTER_LONG',
                'adx': adx,
                'ma_fast': ma_fast,
                'ma_slow': ma_slow,
                'entry_price': price
            }
        )

    def _enter_short(
        self,
        symbol: str,
        timestamp,
        price: float,
        adx: float,
        ma_fast: float,
        ma_slow: float
    ) -> TargetPosition:
        """Enter short position."""
        self.state['position'] = -1.0
        self.state['entry_price'] = price
        self.state['entry_time'] = timestamp
        self.state['bars_held'] = 0

        return TargetPosition(
            symbol=symbol,
            timestamp=timestamp,
            target_exposure=-1.0,
            confidence=min(adx / 50.0, 1.0),
            strategy_name=self.name,
            timeframe='1h',
            metadata={
                'action': 'ENTER_SHORT',
                'adx': adx,
                'ma_fast': ma_fast,
                'ma_slow': ma_slow,
                'entry_price': price
            }
        )

    def _exit_position(
        self,
        symbol: str,
        timestamp,
        reason: str
    ) -> TargetPosition:
        """Exit current position."""
        previous_position = self.state['position']

        self.state['position'] = 0.0
        self.state['entry_price'] = None
        self.state['entry_time'] = None
        self.state['bars_held'] = 0

        return TargetPosition(
            symbol=symbol,
            timestamp=timestamp,
            target_exposure=0.0,
            confidence=0.0,
            strategy_name=self.name,
            timeframe='1h',
            metadata={
                'action': 'EXIT',
                'reason': reason,
                'previous_position': previous_position
            }
        )

    def _hold_position(
        self,
        symbol: str,
        timestamp,
        adx: float,
        reason: str
    ) -> TargetPosition:
        """Continue holding current position."""
        return TargetPosition(
            symbol=symbol,
            timestamp=timestamp,
            target_exposure=self.state['position'],
            confidence=min(adx / 50.0, 1.0),
            strategy_name=self.name,
            timeframe='1h',
            metadata={
                'action': 'HOLD',
                'reason': reason,
                'bars_held': self.state['bars_held'],
                'adx': adx
            }
        )

    def _no_position(
        self,
        symbol: str,
        timestamp,
        reason: str
    ) -> TargetPosition:
        """No position (flat)."""
        return TargetPosition(
            symbol=symbol,
            timestamp=timestamp,
            target_exposure=0.0,
            confidence=0.0,
            strategy_name=self.name,
            timeframe='1h',
            metadata={
                'action': 'NO_POSITION',
                'reason': reason
            }
        )

    def get_required_indicators(self) -> list[str]:
        """Get list of required indicators."""
        return ['sma_fast', 'sma_slow', 'adx', 'atr']

    def get_warmup_period(self) -> int:
        """Get warmup period."""
        # Need enough bars for slow MA and ADX double smoothing
        return max(self.ma_slow_period, 14 * 2) + 20
