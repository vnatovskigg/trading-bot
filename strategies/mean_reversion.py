"""Mean reversion strategy implementation."""

from typing import Dict
import pandas as pd

from data.models import BarSeries
from strategies.base import Strategy
from strategies.models import TargetPosition


class MeanReversionStrategy(Strategy):
    """Mean reversion strategy using RSI and Bollinger Bands.

    Entry Logic:
    - Long: RSI < oversold AND price < lower_BB AND no strong downtrend
    - Short: RSI > overbought AND price > upper_BB AND no strong uptrend

    Exit Logic:
    - Price reverts to BB middle band
    - RSI crosses back to 50 (neutral)
    - Max holding period exceeded

    Trend Protection:
    - Blocks trades against strong higher-timeframe trends
    - Uses moving average deviation to detect strong trends

    Position Management:
    - Single position at a time
    - Cooldown period after exit
    - Quick in-and-out (mean reversion nature)
    """

    def __init__(self, config: dict):
        """Initialize mean reversion strategy.

        Expected config keys:
        - rsi_period: RSI calculation period (default: 14)
        - rsi_oversold: RSI oversold threshold (default: 30)
        - rsi_overbought: RSI overbought threshold (default: 70)
        - bb_period: Bollinger Bands period (default: 20)
        - bb_std: BB standard deviation multiplier (default: 2.0)
        - trend_filter_ma: MA period for trend filter (default: 50)
        - trend_filter_threshold: % deviation threshold (default: 0.02)
        - max_hold_bars: Max bars to hold (default: 50)
        - cooldown_bars: Bars to wait after exit (default: 10)
        """
        super().__init__(config, name="MeanReversion")

        # Parameters
        self.rsi_oversold = config.get('rsi_oversold', 30)
        self.rsi_overbought = config.get('rsi_overbought', 70)
        self.trend_filter_threshold = config.get('trend_filter_threshold', 0.02)
        self.max_hold_bars = config.get('max_hold_bars', 50)
        self.cooldown_bars = config.get('cooldown_bars', 10)

        # State
        self.reset_state()

    def reset_state(self):
        """Reset strategy state."""
        self.state = {
            'position': 0.0,
            'entry_price': None,
            'entry_time': None,
            'bars_held': 0,
            'bars_since_exit': 999  # Start with high value (no cooldown)
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
        # Get current values
        close = bars.get_closes().iloc[-1]
        rsi = indicators['rsi'].iloc[-1]
        bb_upper = indicators['bb_upper'].iloc[-1]
        bb_lower = indicators['bb_lower'].iloc[-1]
        bb_middle = indicators['bb_middle'].iloc[-1]
        ma_trend = indicators['sma_trend'].iloc[-1]

        # Get current timestamp
        timestamp = bars.df.index[-1]

        # Check if indicators are valid
        if pd.isna(rsi) or pd.isna(bb_upper) or pd.isna(bb_lower) or pd.isna(ma_trend):
            return self._no_position(symbol, timestamp, "Indicators not ready")

        # Update counters
        if self.state['position'] != 0:
            self.state['bars_held'] += 1
        else:
            self.state['bars_since_exit'] += 1

        # Trend filter: check if strong trend exists
        trend_deviation = (close - ma_trend) / ma_trend
        strong_uptrend = trend_deviation > self.trend_filter_threshold
        strong_downtrend = trend_deviation < -self.trend_filter_threshold

        # === EXIT LOGIC (check first) ===

        # Max hold time check
        if self.state['bars_held'] >= self.max_hold_bars:
            return self._exit_position(
                symbol,
                timestamp,
                f"Max hold time ({self.max_hold_bars} bars)"
            )

        # Exit long position
        if self.state['position'] > 0:
            # Mean reversion complete
            if close >= bb_middle or rsi >= 50:
                return self._exit_position(
                    symbol,
                    timestamp,
                    f"Mean reversion (long): price={close:.2f} BB_mid={bb_middle:.2f} RSI={rsi:.1f}"
                )

        # Exit short position
        elif self.state['position'] < 0:
            # Mean reversion complete
            if close <= bb_middle or rsi <= 50:
                return self._exit_position(
                    symbol,
                    timestamp,
                    f"Mean reversion (short): price={close:.2f} BB_mid={bb_middle:.2f} RSI={rsi:.1f}"
                )

        # === ENTRY LOGIC (only if not in cooldown) ===

        if self.state['position'] == 0 and self.state['bars_since_exit'] >= self.cooldown_bars:

            # Long entry: oversold + touched lower BB + no strong downtrend
            if (rsi < self.rsi_oversold and
                close < bb_lower and
                not strong_downtrend):
                return self._enter_long(
                    symbol,
                    timestamp,
                    close,
                    rsi,
                    bb_lower,
                    bb_middle
                )

            # Short entry: overbought + touched upper BB + no strong uptrend
            if (rsi > self.rsi_overbought and
                close > bb_upper and
                not strong_uptrend):
                return self._enter_short(
                    symbol,
                    timestamp,
                    close,
                    rsi,
                    bb_upper,
                    bb_middle
                )

        # === HOLD OR NO POSITION ===

        if self.state['position'] != 0:
            # Holding position
            return self._hold_position(
                symbol,
                timestamp,
                rsi,
                f"Waiting for reversion (bars held: {self.state['bars_held']})"
            )
        else:
            # No position
            if self.state['bars_since_exit'] < self.cooldown_bars:
                reason = f"Cooldown ({self.state['bars_since_exit']}/{self.cooldown_bars} bars)"
            elif strong_uptrend:
                reason = f"Strong uptrend detected (deviation: {trend_deviation:.1%})"
            elif strong_downtrend:
                reason = f"Strong downtrend detected (deviation: {trend_deviation:.1%})"
            else:
                reason = f"Waiting for signal (RSI: {rsi:.1f})"

            return self._no_position(symbol, timestamp, reason)

    def _enter_long(
        self,
        symbol: str,
        timestamp,
        price: float,
        rsi: float,
        bb_lower: float,
        bb_middle: float
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
            confidence=0.6,  # Mean reversion typically moderate confidence
            strategy_name=self.name,
            timeframe='15m',
            metadata={
                'action': 'ENTER_LONG',
                'rsi': rsi,
                'price': price,
                'bb_lower': bb_lower,
                'bb_middle': bb_middle,
                'reason': 'Oversold at lower BB'
            }
        )

    def _enter_short(
        self,
        symbol: str,
        timestamp,
        price: float,
        rsi: float,
        bb_upper: float,
        bb_middle: float
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
            confidence=0.6,
            strategy_name=self.name,
            timeframe='15m',
            metadata={
                'action': 'ENTER_SHORT',
                'rsi': rsi,
                'price': price,
                'bb_upper': bb_upper,
                'bb_middle': bb_middle,
                'reason': 'Overbought at upper BB'
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
        self.state['bars_since_exit'] = 0  # Reset cooldown

        return TargetPosition(
            symbol=symbol,
            timestamp=timestamp,
            target_exposure=0.0,
            confidence=0.0,
            strategy_name=self.name,
            timeframe='15m',
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
        rsi: float,
        reason: str
    ) -> TargetPosition:
        """Continue holding current position."""
        return TargetPosition(
            symbol=symbol,
            timestamp=timestamp,
            target_exposure=self.state['position'],
            confidence=0.5,
            strategy_name=self.name,
            timeframe='15m',
            metadata={
                'action': 'HOLD',
                'reason': reason,
                'bars_held': self.state['bars_held'],
                'rsi': rsi
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
            timeframe='15m',
            metadata={
                'action': 'NO_POSITION',
                'reason': reason
            }
        )

    def get_required_indicators(self) -> list[str]:
        """Get list of required indicators."""
        return ['rsi', 'bb_upper', 'bb_lower', 'bb_middle', 'sma_trend']

    def get_warmup_period(self) -> int:
        """Get warmup period."""
        # Need enough bars for RSI and BB calculations
        return max(self.config.get('rsi_period', 14), self.config.get('bb_period', 20)) + 20
