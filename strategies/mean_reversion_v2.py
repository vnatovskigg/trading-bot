"""Mean reversion strategy v2 with market regime filter.

Major change from v1: Only trades during ranging markets (low ADX on higher timeframe).
"""

from typing import Dict
import pandas as pd

from data.models import BarSeries
from strategies.base import Strategy
from strategies.models import TargetPosition


class MeanReversionV2Strategy(Strategy):
    """Mean reversion with market regime awareness.

    Key Innovation:
    - Uses higher timeframe (4h) ADX to detect market regime
    - Only trades mean reversion in ranging markets (ADX < threshold)
    - Avoids counter-trend trades during strong directional moves

    This addresses the core issue: mean reversion loses money fighting trends.
    """

    def __init__(self, config: dict):
        """Initialize mean reversion v2 strategy.

        New config keys:
        - regime_timeframe: Higher TF for regime detection (default: '4h')
        - regime_adx_threshold: Max ADX for ranging market (default: 25)
        """
        super().__init__(config, name="MeanReversionV2")

        # Existing parameters
        self.rsi_oversold = config.get('rsi_oversold', 25)
        self.rsi_overbought = config.get('rsi_overbought', 75)
        self.trend_filter_threshold = config.get('trend_filter_threshold', 0.03)
        self.max_hold_bars = config.get('max_hold_bars', 40)
        self.cooldown_bars = config.get('cooldown_bars', 25)

        # NEW: Market regime parameters
        self.regime_adx_threshold = config.get('regime_adx_threshold', 25)
        self.use_regime_filter = config.get('use_regime_filter', True)

        # State
        self.reset_state()

    def reset_state(self):
        """Reset strategy state."""
        self.state = {
            'position': 0.0,
            'entry_price': None,
            'entry_time': None,
            'bars_held': 0,
            'bars_since_exit': 999,
            'regime': 'unknown'  # Track current regime
        }

    def on_new_candle(
        self,
        symbol: str,
        timeframe: str,
        bars: BarSeries,
        indicators: Dict[str, pd.Series]
    ) -> TargetPosition:
        """Process new candle with regime awareness.

        Args:
            symbol: Trading pair
            timeframe: Timeframe
            bars: Historical bars
            indicators: Calculated indicators (must include 'regime_adx' if using filter)

        Returns:
            TargetPosition for this strategy
        """
        # Get current values
        close = bars.get_closes().iloc[-1]
        rsi = indicators['rsi'].iloc[-1]
        bb_upper = indicators['bb_upper'].iloc[-1]
        bb_lower = indicators['bb_lower'].iloc[-1]
        bb_middle = indicators['bb_middle'].iloc[-1]
        ma_trend = indicators.get('sma_trend', indicators['bb_middle']).iloc[-1]

        timestamp = bars.df.index[-1]

        # Check validity
        if pd.isna(rsi) or pd.isna(bb_upper) or pd.isna(bb_lower) or pd.isna(ma_trend):
            return self._no_position(symbol, timestamp, "Indicators not ready")

        # === NEW: MARKET REGIME FILTER ===
        if self.use_regime_filter:
            # Get regime ADX (from higher timeframe, e.g., 4h)
            regime_adx = indicators.get('regime_adx')

            if regime_adx is not None and not pd.isna(regime_adx.iloc[-1]):
                current_regime_adx = regime_adx.iloc[-1]

                # Determine regime
                if current_regime_adx > self.regime_adx_threshold:
                    self.state['regime'] = 'trending'
                else:
                    self.state['regime'] = 'ranging'

                # Exit existing position if market starts trending
                if self.state['position'] != 0 and self.state['regime'] == 'trending':
                    return self._exit_position(
                        symbol,
                        timestamp,
                        f"Market regime changed to trending (ADX={current_regime_adx:.1f})"
                    )

                # Block new entries during trending markets
                if self.state['position'] == 0 and self.state['regime'] == 'trending':
                    return self._no_position(
                        symbol,
                        timestamp,
                        f"Trending market (4h ADX={current_regime_adx:.1f} > {self.regime_adx_threshold})"
                    )

        # Update counters
        if self.state['position'] != 0:
            self.state['bars_held'] += 1
        else:
            self.state['bars_since_exit'] += 1

        # Trend filter on trading timeframe
        trend_deviation = (close - ma_trend) / ma_trend
        strong_uptrend = trend_deviation > self.trend_filter_threshold
        strong_downtrend = trend_deviation < -self.trend_filter_threshold

        # === EXIT LOGIC ===
        if self.state['bars_held'] >= self.max_hold_bars:
            return self._exit_position(
                symbol,
                timestamp,
                f"Max hold time ({self.max_hold_bars} bars)"
            )

        if self.state['position'] > 0:
            if close >= bb_middle or rsi >= 50:
                return self._exit_position(
                    symbol,
                    timestamp,
                    f"Mean reversion complete (long): price={close:.2f} BB_mid={bb_middle:.2f} RSI={rsi:.1f}"
                )

        elif self.state['position'] < 0:
            if close <= bb_middle or rsi <= 50:
                return self._exit_position(
                    symbol,
                    timestamp,
                    f"Mean reversion complete (short): price={close:.2f} BB_mid={bb_middle:.2f} RSI={rsi:.1f}"
                )

        # === ENTRY LOGIC ===
        if self.state['position'] == 0 and self.state['bars_since_exit'] >= self.cooldown_bars:

            # Long entry
            if (rsi < self.rsi_oversold and
                close < bb_lower and
                not strong_downtrend):
                return self._enter_long(symbol, timestamp, close, rsi, bb_lower, bb_middle)

            # Short entry
            if (rsi > self.rsi_overbought and
                close > bb_upper and
                not strong_uptrend):
                return self._enter_short(symbol, timestamp, close, rsi, bb_upper, bb_middle)

        # === HOLD OR NO POSITION ===
        if self.state['position'] != 0:
            return self._hold_position(
                symbol,
                timestamp,
                rsi,
                f"Waiting for reversion (bars held: {self.state['bars_held']}, regime: {self.state['regime']})"
            )
        else:
            if self.state['bars_since_exit'] < self.cooldown_bars:
                reason = f"Cooldown ({self.state['bars_since_exit']}/{self.cooldown_bars})"
            elif self.state['regime'] == 'trending':
                reason = f"Trending market (regime filter active)"
            else:
                reason = f"Waiting for signal (RSI: {rsi:.1f}, regime: {self.state['regime']})"

            return self._no_position(symbol, timestamp, reason)

    def _enter_long(self, symbol, timestamp, price, rsi, bb_lower, bb_middle):
        """Enter long position."""
        self.state['position'] = 1.0
        self.state['entry_price'] = price
        self.state['entry_time'] = timestamp
        self.state['bars_held'] = 0

        return TargetPosition(
            symbol=symbol,
            timestamp=timestamp,
            target_exposure=1.0,
            confidence=0.7,  # Higher confidence due to regime filter
            strategy_name=self.name,
            timeframe='15m',
            metadata={
                'action': 'ENTER_LONG',
                'rsi': rsi,
                'price': price,
                'bb_lower': bb_lower,
                'bb_middle': bb_middle,
                'regime': self.state['regime'],
                'reason': 'Oversold in ranging market'
            }
        )

    def _enter_short(self, symbol, timestamp, price, rsi, bb_upper, bb_middle):
        """Enter short position."""
        self.state['position'] = -1.0
        self.state['entry_price'] = price
        self.state['entry_time'] = timestamp
        self.state['bars_held'] = 0

        return TargetPosition(
            symbol=symbol,
            timestamp=timestamp,
            target_exposure=-1.0,
            confidence=0.7,
            strategy_name=self.name,
            timeframe='15m',
            metadata={
                'action': 'ENTER_SHORT',
                'rsi': rsi,
                'price': price,
                'bb_upper': bb_upper,
                'bb_middle': bb_middle,
                'regime': self.state['regime'],
                'reason': 'Overbought in ranging market'
            }
        )

    def _exit_position(self, symbol, timestamp, reason):
        """Exit current position."""
        previous_position = self.state['position']

        self.state['position'] = 0.0
        self.state['entry_price'] = None
        self.state['entry_time'] = None
        self.state['bars_held'] = 0
        self.state['bars_since_exit'] = 0

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
                'previous_position': previous_position,
                'regime': self.state['regime']
            }
        )

    def _hold_position(self, symbol, timestamp, rsi, reason):
        """Continue holding current position."""
        return TargetPosition(
            symbol=symbol,
            timestamp=timestamp,
            target_exposure=self.state['position'],
            confidence=0.6,
            strategy_name=self.name,
            timeframe='15m',
            metadata={
                'action': 'HOLD',
                'reason': reason,
                'bars_held': self.state['bars_held'],
                'rsi': rsi,
                'regime': self.state['regime']
            }
        )

    def _no_position(self, symbol, timestamp, reason):
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
                'reason': reason,
                'regime': self.state['regime']
            }
        )

    def get_required_indicators(self) -> list[str]:
        """Get list of required indicators."""
        indicators = ['rsi', 'bb_upper', 'bb_lower', 'bb_middle', 'sma_trend']

        if self.use_regime_filter:
            indicators.append('regime_adx')

        return indicators

    def get_warmup_period(self) -> int:
        """Get warmup period."""
        return max(
            self.config.get('rsi_period', 14),
            self.config.get('bb_period', 20)
        ) + 20
