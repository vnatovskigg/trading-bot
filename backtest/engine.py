"""Core backtesting engine."""

from datetime import datetime
from typing import Optional, Dict
import pandas as pd

from data.provider import DataProvider
from data.models import BarSeries
from strategies.base import Strategy
from indicators import SMA, EMA, ADX, RSI, BollingerBands, ATR
from backtest.execution_sim import ExecutionSimulator, ExecutionConfig
from backtest.portfolio import Portfolio
from backtest.models import BacktestResult
from backtest.metrics import calculate_metrics, PerformanceMetrics


class BacktestEngine:
    """Event-driven backtesting engine.

    Simulates trading strategies on historical data with realistic
    execution and costs.

    Key features:
    - No look-ahead bias
    - Realistic fill simulation
    - Transaction costs (fees + slippage)
    - Position tracking
    - Performance metrics
    """

    def __init__(
        self,
        data_provider: DataProvider,
        strategy: Strategy,
        execution_config: Optional[ExecutionConfig] = None,
        initial_capital: float = 10000,
        position_size_pct: float = 0.20
    ):
        """Initialize backtest engine.

        Args:
            data_provider: Data provider for market data
            strategy: Trading strategy to test
            execution_config: Execution simulation config
            initial_capital: Starting capital
            position_size_pct: Position size as % of equity
        """
        self.data_provider = data_provider
        self.strategy = strategy
        self.execution_sim = ExecutionSimulator(execution_config or ExecutionConfig())
        self.position_size_pct = position_size_pct

        self.portfolio = Portfolio(initial_capital)

    def run(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
        regime_timeframe: Optional[str] = None
    ) -> BacktestResult:
        """Run backtest for a symbol and time range.

        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe for strategy
            start_date: Backtest start date
            end_date: Backtest end date
            regime_timeframe: Optional higher timeframe for regime detection

        Returns:
            BacktestResult with performance metrics
        """
        print(f"\n{'='*60}")
        print(f"BACKTEST: {symbol} on {timeframe}")
        print(f"Strategy: {self.strategy.name}")
        print(f"Period: {start_date.date()} to {end_date.date()}")
        if regime_timeframe:
            print(f"Regime timeframe: {regime_timeframe}")
        print(f"{'='*60}\n")

        # Reset strategy state
        self.strategy.reset_state()

        # Get all historical data
        all_bars = self.data_provider.get_bars(symbol, timeframe, start_date, end_date)

        if len(all_bars) == 0:
            print("ERROR: No data available for this period")
            return None

        print(f"Loaded {len(all_bars)} candles")

        # Get higher timeframe data if needed for regime filter
        regime_bars = None
        if regime_timeframe and self.strategy.config.get('use_regime_filter', False):
            regime_bars = self.data_provider.get_bars(symbol, regime_timeframe, start_date, end_date)
            print(f"Loaded {len(regime_bars)} regime candles ({regime_timeframe})")

        # Determine warmup period
        warmup_bars = self.strategy.get_warmup_period()
        print(f"Warmup period: {warmup_bars} bars\n")

        # Initialize indicators based on strategy config
        indicators = self._initialize_indicators()

        # Event loop
        print("Running backtest...")
        signals_generated = 0

        for i in range(warmup_bars, len(all_bars.df) - 1):  # -1 to have "next" candle
            current_timestamp = all_bars.df.index[i]

            # Get historical window (up to current bar)
            historical_bars = BarSeries(all_bars.df.iloc[:i+1])

            # Calculate indicators
            calculated_indicators = self._calculate_indicators(
                historical_bars,
                indicators
            )

            # Add regime indicators if available
            if regime_bars is not None:
                regime_indicators = self._calculate_regime_indicators(
                    regime_bars,
                    current_timestamp,
                    indicators
                )
                calculated_indicators.update(regime_indicators)

            # Get strategy signal
            target_position = self.strategy.on_new_candle(
                symbol,
                timeframe,
                historical_bars,
                calculated_indicators
            )

            signals_generated += 1

            # Determine trade needed
            current_position = self.portfolio.get_position(symbol)
            current_exposure = self.portfolio.get_exposure(symbol)
            target_exposure = target_position.target_exposure

            # Calculate trade if exposure change needed
            if abs(target_exposure - current_exposure) > 0.01:  # Threshold to avoid tiny trades
                trade = self._calculate_trade(
                    symbol,
                    current_exposure,
                    target_exposure,
                    current_position,
                    all_bars.df.iloc[i+1],  # Next candle for fill
                    target_position.strategy_name
                )

                if trade:
                    self.portfolio.execute_trade(trade)

            # Mark to market
            current_price = float(all_bars.df.iloc[i]['close'])
            self.portfolio.mark_to_market(current_timestamp, {symbol: current_price})

        # Generate result
        print(f"\nBacktest complete!")
        print(f"Signals generated: {signals_generated}")
        print(f"Trades executed: {len(self.portfolio.trades)}")

        result = self._generate_result(symbol, timeframe, start_date, end_date)

        return result

    def _initialize_indicators(self) -> Dict:
        """Initialize indicators based on strategy configuration."""
        config = self.strategy.config

        indicators = {}

        # Trend following indicators
        if 'ma_fast' in config:
            indicators['sma_fast'] = SMA(config['ma_fast'])
            indicators['sma_slow'] = SMA(config['ma_slow'])
            indicators['adx'] = ADX(config.get('adx_period', 14))
            indicators['atr'] = ATR(config.get('atr_period', 14))

        # Mean reversion indicators
        if 'rsi_period' in config:
            indicators['rsi'] = RSI(config['rsi_period'])
            bb_period = config.get('bb_period', 20)
            bb_std = config.get('bb_std', 2.0)
            indicators['bb'] = BollingerBands(bb_period, bb_std)
            indicators['sma_trend'] = SMA(config.get('trend_filter_ma', 50))
            indicators['atr'] = ATR(config.get('atr_period', 14))

            # Market regime filter (for v2 strategy)
            if config.get('use_regime_filter', False):
                indicators['regime_adx'] = ADX(config.get('adx_period', 14))

        return indicators

    def _calculate_indicators(
        self,
        bars: BarSeries,
        indicators: Dict
    ) -> Dict[str, pd.Series]:
        """Calculate all indicators on the bar data."""
        result = {}

        closes = bars.get_closes()
        highs = bars.get_highs()
        lows = bars.get_lows()

        for name, indicator in indicators.items():
            if name == 'regime_adx':
                # Skip regime ADX here - calculated separately
                continue
            elif name.startswith('sma_'):
                result[name] = indicator.calculate(closes)
            elif name == 'adx':
                adx, plus_di, minus_di = indicator.calculate(highs, lows, closes)
                result['adx'] = adx
                result['plus_di'] = plus_di
                result['minus_di'] = minus_di
            elif name == 'rsi':
                result['rsi'] = indicator.calculate(closes)
            elif name == 'bb':
                middle, upper, lower = indicator.calculate(closes)
                result['bb_middle'] = middle
                result['bb_upper'] = upper
                result['bb_lower'] = lower
            elif name == 'atr':
                result['atr'] = indicator.calculate(highs, lows, closes)

        return result

    def _calculate_regime_indicators(
        self,
        regime_bars: BarSeries,
        current_timestamp: datetime,
        indicators: Dict
    ) -> Dict[str, pd.Series]:
        """Calculate regime indicators from higher timeframe data.

        Args:
            regime_bars: Higher timeframe bar data
            current_timestamp: Current timestamp from lower timeframe
            indicators: Indicator objects

        Returns:
            Dict with regime indicator values aligned to current timestamp
        """
        result = {}

        if 'regime_adx' not in indicators:
            return result

        # Get regime bars up to current time (no look-ahead)
        regime_df = regime_bars.df[regime_bars.df.index <= current_timestamp]

        if len(regime_df) == 0:
            return result

        # Calculate regime ADX
        regime_bar_series = BarSeries(regime_df)
        highs = regime_bar_series.get_highs()
        lows = regime_bar_series.get_lows()
        closes = regime_bar_series.get_closes()

        adx, _, _ = indicators['regime_adx'].calculate(highs, lows, closes)

        # Return the most recent value as a series (for compatibility)
        result['regime_adx'] = adx

        return result

    def _calculate_trade(
        self,
        symbol: str,
        current_exposure: float,
        target_exposure: float,
        current_position,
        next_candle_data,
        strategy_name: str
    ):
        """Determine trade needed to move from current to target exposure."""
        from data.models import Candle

        equity = self.portfolio.get_equity()

        # Convert next candle data to Candle object
        next_candle = Candle(
            timestamp=next_candle_data.name,
            open=next_candle_data['open'],
            high=next_candle_data['high'],
            low=next_candle_data['low'],
            close=next_candle_data['close'],
            volume=next_candle_data['volume']
        )

        # Determine what action is needed
        if target_exposure == 0 and current_position:
            # Close entire position
            side = 'sell' if current_position.is_long else 'buy'
            quantity = abs(current_position.quantity)

        elif target_exposure > 0 and (not current_position or current_position.is_short):
            # Need to be long
            if current_position and current_position.is_short:
                # Close short first
                quantity = abs(current_position.quantity)
                side = 'buy'
            else:
                # Enter long
                target_notional = equity * abs(target_exposure) * self.position_size_pct
                quantity = target_notional / float(next_candle.open)
                side = 'buy'

        elif target_exposure < 0 and (not current_position or current_position.is_long):
            # Need to be short
            if current_position and current_position.is_long:
                # Close long first
                quantity = current_position.quantity
                side = 'sell'
            else:
                # Enter short
                target_notional = equity * abs(target_exposure) * self.position_size_pct
                quantity = target_notional / float(next_candle.open)
                side = 'sell'

        else:
            # No trade needed
            return None

        # Simulate fill
        return self.execution_sim.simulate_market_order(
            side, quantity, next_candle, symbol, strategy_name
        )

    def _generate_result(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime
    ) -> BacktestResult:
        """Generate backtest result with metrics."""
        metrics = calculate_metrics(
            self.portfolio.equity_history,
            self.portfolio.trades,
            self.portfolio.initial_capital
        )

        return BacktestResult(
            symbol=symbol,
            timeframe=timeframe,
            strategy_name=self.strategy.name,
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.portfolio.initial_capital,
            final_equity=self.portfolio.get_equity(),
            metrics=metrics,
            trades=self.portfolio.trades,
            equity_curve=self.portfolio.equity_history
        )
