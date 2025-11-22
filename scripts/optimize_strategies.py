"""Strategy parameter optimization script."""

import sys
import os
from datetime import datetime, timedelta
from itertools import product
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.provider import DataProvider
from data.storage import OHLCVStorage
from strategies.trend_following import TrendFollowingStrategy
from strategies.mean_reversion import MeanReversionStrategy
from backtest.engine import BacktestEngine
from backtest.execution_sim import ExecutionConfig


def optimize_trend_following():
    """Optimize trend following strategy parameters."""
    print("="*60)
    print("OPTIMIZING TREND FOLLOWING STRATEGY")
    print("="*60)

    # Parameter grid
    ma_fast_options = [10, 20, 30]
    ma_slow_options = [50, 100, 150]
    adx_threshold_options = [20, 25, 30, 35]

    # Data setup
    storage = OHLCVStorage()
    data_provider = DataProvider(storage, mode='backtest')

    # Get date range (last 1 year)
    earliest, latest = data_provider.get_date_range('BTC/USDT', '1h')
    end_date = latest
    start_date = end_date - timedelta(days=365)

    exec_config = ExecutionConfig(
        slippage_bps=10,
        maker_fee=0.001,
        taker_fee=0.001
    )

    results = []

    print(f"\nTesting {len(ma_fast_options) * len(ma_slow_options) * len(adx_threshold_options)} parameter combinations...\n")

    for ma_fast, ma_slow, adx_threshold in product(ma_fast_options, ma_slow_options, adx_threshold_options):
        # Skip invalid combinations
        if ma_fast >= ma_slow:
            continue

        config = {
            'ma_fast': ma_fast,
            'ma_slow': ma_slow,
            'adx_period': 14,
            'adx_threshold': adx_threshold,
            'atr_period': 14,
            'atr_critical_multiplier': 3.0,
            'max_hold_bars': 200
        }

        strategy = TrendFollowingStrategy(config)
        engine = BacktestEngine(
            data_provider=data_provider,
            strategy=strategy,
            execution_config=exec_config,
            initial_capital=10000,
            position_size_pct=0.20
        )

        try:
            result = engine.run('BTC/USDT', '1h', start_date, end_date)

            if result and result.metrics.total_trades > 0:
                results.append({
                    'ma_fast': ma_fast,
                    'ma_slow': ma_slow,
                    'adx_threshold': adx_threshold,
                    'total_return': result.metrics.total_return,
                    'cagr': result.metrics.cagr,
                    'sharpe': result.metrics.sharpe_ratio,
                    'max_dd': result.metrics.max_drawdown,
                    'win_rate': result.metrics.win_rate,
                    'total_trades': result.metrics.total_trades,
                    'profit_factor': result.metrics.profit_factor
                })

                print(f"MA({ma_fast}/{ma_slow}) ADX>{adx_threshold}: "
                      f"Return={result.metrics.total_return:.1%}, "
                      f"Sharpe={result.metrics.sharpe_ratio:.2f}, "
                      f"Trades={result.metrics.total_trades}")
        except Exception as e:
            print(f"Error with params {ma_fast}/{ma_slow}/{adx_threshold}: {e}")
            continue

    # Convert to DataFrame and sort
    df = pd.DataFrame(results)

    print("\n" + "="*60)
    print("TOP 10 PARAMETER COMBINATIONS BY SHARPE RATIO")
    print("="*60)
    print(df.nlargest(10, 'sharpe').to_string(index=False))

    print("\n" + "="*60)
    print("TOP 10 PARAMETER COMBINATIONS BY TOTAL RETURN")
    print("="*60)
    print(df.nlargest(10, 'total_return').to_string(index=False))

    # Save results
    output_path = 'results/trend_optimization.csv'
    df.to_csv(output_path, index=False)
    print(f"\nFull results saved to: {output_path}")

    return df


def optimize_mean_reversion():
    """Optimize mean reversion strategy parameters."""
    print("\n" + "="*60)
    print("OPTIMIZING MEAN REVERSION STRATEGY")
    print("="*60)

    # Parameter grid - focus on reducing trade frequency
    rsi_oversold_options = [25, 30, 35]
    rsi_overbought_options = [65, 70, 75]
    max_hold_options = [30, 50, 70]
    cooldown_options = [10, 20, 30]

    storage = OHLCVStorage()
    data_provider = DataProvider(storage, mode='backtest')

    earliest, latest = data_provider.get_date_range('BTC/USDT', '15m')
    end_date = latest
    start_date = end_date - timedelta(days=365)

    exec_config = ExecutionConfig(
        slippage_bps=10,
        maker_fee=0.001,
        taker_fee=0.001
    )

    results = []

    print(f"\nTesting {len(rsi_oversold_options) * len(rsi_overbought_options) * len(max_hold_options) * len(cooldown_options)} parameter combinations...\n")

    for rsi_os, rsi_ob, max_hold, cooldown in product(
        rsi_oversold_options, rsi_overbought_options, max_hold_options, cooldown_options
    ):
        config = {
            'rsi_period': 14,
            'rsi_oversold': rsi_os,
            'rsi_overbought': rsi_ob,
            'bb_period': 20,
            'bb_std': 2.0,
            'trend_filter_ma': 50,
            'trend_filter_threshold': 0.02,
            'max_hold_bars': max_hold,
            'cooldown_bars': cooldown
        }

        strategy = MeanReversionStrategy(config)
        engine = BacktestEngine(
            data_provider=data_provider,
            strategy=strategy,
            execution_config=exec_config,
            initial_capital=10000,
            position_size_pct=0.20
        )

        try:
            result = engine.run('BTC/USDT', '15m', start_date, end_date)

            if result and result.metrics.total_trades > 0:
                results.append({
                    'rsi_oversold': rsi_os,
                    'rsi_overbought': rsi_ob,
                    'max_hold': max_hold,
                    'cooldown': cooldown,
                    'total_return': result.metrics.total_return,
                    'cagr': result.metrics.cagr,
                    'sharpe': result.metrics.sharpe_ratio,
                    'max_dd': result.metrics.max_drawdown,
                    'win_rate': result.metrics.win_rate,
                    'total_trades': result.metrics.total_trades,
                    'profit_factor': result.metrics.profit_factor
                })

                print(f"RSI({rsi_os}/{rsi_ob}) Hold:{max_hold} Cool:{cooldown}: "
                      f"Return={result.metrics.total_return:.1%}, "
                      f"Sharpe={result.metrics.sharpe_ratio:.2f}, "
                      f"Trades={result.metrics.total_trades}")
        except Exception as e:
            print(f"Error: {e}")
            continue

    df = pd.DataFrame(results)

    print("\n" + "="*60)
    print("TOP 10 PARAMETER COMBINATIONS BY SHARPE RATIO")
    print("="*60)
    print(df.nlargest(10, 'sharpe').to_string(index=False))

    print("\n" + "="*60)
    print("TOP 10 PARAMETER COMBINATIONS BY TOTAL RETURN")
    print("="*60)
    print(df.nlargest(10, 'total_return').to_string(index=False))

    output_path = 'results/mean_reversion_optimization.csv'
    df.to_csv(output_path, index=False)
    print(f"\nFull results saved to: {output_path}")

    return df


def main():
    """Run optimization for both strategies."""
    print("="*60)
    print("STRATEGY PARAMETER OPTIMIZATION")
    print("="*60)
    print("\nThis will test multiple parameter combinations to find optimal settings.")
    print("Note: This may take several minutes...\n")

    # Optimize trend following
    df_trend = optimize_trend_following()

    # Optimize mean reversion
    df_mr = optimize_mean_reversion()

    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETE!")
    print("="*60)
    print("\nReview the results above and update config/default.yaml with")
    print("the best parameter combinations.")

    return 0


if __name__ == '__main__':
    sys.exit(main())
