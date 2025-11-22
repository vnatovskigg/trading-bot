"""Test manually optimized parameters."""

import sys
import os
import yaml
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.provider import DataProvider
from data.storage import OHLCVStorage
from strategies.trend_following import TrendFollowingStrategy
from strategies.mean_reversion import MeanReversionStrategy
from backtest.engine import BacktestEngine
from backtest.execution_sim import ExecutionConfig
from backtest.visualizer import generate_report


def load_config(config_file='optimized.yaml'):
    """Load configuration."""
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'config',
        config_file
    )
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    print("="*60)
    print("TESTING OPTIMIZED PARAMETERS")
    print("="*60)

    # Load optimized config
    config = load_config('optimized.yaml')

    storage = OHLCVStorage()
    data_provider = DataProvider(storage, mode='backtest')

    # Test on same period as baseline for comparison
    earliest, latest = data_provider.get_date_range('BTC/USDT', '1h')
    end_date = latest
    start_date = end_date - timedelta(days=365)

    exec_config = ExecutionConfig(
        slippage_bps=config['backtest']['slippage_bps'],
        maker_fee=config['backtest']['maker_fee'],
        taker_fee=config['backtest']['taker_fee']
    )

    # Test 1: Trend Following with optimized params
    print("\n" + "="*60)
    print("TEST 1: OPTIMIZED TREND FOLLOWING")
    print("="*60)
    print("\nOptimized parameters:")
    for key, value in config['strategies']['trend_following'].items():
        print(f"  {key}: {value}")

    strategy = TrendFollowingStrategy(config['strategies']['trend_following'])
    engine = BacktestEngine(
        data_provider=data_provider,
        strategy=strategy,
        execution_config=exec_config,
        initial_capital=config['backtest']['initial_capital'],
        position_size_pct=config['backtest']['position_size_pct']
    )

    result_trend = engine.run('BTC/USDT', '1h', start_date, end_date)

    if result_trend:
        print(result_trend.metrics)

        # Generate report
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = f'results/trend_optimized_{timestamp}'
        generate_report(result_trend, output_dir)

    # Test 2: Mean Reversion with optimized params
    print("\n" + "="*60)
    print("TEST 2: OPTIMIZED MEAN REVERSION")
    print("="*60)
    print("\nOptimized parameters:")
    for key, value in config['strategies']['mean_reversion'].items():
        print(f"  {key}: {value}")

    strategy_mr = MeanReversionStrategy(config['strategies']['mean_reversion'])
    engine_mr = BacktestEngine(
        data_provider=data_provider,
        strategy=strategy_mr,
        execution_config=exec_config,
        initial_capital=config['backtest']['initial_capital'],
        position_size_pct=config['backtest']['position_size_pct']
    )

    result_mr = engine_mr.run('BTC/USDT', '15m', start_date, end_date)

    if result_mr:
        print(result_mr.metrics)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = f'results/mean_reversion_optimized_{timestamp}'
        generate_report(result_mr, output_dir)

    # Comparison
    print("\n" + "="*60)
    print("COMPARISON: BASELINE vs OPTIMIZED")
    print("="*60)

    print("\nTREND FOLLOWING:")
    print("  Baseline:  Return=-1.99%, Sharpe=-0.01, Trades=165, Win Rate=24.24%")
    if result_trend:
        print(f"  Optimized: Return={result_trend.metrics.total_return:.2%}, "
              f"Sharpe={result_trend.metrics.sharpe_ratio:.2f}, "
              f"Trades={result_trend.metrics.total_trades}, "
              f"Win Rate={result_trend.metrics.win_rate:.2%}")

    print("\nMEAN REVERSION:")
    print("  Baseline:  Return=-23.29%, Sharpe=-0.08, Trades=364, Win Rate=42.58%")
    if result_mr:
        print(f"  Optimized: Return={result_mr.metrics.total_return:.2%}, "
              f"Sharpe={result_mr.metrics.sharpe_ratio:.2f}, "
              f"Trades={result_mr.metrics.total_trades}, "
              f"Win Rate={result_mr.metrics.win_rate:.2%}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
