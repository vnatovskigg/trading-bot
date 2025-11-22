"""Main script to run backtests."""

import sys
import os
import yaml
from datetime import datetime, timedelta

from data.provider import DataProvider
from data.storage import OHLCVStorage
from strategies.trend_following import TrendFollowingStrategy
from strategies.mean_reversion import MeanReversionStrategy
from backtest.engine import BacktestEngine
from backtest.execution_sim import ExecutionConfig
from backtest.visualizer import generate_report


def load_config() -> dict:
    """Load configuration from YAML file."""
    config_path = os.path.join(os.path.dirname(__file__), 'config', 'default.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_backtest(
    symbol: str,
    timeframe: str,
    strategy_name: str,
    config: dict,
    start_date: datetime,
    end_date: datetime
):
    """Run a single backtest.

    Args:
        symbol: Trading pair symbol
        timeframe: Timeframe for strategy
        strategy_name: Name of strategy ('trend_following' or 'mean_reversion')
        config: Configuration dictionary
        start_date: Backtest start date
        end_date: Backtest end date
    """
    # Initialize data provider
    storage = OHLCVStorage()
    data_provider = DataProvider(storage, mode='backtest')

    # Initialize strategy
    if strategy_name == 'trend_following':
        strategy = TrendFollowingStrategy(config['strategies']['trend_following'])
    elif strategy_name == 'mean_reversion':
        strategy = MeanReversionStrategy(config['strategies']['mean_reversion'])
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    # Initialize execution config
    exec_config = ExecutionConfig(
        slippage_bps=config['backtest']['slippage_bps'],
        maker_fee=config['backtest']['maker_fee'],
        taker_fee=config['backtest']['taker_fee']
    )

    # Initialize backtest engine
    engine = BacktestEngine(
        data_provider=data_provider,
        strategy=strategy,
        execution_config=exec_config,
        initial_capital=config['backtest']['initial_capital'],
        position_size_pct=config['backtest']['position_size_pct']
    )

    # Run backtest
    result = engine.run(
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date
    )

    if result:
        # Print metrics
        print(result.metrics)

        # Generate report
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(
            'results',
            f'{strategy_name}_{symbol.replace("/", "_")}_{timeframe}_{timestamp}'
        )

        generate_report(result, output_dir)

        return result

    return None


def main():
    """Main entry point."""
    print("="*60)
    print("CRYPTO TRADING BOT - BACKTEST")
    print("="*60)

    # Load config
    config = load_config()

    # Determine date range
    storage = OHLCVStorage()
    data_provider = DataProvider(storage)

    # Get available data range for BTC/USDT 1h (as reference)
    earliest, latest = data_provider.get_date_range('BTC/USDT', '1h')

    if not earliest or not latest:
        print("ERROR: No data found. Please run scripts/download_data.py first.")
        return 1

    print(f"\nAvailable data: {earliest.date()} to {latest.date()}")

    # Use most recent 1 year for backtest (to have faster initial results)
    end_date = latest
    start_date = end_date - timedelta(days=365)

    print(f"Backtest period: {start_date.date()} to {end_date.date()}\n")

    # Run backtests for both strategies
    results = {}

    # 1. Trend Following on BTC/USDT 1h
    print("\n" + "="*60)
    print("BACKTEST 1: Trend Following Strategy")
    print("="*60)
    result = run_backtest(
        symbol='BTC/USDT',
        timeframe='1h',
        strategy_name='trend_following',
        config=config,
        start_date=start_date,
        end_date=end_date
    )
    if result:
        results['trend_btc'] = result

    # 2. Mean Reversion on BTC/USDT 15m
    print("\n" + "="*60)
    print("BACKTEST 2: Mean Reversion Strategy")
    print("="*60)
    result = run_backtest(
        symbol='BTC/USDT',
        timeframe='15m',
        strategy_name='mean_reversion',
        config=config,
        start_date=start_date,
        end_date=end_date
    )
    if result:
        results['mr_btc'] = result

    # Summary comparison
    print("\n" + "="*60)
    print("SUMMARY COMPARISON")
    print("="*60)

    for name, result in results.items():
        print(f"\n{name} ({result.strategy_name}):")
        print(f"  Total Return: {result.metrics.total_return:.2%}")
        print(f"  CAGR: {result.metrics.cagr:.2%}")
        print(f"  Sharpe Ratio: {result.metrics.sharpe_ratio:.2f}")
        print(f"  Max Drawdown: {result.metrics.max_drawdown:.2%}")
        print(f"  Win Rate: {result.metrics.win_rate:.2%}")
        print(f"  Total Trades: {result.metrics.total_trades}")

    print("\n" + "="*60)
    print("ALL BACKTESTS COMPLETE!")
    print("="*60)

    return 0


if __name__ == '__main__':
    sys.exit(main())
