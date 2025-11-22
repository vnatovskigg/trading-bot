"""Test mean reversion v1.2 with market regime filter."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

import yaml
from datetime import datetime, timedelta
from database.connection import get_engine
from data.storage import OHLCVStorage
from data.provider import DataProvider
from strategies.mean_reversion_v2 import MeanReversionV2Strategy
from backtest.engine import BacktestEngine
from backtest.execution_sim import ExecutionConfig
from backtest.visualizer import generate_report


def main():
    """Run mean reversion v1.2 backtest."""

    # Load v1.2 configuration
    with open('config/versions/v1.2_regime_filter.yaml', 'r') as f:
        config = yaml.safe_load(f)

    print("="*60)
    print("MEAN REVERSION V1.2 - MARKET REGIME FILTER")
    print("="*60)
    print("\nConfiguration changes from v1.1:")
    print("  + use_regime_filter: true")
    print("  + regime_adx_threshold: 25")
    print("  + regime_timeframe: 4h")
    print("\nHypothesis: Mean reversion only works in ranging markets.")
    print("Strategy will exit positions and block new entries when 4h ADX > 25.\n")

    # Initialize database and data provider
    storage = OHLCVStorage()
    data_provider = DataProvider(storage, mode='backtest')

    # Create mean reversion v2 strategy
    mean_reversion_config = config['strategies']['mean_reversion']
    strategy = MeanReversionV2Strategy(mean_reversion_config)

    # Create execution config
    execution_config = ExecutionConfig(
        slippage_bps=config['backtest']['slippage_bps'],
        maker_fee=config['backtest']['maker_fee'],
        taker_fee=config['backtest']['taker_fee']
    )

    # Create backtest engine
    engine = BacktestEngine(
        data_provider=data_provider,
        strategy=strategy,
        execution_config=execution_config,
        initial_capital=config['backtest']['initial_capital'],
        position_size_pct=config['backtest']['position_size_pct']
    )

    # Define backtest period
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)

    # Run backtest for BTC/USDT
    symbol = 'BTC/USDT'
    timeframe = '15m'
    regime_timeframe = '4h'

    result = engine.run(
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        regime_timeframe=regime_timeframe
    )

    if result is None:
        print("ERROR: Backtest failed")
        return

    # Print results
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"\nTotal Return:       {result.metrics.total_return:+.2f}%")
    print(f"CAGR:              {result.metrics.cagr:+.2f}%")
    print(f"Sharpe Ratio:       {result.metrics.sharpe_ratio:.2f}")
    print(f"Sortino Ratio:      {result.metrics.sortino_ratio:.2f}")
    print(f"Max Drawdown:       {result.metrics.max_drawdown:.2f}%")
    print(f"Win Rate:           {result.metrics.win_rate:.2f}%")
    print(f"Total Trades:       {result.metrics.total_trades}")
    print(f"Profit Factor:       {result.metrics.profit_factor:.2f}")
    print(f"Avg Win:            ${result.metrics.avg_win:.2f}")
    print(f"Avg Loss:           -${abs(result.metrics.avg_loss):.2f}")

    # Save results to experiment directory
    output_dir = str(Path('experiments/mean_reversion/v1.2_market_regime_filter'))

    # Generate all visualizations and reports
    print(f"\nSaving results to {output_dir}/")
    generate_report(result, output_dir)

    print("\n" + "="*60)
    print("COMPARISON TO v1.1")
    print("="*60)
    print("\nv1.1 Results (Reduced Frequency):")
    print("  Total Return:   -17.47%")
    print("  Total Trades:   155")
    print("  Win Rate:       43.23%")
    print("  Profit Factor:  0.46")

    print(f"\nv1.2 Results (Market Regime Filter):")
    print(f"  Total Return:   {result.metrics.total_return:+.2f}%")
    print(f"  Total Trades:   {result.metrics.total_trades}")
    print(f"  Win Rate:       {result.metrics.win_rate:.2f}%")
    print(f"  Profit Factor:   {result.metrics.profit_factor:.2f}")

    # Calculate improvement
    trade_reduction = (155 - result.metrics.total_trades) / 155 * 100
    return_improvement = result.metrics.total_return - (-17.47)

    print(f"\nChanges:")
    print(f"  Trade Count:    {result.metrics.total_trades - 155:+d} ({trade_reduction:+.1f}%)")
    print(f"  Return:         {return_improvement:+.2f}pp")

    if result.metrics.total_return > 0:
        print("\n✅ PROFITABLE! Market regime filter works!")
    elif result.metrics.total_return > -17.47:
        print("\n⚠️  Improved but still losing. Need more iterations.")
    else:
        print("\n❌ Worse than v1.1. Regime filter didn't help.")

    print("\nResults saved to experiments/mean_reversion/v1.2_market_regime_filter/")
    print("Files:")
    print("  - results.json")
    print("  - equity_curve.png")
    print("  - drawdown.png")
    print("  - monthly_returns.png")
    print("  - trades.csv")
    print("  - equity_curve.csv")
    print("  - report.txt")


if __name__ == '__main__':
    main()
