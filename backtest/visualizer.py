"""Visualization for backtest results."""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from datetime import datetime
from typing import List, Tuple
import os

from backtest.models import BacktestResult


def plot_equity_curve(result: BacktestResult, output_path: str = None):
    """Plot equity curve.

    Args:
        result: Backtest result
        output_path: Path to save plot (optional)
    """
    df = pd.DataFrame(result.equity_curve, columns=['timestamp', 'equity'])
    df.set_index('timestamp', inplace=True)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot equity
    ax.plot(df.index, df['equity'], label='Strategy Equity', linewidth=2)

    # Plot buy & hold for comparison
    # (Simplified - assumes starting price same as strategy's first trade)
    if result.trades:
        first_price = result.trades[0].price
        last_price = result.trades[-1].price
        buy_hold_return = (last_price / first_price) - 1
        buy_hold_equity = result.initial_capital * (1 + buy_hold_return)

        ax.axhline(y=result.initial_capital, color='gray', linestyle='--',
                   alpha=0.5, label='Initial Capital')

        # Simple buy & hold line
        ax.plot([df.index[0], df.index[-1]],
                [result.initial_capital, buy_hold_equity],
                color='orange', linestyle='--', alpha=0.7,
                label=f'Buy & Hold (~{buy_hold_return:.1%})')

    # Formatting
    ax.set_xlabel('Date')
    ax.set_ylabel('Equity ($)')
    ax.set_title(f'{result.strategy_name} - {result.symbol} ({result.timeframe})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Equity curve saved to: {output_path}")
    else:
        plt.show()

    plt.close()


def plot_drawdown(result: BacktestResult, output_path: str = None):
    """Plot drawdown curve.

    Args:
        result: Backtest result
        output_path: Path to save plot (optional)
    """
    df = pd.DataFrame(result.equity_curve, columns=['timestamp', 'equity'])
    df.set_index('timestamp', inplace=True)

    # Calculate drawdown
    cummax = df['equity'].cummax()
    drawdown = (df['equity'] - cummax) / cummax

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 4))

    # Plot drawdown
    ax.fill_between(df.index, 0, drawdown * 100, color='red', alpha=0.3)
    ax.plot(df.index, drawdown * 100, color='red', linewidth=1)

    # Formatting
    ax.set_xlabel('Date')
    ax.set_ylabel('Drawdown (%)')
    ax.set_title(f'Drawdown - {result.strategy_name}')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Drawdown plot saved to: {output_path}")
    else:
        plt.show()

    plt.close()


def generate_report(result: BacktestResult, output_dir: str):
    """Generate complete backtest report with plots.

    Args:
        result: Backtest result
        output_dir: Directory to save report files
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Generate plots
    equity_path = os.path.join(output_dir, 'equity_curve.png')
    drawdown_path = os.path.join(output_dir, 'drawdown.png')

    plot_equity_curve(result, equity_path)
    plot_drawdown(result, drawdown_path)

    # Save trades to CSV
    trades_path = os.path.join(output_dir, 'trades.csv')
    trades_df = pd.DataFrame([{
        'timestamp': t.timestamp,
        'symbol': t.symbol,
        'side': t.side,
        'quantity': t.quantity,
        'price': t.price,
        'fee': t.fee,
        'realized_pnl': t.realized_pnl,
        'strategy': t.strategy_name
    } for t in result.trades])
    trades_df.to_csv(trades_path, index=False)
    print(f"Trades saved to: {trades_path}")

    # Save equity curve to CSV
    equity_path_csv = os.path.join(output_dir, 'equity_curve.csv')
    equity_df = pd.DataFrame(result.equity_curve, columns=['timestamp', 'equity'])
    equity_df.to_csv(equity_path_csv, index=False)
    print(f"Equity curve saved to: {equity_path_csv}")

    # Save metrics to text file
    report_path = os.path.join(output_dir, 'report.txt')
    with open(report_path, 'w') as f:
        f.write(f"{'='*60}\n")
        f.write(f"BACKTEST REPORT\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"Strategy: {result.strategy_name}\n")
        f.write(f"Symbol: {result.symbol}\n")
        f.write(f"Timeframe: {result.timeframe}\n")
        f.write(f"Period: {result.start_date.date()} to {result.end_date.date()}\n")
        f.write(f"Initial Capital: ${result.initial_capital:,.2f}\n")
        f.write(f"Final Equity: ${result.final_equity:,.2f}\n")
        f.write(f"\n{result.metrics}\n")

    print(f"Report saved to: {report_path}")
    print(f"\nAll results saved to: {output_dir}")
