"""Performance metrics calculation."""

from dataclasses import dataclass
from typing import List, Tuple
from datetime import datetime
import pandas as pd
import numpy as np

from backtest.models import Trade


@dataclass
class PerformanceMetrics:
    """Performance metrics for a backtest."""
    # Returns
    total_return: float
    cagr: float  # Compound Annual Growth Rate

    # Risk metrics
    volatility: float  # Annualized
    sharpe_ratio: float
    sortino_ratio: float

    # Drawdown
    max_drawdown: float
    max_drawdown_duration: int  # Number of periods

    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float

    # Exposure
    avg_time_in_market: float

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'total_return': self.total_return,
            'cagr': self.cagr,
            'volatility': self.volatility,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'max_drawdown': self.max_drawdown,
            'max_drawdown_duration': self.max_drawdown_duration,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': self.win_rate,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss,
            'profit_factor': self.profit_factor,
            'avg_time_in_market': self.avg_time_in_market
        }

    def __str__(self) -> str:
        """Get formatted string representation."""
        return f"""
Performance Metrics:
====================
Returns:
  Total Return:      {self.total_return:>10.2%}
  CAGR:              {self.cagr:>10.2%}

Risk-Adjusted:
  Volatility (ann.): {self.volatility:>10.2%}
  Sharpe Ratio:      {self.sharpe_ratio:>10.2f}
  Sortino Ratio:     {self.sortino_ratio:>10.2f}

Drawdown:
  Max Drawdown:      {self.max_drawdown:>10.2%}
  Max DD Duration:   {self.max_drawdown_duration:>10} periods

Trades:
  Total Trades:      {self.total_trades:>10}
  Win Rate:          {self.win_rate:>10.2%}
  Avg Win:           ${self.avg_win:>10.2f}
  Avg Loss:          ${self.avg_loss:>10.2f}
  Profit Factor:     {self.profit_factor:>10.2f}

Exposure:
  Time in Market:    {self.avg_time_in_market:>10.2%}
"""


def calculate_metrics(
    equity_curve: List[Tuple[datetime, float]],
    trades: List[Trade],
    initial_capital: float
) -> PerformanceMetrics:
    """Calculate all performance metrics.

    Args:
        equity_curve: List of (timestamp, equity) tuples
        trades: List of executed trades
        initial_capital: Starting capital

    Returns:
        PerformanceMetrics object
    """
    if not equity_curve:
        # Return zero metrics
        return PerformanceMetrics(
            total_return=0.0, cagr=0.0, volatility=0.0,
            sharpe_ratio=0.0, sortino_ratio=0.0,
            max_drawdown=0.0, max_drawdown_duration=0,
            total_trades=0, winning_trades=0, losing_trades=0,
            win_rate=0.0, avg_win=0.0, avg_loss=0.0,
            profit_factor=0.0, avg_time_in_market=0.0
        )

    # Convert to DataFrame
    df = pd.DataFrame(equity_curve, columns=['timestamp', 'equity'])
    df.set_index('timestamp', inplace=True)
    df['returns'] = df['equity'].pct_change()

    # === RETURNS ===
    total_return = (df['equity'].iloc[-1] / initial_capital) - 1.0

    # Calculate CAGR
    days = (df.index[-1] - df.index[0]).days
    years = days / 365.25
    if years > 0:
        cagr = (1 + total_return) ** (1 / years) - 1
    else:
        cagr = 0.0

    # === VOLATILITY ===
    # Annualized volatility
    returns_std = df['returns'].std()
    # Assume hourly data for now (can be made configurable)
    periods_per_year = 365 * 24  # Hours per year
    volatility = returns_std * np.sqrt(periods_per_year)

    # === SHARPE RATIO ===
    # Assuming 0% risk-free rate for crypto
    if volatility > 0:
        sharpe_ratio = cagr / volatility
    else:
        sharpe_ratio = 0.0

    # === SORTINO RATIO ===
    # Only consider downside volatility
    downside_returns = df['returns'][df['returns'] < 0]
    if len(downside_returns) > 0:
        downside_std = downside_returns.std() * np.sqrt(periods_per_year)
        sortino_ratio = cagr / downside_std if downside_std > 0 else 0.0
    else:
        sortino_ratio = 0.0

    # === DRAWDOWN ===
    # Calculate running maximum
    cummax = df['equity'].cummax()
    drawdown = (df['equity'] - cummax) / cummax
    max_drawdown = abs(drawdown.min())

    # Calculate drawdown duration
    is_underwater = drawdown < -0.001  # Small threshold to avoid noise
    if is_underwater.any():
        # Group consecutive underwater periods
        underwater_groups = (is_underwater != is_underwater.shift()).cumsum()
        underwater_periods = is_underwater.groupby(underwater_groups).sum()
        max_drawdown_duration = int(underwater_periods.max())
    else:
        max_drawdown_duration = 0

    # === TRADE STATISTICS ===
    # Filter closing trades (those with realized P&L)
    closing_trades = [t for t in trades if t.realized_pnl != 0]

    winning_trades_list = [t for t in closing_trades if t.realized_pnl > 0]
    losing_trades_list = [t for t in closing_trades if t.realized_pnl < 0]

    total_trades = len(closing_trades)
    winning_trades = len(winning_trades_list)
    losing_trades = len(losing_trades_list)

    win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

    avg_win = np.mean([t.realized_pnl for t in winning_trades_list]) if winning_trades_list else 0.0
    avg_loss = np.mean([t.realized_pnl for t in losing_trades_list]) if losing_trades_list else 0.0

    # Profit factor
    gross_profit = sum(t.realized_pnl for t in winning_trades_list)
    gross_loss = abs(sum(t.realized_pnl for t in losing_trades_list))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

    # === TIME IN MARKET ===
    # Rough estimate: count trades / 2 (entry + exit) to get number of positions
    # Then estimate average holding time
    # For now, simplified placeholder
    avg_time_in_market = 0.5  # 50% placeholder

    return PerformanceMetrics(
        total_return=total_return,
        cagr=cagr,
        volatility=volatility,
        sharpe_ratio=sharpe_ratio,
        sortino_ratio=sortino_ratio,
        max_drawdown=max_drawdown,
        max_drawdown_duration=max_drawdown_duration,
        total_trades=total_trades,
        winning_trades=winning_trades,
        losing_trades=losing_trades,
        win_rate=win_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        profit_factor=profit_factor,
        avg_time_in_market=avg_time_in_market
    )
