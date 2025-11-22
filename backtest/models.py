"""Data models for backtesting."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Tuple, Optional


@dataclass
class Trade:
    """Represents a single trade execution."""
    timestamp: datetime
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    price: float
    fee: float
    strategy_name: str
    realized_pnl: float = 0.0
    metadata: dict = field(default_factory=dict)

    def get_cost(self) -> float:
        """Get total cost including fees."""
        return (self.price * self.quantity) + self.fee


@dataclass
class Position:
    """Represents an open position."""
    symbol: str
    quantity: float  # positive = long, negative = short
    avg_entry_price: float
    entry_timestamp: datetime
    strategy_name: str
    unrealized_pnl: float = 0.0

    def update_pnl(self, current_price: float):
        """Update unrealized P&L based on current price."""
        if self.quantity > 0:  # Long position
            self.unrealized_pnl = (current_price - self.avg_entry_price) * self.quantity
        else:  # Short position
            self.unrealized_pnl = (self.avg_entry_price - current_price) * abs(self.quantity)

    @property
    def is_long(self) -> bool:
        return self.quantity > 0

    @property
    def is_short(self) -> bool:
        return self.quantity < 0

    @property
    def notional_value(self) -> float:
        """Get position value at entry price."""
        return abs(self.quantity * self.avg_entry_price)


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    symbol: str
    timeframe: str
    strategy_name: str
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_equity: float
    metrics: 'PerformanceMetrics'
    trades: List[Trade]
    equity_curve: List[Tuple[datetime, float]]
