"""Portfolio management for backtesting."""

from typing import Dict, List, Tuple, Optional
from datetime import datetime

from backtest.models import Position, Trade


class Portfolio:
    """Manages portfolio state during backtesting.

    Tracks:
    - Cash balance
    - Open positions
    - Equity (cash + position values)
    - Trade history
    - Equity curve
    """

    def __init__(self, initial_capital: float):
        """Initialize portfolio.

        Args:
            initial_capital: Starting cash balance
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_history: List[Tuple[datetime, float]] = []

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get current position for a symbol.

        Args:
            symbol: Trading pair symbol

        Returns:
            Position if exists, None otherwise
        """
        return self.positions.get(symbol)

    def get_equity(self) -> float:
        """Calculate total portfolio equity.

        Returns:
            Total equity (cash + unrealized P&L)
        """
        positions_value = sum(p.unrealized_pnl for p in self.positions.values())
        return self.cash + positions_value

    def execute_trade(self, trade: Trade):
        """Execute a trade and update portfolio state.

        Args:
            trade: Trade to execute
        """
        symbol = trade.symbol

        # Handle existing position
        if symbol in self.positions:
            position = self.positions[symbol]

            # Determine if closing/reducing or adding to position
            if (position.is_long and trade.side == 'sell') or (position.is_short and trade.side == 'buy'):
                # Closing or reducing position
                close_quantity = min(abs(position.quantity), trade.quantity)

                # Calculate realized P&L
                if position.is_long:
                    # Closing long: profit = (sell_price - entry_price) * quantity
                    trade.realized_pnl = (trade.price - position.avg_entry_price) * close_quantity
                else:
                    # Closing short: profit = (entry_price - buy_price) * quantity
                    trade.realized_pnl = (position.avg_entry_price - trade.price) * close_quantity

                # Subtract fees from realized P&L
                trade.realized_pnl -= trade.fee

                # Update or remove position
                if abs(position.quantity) <= trade.quantity:
                    # Fully closing
                    del self.positions[symbol]
                else:
                    # Partial close
                    if position.is_long:
                        position.quantity -= trade.quantity
                    else:
                        position.quantity += trade.quantity

            else:
                # Adding to position (same direction)
                new_quantity = abs(position.quantity) + trade.quantity
                # Recalculate average entry price
                position.avg_entry_price = (
                    (position.avg_entry_price * abs(position.quantity) +
                     trade.price * trade.quantity) / new_quantity
                )

                if position.is_long:
                    position.quantity += trade.quantity
                else:
                    position.quantity -= trade.quantity

        else:
            # New position
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=trade.quantity if trade.side == 'buy' else -trade.quantity,
                avg_entry_price=trade.price,
                entry_timestamp=trade.timestamp,
                strategy_name=trade.strategy_name
            )

        # Update cash
        if trade.side == 'buy':
            # Buying costs cash
            self.cash -= (trade.price * trade.quantity + trade.fee)
        else:
            # Selling adds cash
            self.cash += (trade.price * trade.quantity - trade.fee)

        # Record trade
        self.trades.append(trade)

    def mark_to_market(self, timestamp: datetime, prices: Dict[str, float]):
        """Update position values and record equity.

        Args:
            timestamp: Current timestamp
            prices: Current prices for each symbol
        """
        # Update unrealized P&L for all positions
        for symbol, position in self.positions.items():
            if symbol in prices:
                position.update_pnl(prices[symbol])

        # Record equity
        equity = self.get_equity()
        self.equity_history.append((timestamp, equity))

    def get_total_return(self) -> float:
        """Calculate total return percentage.

        Returns:
            Total return as decimal (e.g., 0.15 = 15%)
        """
        return (self.get_equity() / self.initial_capital) - 1.0

    def get_position_count(self) -> int:
        """Get number of open positions.

        Returns:
            Number of open positions
        """
        return len(self.positions)

    def get_exposure(self, symbol: str) -> float:
        """Get current exposure to a symbol as fraction of equity.

        Args:
            symbol: Trading pair symbol

        Returns:
            Exposure as fraction (-1 to +1, where 1 = 100% of equity)
        """
        position = self.get_position(symbol)
        if not position:
            return 0.0

        equity = self.get_equity()
        if equity <= 0:
            return 0.0

        position_value = abs(position.quantity * position.avg_entry_price)
        exposure = position_value / equity

        # Make negative for short positions
        if position.is_short:
            exposure = -exposure

        return exposure
