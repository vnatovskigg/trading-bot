"""Execution simulator for backtesting."""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Optional

from data.models import Candle
from backtest.models import Trade


@dataclass
class ExecutionConfig:
    """Configuration for execution simulation."""
    slippage_bps: float = 10.0  # Basis points (0.1%)
    maker_fee: float = 0.001  # 0.1%
    taker_fee: float = 0.001  # 0.1%


class ExecutionSimulator:
    """Simulates order execution for backtesting.

    Fills market orders at the next candle's open price with slippage.
    Applies trading fees based on configuration.
    """

    def __init__(self, config: Optional[ExecutionConfig] = None):
        """Initialize execution simulator.

        Args:
            config: Execution configuration, uses defaults if None
        """
        self.config = config or ExecutionConfig()

    def simulate_market_order(
        self,
        side: str,
        quantity: float,
        next_candle: Candle,
        symbol: str,
        strategy_name: str
    ) -> Trade:
        """Simulate a market order fill.

        The order is filled at the next candle's open price plus slippage.
        This ensures no look-ahead bias.

        Args:
            side: 'buy' or 'sell'
            quantity: Amount to trade (always positive)
            next_candle: The next candle after signal (for fill price)
            symbol: Trading pair symbol
            strategy_name: Name of strategy placing order

        Returns:
            Trade object with execution details
        """
        if side not in ['buy', 'sell']:
            raise ValueError(f"Invalid side: {side}. Must be 'buy' or 'sell'")

        if quantity <= 0:
            raise ValueError(f"Quantity must be positive, got {quantity}")

        # Calculate fill price with slippage
        slippage_factor = self.config.slippage_bps / 10000.0

        if side == 'buy':
            # Buying: slippage increases price
            fill_price = float(next_candle.open) * (1 + slippage_factor)
        else:
            # Selling: slippage decreases price
            fill_price = float(next_candle.open) * (1 - slippage_factor)

        # Calculate fee (use taker fee for market orders)
        notional = fill_price * quantity
        fee = notional * self.config.taker_fee

        return Trade(
            timestamp=next_candle.timestamp,
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=fill_price,
            fee=fee,
            strategy_name=strategy_name,
            metadata={
                'slippage_bps': self.config.slippage_bps,
                'fee_rate': self.config.taker_fee
            }
        )
