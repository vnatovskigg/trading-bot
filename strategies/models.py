"""Data models for trading strategies."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class TargetPosition:
    """Represents a strategy's desired position.

    Attributes:
        symbol: Trading pair symbol
        timestamp: When this position was generated
        target_exposure: Desired exposure from -1.0 (full short) to +1.0 (full long)
        confidence: Strategy confidence in this position (0.0 to 1.0)
        strategy_name: Name of strategy generating this position
        timeframe: Timeframe used for analysis
        metadata: Additional information (indicator values, reasoning, etc.)
    """
    symbol: str
    timestamp: datetime
    target_exposure: float  # -1.0 to +1.0
    confidence: float  # 0.0 to 1.0
    strategy_name: str
    timeframe: str
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        """Validate the position."""
        if not -1.0 <= self.target_exposure <= 1.0:
            raise ValueError(
                f"target_exposure must be between -1.0 and 1.0, got {self.target_exposure}"
            )

        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"confidence must be between 0.0 and 1.0, got {self.confidence}"
            )

    @property
    def is_long(self) -> bool:
        """Check if position is long."""
        return self.target_exposure > 0

    @property
    def is_short(self) -> bool:
        """Check if position is short."""
        return self.target_exposure < 0

    @property
    def is_flat(self) -> bool:
        """Check if position is flat (no exposure)."""
        return self.target_exposure == 0

    def __repr__(self) -> str:
        direction = "LONG" if self.is_long else "SHORT" if self.is_short else "FLAT"
        return (
            f"TargetPosition({self.symbol} {direction} "
            f"exposure={self.target_exposure:.2f} "
            f"confidence={self.confidence:.2f})"
        )
