"""Historical data fetcher from exchange APIs."""

import ccxt
from datetime import datetime, timedelta
from typing import List, Optional
import time
from data.models import Candle


class HistoricalDataFetcher:
    """Fetches historical OHLCV data from cryptocurrency exchanges.

    Uses CCXT library to support multiple exchanges. Handles:
    - Pagination for large date ranges
    - Rate limiting
    - Error handling and retries
    """

    # Timeframe mapping from our format to CCXT format
    TIMEFRAME_MAP = {
        '1m': '1m',
        '5m': '5m',
        '15m': '15m',
        '30m': '30m',
        '1h': '1h',
        '4h': '4h',
        '1d': '1d',
        '1w': '1w'
    }

    def __init__(self, exchange_id: str = 'binance'):
        """Initialize fetcher for specified exchange.

        Args:
            exchange_id: Exchange identifier (e.g., 'binance', 'coinbase')
        """
        self.exchange_id = exchange_id
        self.exchange = self._init_exchange(exchange_id)

    def _init_exchange(self, exchange_id: str) -> ccxt.Exchange:
        """Initialize CCXT exchange object.

        Args:
            exchange_id: Exchange identifier

        Returns:
            CCXT exchange instance
        """
        exchange_class = getattr(ccxt, exchange_id)
        exchange = exchange_class({
            'enableRateLimit': True,  # Respect rate limits
            'options': {
                'defaultType': 'spot'  # Use spot market
            }
        })

        return exchange

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: Optional[datetime] = None,
        limit_per_request: int = 1000
    ) -> List[Candle]:
        """Fetch OHLCV data for a symbol and timeframe.

        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Timeframe string (e.g., '1h', '15m')
            start: Start datetime (inclusive)
            end: End datetime (exclusive), None for now
            limit_per_request: Candles per API request (max 1000 for Binance)

        Returns:
            List of Candle objects sorted by timestamp

        Raises:
            ValueError: If timeframe not supported
            ccxt.NetworkError: On network issues
            ccxt.ExchangeError: On exchange-specific errors
        """
        if timeframe not in self.TIMEFRAME_MAP:
            raise ValueError(f"Unsupported timeframe: {timeframe}. Use one of {list(self.TIMEFRAME_MAP.keys())}")

        if end is None:
            end = datetime.utcnow()

        ccxt_timeframe = self.TIMEFRAME_MAP[timeframe]

        # Convert to milliseconds
        since = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)

        all_candles = []
        current_since = since

        print(f"Fetching {symbol} {timeframe} from {start} to {end}...")

        while current_since < end_ms:
            try:
                # Fetch batch
                ohlcv = self.exchange.fetch_ohlcv(
                    symbol,
                    ccxt_timeframe,
                    since=current_since,
                    limit=limit_per_request
                )

                if not ohlcv:
                    print(f"No more data available after {datetime.fromtimestamp(current_since / 1000)}")
                    break

                # Convert to Candle objects
                batch_candles = [Candle.from_ccxt(data) for data in ohlcv]
                all_candles.extend(batch_candles)

                # Update since for next batch (use timestamp of last candle + 1ms)
                last_timestamp = ohlcv[-1][0]

                # If we got less than requested, we've reached the end
                if len(ohlcv) < limit_per_request:
                    break

                # Move to next batch
                # Add 1ms to avoid re-fetching the last candle
                current_since = last_timestamp + 1

                # Stop if we've passed the end time
                if current_since >= end_ms:
                    break

                # Progress indicator
                progress_date = datetime.fromtimestamp(last_timestamp / 1000)
                print(f"  Fetched up to {progress_date} ({len(all_candles)} candles so far)")

                # Small delay to be nice to the exchange (rate limiter handles this, but extra safety)
                time.sleep(0.1)

            except ccxt.RateLimitExceeded as e:
                print(f"Rate limit exceeded, waiting 60 seconds...")
                time.sleep(60)
                continue

            except ccxt.NetworkError as e:
                print(f"Network error: {e}, retrying in 5 seconds...")
                time.sleep(5)
                continue

            except Exception as e:
                print(f"Error fetching data: {e}")
                raise

        # Filter to exact range and remove duplicates
        unique_candles = {}
        for candle in all_candles:
            if start <= candle.timestamp < end:
                # Use timestamp as key to deduplicate
                unique_candles[candle.timestamp] = candle

        # Sort by timestamp
        result = sorted(unique_candles.values(), key=lambda c: c.timestamp)

        print(f"Fetched {len(result)} unique candles for {symbol} {timeframe}")

        return result

    def fetch_recent(
        self,
        symbol: str,
        timeframe: str,
        count: int = 500
    ) -> List[Candle]:
        """Fetch the most recent N candles.

        Args:
            symbol: Trading pair
            timeframe: Timeframe string
            count: Number of candles to fetch

        Returns:
            List of recent Candle objects
        """
        if timeframe not in self.TIMEFRAME_MAP:
            raise ValueError(f"Unsupported timeframe: {timeframe}")

        ccxt_timeframe = self.TIMEFRAME_MAP[timeframe]

        try:
            ohlcv = self.exchange.fetch_ohlcv(
                symbol,
                ccxt_timeframe,
                limit=count
            )

            candles = [Candle.from_ccxt(data) for data in ohlcv]

            return candles

        except Exception as e:
            print(f"Error fetching recent data: {e}")
            raise

    def get_timeframe_delta(self, timeframe: str) -> timedelta:
        """Get timedelta for a timeframe string.

        Args:
            timeframe: Timeframe string (e.g., '1h', '15m')

        Returns:
            timedelta representing the timeframe

        Raises:
            ValueError: If timeframe format is invalid
        """
        # Parse timeframe
        unit = timeframe[-1]
        value = int(timeframe[:-1])

        if unit == 'm':
            return timedelta(minutes=value)
        elif unit == 'h':
            return timedelta(hours=value)
        elif unit == 'd':
            return timedelta(days=value)
        elif unit == 'w':
            return timedelta(weeks=value)
        else:
            raise ValueError(f"Invalid timeframe unit: {unit}")

    def calculate_expected_candles(
        self,
        timeframe: str,
        start: datetime,
        end: datetime
    ) -> int:
        """Calculate expected number of candles for a date range.

        Args:
            timeframe: Timeframe string
            start: Start datetime
            end: End datetime

        Returns:
            Approximate number of candles expected
        """
        delta = self.get_timeframe_delta(timeframe)
        duration = end - start

        return int(duration / delta)

    def get_exchange_info(self) -> dict:
        """Get exchange information and limits.

        Returns:
            Dictionary with exchange info
        """
        return {
            'id': self.exchange_id,
            'name': self.exchange.name,
            'rate_limit': self.exchange.rateLimit,
            'has_fetch_ohlcv': self.exchange.has['fetchOHLCV'],
            'timeframes': self.exchange.timeframes if hasattr(self.exchange, 'timeframes') else {}
        }
