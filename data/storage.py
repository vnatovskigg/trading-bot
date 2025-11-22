"""Database storage for OHLCV data."""

from datetime import datetime
from typing import Optional, List
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from data.models import Candle, BarSeries
from database.connection import get_engine


class OHLCVStorage:
    """Handles reading and writing OHLCV data to TimescaleDB.

    Provides methods for:
    - Inserting candles (with upsert logic)
    - Querying candles by time range
    - Detecting data gaps
    - Getting latest timestamps
    """

    def __init__(self, engine: Optional[Engine] = None):
        """Initialize storage with database engine.

        Args:
            engine: SQLAlchemy engine, uses default if None
        """
        self.engine = engine or get_engine()

    def insert_candles(
        self,
        symbol: str,
        timeframe: str,
        candles: List[Candle],
        batch_size: int = 1000
    ) -> int:
        """Insert candles into database with upsert logic.

        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            timeframe: Timeframe string (e.g., '1h', '15m')
            candles: List of Candle objects
            batch_size: Number of candles per batch insert

        Returns:
            Number of candles inserted/updated
        """
        if not candles:
            return 0

        # Prepare data for insertion
        data = []
        for candle in candles:
            data.append({
                'time': candle.timestamp,
                'symbol': symbol,
                'timeframe': timeframe,
                'open': float(candle.open),
                'high': float(candle.high),
                'low': float(candle.low),
                'close': float(candle.close),
                'volume': float(candle.volume)
            })

        # Insert in batches
        total_inserted = 0
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            df = pd.DataFrame(batch)

            # Use ON CONFLICT DO UPDATE for upsert
            with self.engine.begin() as conn:
                # Insert using pandas to_sql with custom method
                df.to_sql(
                    'ohlcv_temp',
                    conn,
                    if_exists='replace',
                    index=False,
                    method='multi'
                )

                # Upsert from temp table
                upsert_query = text("""
                    INSERT INTO ohlcv (time, symbol, timeframe, open, high, low, close, volume)
                    SELECT time, symbol, timeframe, open, high, low, close, volume
                    FROM ohlcv_temp
                    ON CONFLICT (time, symbol, timeframe)
                    DO UPDATE SET
                        open = EXCLUDED.open,
                        high = EXCLUDED.high,
                        low = EXCLUDED.low,
                        close = EXCLUDED.close,
                        volume = EXCLUDED.volume
                """)

                result = conn.execute(upsert_query)
                total_inserted += len(batch)

        return total_inserted

    def get_candles(
        self,
        symbol: str,
        timeframe: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> BarSeries:
        """Query candles from database.

        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe string
            start: Start datetime (inclusive), None for all
            end: End datetime (exclusive), None for all
            limit: Maximum number of candles to return

        Returns:
            BarSeries containing the requested candles
        """
        # Build query
        query = """
            SELECT time, open, high, low, close, volume
            FROM ohlcv
            WHERE symbol = :symbol AND timeframe = :timeframe
        """

        params = {'symbol': symbol, 'timeframe': timeframe}

        if start is not None:
            query += " AND time >= :start"
            params['start'] = start

        if end is not None:
            query += " AND time < :end"
            params['end'] = end

        query += " ORDER BY time ASC"

        if limit is not None:
            query += " LIMIT :limit"
            params['limit'] = limit

        # Execute query
        with self.engine.connect() as conn:
            df = pd.read_sql(
                text(query),
                conn,
                params=params,
                parse_dates=['time'],
                index_col='time'
            )

        # Return empty BarSeries if no data
        if df.empty:
            empty_df = pd.DataFrame(columns=BarSeries.REQUIRED_COLUMNS)
            empty_df.index = pd.DatetimeIndex([])
            empty_df.index.name = 'time'
            return BarSeries(empty_df)

        return BarSeries(df)

    def get_latest_timestamp(
        self,
        symbol: str,
        timeframe: str
    ) -> Optional[datetime]:
        """Get the latest (most recent) timestamp for a symbol/timeframe.

        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe string

        Returns:
            Latest timestamp, or None if no data exists
        """
        query = text("""
            SELECT MAX(time) as latest
            FROM ohlcv
            WHERE symbol = :symbol AND timeframe = :timeframe
        """)

        with self.engine.connect() as conn:
            result = conn.execute(query, {'symbol': symbol, 'timeframe': timeframe})
            row = result.fetchone()

            return row[0] if row and row[0] else None

    def get_earliest_timestamp(
        self,
        symbol: str,
        timeframe: str
    ) -> Optional[datetime]:
        """Get the earliest timestamp for a symbol/timeframe.

        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe string

        Returns:
            Earliest timestamp, or None if no data exists
        """
        query = text("""
            SELECT MIN(time) as earliest
            FROM ohlcv
            WHERE symbol = :symbol AND timeframe = :timeframe
        """)

        with self.engine.connect() as conn:
            result = conn.execute(query, {'symbol': symbol, 'timeframe': timeframe})
            row = result.fetchone()

            return row[0] if row and row[0] else None

    def get_row_count(
        self,
        symbol: str,
        timeframe: str
    ) -> int:
        """Get number of candles for a symbol/timeframe.

        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe string

        Returns:
            Number of candles in database
        """
        query = text("""
            SELECT COUNT(*) as count
            FROM ohlcv
            WHERE symbol = :symbol AND timeframe = :timeframe
        """)

        with self.engine.connect() as conn:
            result = conn.execute(query, {'symbol': symbol, 'timeframe': timeframe})
            row = result.fetchone()

            return row[0] if row else 0

    def detect_gaps(
        self,
        symbol: str,
        timeframe: str,
        expected_interval_minutes: int,
        max_gap_multiplier: float = 2.0
    ) -> List[tuple[datetime, datetime]]:
        """Detect gaps in stored data.

        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe string
            expected_interval_minutes: Expected minutes between candles
            max_gap_multiplier: Multiplier for gap detection

        Returns:
            List of (start, end) tuples representing gaps
        """
        bars = self.get_candles(symbol, timeframe)

        if len(bars) == 0:
            return []

        return bars.check_gaps(expected_interval_minutes, max_gap_multiplier)

    def delete_candles(
        self,
        symbol: str,
        timeframe: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None
    ) -> int:
        """Delete candles from database.

        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe string
            start: Start datetime (inclusive), None for all
            end: End datetime (exclusive), None for all

        Returns:
            Number of candles deleted
        """
        query = """
            DELETE FROM ohlcv
            WHERE symbol = :symbol AND timeframe = :timeframe
        """

        params = {'symbol': symbol, 'timeframe': timeframe}

        if start is not None:
            query += " AND time >= :start"
            params['start'] = start

        if end is not None:
            query += " AND time < :end"
            params['end'] = end

        with self.engine.begin() as conn:
            result = conn.execute(text(query), params)
            return result.rowcount

    def get_summary(self) -> pd.DataFrame:
        """Get summary of all data in database.

        Returns:
            DataFrame with symbol, timeframe, count, start, end
        """
        query = text("""
            SELECT
                symbol,
                timeframe,
                COUNT(*) as count,
                MIN(time) as start_time,
                MAX(time) as end_time
            FROM ohlcv
            GROUP BY symbol, timeframe
            ORDER BY symbol, timeframe
        """)

        with self.engine.connect() as conn:
            df = pd.read_sql(query, conn, parse_dates=['start_time', 'end_time'])

        return df
