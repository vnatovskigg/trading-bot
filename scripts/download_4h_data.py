"""Download 4h historical data for regime detection."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from database.connection import get_engine
from data.storage import OHLCVStorage
from data.fetcher import HistoricalDataFetcher


def main():
    """Download 4h data for BTC/USDT and ETH/USDT."""

    print("="*60)
    print("DOWNLOADING 4H DATA FOR REGIME DETECTION")
    print("="*60)

    # Initialize
    storage = OHLCVStorage()
    fetcher = HistoricalDataFetcher(exchange_id='binance')

    symbols = ['BTC/USDT', 'ETH/USDT']
    timeframe = '4h'
    years = 2

    from datetime import datetime, timedelta
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=365 * years)

    for symbol in symbols:
        print(f"\nDownloading {symbol} {timeframe} data ({years} years)...")
        try:
            candles = fetcher.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                start=start_date,
                end=end_date
            )
            print(f"  Fetched {len(candles)} candles")

            # Store to database
            inserted = storage.insert_candles(
                symbol=symbol,
                timeframe=timeframe,
                candles=candles
            )
            print(f"✓ Stored {inserted} candles")
        except Exception as e:
            print(f"✗ Error downloading {symbol} {timeframe}: {e}")

    print("\n" + "="*60)
    print("DOWNLOAD COMPLETE")
    print("="*60)


if __name__ == '__main__':
    main()
