"""Script to download historical data from exchange."""

import sys
import os
from datetime import datetime, timedelta
import yaml

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.fetcher import HistoricalDataFetcher
from data.storage import OHLCVStorage
from database.connection import test_connection


def load_config() -> dict:
    """Load configuration from YAML file."""
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'config',
        'default.yaml'
    )

    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def download_historical_data():
    """Download historical data for configured symbols and timeframes."""
    print("="* 60)
    print("Historical Data Download Script")
    print("=" * 60)

    # Test database connection
    print("\n1. Testing database connection...")
    if not test_connection():
        print("ERROR: Cannot connect to database!")
        print("Make sure Docker container is running: docker-compose up -d")
        return 1

    print("✓ Database connection successful")

    # Load config
    print("\n2. Loading configuration...")
    config = load_config()

    symbols = config['symbols']
    timeframes = list(set([
        config['timeframes']['trend'],
        config['timeframes']['mean_reversion']
    ]))
    history_years = config['data']['history_years']

    print(f"✓ Config loaded:")
    print(f"  - Symbols: {symbols}")
    print(f"  - Timeframes: {timeframes}")
    print(f"  - History: {history_years} years")

    # Initialize components
    print("\n3. Initializing fetcher and storage...")
    fetcher = HistoricalDataFetcher(exchange_id=config['exchange']['name'])
    storage = OHLCVStorage()

    exchange_info = fetcher.get_exchange_info()
    print(f"✓ Connected to {exchange_info['name']}")
    print(f"  - Rate limit: {exchange_info['rate_limit']} ms")

    # Calculate date range
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=365 * history_years)

    print(f"\n4. Download range:")
    print(f"  - Start: {start_date}")
    print(f"  - End: {end_date}")

    # Download data for each symbol and timeframe
    print(f"\n5. Downloading data...")
    print("=" * 60)

    total_candles = 0

    for symbol in symbols:
        for timeframe in timeframes:
            print(f"\n📥 {symbol} - {timeframe}")
            print("-" * 40)

            try:
                # Check existing data
                existing_start = storage.get_earliest_timestamp(symbol, timeframe)
                existing_end = storage.get_latest_timestamp(symbol, timeframe)
                existing_count = storage.get_row_count(symbol, timeframe)

                if existing_count > 0:
                    print(f"  Existing data: {existing_count} candles")
                    print(f"  Range: {existing_start} to {existing_end}")

                # Fetch data
                candles = fetcher.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    start=start_date,
                    end=end_date
                )

                if not candles:
                    print(f"  ⚠️  No data available")
                    continue

                # Calculate expected candles
                expected = fetcher.calculate_expected_candles(timeframe, start_date, end_date)
                coverage_pct = (len(candles) / expected * 100) if expected > 0 else 0

                print(f"  Fetched: {len(candles)} candles")
                print(f"  Expected: ~{expected} candles")
                print(f"  Coverage: {coverage_pct:.1f}%")

                # Store in database
                print(f"  Storing to database...")
                inserted = storage.insert_candles(
                    symbol=symbol,
                    timeframe=timeframe,
                    candles=candles
                )

                print(f"  ✓ Stored {inserted} candles")

                # Check for gaps
                gaps = storage.detect_gaps(
                    symbol=symbol,
                    timeframe=timeframe,
                    expected_interval_minutes=int(timeframe[:-1]) if timeframe[-1] == 'm' else int(timeframe[:-1]) * 60
                )

                if gaps:
                    print(f"  ⚠️  Found {len(gaps)} gap(s) in data:")
                    for i, (gap_start, gap_end) in enumerate(gaps[:5]):  # Show first 5
                        duration = gap_end - gap_start
                        print(f"     Gap {i+1}: {gap_start} to {gap_end} ({duration})")
                    if len(gaps) > 5:
                        print(f"     ... and {len(gaps) - 5} more")
                else:
                    print(f"  ✓ No gaps detected")

                total_candles += len(candles)

            except Exception as e:
                print(f"  ❌ Error: {e}")
                continue

    # Final summary
    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE")
    print("=" * 60)
    print(f"\nTotal candles downloaded: {total_candles}")

    print("\nData summary:")
    summary = storage.get_summary()
    print(summary.to_string(index=False))

    print("\n✓ All done!")

    return 0


if __name__ == '__main__':
    sys.exit(download_historical_data())
