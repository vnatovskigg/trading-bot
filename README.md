# Crypto Trading Bot

A Python-based automated trading system that combines trend-following and mean-reversion strategies for cryptocurrency markets.

## Features

- **Dual Strategy System**: Trend Following (1h) + Mean Reversion (15m)
- **Robust Backtesting**: Event-driven engine with realistic costs
- **TimescaleDB**: Optimized time-series data storage
- **Paper Trading**: Test on Binance testnet before going live
- **Risk Management**: Portfolio-level controls and position sizing

## Quick Start

### 1. Start the Database

```bash
docker-compose up -d
```

### 2. Set Up Python Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure

Copy `.env.example` to `.env` and adjust settings:

```bash
cp .env.example .env
```

### 4. Download Historical Data

```bash
python scripts/download_data.py
```

### 5. Run Backtest

```bash
python main_backtest.py
```

## Project Structure

```
trader/
├── config/              # Configuration files
├── data/                # Data layer (fetcher, storage, provider)
├── indicators/          # Technical indicators
├── strategies/          # Trading strategies
├── backtest/            # Backtesting engine
├── database/            # Database connection utilities
├── logs/                # Log files
├── results/             # Backtest results
├── tests/               # Unit and integration tests
└── docker/              # Docker configuration
```

## Testing

```bash
pytest tests/
```

## License

Proprietary - Not for distribution
