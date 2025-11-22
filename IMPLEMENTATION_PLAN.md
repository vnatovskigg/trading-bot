# Crypto Trading Bot - Implementation Plan

**Last Updated:** 2025-11-22
**Current Phase:** Not Started
**Current Status:** Planning Complete

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Technology Stack](#technology-stack)
3. [MVP Scope (Phase 1)](#mvp-scope-phase-1)
4. [Extended Scope (Phase 2+)](#extended-scope-phase-2)
5. [Implementation Progress Tracker](#implementation-progress-tracker)
6. [Detailed Component Specifications](#detailed-component-specifications)
7. [Testing Strategy](#testing-strategy)
8. [Deployment Plan](#deployment-plan)

---

## Project Overview

### Goal
Build an automated crypto trading system that combines two complementary strategies:
- **Trend Following** (medium timeframe): Capture directional moves in trending markets
- **Mean Reversion** (short timeframe): Exploit short-term overreactions in ranging markets

### Core Principles
- **No curve-fitting**: Focus on robust strategies, realistic execution, proper risk management
- **Backtested & validated**: Test thoroughly before risking capital
- **Progressive deployment**: Backtest → Paper trading → Small live → Scale up
- **Monitoring-first**: Know what's happening and why at all times

### Initial Scope
- **Symbols**: BTC/USDT, ETH/USDT (liquid pairs)
- **Exchange**: Binance (testnet for paper, low fees for live)
- **Market**: Spot trading (can extend to perps later)
- **Capital**: Start testing with $10k equivalent
- **Positions**: Single position per symbol per strategy

---

## Technology Stack

### Core Technologies
- **Language**: Python 3.11+
- **Database**: PostgreSQL 15+ with TimescaleDB extension
  - Rationale: Time-series optimized, scales to tick data, handles concurrent writes for live trading
  - Setup: Docker container for development, managed instance for production
- **Exchange API**: CCXT library (unified interface across exchanges)
- **Data Processing**: pandas, numpy
- **Technical Indicators**: Custom implementation (avoid TA-Lib dependency issues)

### Key Libraries
```
ccxt>=4.0.0           # Exchange API abstraction
pandas>=2.0.0         # Data manipulation
numpy>=1.24.0         # Numerical computations
psycopg2-binary       # PostgreSQL adapter
sqlalchemy>=2.0.0     # ORM and query builder
pydantic>=2.0.0       # Config validation and data models
pyyaml>=6.0           # Config file parsing
pytest>=7.4.0         # Testing
matplotlib>=3.7.0     # Visualization
python-dotenv>=1.0.0  # Environment variables
```

### Development Tools
- **Code Quality**: black, ruff, mypy
- **Version Control**: git
- **Container**: Docker & docker-compose
- **Secrets**: .env files (local), environment variables (production)

---

## MVP Scope (Phase 1)

### Components to Implement
1. **Data Layer** (Component 2)
2. **Indicators & Features** (Component 3)
3. **Strategy Engine** (Component 4)
4. **Backtesting Engine** (Component 8)

### What's NOT in MVP
- Signal orchestrator (strategies tested independently)
- Risk management module (simple fixed % position sizing)
- Execution layer (simulated fills only)
- Live trading capabilities
- Monitoring dashboards (basic logging only)
- Advanced configuration management

### MVP Success Criteria
✅ Download and store 2 years of historical OHLCV data
✅ Calculate all required indicators correctly
✅ Generate signals from both strategies
✅ Run complete backtests with realistic costs
✅ Produce equity curves and performance metrics
✅ Have clean, testable, extensible code structure
✅ Clear documentation of what works and what doesn't

---

## Extended Scope (Phase 2+)

### Phase 2: Risk & Portfolio Management
- **Component 5**: Signal Orchestrator / Portfolio Combiner
- **Component 6**: Risk Management & Position Sizing
  - Per-trade risk limits
  - Portfolio-level exposure limits
  - Volatility-based position sizing
  - Drawdown circuit breakers

### Phase 3: Live Trading Infrastructure
- **Component 7**: Execution Layer
  - Real exchange adapter (Binance testnet)
  - Order management (place, cancel, track)
  - Position reconciliation
  - Error handling & retries
- **Component 9**: Monitoring & Logging
  - Structured logging
  - Metrics collection
  - Alerting system
- **Component 11**: Deployment & Ops
  - Paper trading mode
  - Process management
  - Graceful shutdown/restart

### Phase 4: Production Readiness
- **Component 10**: Configuration & Experiment Management
- Advanced features:
  - Multiple symbols & timeframes
  - Strategy parameter optimization
  - Walk-forward analysis
  - Live performance tracking vs backtest

### Phase 5: Advanced Features (Future)
- Perpetual futures support
- Additional strategies
- Machine learning features
- Multi-exchange arbitrage
- Advanced order types (iceberg, TWAP)

---

## Implementation Progress Tracker

### Phase 1: MVP - Data Layer ❌ NOT STARTED

#### 1.1 Database Setup
- [ ] Set up PostgreSQL + TimescaleDB Docker container
- [ ] Create database schema
- [ ] Write migration scripts
- [ ] Test time-series queries and performance

**Files to create:**
- `docker-compose.yml`
- `database/schema.sql`
- `database/migrations/001_initial_schema.sql`
- `database/connection.py`

#### 1.2 Data Models
- [ ] Implement `Candle` dataclass
- [ ] Implement `BarSeries` class
- [ ] Add validation and type hints
- [ ] Unit tests for data models

**Files to create:**
- `data/models.py`
- `tests/test_data_models.py`

#### 1.3 Historical Data Fetcher
- [ ] CCXT integration for Binance
- [ ] Download OHLCV for configured symbols/timeframes
- [ ] Handle rate limits and pagination
- [ ] Gap detection and reporting
- [ ] Progress tracking for long downloads

**Files to create:**
- `data/fetcher.py`
- `tests/test_fetcher.py`

**Implementation notes:**
- Use `ccxt.binance.fetch_ohlcv()`
- Binance limit: 1000 candles per request
- Rate limit: ~1200 requests/minute (weight-based)
- Store raw data with exchange timestamp + UTC conversion
- Detect gaps > 2x candle period

#### 1.4 Data Storage
- [ ] Insert OHLCV to TimescaleDB
- [ ] Upsert logic (handle re-downloads)
- [ ] Batch inserts for performance
- [ ] Query interface: get_bars(symbol, timeframe, start, end)
- [ ] Handle missing data gracefully

**Files to create:**
- `data/storage.py`
- `tests/test_storage.py`

**Database schema:**
```sql
CREATE TABLE ohlcv (
    time TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    open NUMERIC NOT NULL,
    high NUMERIC NOT NULL,
    low NUMERIC NOT NULL,
    close NUMERIC NOT NULL,
    volume NUMERIC NOT NULL,
    PRIMARY KEY (time, symbol, timeframe)
);

SELECT create_hypertable('ohlcv', 'time');
CREATE INDEX idx_symbol_timeframe ON ohlcv (symbol, timeframe, time DESC);
```

#### 1.5 Data Provider
- [ ] Unified interface for backtest & live modes
- [ ] Time alignment across timeframes
- [ ] No look-ahead bias enforcement
- [ ] Candle completion validation
- [ ] Efficient windowing (last N bars)

**Files to create:**
- `data/provider.py`
- `tests/test_provider.py`

**Key methods:**
```python
class DataProvider:
    def get_bars(symbol, timeframe, start, end) -> BarSeries
    def get_latest_bars(symbol, timeframe, count) -> BarSeries
    def get_aligned_timeframes(symbol, timeframes, timestamp) -> Dict[str, BarSeries]
    def validate_completeness(bars) -> bool
```

### Phase 1: MVP - Indicators ❌ NOT STARTED

#### 2.1 Base Indicator Framework
- [ ] Abstract `Indicator` base class
- [ ] Caching mechanism
- [ ] Validity tracking (enough data for calculation)
- [ ] Consistent pandas Series input/output

**Files to create:**
- `indicators/base.py`
- `tests/test_indicator_base.py`

#### 2.2 Trend Indicators
- [ ] Simple Moving Average (SMA)
- [ ] Exponential Moving Average (EMA)
- [ ] ADX (Average Directional Index)
- [ ] Donchian Channels (optional for MVP)

**Files to create:**
- `indicators/trend.py`
- `tests/test_trend_indicators.py`

**Implementation notes:**
- SMA: `series.rolling(window=period).mean()`
- EMA: `series.ewm(span=period, adjust=False).mean()`
- ADX: Calculate +DI, -DI, DX, then smooth for ADX
- Mark first N values as invalid (warmup period)

#### 2.3 Mean Reversion Indicators
- [ ] RSI (Relative Strength Index)
- [ ] Bollinger Bands (middle, upper, lower)
- [ ] Z-Score

**Files to create:**
- `indicators/mean_reversion.py`
- `tests/test_mean_reversion_indicators.py`

**Implementation notes:**
- RSI: Wilder's smoothing method, 14-period default
- BB: 20-period SMA ± 2 standard deviations
- Z-Score: (price - mean) / std_dev

#### 2.4 Volatility Indicators
- [ ] ATR (Average True Range)
- [ ] Realized Volatility (log returns std dev)
- [ ] Bollinger Band Width

**Files to create:**
- `indicators/volatility.py`
- `tests/test_volatility_indicators.py`

**Implementation notes:**
- ATR: Wilder's smoothing of True Range
- Realized Vol: `np.log(close/close.shift(1)).std() * np.sqrt(periods_per_year)`
- BB Width: (upper_band - lower_band) / middle_band

### Phase 1: MVP - Strategies ❌ NOT STARTED

#### 3.1 Strategy Base Class
- [ ] Abstract `Strategy` interface
- [ ] `on_new_candle()` method signature
- [ ] `TargetPosition` output model
- [ ] Strategy state management

**Files to create:**
- `strategies/base.py`
- `strategies/models.py`
- `tests/test_strategy_base.py`

**TargetPosition model:**
```python
@dataclass
class TargetPosition:
    symbol: str
    timestamp: datetime
    target_exposure: float  # -1.0 to +1.0 (short to long)
    confidence: float  # 0.0 to 1.0
    strategy_name: str
    timeframe: str
    metadata: dict  # Store reasoning, indicator values, etc.
```

#### 3.2 Trend Following Strategy
- [ ] MA crossover logic (fast > slow = bullish)
- [ ] ADX trend strength filter
- [ ] Volatility guard (high ATR = reduce/skip)
- [ ] Entry conditions
- [ ] Exit conditions
- [ ] State tracking (in position, entry price, bars held)

**Files to create:**
- `strategies/trend_following.py`
- `tests/test_trend_following.py`

**Logic specification:**
```
ENTRY (Long):
- fast_MA > slow_MA
- price > both MAs
- ADX > threshold (e.g., 25)
- ATR < critical_threshold (e.g., 3x normal)
→ target_exposure = +1.0

ENTRY (Short):
- fast_MA < slow_MA
- price < both MAs
- ADX > threshold
- ATR < critical_threshold
→ target_exposure = -1.0

EXIT:
- MA cross reversal (fast crosses back through slow)
- Hold time > max_bars (e.g., 200)
→ target_exposure = 0.0

NO POSITION:
- ADX < threshold (choppy market)
- ATR > critical (too volatile)
→ target_exposure = 0.0
```

**Default parameters:**
- `ma_fast`: 20
- `ma_slow`: 50
- `adx_period`: 14
- `adx_threshold`: 25
- `atr_period`: 14
- `atr_critical_multiplier`: 3.0
- `max_hold_bars`: 200

#### 3.3 Mean Reversion Strategy
- [ ] RSI overbought/oversold detection
- [ ] Bollinger Band touch logic
- [ ] Trend filter (higher TF awareness)
- [ ] Entry conditions
- [ ] Exit conditions (snapback or max hold)
- [ ] Cooldown between trades

**Files to create:**
- `strategies/mean_reversion.py`
- `tests/test_mean_reversion.py`

**Logic specification:**
```
TREND FILTER (using 50 MA on same timeframe):
- strong_uptrend: price > MA_50 and (price - MA_50) / MA_50 > threshold (e.g., 2%)
- strong_downtrend: price < MA_50 and (MA_50 - price) / MA_50 > threshold
- ranging: else

ENTRY (Long):
- RSI < oversold_threshold (e.g., 30)
- price < lower_BB
- NOT strong_downtrend
→ target_exposure = +1.0

ENTRY (Short):
- RSI > overbought_threshold (e.g., 70)
- price > upper_BB
- NOT strong_uptrend
→ target_exposure = -1.0

EXIT:
- Price crosses back to BB middle
- RSI crosses back to 50
- Hold time > max_hold_bars (e.g., 50)
→ target_exposure = 0.0

COOLDOWN:
- After exit, wait N bars before next signal
```

**Default parameters:**
- `rsi_period`: 14
- `rsi_oversold`: 30
- `rsi_overbought`: 70
- `bb_period`: 20
- `bb_std`: 2.0
- `trend_filter_ma`: 50
- `trend_filter_threshold`: 0.02 (2%)
- `max_hold_bars`: 50
- `cooldown_bars`: 10

### Phase 1: MVP - Backtesting Engine ❌ NOT STARTED

#### 4.1 Execution Simulator
- [ ] Simulate market order fills
- [ ] Apply slippage model
- [ ] Apply trading fees
- [ ] Track filled price, fee, net proceeds
- [ ] Handle partial fills (for MVP: assume full fills)

**Files to create:**
- `backtest/execution_sim.py`
- `tests/test_execution_sim.py`

**Fill logic:**
```
Market Order Fill:
- Fill price = next_candle.open * (1 + slippage_factor)
  - Long: next_candle.open * (1 + slippage_bps/10000)
  - Short: next_candle.open * (1 - slippage_bps/10000)
- Fee = fill_price * quantity * fee_rate
- Net proceeds = fill_price * quantity ± fee

Slippage model (simple):
- Fixed basis points (e.g., 10 bps = 0.1%)
- Can extend to volatility-dependent later
```

**Default parameters:**
- `slippage_bps`: 10 (0.1%)
- `maker_fee`: 0.001 (0.1%)
- `taker_fee`: 0.001 (0.1%)

#### 4.2 Backtest Engine Core
- [ ] Event-driven loop through candles
- [ ] Call strategies with proper data windows
- [ ] Translate target positions to trades
- [ ] Track portfolio state (cash, positions, equity)
- [ ] Record all trades with metadata
- [ ] Handle position entry/exit/hold logic

**Files to create:**
- `backtest/engine.py`
- `backtest/portfolio.py`
- `tests/test_backtest_engine.py`

**Engine loop pseudocode:**
```python
for timestamp in sorted_timestamps:
    # Get market data up to (not including) current candle
    historical_data = provider.get_bars(symbol, timeframe, start, timestamp)

    # Compute indicators on historical data
    indicators = calculate_indicators(historical_data)

    # Get strategy signal (based on last CLOSED candle)
    target_position = strategy.on_new_candle(symbol, timeframe, historical_data, indicators)

    # Determine trade needed (if any)
    current_position = portfolio.get_position(symbol)
    trade = calculate_trade(current_position, target_position)

    # Simulate fill at NEXT candle's open (no look-ahead)
    if trade:
        fill = execution_sim.fill_market_order(trade, next_candle)
        portfolio.update(fill)

    # Update portfolio value with current prices
    portfolio.mark_to_market(current_candle.close)

    # Record state
    equity_curve.append(timestamp, portfolio.equity)
```

**Portfolio tracking:**
```python
class Portfolio:
    cash: float
    positions: Dict[str, Position]  # symbol -> Position
    initial_capital: float

    def get_equity() -> float
    def get_position(symbol) -> Position
    def update(fill: Fill) -> None
    def mark_to_market(prices: Dict[str, float]) -> None
```

**Position model:**
```python
@dataclass
class Position:
    symbol: str
    quantity: float  # positive = long, negative = short
    avg_entry_price: float
    entry_timestamp: datetime
    strategy_name: str
    unrealized_pnl: float
```

#### 4.3 Performance Metrics
- [ ] Equity curve tracking
- [ ] Returns calculation (total, annualized)
- [ ] Sharpe ratio, Sortino ratio
- [ ] Maximum drawdown, drawdown duration
- [ ] Win rate, average win/loss
- [ ] Profit factor
- [ ] Per-trade statistics

**Files to create:**
- `backtest/metrics.py`
- `tests/test_metrics.py`

**Metrics to calculate:**
```python
class PerformanceMetrics:
    # Returns
    total_return: float
    cagr: float  # Compound Annual Growth Rate

    # Risk-adjusted returns
    sharpe_ratio: float  # (return - rf_rate) / volatility
    sortino_ratio: float  # (return - rf_rate) / downside_deviation

    # Drawdown
    max_drawdown: float  # Largest peak-to-trough decline
    max_drawdown_duration: int  # Days underwater

    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float  # gross_profit / gross_loss

    # Exposure
    avg_time_in_market: float
    max_concurrent_positions: int
```

#### 4.4 Visualization
- [ ] Equity curve plot
- [ ] Drawdown plot
- [ ] Monthly returns heatmap
- [ ] Trade markers on price chart
- [ ] Indicator overlays

**Files to create:**
- `backtest/visualizer.py`
- `tests/test_visualizer.py`

**Plots to generate:**
1. Equity curve with buy/hold comparison
2. Underwater (drawdown) chart
3. Monthly returns table
4. Price + indicators + trade signals

#### 4.5 Backtest Runner & Reports
- [ ] Main backtest orchestration script
- [ ] Load config, initialize components
- [ ] Run backtest for each strategy independently
- [ ] Generate performance reports
- [ ] Save results (metrics, trades, equity curve)

**Files to create:**
- `backtest/runner.py`
- `main_backtest.py` (entry point)
- `tests/test_backtest_integration.py`

**Output artifacts:**
```
results/
├── backtest_YYYYMMDD_HHMMSS/
│   ├── config.yaml (snapshot of config used)
│   ├── metrics.json
│   ├── trades.csv
│   ├── equity_curve.csv
│   ├── equity_plot.png
│   ├── drawdown_plot.png
│   └── report.txt (human-readable summary)
```

### Phase 1: MVP - Configuration ❌ NOT STARTED

#### 5.1 Configuration File
- [ ] Create `config/default.yaml`
- [ ] Validation with Pydantic
- [ ] Environment-specific overrides

**Files to create:**
- `config/default.yaml`
- `config/models.py` (Pydantic models)
- `config/loader.py`
- `tests/test_config.py`

**Config structure:**
```yaml
# Exchange settings
exchange:
  name: binance
  mode: backtest  # backtest | paper | live

# Trading pairs
symbols:
  - BTC/USDT
  - ETH/USDT

# Timeframes
timeframes:
  trend: 1h
  mean_reversion: 15m

# Data settings
data:
  history_years: 2
  database_url: postgresql://user:pass@localhost:5432/trading

# Strategy: Trend Following
strategies:
  trend_following:
    enabled: true
    ma_fast: 20
    ma_slow: 50
    adx_period: 14
    adx_threshold: 25
    atr_period: 14
    atr_critical_multiplier: 3.0
    max_hold_bars: 200

  # Strategy: Mean Reversion
  mean_reversion:
    enabled: true
    rsi_period: 14
    rsi_oversold: 30
    rsi_overbought: 70
    bb_period: 20
    bb_std: 2.0
    trend_filter_ma: 50
    trend_filter_threshold: 0.02
    max_hold_bars: 50
    cooldown_bars: 10

# Backtesting settings
backtest:
  initial_capital: 10000
  position_size_pct: 0.20  # 20% of equity per position
  slippage_bps: 10
  maker_fee: 0.001
  taker_fee: 0.001
  start_date: null  # null = auto (data_history_years ago)
  end_date: null  # null = latest available

# Logging
logging:
  level: INFO
  file: logs/trading.log
```

### Phase 1: MVP - Infrastructure ❌ NOT STARTED

#### 6.1 Project Setup
- [ ] Create directory structure
- [ ] Initialize git repository
- [ ] Create .gitignore
- [ ] Create requirements.txt
- [ ] Create README.md
- [ ] Set up virtual environment

**Files to create:**
- `.gitignore`
- `requirements.txt`
- `README.md`
- `.env.example`

#### 6.2 Docker Setup
- [ ] PostgreSQL + TimescaleDB container
- [ ] docker-compose.yml
- [ ] Database initialization scripts
- [ ] Volume mounts for persistence

**Files to create:**
- `docker-compose.yml`
- `docker/postgres/init.sql`

#### 6.3 Testing Setup
- [ ] pytest configuration
- [ ] Test fixtures for data
- [ ] Mock exchange responses
- [ ] Test database setup/teardown

**Files to create:**
- `pytest.ini`
- `tests/conftest.py`
- `tests/fixtures/`

---

## Detailed Component Specifications

### Component 2: Data Layer

**Responsibilities:**
1. Fetch historical OHLCV from exchange
2. Store in TimescaleDB with proper indexing
3. Provide unified query interface
4. Ensure time alignment and data quality
5. Prevent look-ahead bias

**Key Classes:**

```python
# data/models.py
@dataclass
class Candle:
    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal

    def __post_init__(self):
        # Validation: high >= low, etc.
        pass

class BarSeries:
    """Wrapper around pandas DataFrame with validation"""
    def __init__(self, df: pd.DataFrame):
        self.df = df  # Index: timestamp, Columns: OHLCV
        self._validate()

    def _validate(self):
        # Check sorted, no gaps, no nulls, etc.
        pass

    def get_closes(self) -> pd.Series:
        return self.df['close']

    def get_highs(self) -> pd.Series:
        return self.df['high']

    # ... etc

# data/fetcher.py
class HistoricalDataFetcher:
    def __init__(self, exchange_id: str):
        self.exchange = ccxt.binance()

    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime
    ) -> List[Candle]:
        """Fetch with pagination, rate limiting, error handling"""
        pass

# data/storage.py
class OHLCVStorage:
    def __init__(self, db_url: str):
        self.engine = create_engine(db_url)

    def insert_candles(self, symbol: str, timeframe: str, candles: List[Candle]):
        """Batch insert with upsert logic"""
        pass

    def get_candles(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime
    ) -> BarSeries:
        """Query with caching"""
        pass

    def get_latest_timestamp(self, symbol: str, timeframe: str) -> Optional[datetime]:
        """For incremental updates"""
        pass

    def detect_gaps(self, symbol: str, timeframe: str) -> List[Tuple[datetime, datetime]]:
        """Find missing data ranges"""
        pass

# data/provider.py
class DataProvider:
    """Unified interface for backtest and live modes"""
    def __init__(self, storage: OHLCVStorage, mode: str = 'backtest'):
        self.storage = storage
        self.mode = mode

    def get_bars(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
        limit: Optional[int] = None
    ) -> BarSeries:
        """Get historical bars, ensuring no look-ahead"""
        pass

    def get_latest_bars(
        self,
        symbol: str,
        timeframe: str,
        count: int
    ) -> BarSeries:
        """Get last N closed bars"""
        pass
```

**Database Schema:**
```sql
-- Create extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- OHLCV table
CREATE TABLE ohlcv (
    time TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    open NUMERIC(20, 8) NOT NULL,
    high NUMERIC(20, 8) NOT NULL,
    low NUMERIC(20, 8) NOT NULL,
    close NUMERIC(20, 8) NOT NULL,
    volume NUMERIC(30, 8) NOT NULL,
    PRIMARY KEY (time, symbol, timeframe)
);

-- Convert to hypertable (TimescaleDB)
SELECT create_hypertable('ohlcv', 'time');

-- Indexes for fast queries
CREATE INDEX idx_symbol_timeframe ON ohlcv (symbol, timeframe, time DESC);
CREATE INDEX idx_symbol_time ON ohlcv (symbol, time DESC);

-- Compression policy (optional, for older data)
ALTER TABLE ohlcv SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol,timeframe'
);

SELECT add_compression_policy('ohlcv', INTERVAL '7 days');
```

**Critical Implementation Details:**

1. **Time Alignment:**
   - All timestamps in UTC
   - Candles aligned to their opening time
   - 1h candle at 14:00 represents [14:00, 15:00)
   - When querying for "end time", exclude that candle (semi-open interval)

2. **No Look-Ahead Bias:**
   - Strategy sees only CLOSED candles
   - Never use current incomplete candle for signals
   - Fills simulated at NEXT candle's open

3. **Gap Handling:**
   - Log warnings for gaps > 2x candle period
   - Option to skip trading during data gaps
   - Backfill on detection

4. **Caching:**
   - Cache recent queries in memory
   - LRU cache with configurable size
   - Invalidate on new data insertion

### Component 3: Indicators & Features

**Responsibilities:**
1. Compute technical indicators from price data
2. Track validity (warmup period)
3. Cache computed values
4. Provide consistent interface

**Key Classes:**

```python
# indicators/base.py
class Indicator(ABC):
    def __init__(self, period: int):
        self.period = period
        self._cache = {}

    @abstractmethod
    def calculate(self, data: pd.Series) -> pd.Series:
        """Calculate indicator values"""
        pass

    def is_valid(self, data: pd.Series) -> bool:
        """Check if enough data for valid calculation"""
        return len(data) >= self.period

    def get_warmup_period(self) -> int:
        """How many bars needed before indicator is valid"""
        return self.period

# indicators/trend.py
class SMA(Indicator):
    def calculate(self, data: pd.Series) -> pd.Series:
        result = data.rolling(window=self.period).mean()
        # Mark first (period-1) values as NaN (invalid)
        return result

class EMA(Indicator):
    def calculate(self, data: pd.Series) -> pd.Series:
        return data.ewm(span=self.period, adjust=False).mean()

class ADX(Indicator):
    def __init__(self, period: int = 14):
        super().__init__(period)

    def calculate(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """
        Calculate ADX
        1. Calculate +DM, -DM
        2. Calculate +DI, -DI
        3. Calculate DX = abs(+DI - -DI) / (+DI + -DI) * 100
        4. ADX = EMA of DX
        """
        # Implementation details...
        pass

# indicators/mean_reversion.py
class RSI(Indicator):
    def calculate(self, data: pd.Series) -> pd.Series:
        """
        Wilder's RSI:
        1. Calculate price changes
        2. Separate gains and losses
        3. Wilder's smoothing (EMA with alpha = 1/period)
        4. RS = avg_gain / avg_loss
        5. RSI = 100 - (100 / (1 + RS))
        """
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).ewm(alpha=1/self.period, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/self.period, adjust=False).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

class BollingerBands:
    def __init__(self, period: int = 20, std_dev: float = 2.0):
        self.period = period
        self.std_dev = std_dev

    def calculate(self, data: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Returns (middle, upper, lower)"""
        middle = data.rolling(window=self.period).mean()
        std = data.rolling(window=self.period).std()
        upper = middle + (std * self.std_dev)
        lower = middle - (std * self.std_dev)
        return middle, upper, lower

# indicators/volatility.py
class ATR(Indicator):
    def calculate(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """
        True Range = max(high - low, abs(high - prev_close), abs(low - prev_close))
        ATR = Wilder's smoothing of TR
        """
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/self.period, adjust=False).mean()
        return atr
```

**Indicator Management:**

```python
# indicators/manager.py
class IndicatorManager:
    """Centralized indicator calculation and caching"""
    def __init__(self, config: dict):
        self.indicators = {}
        self._initialize_indicators(config)

    def calculate_all(self, bars: BarSeries) -> Dict[str, pd.Series]:
        """Calculate all configured indicators"""
        results = {}

        # Trend indicators
        results['sma_fast'] = self.indicators['sma_fast'].calculate(bars.get_closes())
        results['sma_slow'] = self.indicators['sma_slow'].calculate(bars.get_closes())
        results['adx'] = self.indicators['adx'].calculate(
            bars.get_highs(),
            bars.get_lows(),
            bars.get_closes()
        )

        # Mean reversion indicators
        results['rsi'] = self.indicators['rsi'].calculate(bars.get_closes())
        bb_mid, bb_upper, bb_lower = self.indicators['bb'].calculate(bars.get_closes())
        results['bb_middle'] = bb_mid
        results['bb_upper'] = bb_upper
        results['bb_lower'] = bb_lower

        # Volatility indicators
        results['atr'] = self.indicators['atr'].calculate(
            bars.get_highs(),
            bars.get_lows(),
            bars.get_closes()
        )

        return results
```

### Component 4: Strategy Engine

**Strategy Interface:**

```python
# strategies/base.py
class Strategy(ABC):
    def __init__(self, config: dict):
        self.config = config
        self.name = self.__class__.__name__
        self.state = {}  # Internal state tracking

    @abstractmethod
    def on_new_candle(
        self,
        symbol: str,
        timeframe: str,
        bars: BarSeries,
        indicators: Dict[str, pd.Series]
    ) -> TargetPosition:
        """Process new candle and return target position"""
        pass

    def reset_state(self):
        """Reset internal state (for new backtest run)"""
        self.state = {}

# strategies/models.py
@dataclass
class TargetPosition:
    symbol: str
    timestamp: datetime
    target_exposure: float  # -1.0 to +1.0
    confidence: float  # 0.0 to 1.0
    strategy_name: str
    timeframe: str
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        assert -1.0 <= self.target_exposure <= 1.0
        assert 0.0 <= self.confidence <= 1.0
```

**Trend Following Implementation:**

```python
# strategies/trend_following.py
class TrendFollowingStrategy(Strategy):
    def __init__(self, config: dict):
        super().__init__(config)

        # Parameters
        self.ma_fast_period = config['ma_fast']
        self.ma_slow_period = config['ma_slow']
        self.adx_threshold = config['adx_threshold']
        self.atr_critical_multiplier = config['atr_critical_multiplier']
        self.max_hold_bars = config.get('max_hold_bars', 200)

        # State
        self.state = {
            'position': 0.0,  # Current exposure
            'entry_price': None,
            'entry_time': None,
            'bars_held': 0
        }

    def on_new_candle(
        self,
        symbol: str,
        timeframe: str,
        bars: BarSeries,
        indicators: Dict[str, pd.Series]
    ) -> TargetPosition:

        # Get latest values
        close = bars.get_closes().iloc[-1]
        ma_fast = indicators['sma_fast'].iloc[-1]
        ma_slow = indicators['sma_slow'].iloc[-1]
        adx = indicators['adx'].iloc[-1]
        atr = indicators['atr'].iloc[-1]

        # Calculate ATR baseline (e.g., 20-bar average)
        atr_baseline = indicators['atr'].iloc[-20:].mean()

        # Check validity
        if pd.isna(ma_fast) or pd.isna(ma_slow) or pd.isna(adx) or pd.isna(atr):
            return self._no_position(symbol, bars.df.index[-1], "Indicators not ready")

        # Volatility guard
        if atr > atr_baseline * self.atr_critical_multiplier:
            return self._no_position(symbol, bars.df.index[-1], "Extreme volatility")

        # Trend strength filter
        if adx < self.adx_threshold:
            return self._exit_position(symbol, bars.df.index[-1], "Weak trend (ADX)")

        # Determine trend direction
        bullish = (ma_fast > ma_slow) and (close > ma_fast) and (close > ma_slow)
        bearish = (ma_fast < ma_slow) and (close < ma_fast) and (close < ma_slow)

        # Update bars held
        if self.state['position'] != 0:
            self.state['bars_held'] += 1

        # Max hold time check
        if self.state['bars_held'] >= self.max_hold_bars:
            return self._exit_position(symbol, bars.df.index[-1], "Max hold time reached")

        # Entry logic
        if self.state['position'] == 0:
            if bullish:
                return self._enter_long(symbol, bars.df.index[-1], close, adx, indicators)
            elif bearish:
                return self._enter_short(symbol, bars.df.index[-1], close, adx, indicators)

        # Exit logic (already in position)
        elif self.state['position'] > 0:  # Long position
            if bearish or ma_fast < ma_slow:
                return self._exit_position(symbol, bars.df.index[-1], "Trend reversal (long exit)")

        elif self.state['position'] < 0:  # Short position
            if bullish or ma_fast > ma_slow:
                return self._exit_position(symbol, bars.df.index[-1], "Trend reversal (short exit)")

        # Hold current position
        return TargetPosition(
            symbol=symbol,
            timestamp=bars.df.index[-1],
            target_exposure=self.state['position'],
            confidence=min(adx / 50.0, 1.0),  # Scale ADX to confidence
            strategy_name=self.name,
            timeframe=timeframe,
            metadata={'reason': 'Holding position', 'adx': adx}
        )

    def _enter_long(self, symbol, timestamp, price, adx, indicators):
        self.state['position'] = 1.0
        self.state['entry_price'] = price
        self.state['entry_time'] = timestamp
        self.state['bars_held'] = 0

        return TargetPosition(
            symbol=symbol,
            timestamp=timestamp,
            target_exposure=1.0,
            confidence=min(adx / 50.0, 1.0),
            strategy_name=self.name,
            timeframe='1h',
            metadata={
                'reason': 'Enter long',
                'adx': adx,
                'ma_fast': indicators['sma_fast'].iloc[-1],
                'ma_slow': indicators['sma_slow'].iloc[-1]
            }
        )

    def _enter_short(self, symbol, timestamp, price, adx, indicators):
        self.state['position'] = -1.0
        self.state['entry_price'] = price
        self.state['entry_time'] = timestamp
        self.state['bars_held'] = 0

        return TargetPosition(
            symbol=symbol,
            timestamp=timestamp,
            target_exposure=-1.0,
            confidence=min(adx / 50.0, 1.0),
            strategy_name=self.name,
            timeframe='1h',
            metadata={
                'reason': 'Enter short',
                'adx': adx,
                'ma_fast': indicators['sma_fast'].iloc[-1],
                'ma_slow': indicators['sma_slow'].iloc[-1]
            }
        )

    def _exit_position(self, symbol, timestamp, reason):
        self.state['position'] = 0.0
        self.state['entry_price'] = None
        self.state['entry_time'] = None
        self.state['bars_held'] = 0

        return TargetPosition(
            symbol=symbol,
            timestamp=timestamp,
            target_exposure=0.0,
            confidence=0.0,
            strategy_name=self.name,
            timeframe='1h',
            metadata={'reason': reason}
        )

    def _no_position(self, symbol, timestamp, reason):
        return TargetPosition(
            symbol=symbol,
            timestamp=timestamp,
            target_exposure=0.0,
            confidence=0.0,
            strategy_name=self.name,
            timeframe='1h',
            metadata={'reason': reason}
        )
```

**Mean Reversion Implementation:**

```python
# strategies/mean_reversion.py
class MeanReversionStrategy(Strategy):
    def __init__(self, config: dict):
        super().__init__(config)

        # Parameters
        self.rsi_oversold = config['rsi_oversold']
        self.rsi_overbought = config['rsi_overbought']
        self.trend_filter_threshold = config['trend_filter_threshold']
        self.max_hold_bars = config['max_hold_bars']
        self.cooldown_bars = config.get('cooldown_bars', 10)

        # State
        self.state = {
            'position': 0.0,
            'entry_price': None,
            'entry_time': None,
            'bars_held': 0,
            'bars_since_exit': 999  # Start with high value (no cooldown)
        }

    def on_new_candle(
        self,
        symbol: str,
        timeframe: str,
        bars: BarSeries,
        indicators: Dict[str, pd.Series]
    ) -> TargetPosition:

        # Get latest values
        close = bars.get_closes().iloc[-1]
        rsi = indicators['rsi'].iloc[-1]
        bb_upper = indicators['bb_upper'].iloc[-1]
        bb_lower = indicators['bb_lower'].iloc[-1]
        bb_middle = indicators['bb_middle'].iloc[-1]
        ma_trend = indicators.get('sma_trend', indicators['bb_middle']).iloc[-1]

        # Check validity
        if pd.isna(rsi) or pd.isna(bb_upper) or pd.isna(bb_lower):
            return self._no_position(symbol, bars.df.index[-1], "Indicators not ready")

        # Update counters
        if self.state['position'] != 0:
            self.state['bars_held'] += 1
        else:
            self.state['bars_since_exit'] += 1

        # Trend filter: check if strong trend exists
        trend_deviation = (close - ma_trend) / ma_trend
        strong_uptrend = trend_deviation > self.trend_filter_threshold
        strong_downtrend = trend_deviation < -self.trend_filter_threshold

        # Max hold time check
        if self.state['bars_held'] >= self.max_hold_bars:
            return self._exit_position(symbol, bars.df.index[-1], "Max hold time")

        # Mean reversion exit logic (if in position)
        if self.state['position'] > 0:  # Long position
            # Exit if reverted to middle or RSI normalized
            if close >= bb_middle or rsi >= 50:
                return self._exit_position(symbol, bars.df.index[-1], "Mean reversion (long)")

        elif self.state['position'] < 0:  # Short position
            if close <= bb_middle or rsi <= 50:
                return self._exit_position(symbol, bars.df.index[-1], "Mean reversion (short)")

        # Entry logic (only if not in cooldown)
        if self.state['position'] == 0 and self.state['bars_since_exit'] >= self.cooldown_bars:

            # Long entry: oversold + touched lower BB + no strong downtrend
            if rsi < self.rsi_oversold and close < bb_lower and not strong_downtrend:
                return self._enter_long(symbol, bars.df.index[-1], close, rsi, indicators)

            # Short entry: overbought + touched upper BB + no strong uptrend
            if rsi > self.rsi_overbought and close > bb_upper and not strong_uptrend:
                return self._enter_short(symbol, bars.df.index[-1], close, rsi, indicators)

        # Hold current position
        return TargetPosition(
            symbol=symbol,
            timestamp=bars.df.index[-1],
            target_exposure=self.state['position'],
            confidence=0.5,  # Mean reversion has moderate confidence
            strategy_name=self.name,
            timeframe=timeframe,
            metadata={
                'reason': 'Holding' if self.state['position'] != 0 else 'No signal',
                'rsi': rsi,
                'bars_held': self.state['bars_held']
            }
        )

    def _enter_long(self, symbol, timestamp, price, rsi, indicators):
        self.state['position'] = 1.0
        self.state['entry_price'] = price
        self.state['entry_time'] = timestamp
        self.state['bars_held'] = 0

        return TargetPosition(
            symbol=symbol,
            timestamp=timestamp,
            target_exposure=1.0,
            confidence=0.6,
            strategy_name=self.name,
            timeframe='15m',
            metadata={
                'reason': 'Enter long (oversold)',
                'rsi': rsi,
                'price': price,
                'bb_lower': indicators['bb_lower'].iloc[-1]
            }
        )

    def _enter_short(self, symbol, timestamp, price, rsi, indicators):
        self.state['position'] = -1.0
        self.state['entry_price'] = price
        self.state['entry_time'] = timestamp
        self.state['bars_held'] = 0

        return TargetPosition(
            symbol=symbol,
            timestamp=timestamp,
            target_exposure=-1.0,
            confidence=0.6,
            strategy_name=self.name,
            timeframe='15m',
            metadata={
                'reason': 'Enter short (overbought)',
                'rsi': rsi,
                'price': price,
                'bb_upper': indicators['bb_upper'].iloc[-1]
            }
        )

    def _exit_position(self, symbol, timestamp, reason):
        self.state['position'] = 0.0
        self.state['entry_price'] = None
        self.state['entry_time'] = None
        self.state['bars_held'] = 0
        self.state['bars_since_exit'] = 0

        return TargetPosition(
            symbol=symbol,
            timestamp=timestamp,
            target_exposure=0.0,
            confidence=0.0,
            strategy_name=self.name,
            timeframe='15m',
            metadata={'reason': reason}
        )

    def _no_position(self, symbol, timestamp, reason):
        return TargetPosition(
            symbol=symbol,
            timestamp=timestamp,
            target_exposure=0.0,
            confidence=0.0,
            strategy_name=self.name,
            timeframe='15m',
            metadata={'reason': reason}
        )
```

### Component 8: Backtesting Engine

**Portfolio & Position Tracking:**

```python
# backtest/portfolio.py
@dataclass
class Position:
    symbol: str
    quantity: float  # positive = long, negative = short
    avg_entry_price: float
    entry_timestamp: datetime
    strategy_name: str
    unrealized_pnl: float = 0.0

    def update_pnl(self, current_price: float):
        if self.quantity > 0:  # Long
            self.unrealized_pnl = (current_price - self.avg_entry_price) * self.quantity
        else:  # Short
            self.unrealized_pnl = (self.avg_entry_price - current_price) * abs(self.quantity)

@dataclass
class Trade:
    timestamp: datetime
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    price: float
    fee: float
    strategy_name: str
    realized_pnl: float = 0.0
    metadata: dict = field(default_factory=dict)

class Portfolio:
    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_history: List[Tuple[datetime, float]] = []

    def get_position(self, symbol: str) -> Optional[Position]:
        return self.positions.get(symbol)

    def get_equity(self) -> float:
        positions_value = sum(p.unrealized_pnl for p in self.positions.values())
        return self.cash + positions_value

    def execute_trade(self, trade: Trade):
        """Update portfolio state after a trade"""
        symbol = trade.symbol

        # Handle existing position
        if symbol in self.positions:
            position = self.positions[symbol]

            # Calculate realized P&L if closing/reducing position
            if (position.quantity > 0 and trade.side == 'sell') or \
               (position.quantity < 0 and trade.side == 'buy'):

                # Closing or reducing
                close_quantity = min(abs(position.quantity), trade.quantity)
                if position.quantity > 0:  # Closing long
                    trade.realized_pnl = (trade.price - position.avg_entry_price) * close_quantity
                else:  # Closing short
                    trade.realized_pnl = (position.avg_entry_price - trade.price) * close_quantity

                # Update position
                if abs(position.quantity) == trade.quantity:
                    # Fully closing
                    del self.positions[symbol]
                else:
                    # Partial close
                    position.quantity -= trade.quantity if trade.side == 'sell' else -trade.quantity

            else:
                # Adding to position (same direction)
                total_quantity = abs(position.quantity) + trade.quantity
                position.avg_entry_price = (
                    (position.avg_entry_price * abs(position.quantity) +
                     trade.price * trade.quantity) / total_quantity
                )
                position.quantity += trade.quantity if trade.side == 'buy' else -trade.quantity

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
        cost = trade.price * trade.quantity + trade.fee
        self.cash -= cost if trade.side == 'buy' else -cost

        # Record trade
        self.trades.append(trade)

    def mark_to_market(self, timestamp: datetime, prices: Dict[str, float]):
        """Update unrealized P&L and record equity"""
        for symbol, position in self.positions.items():
            if symbol in prices:
                position.update_pnl(prices[symbol])

        equity = self.get_equity()
        self.equity_history.append((timestamp, equity))
```

**Execution Simulator:**

```python
# backtest/execution_sim.py
@dataclass
class ExecutionConfig:
    slippage_bps: float = 10.0  # 0.1%
    maker_fee: float = 0.001  # 0.1%
    taker_fee: float = 0.001  # 0.1%

class ExecutionSimulator:
    def __init__(self, config: ExecutionConfig):
        self.config = config

    def simulate_market_order(
        self,
        side: str,
        quantity: float,
        next_candle: Candle,
        strategy_name: str
    ) -> Trade:
        """
        Simulate market order fill at next candle's open
        """
        # Fill price includes slippage
        slippage_factor = self.config.slippage_bps / 10000.0

        if side == 'buy':
            fill_price = next_candle.open * (1 + slippage_factor)
        else:
            fill_price = next_candle.open * (1 - slippage_factor)

        # Fee (use taker fee for market orders)
        notional = fill_price * quantity
        fee = notional * self.config.taker_fee

        return Trade(
            timestamp=next_candle.timestamp,
            symbol='',  # Will be set by caller
            side=side,
            quantity=quantity,
            price=float(fill_price),
            fee=float(fee),
            strategy_name=strategy_name,
            metadata={'slippage_bps': self.config.slippage_bps}
        )
```

**Backtest Engine:**

```python
# backtest/engine.py
class BacktestEngine:
    def __init__(
        self,
        data_provider: DataProvider,
        strategy: Strategy,
        indicator_manager: IndicatorManager,
        execution_sim: ExecutionSimulator,
        initial_capital: float,
        position_size_pct: float = 0.2
    ):
        self.data_provider = data_provider
        self.strategy = strategy
        self.indicator_manager = indicator_manager
        self.execution_sim = execution_sim
        self.position_size_pct = position_size_pct

        self.portfolio = Portfolio(initial_capital)

    def run(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime
    ) -> BacktestResult:
        """Run backtest for single symbol/strategy"""

        # Reset strategy state
        self.strategy.reset_state()

        # Get all historical data
        all_bars = self.data_provider.get_bars(symbol, timeframe, start_date, end_date)

        # Determine warmup period (max indicator period)
        warmup_bars = max(
            self.strategy.config.get('ma_slow', 50),
            self.strategy.config.get('bb_period', 20)
        ) + 50  # Extra buffer

        print(f"Running backtest: {symbol} on {timeframe} from {start_date} to {end_date}")
        print(f"Total bars: {len(all_bars.df)}, Warmup: {warmup_bars}")

        # Event loop
        for i in range(warmup_bars, len(all_bars.df) - 1):  # -1 to have "next" candle
            current_timestamp = all_bars.df.index[i]

            # Get historical window (up to current bar)
            historical_bars = BarSeries(all_bars.df.iloc[:i+1])

            # Calculate indicators
            indicators = self.indicator_manager.calculate_all(historical_bars)

            # Get strategy signal
            target_position = self.strategy.on_new_candle(
                symbol,
                timeframe,
                historical_bars,
                indicators
            )

            # Determine trade needed
            current_position = self.portfolio.get_position(symbol)
            current_exposure = 0.0
            if current_position:
                position_value = abs(current_position.quantity * current_position.avg_entry_price)
                current_exposure = position_value / self.portfolio.get_equity()
                if current_position.quantity < 0:
                    current_exposure *= -1

            target_exposure = target_position.target_exposure

            # Calculate trade size
            if abs(target_exposure - current_exposure) > 0.01:  # Threshold to avoid tiny trades
                trade = self._calculate_trade(
                    symbol,
                    current_exposure,
                    target_exposure,
                    current_position,
                    all_bars.df.iloc[i+1],  # Next candle for fill
                    target_position.strategy_name
                )

                if trade:
                    trade.symbol = symbol
                    self.portfolio.execute_trade(trade)

            # Mark to market
            current_price = float(all_bars.df.iloc[i]['close'])
            self.portfolio.mark_to_market(current_timestamp, {symbol: current_price})

        # Calculate metrics
        result = self._generate_result(symbol, timeframe, start_date, end_date)
        return result

    def _calculate_trade(
        self,
        symbol: str,
        current_exposure: float,
        target_exposure: float,
        current_position: Optional[Position],
        next_candle: Candle,
        strategy_name: str
    ) -> Optional[Trade]:
        """Determine trade needed to move from current to target exposure"""

        equity = self.portfolio.get_equity()

        # Calculate target position size in dollars
        target_notional = equity * abs(target_exposure) * self.position_size_pct
        current_notional = 0.0
        if current_position:
            current_notional = abs(current_position.quantity * current_position.avg_entry_price)

        # Determine action
        if target_exposure == 0 and current_position:
            # Close position
            side = 'sell' if current_position.quantity > 0 else 'buy'
            quantity = abs(current_position.quantity)

        elif target_exposure > 0 and (not current_position or current_position.quantity <= 0):
            # Enter or flip to long
            if current_position and current_position.quantity < 0:
                # First close short
                quantity = abs(current_position.quantity)
                side = 'buy'
            else:
                # Enter long
                quantity = target_notional / float(next_candle.open)
                side = 'buy'

        elif target_exposure < 0 and (not current_position or current_position.quantity >= 0):
            # Enter or flip to short
            if current_position and current_position.quantity > 0:
                # First close long
                quantity = current_position.quantity
                side = 'sell'
            else:
                # Enter short
                quantity = target_notional / float(next_candle.open)
                side = 'sell'

        else:
            # No trade needed
            return None

        # Simulate fill
        return self.execution_sim.simulate_market_order(
            side, quantity, next_candle, strategy_name
        )

    def _generate_result(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime
    ) -> 'BacktestResult':
        """Generate backtest result with metrics"""

        from backtest.metrics import calculate_metrics

        metrics = calculate_metrics(
            self.portfolio.equity_history,
            self.portfolio.trades,
            self.portfolio.initial_capital
        )

        return BacktestResult(
            symbol=symbol,
            timeframe=timeframe,
            strategy_name=self.strategy.name,
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.portfolio.initial_capital,
            final_equity=self.portfolio.get_equity(),
            metrics=metrics,
            trades=self.portfolio.trades,
            equity_curve=self.portfolio.equity_history
        )

@dataclass
class BacktestResult:
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
```

**Performance Metrics:**

```python
# backtest/metrics.py
@dataclass
class PerformanceMetrics:
    # Returns
    total_return: float
    cagr: float

    # Risk metrics
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float

    # Drawdown
    max_drawdown: float
    max_drawdown_duration: int  # days

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

def calculate_metrics(
    equity_curve: List[Tuple[datetime, float]],
    trades: List[Trade],
    initial_capital: float
) -> PerformanceMetrics:
    """Calculate all performance metrics"""

    if not equity_curve:
        return PerformanceMetrics(
            total_return=0, cagr=0, volatility=0, sharpe_ratio=0,
            sortino_ratio=0, max_drawdown=0, max_drawdown_duration=0,
            total_trades=0, winning_trades=0, losing_trades=0,
            win_rate=0, avg_win=0, avg_loss=0, profit_factor=0,
            avg_time_in_market=0
        )

    # Convert to DataFrame
    df = pd.DataFrame(equity_curve, columns=['timestamp', 'equity'])
    df['returns'] = df['equity'].pct_change()

    # Returns
    total_return = (df['equity'].iloc[-1] / initial_capital) - 1
    days = (df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).days
    years = days / 365.25
    cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

    # Volatility (annualized)
    volatility = df['returns'].std() * np.sqrt(252)  # Assuming daily-ish frequency

    # Sharpe (assuming 0% risk-free rate for crypto)
    sharpe_ratio = cagr / volatility if volatility > 0 else 0

    # Sortino (downside deviation)
    downside_returns = df['returns'][df['returns'] < 0]
    downside_std = downside_returns.std() * np.sqrt(252)
    sortino_ratio = cagr / downside_std if downside_std > 0 else 0

    # Drawdown
    cummax = df['equity'].cummax()
    drawdown = (df['equity'] - cummax) / cummax
    max_drawdown = abs(drawdown.min())

    # Drawdown duration
    is_underwater = drawdown < 0
    underwater_periods = is_underwater.astype(int).groupby((~is_underwater).cumsum()).sum()
    max_drawdown_duration = int(underwater_periods.max()) if len(underwater_periods) > 0 else 0

    # Trade statistics
    winning_trades_list = [t for t in trades if t.realized_pnl > 0]
    losing_trades_list = [t for t in trades if t.realized_pnl < 0]

    total_trades = len(trades)
    winning_trades = len(winning_trades_list)
    losing_trades = len(losing_trades_list)
    win_rate = winning_trades / total_trades if total_trades > 0 else 0

    avg_win = np.mean([t.realized_pnl for t in winning_trades_list]) if winning_trades_list else 0
    avg_loss = np.mean([t.realized_pnl for t in losing_trades_list]) if losing_trades_list else 0

    gross_profit = sum(t.realized_pnl for t in winning_trades_list)
    gross_loss = abs(sum(t.realized_pnl for t in losing_trades_list))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

    # Time in market (rough estimate)
    # TODO: More accurate calculation based on position tracking
    avg_time_in_market = 0.5  # Placeholder

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
```

---

## Testing Strategy

### Unit Tests
- [ ] Data models validation
- [ ] Each indicator calculation (compare with known values)
- [ ] Strategy logic (specific scenarios)
- [ ] Portfolio accounting
- [ ] Metrics calculations

### Integration Tests
- [ ] End-to-end backtest with mock data
- [ ] Data fetching and storage
- [ ] Multi-symbol backtests

### Validation Tests
- [ ] No look-ahead bias verification
- [ ] Fee/slippage application correctness
- [ ] Position sizing logic
- [ ] Trade reconciliation

---

## Deployment Plan

### Phase 1: MVP (Weeks 1-2)
- Set up infrastructure (DB, Docker)
- Implement data layer
- Implement indicators
- Implement strategies
- Implement backtesting engine
- Run initial backtests
- **Deliverable**: Backtest results showing if strategies have edge

### Phase 2: Risk & Portfolio (Weeks 3-4)
- Implement signal orchestrator
- Implement risk management
- Advanced position sizing
- Walk-forward testing
- **Deliverable**: Robust combined strategy with risk controls

### Phase 3: Paper Trading (Weeks 5-6)
- Implement execution layer
- Binance testnet integration
- Real-time data streaming
- Monitoring & logging
- **Deliverable**: Bot running on testnet

### Phase 4: Live (Weeks 7+)
- Small capital deployment
- Real-time monitoring
- Performance tracking
- Gradual capital increase
- **Deliverable**: Profitable live trading system

---

## Current Status: MVP Fully Implemented and Tested ✅

**Next Steps:**
1. Create directory structure
2. Set up PostgreSQL + TimescaleDB
3. Implement data layer
4. Download historical data
5. Implement indicators
6. Implement strategies
7. Run first backtest

**Notes:**
- All decisions documented
- Ready to start implementation
- Code will be structured for easy extension
- Testing strategy in place
- Clear path from MVP to production
