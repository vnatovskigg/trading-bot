-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Create OHLCV table
CREATE TABLE IF NOT EXISTS ohlcv (
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
SELECT create_hypertable('ohlcv', 'time', if_not_exists => TRUE);

-- Create indexes for fast queries
CREATE INDEX IF NOT EXISTS idx_symbol_timeframe ON ohlcv (symbol, timeframe, time DESC);
CREATE INDEX IF NOT EXISTS idx_symbol_time ON ohlcv (symbol, time DESC);

-- Enable compression for older data (optional, saves space)
ALTER TABLE ohlcv SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol,timeframe'
);

-- Add compression policy: compress data older than 7 days
SELECT add_compression_policy('ohlcv', INTERVAL '7 days', if_not_exists => TRUE);

-- Grant permissions
GRANT ALL PRIVILEGES ON TABLE ohlcv TO trader;
