-- ===========================
-- Daily Market Insights (Postgres schema)
-- Safe to paste into Railway's SQL console
-- ===========================

-- Extensions (optional but handy)
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- 1) Instruments master
CREATE TABLE IF NOT EXISTS instruments (
  symbol      TEXT PRIMARY KEY,              -- e.g., 'AAPL', 'MSFT', 'SPY', 'BTC-USD'
  name        TEXT,
  created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 2) Raw daily prices (normalized fields + raw JSON)
CREATE TABLE IF NOT EXISTS daily_prices (
  trade_date  DATE  NOT NULL,
  symbol      TEXT  NOT NULL REFERENCES instruments(symbol) ON DELETE CASCADE,
  open        NUMERIC,
  high        NUMERIC,
  low         NUMERIC,
  close       NUMERIC,
  adj_close   NUMERIC,
  volume      BIGINT,
  source      TEXT  NOT NULL DEFAULT 'alphavantage',
  raw_json    JSONB,                         -- keep original payload for audit/debug
  created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  PRIMARY KEY (trade_date, symbol)
);

-- Helpful indexes for queries and time-series lookups
CREATE INDEX IF NOT EXISTS idx_daily_prices_symbol_date
  ON daily_prices (symbol, trade_date DESC);
CREATE INDEX IF NOT EXISTS idx_daily_prices_date
  ON daily_prices (trade_date);
CREATE INDEX IF NOT EXISTS idx_daily_prices_rawjson_gin
  ON daily_prices USING GIN (raw_json);

-- 3) Derived daily metrics (what we compute locally)
CREATE TABLE IF NOT EXISTS daily_metrics (
  trade_date  DATE  NOT NULL,
  symbol      TEXT  NOT NULL REFERENCES instruments(symbol) ON DELETE CASCADE,
  return_pct  NUMERIC,   -- day-over-day return (%)
  ma7         NUMERIC,   -- 7-day moving average (close)
  ma30        NUMERIC,   -- 30-day moving average (close)
  rsi14       NUMERIC,   -- 14-day RSI
  vol7        NUMERIC,   -- 7-day stddev of returns (volatility proxy)
  high20      NUMERIC,   -- 20-day rolling high
  low20       NUMERIC,   -- 20-day rolling low
  created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  PRIMARY KEY (trade_date, symbol)
);

CREATE INDEX IF NOT EXISTS idx_daily_metrics_symbol_date
  ON daily_metrics (symbol, trade_date DESC);
CREATE INDEX IF NOT EXISTS idx_daily_metrics_date
  ON daily_metrics (trade_date);

-- 4) LLM outputs (daily analyst summary + 3 recs)
CREATE TABLE IF NOT EXISTS daily_recommendations (
  id            BIGSERIAL PRIMARY KEY,
  for_date      DATE NOT NULL,             -- date the insight refers to (usually yesterday)
  scope         TEXT NOT NULL,             -- 'portfolio' or a specific symbol
  summary       TEXT,
  recommendations JSONB,                   -- e.g. [{"title": "...", "action": "..."}]
  model         TEXT,                      -- e.g., "gpt-4o-mini"
  raw_prompt    TEXT,                      -- stored for reproducibility
  created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  UNIQUE (for_date, scope)
);

CREATE INDEX IF NOT EXISTS idx_daily_recs_for_date
  ON daily_recommendations (for_date DESC);
CREATE INDEX IF NOT EXISTS idx_daily_recs_recommendations_gin
  ON daily_recommendations USING GIN (recommendations);

-- 5) (Optional) ETL run log for observability
CREATE TABLE IF NOT EXISTS etl_runs (
  id        BIGSERIAL PRIMARY KEY,
  run_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  status    TEXT NOT NULL CHECK (status IN ('success','partial','failure')),
  details   JSONB
);

-- ===========================
-- Optional seed: preload your watchlist symbols
-- Edit names as you like; safe to re-run due to ON CONFLICT DO NOTHING
-- ===========================
INSERT INTO instruments (symbol, name) VALUES
  ('AAPL', 'Apple Inc.'),
  ('MSFT', 'Microsoft Corp.'),
  ('SPY',  'SPDR S&P 500 ETF'),
  ('BTC-USD', 'Bitcoin / USD')
ON CONFLICT (symbol) DO NOTHING;
