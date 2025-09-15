# RailPulse Market Data API

A production-ready FastAPI service that ingests present-day intraday data from Alpha Vantage, aggregates it to daily OHLCV per symbol, stores it in Postgres, computes KPIs, and generates LLM-powered market summaries.

## Features

- **Intraday Data Ingestion**: Fetches TIME_SERIES_INTRADAY from Alpha Vantage with configurable intervals
- **Daily Aggregation**: Converts intraday bars to daily OHLCV data (Asia/Kolkata timezone)
- **Database Storage**: Upserts to existing Postgres tables with idempotent operations
- **KPI Calculation**: Computes daily return percentages with hooks for future metrics
- **LLM Analysis**: Generates market summaries and recommendations using OpenAI
- **REST API**: Clean endpoints for Next.js dashboard integration
- **Rate Limiting**: Respects Alpha Vantage free tier limits with polite backoff
- **Observability**: Structured logging and error handling

## Project Structure

```
app/
├── __init__.py
├── config.py          # Environment configuration
├── db.py              # Database engine and helpers
├── alpha_vantage.py   # Alpha Vantage API client
├── etl.py             # ETL pipeline orchestration
├── metrics.py         # KPI calculations
├── llm.py             # OpenAI integration
├── routes.py          # FastAPI endpoints
└── main.py            # Application entrypoint

scripts/
├── __init__.py
└── run_today.py       # CLI runner for Railway scheduler

requirements.txt       # Python dependencies
README.md             # This file
```

## Environment Variables

Create a `.env` file or set the following environment variables:

```bash
# Database (Railway Postgres)
DATABASE_URL=postgresql://user:password@host:port/database

# Alpha Vantage API
ALPHAVANTAGE_API_KEY=your_alpha_vantage_key
ALPHAVANTAGE_INTERVAL=5min  # 1min,5min,15min,30min,60min

# OpenAI API
OPENAI_API_KEY=your_openai_key

# Stock Symbols (comma-separated)
TICKERS=AAPL,MSFT,SPY,BTC-USD

# Timezone
TIMEZONE=Asia/Kolkata

# Security
APP_SECRET=your_secure_secret_key
```

## Database Schema

The service expects these existing Postgres tables:

```sql
-- Instruments table
CREATE TABLE instruments (
    symbol VARCHAR PRIMARY KEY,
    -- other columns...
);

-- Daily prices table
CREATE TABLE daily_prices (
    trade_date DATE,
    symbol VARCHAR,
    open DECIMAL,
    high DECIMAL,
    low DECIMAL,
    close DECIMAL,
    adj_close DECIMAL,
    volume BIGINT,
    source VARCHAR,
    raw_json JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (trade_date, symbol)
);

-- Daily metrics table
CREATE TABLE daily_metrics (
    trade_date DATE,
    symbol VARCHAR,
    return_pct DECIMAL,
    ma7 DECIMAL,
    ma30 DECIMAL,
    rsi14 DECIMAL,
    vol7 DECIMAL,
    high20 DECIMAL,
    low20 DECIMAL,
    created_at TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (trade_date, symbol)
);

-- Daily recommendations table
CREATE TABLE daily_recommendations (
    for_date DATE,
    scope VARCHAR,
    summary TEXT,
    recommendations JSONB,
    model VARCHAR,
    raw_prompt TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE (for_date, scope)
);
```

## Installation

1. **Clone and install dependencies:**
```bash
cd RailPulse
pip install -r requirements.txt
```

2. **Set environment variables:**
```bash
# Copy and edit the environment file
cp .env.example .env
# Edit .env with your actual values
```

3. **Verify database connection:**
```bash
# Test that your DATABASE_URL is correct
python -c "from app.db import execute_query; execute_query('SELECT 1')"
```

## Usage

### Running the FastAPI Server

```bash
# Development mode
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Production mode
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Running the ETL Pipeline

```bash
# CLI runner (for Railway scheduler or manual execution)
python scripts/run_today.py

# Or via API endpoint (requires APP_SECRET)
curl -X POST "http://localhost:8000/admin/run-today?app_secret=your_secret"
```

## API Endpoints

### Public Endpoints

- `GET /` - API information
- `GET /healthz` - Health check
- `GET /metrics?date=YYYY-MM-DD&symbol=AAPL,MSFT` - Get daily prices and metrics
- `GET /recommendations?date=YYYY-MM-DD` - Get LLM recommendations

### Admin Endpoints (require APP_SECRET)

- `POST /admin/run-today` - Trigger ETL pipeline

### Example API Calls

```bash
# Health check
curl http://localhost:8000/healthz

# Get today's metrics for all symbols
curl http://localhost:8000/metrics

# Get specific date and symbols
curl "http://localhost:8000/metrics?date=2024-01-15&symbol=AAPL,MSFT"

# Get recommendations
curl http://localhost:8000/recommendations

# Trigger ETL (admin)
curl -X POST "http://localhost:8000/admin/run-today" \
  -H "x-app-secret: your_secret"
```

## Data Flow

1. **Fetch**: Alpha Vantage TIME_SERIES_INTRADAY for each symbol
2. **Filter**: Only bars from today (Asia/Kolkata timezone)
3. **Aggregate**: Convert intraday bars to daily OHLCV
4. **Store**: Upsert to `daily_prices` and `daily_metrics` tables
5. **Analyze**: Generate portfolio summary and LLM recommendations
6. **Serve**: Expose data via REST API endpoints

## Rate Limiting

- Alpha Vantage free tier: 5 calls per minute
- Service waits 12 seconds between symbol requests
- Implements exponential backoff for API errors
- Graceful handling of rate limit responses

## Deployment

### Railway Deployment

1. **Connect your repository to Railway**
2. **Set environment variables in Railway dashboard**
3. **Deploy the service**
4. **Schedule the ETL pipeline:**
```bash
# Add to Railway cron jobs
0 16 * * 1-5 cd /app && python scripts/run_today.py
```

### Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Monitoring

- Structured JSON logging for all operations
- Health check endpoint for uptime monitoring
- Database connection pooling with pre-ping
- Graceful error handling with detailed messages

## Next.js Integration

The API is designed for seamless integration with Next.js dashboards:

```javascript
// Fetch today's metrics
const response = await fetch('/api/metrics');
const data = await response.json();

// Fetch recommendations
const recommendations = await fetch('/api/recommendations');
const analysis = await recommendations.json();
```

## Troubleshooting

### Common Issues

1. **Database Connection Errors**
   - Verify DATABASE_URL format and credentials
   - Check network connectivity to Railway Postgres

2. **Alpha Vantage API Errors**
   - Verify ALPHAVANTAGE_API_KEY is valid
   - Check rate limiting (5 calls per minute)
   - Ensure symbols are valid (use Yahoo Finance format for crypto)

3. **No Data for Today**
   - Markets may be closed (weekends/holidays)
   - Check timezone configuration (Asia/Kolkata)
   - Verify symbol format matches Alpha Vantage requirements

4. **OpenAI API Errors**
   - Verify OPENAI_API_KEY is valid and has credits
   - Check API rate limits and quotas

### Logs

Check application logs for detailed error information:
```bash
# View logs in Railway dashboard or local output
tail -f app.log
```

## License

MIT License - see LICENSE file for details.
