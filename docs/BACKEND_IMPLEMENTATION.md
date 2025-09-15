# RailPulse Backend Implementation

## Table of Contents
1. [System Architecture](#system-architecture)
2. [Data Flow](#data-flow)
3. [Component Design](#component-design)
4. [Database Schema](#database-schema)
5. [API Endpoints](#api-endpoints)
6. [ETL Pipeline](#etl-pipeline)
7. [LLM Integration](#llm-integration)
8. [Error Handling & Resilience](#error-handling--resilience)
9. [Performance Optimizations](#performance-optimizations)
10. [Deployment Strategy](#deployment-strategy)
11. [Monitoring & Logging](#monitoring--logging)
12. [Security Considerations](#security-considerations)
13. [Future Enhancements](#future-enhancements)

## System Architecture

### High-Level Architecture

RailPulse is a FastAPI-based service that fetches financial market data, processes it, stores it in a PostgreSQL database, and provides API endpoints for a frontend dashboard. The system follows a modular design with clear separation of concerns:

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│  Alpha Vantage  │────▶│  RailPulse ETL  │────▶│  PostgreSQL DB  │
│     API         │     │    Pipeline     │     │                 │
│                 │     │                 │     │                 │
└─────────────────┘     └────────┬────────┘     └────────▲────────┘
                                 │                       │
                                 ▼                       │
                        ┌─────────────────┐     ┌────────┴────────┐
                        │                 │     │                 │
                        │   OpenAI API    │     │  FastAPI REST   │
                        │  (LLM Analysis) │     │    Endpoints    │
                        │                 │     │                 │
                        └─────────────────┘     └────────┬────────┘
                                                         │
                                                         ▼
                                                ┌─────────────────┐
                                                │                 │
                                                │   Next.js UI    │
                                                │   Dashboard     │
                                                │                 │
                                                └─────────────────┘
```

### Core Components

1. **ETL Pipeline**: Fetches, transforms, and loads market data
2. **Database Layer**: Stores normalized data and computed metrics
3. **LLM Integration**: Generates market insights using OpenAI
4. **API Layer**: Exposes data and insights via REST endpoints
5. **Scheduler**: Runs daily data collection via Railway cron jobs

### Technology Stack

- **Backend Framework**: FastAPI (Python 3.11+)
- **Database**: PostgreSQL (hosted on Railway)
- **HTTP Client**: httpx (async)
- **Database ORM**: SQLAlchemy Core
- **LLM Provider**: OpenAI
- **Deployment**: Railway
- **Scheduler**: Railway cron jobs
- **Frontend**: Next.js + Tailwind (separate service)

## Data Flow

### Daily ETL Process

1. **Trigger**: Railway cron job or manual admin endpoint
2. **Data Acquisition**:
   - Fetch intraday data from Alpha Vantage
   - Apply rate limiting (5 calls/minute, max 25/day)
   - Handle API errors with retry logic
3. **Data Transformation**:
   - Convert UTC timestamps to IST timezone
   - Aggregate intraday bars to daily OHLCV data
   - Calculate technical indicators and metrics
4. **Data Loading**:
   - Upsert to `daily_prices` table (idempotent)
   - Upsert to `daily_metrics` table (idempotent)
5. **LLM Processing**:
   - Prepare market data summary
   - Send to OpenAI for analysis
   - Store results in `daily_recommendations` table

### API Request Flow

1. **Client Request**: Next.js dashboard calls API endpoints
2. **API Processing**:
   - Validate request parameters
   - Query database for requested data
   - Format response according to API contract
3. **Client Rendering**: Dashboard displays data and insights

## Component Design

### Configuration Management (`app/config.py`)

- Environment-based configuration using Pydantic
- Timezone handling (Asia/Kolkata)
- API keys and credentials management
- Feature flags for development/testing

### Database Layer (`app/db.py`)

- Connection pooling with SQLAlchemy
- Parameterized queries for security
- Upsert helpers for idempotent operations
- Transaction management and error handling

### Alpha Vantage Client (`app/alpha_vantage.py`)

- Async HTTP client with timeout handling
- Rate limiting implementation (token bucket)
- Retry logic with exponential backoff
- Error classification and handling
- Data normalization and transformation

### ETL Orchestration (`app/etl.py`)

- Pipeline coordination and scheduling
- Parallel processing of multiple symbols
- Error handling and partial success tracking
- Metrics calculation and aggregation

### LLM Integration (`app/llm.py`)

- OpenAI client configuration
- Prompt template management
- Response parsing and validation
- Error handling and fallbacks

### API Routes (`app/routes.py`)

- REST endpoint definitions
- Request validation
- Response formatting
- Error handling and status codes

### Main Application (`app/main.py`)

- FastAPI application setup
- Middleware configuration (CORS, logging)
- Exception handlers
- Health checks and diagnostics

## Database Schema

### Tables

#### `instruments`
```sql
CREATE TABLE instruments (
    symbol TEXT PRIMARY KEY,
    name TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

#### `daily_prices`
```sql
CREATE TABLE daily_prices (
    trade_date DATE,
    symbol TEXT,
    open NUMERIC,
    high NUMERIC,
    low NUMERIC,
    close NUMERIC,
    adj_close NUMERIC,
    volume BIGINT,
    source TEXT DEFAULT 'alphavantage',
    raw_json JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY(trade_date, symbol)
);
```

#### `daily_metrics`
```sql
CREATE TABLE daily_metrics (
    trade_date DATE,
    symbol TEXT,
    return_pct NUMERIC,
    ma7 NUMERIC,
    ma30 NUMERIC,
    rsi14 NUMERIC,
    vol7 NUMERIC,
    high20 NUMERIC,
    low20 NUMERIC,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY(trade_date, symbol)
);
```

#### `daily_recommendations`
```sql
CREATE TABLE daily_recommendations (
    id BIGSERIAL PRIMARY KEY,
    for_date DATE,
    scope TEXT,
    summary TEXT,
    recommendations JSONB,
    model TEXT,
    raw_prompt TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(for_date, scope)
);
```

#### `etl_runs`
```sql
CREATE TABLE etl_runs (
    id BIGSERIAL PRIMARY KEY,
    run_at TIMESTAMPTZ DEFAULT NOW(),
    status TEXT CHECK (status IN ('success','partial','failure')),
    details JSONB
);
```

### Key Relationships

- `daily_prices` and `daily_metrics` share the same composite primary key (trade_date, symbol)
- `daily_recommendations` has a unique constraint on (for_date, scope)
- All tables include timestamps for auditing and debugging

## API Endpoints

### Health Check

```
GET /healthz
```

**Response:**
```json
{
  "status": "ok",
  "time": "2025-09-16T02:21:23+05:30"
}
```

### Admin Trigger

```
POST /admin/run-daily?secret=APP_SECRET
```

**Response:**
```json
{
  "status": "success",
  "symbols_processed": 5,
  "symbols_failed": 0,
  "run_id": 123
}
```

### Metrics Endpoint

```
GET /metrics?from=YYYY-MM-DD&to=YYYY-MM-DD&symbol=AAPL,MSFT
```

**Response:**
```json
{
  "range": {"from": "2025-08-16", "to": "2025-09-15"},
  "symbols": ["AAPL", "MSFT", "SPY", "BTC-USD"],
  "prices": [
    {"trade_date": "2025-09-12", "symbol": "AAPL", "open": 222.50, "high": 224.95, "low": 221.60, "close": 222.77, "adj_close": 222.77, "volume": 46388779}
  ],
  "metrics": [
    {"trade_date": "2025-09-12", "symbol": "AAPL", "return_pct": 0.84, "ma7": 224.9, "ma30": 219.2, "rsi14": 58.7, "vol7": 1.3, "high20": 229.1, "low20": 214.4}
  ]
}
```

### Recommendations Endpoint

```
GET /recommendations?date=YYYY-MM-DD
```

**Response:**
```json
{
  "for_date": "2025-09-12",
  "scope": "portfolio",
  "summary": "Major indices closed higher with tech leadership; BTC softened intraday.",
  "recommendations": [
    {"title": "Lean Tech Overweight", "action": "Increase AAPL/MSFT weighting modestly if RSI < 65 and price above MA30."},
    {"title": "Volatility Buffer", "action": "Set tighter stops on BTC-USD given elevated 7D vol vs. equities."},
    {"title": "Trend Confirmation", "action": "Hold positions trending above MA30; reassess losers near 20D lows."}
  ],
  "model": "gpt-4o-mini"
}
```

## ETL Pipeline

### Pipeline Stages

1. **Initialization**:
   - Load configuration
   - Validate environment variables
   - Set up logging

2. **Data Acquisition**:
   - Check cache for recent data
   - Apply rate limiting
   - Fetch data from Alpha Vantage
   - Handle API errors and retries

3. **Data Transformation**:
   - Parse JSON responses
   - Convert timestamps to IST
   - Normalize data structure
   - Aggregate intraday to daily OHLCV

4. **Data Loading**:
   - Upsert to `daily_prices`
   - Calculate technical indicators
   - Upsert to `daily_metrics`

5. **LLM Processing**:
   - Prepare market data summary
   - Generate prompt for OpenAI
   - Parse and validate response
   - Store in `daily_recommendations`

6. **Finalization**:
   - Log completion status
   - Record ETL run details
   - Return summary statistics

### Scheduling

- **Daily Job**: Runs at 11:30 UTC (5:00 PM IST) on weekdays
- **Manual Trigger**: Available via admin endpoint with secret key
- **Idempotency**: Safe to run multiple times per day

## LLM Integration

### OpenAI Integration

- **Model**: gpt-4o-mini (configurable)
- **Temperature**: 0.2 (low for consistent outputs)
- **Max Tokens**: 500 (sufficient for summary + recommendations)

### Prompt Template

```
You are a market analyst. Based on the following performance data for [YYYY-MM-DD IST],
write a concise 3–4 sentence summary and exactly 3 actionable recommendations.
Be specific, avoid generic advice, and reference symbols when relevant.
Data JSON:
{KPIS_JSON}
```

### Response Structure

```json
{
  "summary": "Concise market summary for the day",
  "recommendations": [
    {
      "title": "Short title for recommendation",
      "action": "Specific actionable advice"
    },
    // Two more recommendations
  ]
}
```

## Error Handling & Resilience

### API Error Handling

- **Rate Limiting**: Exponential backoff with jitter
- **Quota Exceeded**: Fallback to cached data
- **Network Errors**: Retry with timeout handling
- **Malformed Responses**: Validation and logging

### Database Error Handling

- **Connection Issues**: Retry with exponential backoff
- **Transaction Management**: Automatic rollback on error
- **Constraint Violations**: Proper error messages
- **Connection Pooling**: Efficient resource usage

### Partial Success Handling

- **Symbol-Level Granularity**: Process each symbol independently
- **Failure Tracking**: Record failed symbols with error details
- **Partial Results**: Return successful data even if some symbols fail
- **Alerting**: Log critical failures for monitoring

## Performance Optimizations

### Parallel Processing

- **Async I/O**: Non-blocking HTTP and database operations
- **Task Concurrency**: Process multiple symbols in parallel
- **Connection Pooling**: Reuse database connections
- **Batch Operations**: Minimize database round trips

### Caching Strategy

- **Data Freshness**: Skip API calls for recently updated data
- **Intelligent Prioritization**: Fetch oldest data first
- **Response Caching**: Cache API responses for common queries
- **Memory Efficiency**: Avoid redundant data in memory

### Database Optimizations

- **Indexing**: Optimize for common query patterns
- **Batch Upserts**: Minimize database operations
- **Connection Pooling**: Efficient connection management
- **Query Optimization**: Use explain plans to optimize queries

## Deployment Strategy

### Railway Deployment

- **Service**: FastAPI Python application
- **Database**: PostgreSQL instance
- **Environment Variables**: Securely stored in Railway
- **Scaling**: Automatic scaling based on load
- **Monitoring**: Built-in logs and metrics

### CI/CD Pipeline

- **Source Control**: GitHub repository
- **Automatic Deployment**: On push to main branch
- **Environment Separation**: Development vs. Production
- **Rollback**: Easy version rollback if needed

### Scheduled Jobs

- **Cron Configuration**: Railway cron jobs
- **Schedule**: Weekdays at 11:30 UTC (5:00 PM IST)
- **Monitoring**: Job success/failure tracking
- **Manual Trigger**: Admin endpoint for on-demand runs

## Monitoring & Logging

### Logging Strategy

- **Structured Logging**: JSON format for machine parsing
- **Log Levels**: DEBUG, INFO, WARNING, ERROR
- **Context Enrichment**: Include request IDs, symbols, timestamps
- **Sensitive Data**: Mask API keys and credentials

### Key Metrics

- **API Success Rate**: Percentage of successful API calls
- **Database Performance**: Query execution times
- **ETL Pipeline**: Success rate, processing time
- **Rate Limit Usage**: API quota consumption
- **Error Rates**: By component and error type

### Alerting

- **Critical Failures**: Alert on pipeline failures
- **Rate Limit Warnings**: Alert when approaching limits
- **Database Issues**: Alert on connection problems
- **API Availability**: Monitor external API health

## Security Considerations

### API Security

- **Authentication**: Secret key for admin endpoints
- **Rate Limiting**: Prevent abuse of public endpoints
- **Input Validation**: Strict parameter validation
- **CORS Policy**: Restrict to authorized origins

### Data Security

- **Environment Variables**: Secure storage of credentials
- **API Keys**: Never logged or exposed in responses
- **Database Access**: Least privilege principle
- **Data Validation**: Prevent injection attacks

### Compliance

- **Data Retention**: Clear policies for historical data
- **Attribution**: Proper citation of data sources
- **Terms of Service**: Compliance with Alpha Vantage ToS
- **Rate Limits**: Respect API provider limits

## Future Enhancements

### Technical Improvements

- **Additional Data Sources**: Integrate alternative providers
- **Advanced Metrics**: Implement more technical indicators
- **Caching Layer**: Redis for improved performance
- **WebSockets**: Real-time updates for dashboard

### Feature Enhancements

- **User Authentication**: Multi-tenant support
- **Custom Watchlists**: User-defined symbol lists
- **Alerts**: Notification system for price movements
- **Historical Analysis**: Backtest trading strategies
- **Export Functionality**: CSV/Excel data export

### Integration Opportunities

- **Slack/Email Notifications**: Daily digest reports
- **Mobile App**: Native mobile experience
- **Trading Platforms**: Integration with brokers
- **Social Sharing**: Share insights and recommendations
