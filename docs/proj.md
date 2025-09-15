# Project: Daily Market Insights → PostgreSQL → LLM (Railway)

A free-only MVP that fetches market data daily (Alpha Vantage), stores normalized snapshots in Railway Postgres, runs an LLM summary + recommendations, and serves a minimal dashboard via a FastAPI API (Next.js dashboard in a separate service).

---

## 1) Objective
Build a fully hosted, automated pipeline on Railway that:
- Fetches **daily market performance data** for a small watchlist of symbols
- Stores **raw JSON + normalized** OHLCV and derived KPIs in Postgres
- Generates **LLM insights** for the previous day (summary + 3 actions)
- Exposes a **clean API** for a simple UI
- Runs **once per day** in IST with idempotent writes

**Scope locked**: Free-only data source (**Alpha Vantage**); keep symbols small to respect free limits.

---

## 2) Non‑Goals (for MVP)
- No paid analytics sources (e.g., Plausible hosted, Clearbit, Plaid, etc.)
- No web engagement KPIs (bounce rate, time on site) in v1
- No user auth / multi-tenancy; simple admin secret for manual trigger

---

## 3) Tech Stack
- **Backend**: FastAPI (Python 3.11+), httpx, SQLAlchemy Core, psycopg2
- **DB**: Railway PostgreSQL
- **LLM**: OpenAI (OpenAI key) — small prompts, structured output
- **Scheduler**: Railway cron (daily ~00:30 IST)
- **Frontend**: Next.js + Tailwind (separate service). Dashboard consumes the FastAPI API

---

## 4) Architecture (High-Level)
```
Alpha Vantage API  ──>  ETL (FastAPI service / daily job)
                         │
                         ├─ upsert → daily_prices (raw + normalized)
                         ├─ compute → daily_metrics (KPIs)
                         └─ query yesterday → LLM → daily_recommendations

FastAPI API  ──>  /metrics, /recommendations  ──>  Next.js Dashboard
```

**Key properties**: idempotent ETL; raw JSON persisted for audit; tight prompts; minimal dependencies.

---

## 5) Data Model (Created in DB)
- `instruments(symbol TEXT PK, name TEXT, created_at TIMESTAMPTZ)`
- `daily_prices(trade_date DATE, symbol TEXT, open NUMERIC, high NUMERIC, low NUMERIC, close NUMERIC, adj_close NUMERIC, volume BIGINT, source TEXT DEFAULT 'alphavantage', raw_json JSONB, created_at TIMESTAMPTZ, PRIMARY KEY(trade_date, symbol))`
- `daily_metrics(trade_date DATE, symbol TEXT, return_pct NUMERIC, ma7 NUMERIC, ma30 NUMERIC, rsi14 NUMERIC, vol7 NUMERIC, high20 NUMERIC, low20 NUMERIC, created_at TIMESTAMPTZ, PRIMARY KEY(trade_date, symbol))`
- `daily_recommendations(id BIGSERIAL PK, for_date DATE, scope TEXT, summary TEXT, recommendations JSONB, model TEXT, raw_prompt TEXT, created_at TIMESTAMPTZ, UNIQUE(for_date, scope))`
- `etl_runs(id BIGSERIAL PK, run_at TIMESTAMPTZ, status TEXT CHECK (status IN ('success','partial','failure')), details JSONB)`

**Notes**:
- Store `raw_json` from Alpha Vantage for each upserted row (traceability)
- All writes are **ON CONFLICT** upserts → safe to re-run daily job

---

## 6) KPIs & Calculations
Per symbol, per `trade_date` (computed locally from closes):
- `return_pct` = (close - prev_close) / prev_close * 100
- `ma7`, `ma30` = simple MAs of close
- `rsi14` = Wilder’s RSI (14 periods)
- `vol7` = stddev of daily return_pct over last 7 obs
- `high20`, `low20` = rolling 20-day high/low of close

Portfolio rollups (for LLM + UI):
- Equal-weighted daily return across symbols
- Top gainer/loser by `return_pct` (yesterday)
- 7D portfolio volatility proxy

---

## 7) API Contracts (FastAPI)
- `GET /healthz` → `{ status: "ok", time: <utc iso> }`
- `POST /admin/run-daily?secret=APP_SECRET` → manually runs full pipeline
- `GET /metrics?from=YYYY-MM-DD&to=YYYY-MM-DD&symbol=AAPL,MSFT` → returns prices + metrics arrays for UI
- `GET /recommendations?date=YYYY-MM-DD` → summary + 3 actions for the requested date (default latest)

**/metrics response shape (example)**
```json
{
  "range": {"from": "2025-08-16", "to": "2025-09-15"},
  "symbols": ["AAPL", "MSFT", "SPY", "BTC-USD"],
  "prices": [
    {"trade_date": "2025-09-12", "symbol": "AAPL", "open": 0, "high": 0, "low": 0, "close": 226.76, "adj_close": 226.76, "volume": 48249321}
  ],
  "metrics": [
    {"trade_date": "2025-09-12", "symbol": "AAPL", "return_pct": 0.84, "ma7": 224.9, "ma30": 219.2, "rsi14": 58.7, "vol7": 1.3, "high20": 229.1, "low20": 214.4}
  ]
}
```

**/recommendations response shape (example)**
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

---

## 8) LLM Prompt Template
```
You are a market analyst. Based on the following performance data for [YYYY-MM-DD IST],
write a concise 3–4 sentence summary and exactly 3 actionable recommendations.
Be specific, avoid generic advice, and reference symbols when relevant.
Data JSON:
{KPIS_JSON}
```

Persist result for the date as one row in `daily_recommendations` with `scope='portfolio'` and the `raw_prompt` used.

---

## 9) Env Vars (Railway)
- `DATABASE_URL` (Railway Postgres)
- `ALPHAVANTAGE_API_KEY`
- `OPENAI_API_KEY`
- `TIMEZONE=Asia/Kolkata`
- `TICKERS=AAPL,MSFT,SPY,BTC-USD`
- `APP_SECRET=<random>`

---

## 10) Implementation Plan (Phases)
**P0 – Repo bootstrap**
- Create FastAPI skeleton with `app/` and `scripts/` folders
- Add `requirements.txt`, `.python-version`, and base logging setup

**P1 – DB connectivity & migration**
- Verify `DATABASE_URL` from Railway locally and in deploy
- Run DDL (already created) to ensure tables exist
- Seed `instruments` for default tickers

**P2 – ETL (Alpha Vantage)**
- Client with httpx + tenacity (timeout/backoff)
- Fetch compact series for each symbol with 60D lookback
- Normalize and upsert into `daily_prices`
- Compute KPIs and upsert into `daily_metrics`
- Return run summary

**P3 – LLM insights**
- Query yesterday KPIs, compute portfolio rollups
- Build compact JSON and prompt OpenAI
- Upsert into `daily_recommendations` (scope='portfolio')

**P4 – Routes**
- `/healthz`, `/admin/run-daily`, `/metrics`, `/recommendations`
- Input validation, pagination/window defaults, error handling

**P5 – Scheduler**
- `scripts/daily_job.py` runs P2 → P3
- Configure Railway cron: daily ~00:30 IST

**P6 – Frontend (separate service)**
- Next.js + Tailwind dashboard consuming API
- Cards + Chart + Table + Analyst box

**P7 – Backfill & Demo**
- First run fetches last 10–30 days so the UI has data
- Verify cron and manual trigger both work

---

## 11) Operational Concerns
- **Idempotency**: Upserts + unique constraints prevent duplicates
- **Rate limits**: Keep watchlist to ~3–5 symbols; use backoff on 429/Note
- **Reliability**: Partial failures recorded in `etl_runs`
- **Logging**: One structured log per stage (fetch, upsert, derive, llm)
- **Secrets**: Only via Railway env vars; never commit

---

## 12) Acceptance Criteria (Demo Ready)
- Deployed FastAPI on Railway; `/healthz` responds
- Daily cron populates `daily_prices`, `daily_metrics`, and `daily_recommendations`
- `/metrics` returns values for last 30 days; `/recommendations` returns latest day
- README documents env setup, deploy, and updating `TICKERS`

---

## 13) Future Enhancements
- Add Stripe for real revenue KPIs (paid processing, free API calls)
- Swap/augment with Plausible or GA4 if paid access is provided
- Slack/Email daily digest
- Re-generate insights for any past date
- Multi-tenant watchlists per user

