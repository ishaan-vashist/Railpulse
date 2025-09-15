"""Alpha Vantage API client for fetching market data."""
import asyncio
import json
import logging
import random
from datetime import datetime, date
from typing import List, Dict, Any, Optional
import httpx
import pytz
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from .config import settings, IST, today_ist_date

logger = logging.getLogger(__name__)

# Token-bucket-ish limiter for per-minute throttling
# We'll allow up to settings.alphavantage_calls_per_minute per 60s window.
_last_minute_reset = 0.0
_minute_count = 0
_minute_lock = asyncio.Lock()

async def throttle_minute():
    """Ensure we don't exceed calls/minute. Adds small jitter to be safe."""
    global _last_minute_reset, _minute_count
    async with _minute_lock:
        now = asyncio.get_event_loop().time()
        if now - _last_minute_reset >= 60.0:
            _last_minute_reset = now
            _minute_count = 0
        if _minute_count >= settings.alphavantage_calls_per_minute:
            # sleep until next minute window
            sleep_for = 60.0 - (now - _last_minute_reset)
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)
            _last_minute_reset = asyncio.get_event_loop().time()
            _minute_count = 0
        _minute_count += 1
    # Add small jitter (0.4–1.2s) so we don't look perfectly periodic
    await asyncio.sleep(0.4 + random.random() * 0.8)

def _is_rate_limited_json(data: Dict[str, Any]) -> bool:
    note = (data.get("Note") or data.get("Information") or "")
    if not isinstance(note, str):
        return False
    # Only treat as retryable rate limits - per-minute limits
    return ("5 calls per minute" in note) or ("Standard API call frequency" in note) or ("Thank you for using Alpha Vantage" in note)

def _is_daily_limit_info(data: Dict[str, Any]) -> bool:
    """Check if response is just informational about daily limits (not an actual limit hit)"""
    info = data.get("Information", "")
    if not isinstance(info, str):
        return False
    # This is just informational about your daily limit, not an actual limit hit
    return ("standard API rate limit is 25 requests per day" in info) and ("detected your API key" in info)

async def _handle_rate_limit_backoff(attempt: int):
    # Exponential-ish backoff with cap, plus jitter
    base = min(60, 5 * (2 ** (attempt - 1)))  # 5,10,20,40,60...
    jitter = random.uniform(0.2, 0.8) * base * 0.1
    await asyncio.sleep(base + jitter)


class Bar:
    """Represents a single OHLCV bar."""
    
    def __init__(self, timestamp: datetime, open_price: float, high: float, 
                 low: float, close: float, volume: int):
        self.timestamp = timestamp
        self.open = open_price
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume
        }


class AlphaVantageError(Exception):
    """Custom exception for Alpha Vantage API errors."""
    pass


class AlphaVantageClient:
    """Client for Alpha Vantage API with async requests and error handling."""
    
    def __init__(self):
        self.api_key = settings.alphavantage_api_key
        self.base_url = "https://www.alphavantage.co/query"
        self.timeout = httpx.Timeout(30.0)  # single default in seconds
    
    async def _handle_api_response(self, data: Dict[str, Any], symbol: str, expected_key: str, attempt: int, max_attempts: int) -> Optional[Dict[str, Any]]:
        """Common handler for API responses to reduce code duplication."""
        if "Error Message" in data:
            raise AlphaVantageError(f"Alpha Vantage error: {data['Error Message']}")

        if _is_rate_limited_json(data):
            logger.warning(f"Alpha Vantage rate limit note on {symbol} (attempt {attempt}): {data.get('Note') or data.get('Information')}")
            if attempt < max_attempts:
                await _handle_rate_limit_backoff(attempt)
                return None  # Signal retry needed
            raise AlphaVantageError("Rate limit persisted after retries")

        # Handle Information-only responses
        if expected_key not in data:
            if "Information" in data:
                info_msg = data["Information"]
                # Check if this is just daily limit info (not an actual limit hit)
                if _is_daily_limit_info(data):
                    logger.info(f"Alpha Vantage daily limit info for {symbol}: {info_msg}")
                    # Don't retry - this is just informational, likely means daily limit reached
                    raise AlphaVantageError(f"Daily limit likely reached: {info_msg}")
                else:
                    logger.warning(f"Alpha Vantage Information response for {symbol}: {info_msg}")
                    # Only retry if it's not the daily limit info message
                    if attempt < max_attempts:
                        await _handle_rate_limit_backoff(attempt)
                        return None  # Signal retry needed
                    raise AlphaVantageError(f"Information response persisted after retries: {info_msg}")
            else:
                raise AlphaVantageError(f"Unexpected response for {symbol}: {list(data.keys())}")

        return data

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8), retry=retry_if_exception_type(httpx.HTTPError))
    async def fetch_daily_data(self, symbol: str) -> Dict[str, Any]:
        """Fetch daily adjusted time series data for a symbol."""
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": symbol,
            "apikey": self.api_key,
            "outputsize": "compact",
            "datatype": "json"
        }
        max_attempts = 4
        for attempt in range(1, max_attempts + 1):
            await throttle_minute()
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                logger.info(f"Fetching data for {symbol}")
                resp = await client.get(self.base_url, params=params)
                resp.raise_for_status()
                data = resp.json()

            # Process response with common handler
            processed_data = await self._handle_api_response(
                data, symbol, "Time Series (Daily)", attempt, max_attempts
            )
            if processed_data is None:
                continue  # Retry needed
            return processed_data
    
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8), retry=retry_if_exception_type(httpx.HTTPError))
    async def fetch_intraday_data(self, symbol: str, interval: str = "5min") -> Dict[str, Any]:
        """Fetch intraday time series data for a symbol."""
        params = {
            "function": "TIME_SERIES_INTRADAY",
            "symbol": symbol,
            "interval": interval,
            "apikey": self.api_key,
            "outputsize": "compact",  # Last 100 data points
            "datatype": "json"
        }
        max_attempts = 4
        time_series_key = f"Time Series ({interval})"
        
        for attempt in range(1, max_attempts + 1):
            await throttle_minute()
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                logger.info(f"Fetching intraday data for {symbol} with {interval} interval")
                resp = await client.get(self.base_url, params=params)
                resp.raise_for_status()
                data = resp.json()

            # Process response with common handler
            processed_data = await self._handle_api_response(
                data, symbol, time_series_key, attempt, max_attempts
            )
            if processed_data is None:
                continue  # Retry needed
            return processed_data


def normalize_daily_data(symbol: str, raw_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Normalize Alpha Vantage daily data into rows for database insertion."""
    time_series = raw_data.get("Time Series (Daily)", {})
    
    rows = []
    for date_str, values in time_series.items():
        try:
            trade_date = datetime.strptime(date_str, "%Y-%m-%d").date()
            
            # Handle TIME_SERIES_DAILY format (free tier)
            close_price = float(values["4. close"])
            row = {
                "trade_date": trade_date,
                "symbol": symbol,
                "open": float(values["1. open"]),
                "high": float(values["2. high"]),
                "low": float(values["3. low"]),
                "close": close_price,
                "adj_close": close_price,  # No adjustment data in free tier, use close price
                "volume": int(values["5. volume"]),  # Standard position for TIME_SERIES_DAILY
                "source": "alphavantage",
                "raw_json": json.dumps(values),
                "created_at": datetime.utcnow()
            }
            rows.append(row)
        except (ValueError, KeyError) as e:
            logger.warning(f"Failed to parse data for {symbol} on {date_str}: {e}")
            continue
    
    # Sort by date ascending
    rows.sort(key=lambda x: x["trade_date"])
    logger.info(f"Normalized {len(rows)} rows for {symbol}")
    return rows


def normalize_intraday_data(symbol: str, raw_data: Dict[str, Any], interval: str) -> List[Bar]:
    """Convert Alpha Vantage intraday data to Bar objects."""
    time_series_key = f"Time Series ({interval})"
    time_series = raw_data.get(time_series_key, {})
    
    bars = []
    today_date = today_ist_date()
    
    for timestamp_str, ohlcv in time_series.items():
        try:
            # Parse timestamp (assumes UTC from Alpha Vantage)
            timestamp_utc = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
            timestamp_utc = timestamp_utc.replace(tzinfo=pytz.UTC)
            
            # Convert to IST
            timestamp_ist = timestamp_utc.astimezone(IST)
            
            bar = Bar(
                timestamp=timestamp_ist,
                open_price=float(ohlcv["1. open"]),
                high=float(ohlcv["2. high"]),
                low=float(ohlcv["3. low"]),
                close=float(ohlcv["4. close"]),
                volume=int(ohlcv["5. volume"])
            )
            
            bars.append(bar)
            
        except (ValueError, KeyError) as e:
            logger.warning(f"Failed to parse bar data for {symbol} at {timestamp_str}: {e}")
            continue
    
    # Sort bars by timestamp ascending
    bars.sort(key=lambda x: x.timestamp)
    
    logger.info(f"Normalized {len(bars)} intraday bars for {symbol}")
    return bars


def aggregate_intraday_to_daily(bars: List[Bar]) -> Optional[Dict[str, Any]]:
    """Aggregate intraday bars into daily OHLCV data."""
    if not bars:
        return None
    
    # Group bars by date
    bars_by_date = {}
    for bar in bars:
        date_ist = bar.timestamp.date()
        if date_ist not in bars_by_date:
            bars_by_date[date_ist] = []
        bars_by_date[date_ist].append(bar)
    
    # Process each date
    daily_data = []
    for date_ist, date_bars in bars_by_date.items():
        if not date_bars:
            continue
            
        # Sort bars by timestamp
        date_bars.sort(key=lambda x: x.timestamp)
        
        # Aggregate OHLCV
        open_price = date_bars[0].open  # First bar's open
        high_price = max(bar.high for bar in date_bars)  # Maximum high
        low_price = min(bar.low for bar in date_bars)  # Minimum low
        close_price = date_bars[-1].close  # Last bar's close
        total_volume = sum(bar.volume for bar in date_bars)  # Sum of volumes
        
        daily_data.append({
            "trade_date": date_ist,
            "open": open_price,
            "high": high_price,
            "low": low_price,
            "close": close_price,
            "adj_close": close_price,  # Assuming no adjustment for intraday
            "volume": total_volume,
            "source": "alphavantage_intraday",
            "raw_json": json.dumps({
                "bars_count": len(date_bars),
                "first_timestamp": date_bars[0].timestamp.isoformat(),
                "last_timestamp": date_bars[-1].timestamp.isoformat()
            }),
            "created_at": datetime.utcnow()
        })
    
    return daily_data


async def fetch_intraday(symbol: str, interval: str = None, max_days_back: int = 5, use_mock: bool = False) -> List[Bar]:
    """
    Fetch intraday data for a symbol and get the most recent available data.
    
    Args:
        symbol: Stock symbol (e.g., 'AAPL', 'MSFT')
        interval: Time interval (1min, 5min, 15min, 30min, 60min)
        max_days_back: Maximum number of days to look back for data
        use_mock: If True, use mock data instead of calling the API
    
    Returns:
        List of Bar objects for the most recent trading session
    """
    if interval is None:
        interval = settings.alphavantage_interval
    
    logger.info(f"Fetching intraday data for {symbol} with {interval} interval")
    
    # Check if we already have cached data for this symbol
    from .db import get_latest_daily_price
    cached_data = get_latest_daily_price(symbol)
    
    if cached_data:
        logger.info(f"Found cached data for {symbol} from {cached_data['trade_date']}")
        # If we have data from the last 7 days, use it instead of making an API call
        # This helps us work within the API limits
        cache_date = datetime.strptime(cached_data['trade_date'], '%Y-%m-%d').replace(tzinfo=pytz.UTC)
        now = datetime.now(pytz.UTC)
        if (now - cache_date).days <= 7:
            logger.info(f"Using cached data for {symbol} (within 7 days)")
            # Return empty list to skip API call but still process the symbol
            # The ETL pipeline will use the cached data from the database
            return []
    
    try:
        # Use the async client to fetch intraday data
        client = AlphaVantageClient()
        try:
            data = await client.fetch_intraday_data(symbol, interval)
        except AlphaVantageError as e:
            logger.warning(f"Alpha Vantage API error: {e}")
            # Don't raise, just return empty list to indicate no data available
            # This allows the ETL pipeline to continue with other symbols
            return []
        
        # Use the normalize_intraday_data function to parse the data
        bars = normalize_intraday_data(symbol, data, interval)
        if not bars:
            logger.warning(f"No bars found for {symbol}")
            return []
            
        # Filter bars by max_days_back
        today_date = today_ist_date()
        recent_bars = [bar for bar in bars if (today_date - bar.timestamp.date()).days <= max_days_back]
        
        if not recent_bars:
            logger.warning(f"No recent data found for {symbol} within {max_days_back} days")
            return []
        
        # Group bars by date
        bars_by_date = {}
        for bar in recent_bars:
            date_ist = bar.timestamp.date()
            if date_ist not in bars_by_date:
                bars_by_date[date_ist] = []
            bars_by_date[date_ist].append(bar)
        
        # If we have today's data, use it
        if today_date in bars_by_date:
            selected_bars = bars_by_date[today_date]
            logger.info(f"Using today's data for {symbol} ({len(selected_bars)} bars)")
        else:
            # Otherwise find the most recent date with data
            most_recent_date = max(bars_by_date.keys())
            selected_bars = bars_by_date[most_recent_date]
            logger.info(f"Using most recent data from {most_recent_date} for {symbol} ({len(selected_bars)} bars)")
        
        # Sort bars by timestamp ascending
        selected_bars.sort(key=lambda x: x.timestamp)
        
        logger.info(f"Fetched {len(selected_bars)} bars for {symbol}")
        return selected_bars
        
    except Exception as e:
        logger.error(f"Failed to fetch intraday data for {symbol}: {e}")
        raise


async def fetch_multiple_symbols_daily(symbols: List[str], force_refresh: bool = False) -> Dict[str, List[Bar]]:
    """
    Fetch daily data for multiple symbols with intelligent rate limiting.
    Uses the TIME_SERIES_DAILY endpoint which has better availability on the free tier.
    
    Args:
        symbols: List of stock symbols
        force_refresh: If True, ignore cached data and fetch fresh data
    
    Returns:
        Dictionary mapping symbol to list of bars
    """
    results = {}
    
    # First, check which symbols have recent data in the database
    from .db import get_latest_daily_price
    
    # Get current date in UTC
    now = datetime.now(pytz.UTC)
    
    # Prioritize symbols that don't have recent data
    symbols_to_fetch = []
    symbols_to_skip = []
    
    # Step 1: Determine which symbols need fetching vs using cache
    for symbol in symbols:
        cached_data = get_latest_daily_price(symbol)
        
        if cached_data and not force_refresh:
            # Parse the date from the database
            try:
                trade_date = cached_data['trade_date']
                if isinstance(trade_date, str):
                    cache_date = datetime.strptime(trade_date, '%Y-%m-%d').replace(tzinfo=pytz.UTC)
                else:
                    # Already a date object
                    cache_date = datetime.combine(trade_date, datetime.min.time()).replace(tzinfo=pytz.UTC)
                days_old = (now - cache_date).days
                
                # If data is less than 1 day old, skip API call and use cached data
                # Daily data only updates once per day, so no need to fetch more frequently
                if days_old < 1:
                    logger.info(f"Using today's cached data for {symbol}")
                    symbols_to_skip.append(symbol)
                    results[symbol] = []
                    continue
                else:
                    # Prioritize symbols with older data
                    symbols_to_fetch.append((symbol, days_old))
            except Exception as e:
                logger.warning(f"Error parsing cache date for {symbol}: {e}")
                symbols_to_fetch.append((symbol, 999))  # High priority if date parsing fails
        else:
            # No cached data or force refresh, highest priority
            priority = 999 if force_refresh else 500
            symbols_to_fetch.append((symbol, priority))
    
    # Sort symbols by priority (highest first)
    symbols_to_fetch.sort(key=lambda x: x[1], reverse=True)
    
    # Determine how many API calls we can make (respecting daily limit)
    max_calls = min(settings.alphavantage_max_daily_calls, len(symbols_to_fetch))
    
    logger.info(f"Will fetch {max_calls} symbols via daily API (skipping {len(symbols_to_skip)} with recent data)")
    
    # Step 2: Fetch data for the highest priority symbols
    for i, (symbol, _) in enumerate(symbols_to_fetch[:max_calls]):
        try:
            bars = await fetch_daily(symbol)
            results[symbol] = bars
            
            # Use our consistent rate limiting between requests
            if i < len(symbols_to_fetch[:max_calls]) - 1:  # Don't throttle after the last request
                await throttle_minute()
                
        except Exception as e:
            logger.error(f"Failed to fetch daily data for {symbol}: {e}")
            results[symbol] = []  # Continue with other symbols
    
    # Step 3: Handle remaining symbols we couldn't fetch due to rate limits
    for symbol, _ in symbols_to_fetch[max_calls:]:
        logger.warning(f"Skipping API call for {symbol} due to daily rate limit")
        results[symbol] = []
    
    return results


# Removed deprecated fetch_multiple_symbols function - use fetch_multiple_symbols_daily directly


async def fetch_daily(symbol: str, outputsize: str = "compact") -> List[Bar]:
    """
    Fetch daily data for a symbol using the async AlphaVantageClient.
    
    Args:
        symbol: Stock symbol (e.g., 'AAPL', 'MSFT')
        outputsize: 'compact' for latest 100 data points, 'full' for 20+ years of data
    
    Returns:
        List of Bar objects for daily data
    """
    logger.info(f"Fetching daily data for {symbol} with outputsize={outputsize}")
    
    try:
        client = AlphaVantageClient()
        
        # Use the async client to fetch daily data
        data = await client.fetch_daily_data(symbol)
        
        # Parse the time series data
        time_series_key = "Time Series (Daily)"
        if time_series_key not in data:
            logger.warning(f"No daily time series data found for {symbol}")
            return []
        
        time_series = data[time_series_key]
        all_bars = []
        
        # Get current date in IST
        today_date = today_ist_date()
        
        for date_str, ohlcv in time_series.items():
            try:
                # Parse date (daily data has date strings like "2023-09-15")
                date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
                
                # Create timestamp at market close time (16:00) in IST
                timestamp_ist = datetime.combine(
                    date_obj, 
                    datetime.min.time().replace(hour=16, minute=0)
                ).replace(tzinfo=IST)
                
                bar = Bar(
                    timestamp=timestamp_ist,
                    open_price=float(ohlcv["1. open"]),
                    high=float(ohlcv["2. high"]),
                    low=float(ohlcv["3. low"]),
                    close=float(ohlcv["4. close"]),
                    volume=int(ohlcv["5. volume"])
                )
                
                all_bars.append(bar)
                
            except (ValueError, KeyError) as e:
                logger.warning(f"Failed to parse daily bar data for {symbol} at {date_str}: {e}")
                continue
        
        # Sort bars by timestamp descending (newest first)
        all_bars.sort(key=lambda x: x.timestamp, reverse=True)
        
        # Return only the most recent bar (today or last trading day)
        if all_bars:
            logger.info(f"Fetched {len(all_bars)} daily bars for {symbol}, using most recent")
            return [all_bars[0]]
        else:
            logger.warning(f"No daily bars found for {symbol}")
            return []
        
    except Exception as e:
        logger.error(f"Failed to fetch daily data for {symbol}: {e}")
        return []


def create_sample_raw_json(bars: List[Bar], metadata: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Create a compact raw JSON representation for storage.
    Includes metadata and first/last few bars to avoid huge database rows.
    """
    if not bars:
        return {"bars": [], "metadata": metadata or {}}
    
    # Take first 3 and last 3 bars, or all if fewer than 6
    if len(bars) <= 6:
        sample_bars = bars
    else:
        sample_bars = bars[:3] + bars[-3:]
    
    return {
        "metadata": metadata or {},
        "total_bars": len(bars),
        "sample_bars": [bar.to_dict() for bar in sample_bars],
        "first_timestamp": bars[0].timestamp.isoformat(),
        "last_timestamp": bars[-1].timestamp.isoformat()
    }


async def fetch_and_process_symbol(symbol: str, target_date: date = None) -> Dict[str, Any]:
    """
    Fetch and process data for a single symbol, with caching and error handling.
    
    Args:
        symbol: Stock symbol to process
        target_date: Target date for data (defaults to today in IST)
        
    Returns:
        Dictionary with processing results
    """
    if target_date is None:
        target_date = today_ist_date()
    
    logger.info(f"Processing symbol {symbol} for date {target_date}")
    
    result = {
        "symbol": symbol,
        "target_date": target_date,
        "success": False,
        "data_source": None,
        "error": None,
        "daily_data": None,
        "metrics": None
    }
    
    # Step 1: Check cache first
    from .db import fetch_one, execute_query
    cache_query = """
        SELECT * FROM daily_prices 
        WHERE symbol = :symbol AND trade_date = :trade_date
    """
    
    try:
        cached_data = fetch_one(cache_query, {"symbol": symbol, "trade_date": target_date})
        
        if cached_data:
            logger.info(f"Using cached data for {symbol} on {target_date}")
            result["success"] = True
            result["data_source"] = "cache"
            result["daily_data"] = cached_data
            return result
    except Exception as e:
        logger.error(f"Error checking cache for {symbol}: {e}")
    
    # Step 2: No cache hit, try fetching from API with fallbacks
    client = AlphaVantageClient()
    
    # Step 2.1: Try daily data first (preferred source)
    try:
        logger.info(f"Fetching daily data for {symbol} from Alpha Vantage")
        daily_data = await client.fetch_daily_data(symbol)
        normalized_data = normalize_daily_data(symbol, daily_data)
        
        logger.info(f"Normalized {len(normalized_data)} rows for {symbol}")
        
        # Find data for target date
        target_data = next((row for row in normalized_data 
                          if row["trade_date"] == target_date), None)
        
        if target_data:
            # Add created_at if not present
            if 'created_at' not in target_data:
                from datetime import datetime
                target_data['created_at'] = datetime.utcnow()
                
            logger.info(f"Found exact date match for {symbol} on {target_date}")
            
            # Store in database using upsert_daily_prices
            from .db import upsert_daily_prices
            upsert_daily_prices(target_data)
            
            result["success"] = True
            result["data_source"] = "alphavantage_daily"
            result["daily_data"] = target_data
            
            # Calculate metrics and return
            await _add_metrics_to_result(result, symbol, target_date)
            return result
            
        # No exact date match, try most recent
        if normalized_data:
            most_recent = max(normalized_data, key=lambda x: x["trade_date"])
            
            # Add created_at if not present
            if 'created_at' not in most_recent:
                from datetime import datetime
                most_recent['created_at'] = datetime.utcnow()
                
            logger.info(f"Using most recent data from {most_recent['trade_date']} for {symbol}")
            
            # Store in database using upsert_daily_prices
            from .db import upsert_daily_prices
            upsert_daily_prices(most_recent)
            
            result["success"] = True
            result["data_source"] = "alphavantage_daily_recent"
            result["daily_data"] = most_recent
            result["warning"] = f"Using most recent data from {most_recent['trade_date']} instead of {target_date}"
            
            # Calculate metrics and return
            await _add_metrics_to_result(result, symbol, target_date)
            return result
    except AlphaVantageError as e:
        # Log but continue to fallback
        logger.warning(f"Daily data fetch failed for {symbol}: {e}. Trying intraday fallback.")
    except Exception as e:
        # Log but continue to fallback
        logger.error(f"Unexpected error fetching daily data for {symbol}: {e}. Trying intraday fallback.")
    
    # Step 2.2: Try intraday as fallback
    try:
        logger.info(f"Fetching intraday data for {symbol} from Alpha Vantage")
        intraday_data = await client.fetch_intraday_data(symbol)
        bars = normalize_intraday_data(symbol, intraday_data, "5min")
        
        if bars:
            logger.info(f"Normalized {len(bars)} intraday bars for {symbol}")
            # Aggregate to daily
            daily_rows = aggregate_intraday_to_daily(bars)
            if daily_rows:
                target_row = next((row for row in daily_rows 
                                if row["trade_date"] == target_date), None)
                
                if target_row:
                    # Add created_at if not present
                    if 'created_at' not in target_row:
                        from datetime import datetime
                        target_row['created_at'] = datetime.utcnow()
                    
                    logger.info(f"Found intraday data for {symbol} on {target_date}")
                    
                    # Store in database using upsert_daily_prices
                    from .db import upsert_daily_prices
                    upsert_daily_prices(target_row)
                    
                    result["success"] = True
                    result["data_source"] = "alphavantage_intraday"
                    result["daily_data"] = target_row
                    
                    # Calculate metrics and return
                    await _add_metrics_to_result(result, symbol, target_date)
                    return result
                else:
                    result["error"] = f"No intraday data for {symbol} on {target_date}"
            else:
                result["error"] = f"Failed to aggregate intraday data for {symbol}"
        else:
            result["error"] = f"No intraday bars for {symbol}"
    except AlphaVantageError as e:
        result["error"] = f"Alpha Vantage API error: {str(e)}"
    except Exception as e:
        result["error"] = f"Failed to fetch data: {str(e)}"
    
    return result


async def _add_metrics_to_result(result: Dict[str, Any], symbol: str, target_date: date) -> None:
    """Helper function to calculate metrics and add to result dict."""
    if result["success"] and result["daily_data"]:
        try:
            logger.info(f"Calculating metrics for {symbol} on {target_date}")
            metrics = calculate_metrics(symbol, target_date, result["daily_data"])
            result["metrics"] = metrics
            
            # Store metrics in database
            from .db import upsert_daily_metrics
            
            # Add created_at if not present
            if 'created_at' not in metrics:
                from datetime import datetime
                metrics['created_at'] = datetime.utcnow()
                
            logger.info(f"Storing metrics for {symbol} on {target_date}: return_pct={metrics.get('return_pct')}")
            upsert_daily_metrics(metrics)
            
            # Verify metrics were stored
            from .db import get_daily_metrics
            stored_metrics = get_daily_metrics(str(target_date), [symbol])
            if stored_metrics:
                logger.info(f"Verified metrics storage for {symbol} on {target_date}")
            else:
                logger.warning(f"Could not verify metrics storage for {symbol} on {target_date}")
                
        except Exception as e:
            logger.error(f"Failed to calculate or store metrics for {symbol}: {e}")


def calculate_metrics(symbol: str, target_date: date, daily_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate key metrics for a symbol based on daily data.
    This data will be used by the LLM component.
    """
    logger.info(f"Calculating metrics for {symbol} on {target_date} with close price {daily_data['close']}")
    
    # Initialize metrics with required fields for daily_metrics table
    metrics = {
        "symbol": symbol,
        "trade_date": target_date,
        "return_pct": None,
        "ma7": None,
        "ma30": None,
        "rsi14": None,
        "vol7": None,
        "high20": None,
        "low20": None,
        # Additional fields for analysis
        "price": daily_data["close"],
        "volume": daily_data["volume"],
        "change": None
    }
    
    # Get previous day's data for comparison
    from .db import fetch_one
    prev_day_query = """
        SELECT close FROM daily_prices 
        WHERE symbol = :symbol AND trade_date < :trade_date
        ORDER BY trade_date DESC LIMIT 1
    """
    
    try:
        prev_day = fetch_one(prev_day_query, {"symbol": symbol, "trade_date": target_date})
        
        if prev_day and prev_day["close"]:
            # Calculate change and return percentage
            metrics["change"] = daily_data["close"] - prev_day["close"]
            metrics["return_pct"] = (metrics["change"] / prev_day["close"]) * 100
            logger.info(f"Calculated return_pct for {symbol}: {metrics['return_pct']:.2f}%")
        else:
            # If no previous day data, use intraday return
            if daily_data["open"] > 0:  # Avoid division by zero
                metrics["return_pct"] = ((daily_data["close"] / daily_data["open"]) - 1) * 100
                logger.info(f"Using intraday return for {symbol}: {metrics['return_pct']:.2f}%")
            else:
                logger.warning(f"Cannot calculate return_pct for {symbol}: no valid reference price")
    except Exception as e:
        logger.error(f"Error calculating return_pct for {symbol}: {e}")
    
    # Try to calculate simple moving averages if we have historical data
    try:
        from .db import execute_query
        ma_query = """
            SELECT AVG(close) as avg_close
            FROM daily_prices
            WHERE symbol = :symbol 
              AND trade_date <= :trade_date
              AND trade_date > (DATE(:trade_date) - INTERVAL '7 days')
        """
        
        ma7_result = execute_query(ma_query, {"symbol": symbol, "trade_date": target_date})
        ma7_row = ma7_result.fetchone()
        if ma7_row and ma7_row[0]:
            metrics["ma7"] = float(ma7_row[0])
            logger.info(f"Calculated MA7 for {symbol}: {metrics['ma7']:.2f}")
    except Exception as e:
        logger.error(f"Error calculating MA7 for {symbol}: {e}")
    
    return metrics


async def run_market_data_pipeline(symbols: List[str] = None, target_date: date = None) -> Dict[str, Any]:
    """
    Run the complete market data pipeline for all symbols.
    This is the main entry point for the ETL process.
    
    Args:
        symbols: List of symbols to process (defaults to settings.tickers_list)
        target_date: Target date for data (defaults to today in IST)
        
    Returns:
        Dictionary with processing results and LLM-generated market summary
    """
    if symbols is None:
        symbols = settings.get_tickers_list()
        
    if target_date is None:
        target_date = today_ist_date()
    
    logger.info(f"Running market data pipeline for {len(symbols)} symbols on {target_date}")
    
    results = {
        "target_date": target_date,
        "symbols_requested": len(symbols),
        "symbols_processed": 0,
        "symbols_failed": 0,
        "processed_symbols": [],
        "failed_symbols": [],
        "market_data": {}
    }
    
    # Process each symbol
    for symbol in symbols:
        try:
            logger.info(f"Processing symbol: {symbol} for date: {target_date}")
            symbol_result = await fetch_and_process_symbol(symbol, target_date)
            
            if symbol_result["success"]:
                results["symbols_processed"] += 1
                results["processed_symbols"].append(symbol)
                results["market_data"][symbol] = symbol_result
                logger.info(f"Successfully processed {symbol}: {symbol_result['data_source']}")
                
                # Verify data was stored in database
                from .db import get_latest_daily_price
                db_data = get_latest_daily_price(symbol)
                if db_data:
                    logger.info(f"Verified database storage for {symbol}: {db_data['trade_date']}")
                else:
                    logger.warning(f"Could not verify database storage for {symbol}")
            else:
                results["symbols_failed"] += 1
                results["failed_symbols"].append({"symbol": symbol, "error": symbol_result["error"]})
                logger.warning(f"Failed to process {symbol}: {symbol_result['error']}")
            
            # Rate limiting between API calls
            if symbol != symbols[-1]:  # Don't sleep after the last symbol
                logger.info(f"Applying rate limiting before processing next symbol")
                await throttle_minute()  # Use our consistent rate limiting function
                
        except Exception as e:
            logger.error(f"Failed to process {symbol}: {e}")
            results["symbols_failed"] += 1
            results["failed_symbols"].append({"symbol": symbol, "error": str(e)})
    
    # Generate market summary for LLM if we have data
    if results["symbols_processed"] > 0:
        try:
            from .llm import generate_market_summary
            logger.info("Generating market summary with LLM")
            summary = await generate_market_summary(results["market_data"], target_date)
            results["market_summary"] = summary
            logger.info("Market summary generated successfully")
        except Exception as e:
            logger.error(f"Failed to generate market summary: {e}")
            results["market_summary_error"] = str(e)
    
    logger.info(f"Market data pipeline completed: {results['symbols_processed']} processed, {results['symbols_failed']} failed")
    return results
