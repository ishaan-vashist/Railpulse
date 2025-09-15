"""Alpha Vantage API client for fetching intraday data."""
import json
import logging
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
import requests
import pytz
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from .config import settings, IST, today_ist_date

logger = logging.getLogger(__name__)


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


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((requests.RequestException, AlphaVantageError))
)
def _make_api_request(url: str, params: Dict[str, str]) -> Dict[str, Any]:
    """Make a request to Alpha Vantage API with retry logic."""
    try:
        response = requests.get(url, params=params, timeout=30)
        
        if response.status_code == 429:
            logger.warning("Rate limit hit, backing off...")
            time.sleep(12)  # Wait 12 seconds for rate limit
            raise AlphaVantageError("Rate limit exceeded")
        
        response.raise_for_status()
        data = response.json()
        
        # Check for API error messages
        if "Error Message" in data:
            raise AlphaVantageError(f"API Error: {data['Error Message']}")
        
        if "Note" in data:
            logger.warning(f"API Note: {data['Note']}")
            raise AlphaVantageError("API rate limit or other issue")
        
        return data
        
    except requests.RequestException as e:
        logger.error(f"Request failed: {e}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON response: {e}")
        raise AlphaVantageError("Invalid JSON response")


def fetch_intraday(symbol: str, interval: str = None) -> List[Bar]:
    """
    Fetch intraday data for a symbol and filter for today's bars in IST.
    
    Args:
        symbol: Stock symbol (e.g., 'AAPL', 'MSFT')
        interval: Time interval (1min, 5min, 15min, 30min, 60min)
    
    Returns:
        List of Bar objects for today's trading session in IST
    """
    if interval is None:
        interval = settings.alphavantage_interval
    
    logger.info(f"Fetching intraday data for {symbol} with {interval} interval")
    
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_INTRADAY",
        "symbol": symbol,
        "interval": interval,
        "outputsize": "compact",
        "apikey": settings.alphavantage_api_key
    }
    
    try:
        data = _make_api_request(url, params)
        
        # Parse the time series data
        time_series_key = f"Time Series ({interval})"
        if time_series_key not in data:
            logger.warning(f"No time series data found for {symbol}")
            return []
        
        time_series = data[time_series_key]
        bars = []
        today_date = today_ist_date()
        
        for timestamp_str, ohlcv in time_series.items():
            try:
                # Parse timestamp (assumes UTC from Alpha Vantage)
                timestamp_utc = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                timestamp_utc = timestamp_utc.replace(tzinfo=pytz.UTC)
                
                # Convert to IST
                timestamp_ist = timestamp_utc.astimezone(IST)
                
                # Only include bars from today (IST)
                if timestamp_ist.date() != today_date:
                    continue
                
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
        
        logger.info(f"Fetched {len(bars)} bars for {symbol} on {today_date}")
        return bars
        
    except Exception as e:
        logger.error(f"Failed to fetch intraday data for {symbol}: {e}")
        raise


def fetch_multiple_symbols(symbols: List[str], interval: str = None) -> Dict[str, List[Bar]]:
    """
    Fetch intraday data for multiple symbols with rate limiting.
    
    Args:
        symbols: List of stock symbols
        interval: Time interval for all symbols
    
    Returns:
        Dictionary mapping symbol to list of bars
    """
    results = {}
    
    for i, symbol in enumerate(symbols):
        try:
            bars = fetch_intraday(symbol, interval)
            results[symbol] = bars
            
            # Rate limiting: sleep between requests (5 calls per minute limit)
            if i < len(symbols) - 1:  # Don't sleep after the last request
                time.sleep(12)  # 12 seconds between requests = 5 per minute
                
        except Exception as e:
            logger.error(f"Failed to fetch data for {symbol}: {e}")
            results[symbol] = []  # Continue with other symbols
    
    return results


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
