"""ETL pipeline for aggregating intraday data to daily OHLCV."""
import json
import logging
from datetime import date
from typing import List, Dict, Any, Optional
from .alpha_vantage import Bar, fetch_multiple_symbols_daily, create_sample_raw_json
from .db import upsert_daily_prices, upsert_daily_metrics
from .config import today_ist_date

logger = logging.getLogger(__name__)


def aggregate_today(bars: List[Bar]) -> Optional[Dict[str, Any]]:
    """
    Aggregate intraday bars into daily OHLCV data.
    
    Args:
        bars: List of intraday bars sorted by timestamp
    
    Returns:
        Dictionary with aggregated OHLCV data or None if no bars
    """
    if not bars:
        return None
    
    # Sort bars by timestamp to ensure correct order
    sorted_bars = sorted(bars, key=lambda x: x.timestamp)
    
    # Aggregate OHLCV
    open_price = sorted_bars[0].open  # First bar's open
    high_price = max(bar.high for bar in sorted_bars)  # Maximum high
    low_price = min(bar.low for bar in sorted_bars)  # Minimum low
    close_price = sorted_bars[-1].close  # Last bar's close
    total_volume = sum(bar.volume for bar in sorted_bars)  # Sum of volumes
    
    # Create sample raw JSON for storage
    sample_raw = create_sample_raw_json(
        sorted_bars, 
        metadata={
            "aggregation_method": "intraday_to_daily",
            "source": "alphavantage",
            "bars_count": len(sorted_bars)
        }
    )
    
    return {
        "open": open_price,
        "high": high_price,
        "low": low_price,
        "close": close_price,
        "adj_close": close_price,  # Assuming no adjustment for intraday data
        "volume": total_volume,
        "sample_raw": sample_raw
    }


def process_symbol(symbol: str, bars: List[Bar], trade_date: date) -> bool:
    """
    Process a single symbol's bars: aggregate and upsert to database.
    
    Args:
        symbol: Stock symbol
        bars: List of intraday bars
        trade_date: Date for the aggregated data (may be overridden if bars are from a different date)
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Aggregate bars to daily OHLCV
        daily_data = aggregate_today(bars)
        
        if not daily_data:
            logger.warning(f"No bars to aggregate for {symbol}")
            return False
        
        # Determine the actual trade date from the bars if available
        actual_trade_date = trade_date
        if bars and len(bars) > 0:
            # Use the date from the first bar (they should all be from same date after our fetch_intraday changes)
            actual_trade_date = bars[0].timestamp.date()
            logger.info(f"Using actual trade date {actual_trade_date} for {symbol} (requested: {trade_date})")
        
        # Prepare row for daily_prices table
        prices_row = {
            "trade_date": actual_trade_date,
            "symbol": symbol,
            "open": daily_data["open"],
            "high": daily_data["high"],
            "low": daily_data["low"],
            "close": daily_data["close"],
            "adj_close": daily_data["adj_close"],
            "volume": daily_data["volume"],
            "source": "alphavantage",
            "raw_json": json.dumps(daily_data["sample_raw"])
        }
        
        # Debug log to verify data before upsert
        logger.info(f"Upserting data for {symbol} on {actual_trade_date}: open={prices_row['open']}, close={prices_row['close']}")
        
        # Upsert to daily_prices
        upsert_daily_prices(prices_row)
        
        # Verify data was inserted by querying it back
        from .db import get_latest_daily_price
        inserted_data = get_latest_daily_price(symbol)
        if inserted_data:
            logger.info(f"Verified data insertion for {symbol}: {inserted_data['trade_date']}")
        else:
            logger.warning(f"Could not verify data insertion for {symbol}")
        
        # Calculate basic metrics
        return_pct = ((daily_data["close"] / daily_data["open"]) - 1) * 100
        
        # Prepare row for daily_metrics table (other metrics as NULL for now)
        metrics_row = {
            "trade_date": actual_trade_date,
            "symbol": symbol,
            "return_pct": return_pct,
            "ma7": None,
            "ma30": None,
            "rsi14": None,
            "vol7": None,
            "high20": None,
            "low20": None
        }
        
        # Upsert to daily_metrics
        upsert_daily_metrics(metrics_row)
        
        logger.info(f"Successfully processed {symbol} for {actual_trade_date}: return_pct={return_pct:.2f}%")
        return True
        
    except Exception as e:
        logger.error(f"Failed to process {symbol}: {e}")
        return False


async def run_today(symbols: List[str], force_refresh: bool = False) -> Dict[str, Any]:
    """
    Run the complete ETL pipeline for today's data.
    
    Args:
        symbols: List of stock symbols to process
    
    Returns:
        Summary of the ETL run
    """
    logger.info(f"Starting ETL pipeline for {len(symbols)} symbols")
    
    trade_date = today_ist_date()
    results = {
        "trade_date": str(trade_date),
        "symbols_requested": len(symbols),
        "symbols_processed": 0,
        "symbols_failed": 0,
        "symbols_no_data": 0,
        "processed_symbols": [],
        "failed_symbols": [],
        "no_data_symbols": []
    }
    
    try:
        # Fetch daily data for all symbols (more reliable with free tier)
        logger.info("Fetching daily data from Alpha Vantage...")
        if force_refresh:
            logger.info("Force refresh enabled - bypassing cache")
        symbol_bars = await fetch_multiple_symbols_daily(symbols, force_refresh=force_refresh)
        
        # Process each symbol
        for symbol in symbols:
            bars = symbol_bars.get(symbol, [])
            
            if not bars:
                # Check if we have cached data in the database
                from .db import get_latest_daily_price
                cached_data = get_latest_daily_price(symbol)
                
                if cached_data:
                    # We have cached data, use it instead
                    logger.info(f"No new data for {symbol}, using cached data from {cached_data['trade_date']}")
                    
                    # Check if the cached data is from today
                    from datetime import datetime
                    import pytz
                    
                    trade_date_value = cached_data['trade_date']
                    if isinstance(trade_date_value, str):
                        cache_date = datetime.strptime(trade_date_value, '%Y-%m-%d').date()
                    else:
                        # Already a date object
                        cache_date = trade_date_value
                    if cache_date == trade_date:
                        # Already have today's data, count as processed
                        logger.info(f"Already have today's data for {symbol}")
                        results["symbols_processed"] += 1
                        results["processed_symbols"].append(symbol)
                    else:
                        # Have older data, count as no new data
                        logger.warning(f"Using cached data from {cache_date} for {symbol}")
                        results["symbols_no_data"] += 1
                        results["no_data_symbols"].append(symbol)
                else:
                    # No cached data and no new data
                    logger.warning(f"No intraday data available for {symbol} and no cached data")
                    results["symbols_no_data"] += 1
                    results["no_data_symbols"].append(symbol)
                continue
            
            # Process the symbol
            success = process_symbol(symbol, bars, trade_date)
            
            if success:
                results["symbols_processed"] += 1
                results["processed_symbols"].append(symbol)
            else:
                results["symbols_failed"] += 1
                results["failed_symbols"].append(symbol)
        
        logger.info(f"ETL pipeline completed. Processed: {results['symbols_processed']}, "
                   f"Failed: {results['symbols_failed']}, No data: {results['symbols_no_data']}")
        
        return results
        
    except Exception as e:
        logger.error(f"ETL pipeline failed: {e}")
        raise


def get_portfolio_summary(trade_date: date) -> Dict[str, Any]:
    """
    Get portfolio summary for LLM analysis.
    
    Args:
        trade_date: Date to analyze
    
    Returns:
        Portfolio summary data
    """
    from .db import get_daily_metrics, get_daily_prices
    import logging
    
    logger = logging.getLogger(__name__)
    
    try:
        # Get all metrics for the date
        metrics = get_daily_metrics(str(trade_date))
        
        # If no metrics are available, try to calculate them from daily prices
        if not metrics:
            logger.info(f"No metrics found for {trade_date}, attempting to calculate from daily prices")
            prices = get_daily_prices(str(trade_date))
            
            if not prices:
                logger.warning(f"No price data found for {trade_date}")
                return {
                    "trade_date": str(trade_date),
                    "symbols_count": 0,
                    "portfolio_return": None,
                    "top_gainer": None,
                    "top_loser": None,
                    "metrics": []
                }
            
            # Calculate simple metrics from price data
            calculated_metrics = []
            for price in prices:
                try:
                    # Calculate intraday return
                    if price["open"] > 0:  # Avoid division by zero
                        return_pct = ((price["close"] / price["open"]) - 1) * 100
                    else:
                        return_pct = 0
                    
                    calculated_metrics.append({
                        "trade_date": price["trade_date"],
                        "symbol": price["symbol"],
                        "return_pct": return_pct,
                        "ma7": None,
                        "ma30": None,
                        "rsi14": None,
                        "vol7": None,
                        "high20": None,
                        "low20": None
                    })
                except Exception as e:
                    logger.error(f"Failed to calculate metrics for {price['symbol']}: {e}")
            
            metrics = calculated_metrics
            logger.info(f"Calculated {len(metrics)} metrics from price data")
        
        # Filter out symbols with null return_pct
        valid_metrics = [m for m in metrics if m["return_pct"] is not None]
        
        if not valid_metrics:
            logger.warning(f"No valid metrics with return_pct for {trade_date}")
            return {
                "trade_date": str(trade_date),
                "symbols_count": len(metrics),
                "portfolio_return": None,
                "top_gainer": None,
                "top_loser": None,
                "metrics": metrics
            }
        
        # Calculate portfolio return (equal-weighted average)
        portfolio_return = sum(m["return_pct"] for m in valid_metrics) / len(valid_metrics)
        
        # Find top gainer and loser
        top_gainer = max(valid_metrics, key=lambda x: x["return_pct"])
        top_loser = min(valid_metrics, key=lambda x: x["return_pct"])
        
        return {
            "trade_date": str(trade_date),
            "symbols_count": len(valid_metrics),
            "portfolio_return": round(portfolio_return, 2),
            "top_gainer": {
                "symbol": top_gainer["symbol"],
                "return_pct": round(top_gainer["return_pct"], 2)
            },
            "top_loser": {
                "symbol": top_loser["symbol"],
                "return_pct": round(top_loser["return_pct"], 2)
            },
            "metrics": valid_metrics
        }
        
    except Exception as e:
        logger.error(f"Failed to get portfolio summary: {e}")
        raise
