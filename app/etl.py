"""ETL pipeline for aggregating intraday data to daily OHLCV."""
import json
import logging
from datetime import date
from typing import List, Dict, Any, Optional
from .alpha_vantage import Bar, fetch_multiple_symbols, create_sample_raw_json
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
        trade_date: Date for the aggregated data
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Aggregate bars to daily OHLCV
        daily_data = aggregate_today(bars)
        
        if not daily_data:
            logger.warning(f"No bars to aggregate for {symbol} on {trade_date}")
            return False
        
        # Prepare row for daily_prices table
        prices_row = {
            "trade_date": trade_date,
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
        
        # Upsert to daily_prices
        upsert_daily_prices(prices_row)
        
        # Calculate basic metrics
        return_pct = ((daily_data["close"] / daily_data["open"]) - 1) * 100
        
        # Prepare row for daily_metrics table (other metrics as NULL for now)
        metrics_row = {
            "trade_date": trade_date,
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
        
        logger.info(f"Successfully processed {symbol}: return_pct={return_pct:.2f}%")
        return True
        
    except Exception as e:
        logger.error(f"Failed to process {symbol}: {e}")
        return False


def run_today(symbols: List[str]) -> Dict[str, Any]:
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
        # Fetch intraday data for all symbols
        logger.info("Fetching intraday data from Alpha Vantage...")
        symbol_bars = fetch_multiple_symbols(symbols)
        
        # Process each symbol
        for symbol in symbols:
            bars = symbol_bars.get(symbol, [])
            
            if not bars:
                logger.warning(f"No intraday data available for {symbol}")
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
    from .db import get_daily_metrics
    
    try:
        # Get all metrics for the date
        metrics = get_daily_metrics(str(trade_date))
        
        if not metrics:
            return {
                "trade_date": str(trade_date),
                "symbols_count": 0,
                "portfolio_return": None,
                "top_gainer": None,
                "top_loser": None,
                "metrics": []
            }
        
        # Filter out symbols with null return_pct
        valid_metrics = [m for m in metrics if m["return_pct"] is not None]
        
        if not valid_metrics:
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
