"""KPI calculations for daily metrics."""
import logging
from typing import List, Dict, Any, Optional
from datetime import date, timedelta
from .db import get_daily_prices, execute_query

logger = logging.getLogger(__name__)


def calculate_return_pct(open_price: float, close_price: float) -> float:
    """Calculate daily return percentage."""
    if open_price == 0:
        return 0.0
    return ((close_price / open_price) - 1) * 100


def calculate_moving_average(symbol: str, trade_date: date, days: int) -> Optional[float]:
    """
    Calculate moving average for a symbol over specified days.
    Note: This is a stub for future implementation when historical data is available.
    """
    # For now, return None as we only have today's data
    # Future implementation would query historical daily_prices
    logger.debug(f"Moving average calculation not implemented yet for {symbol}")
    return None


def calculate_rsi(symbol: str, trade_date: date, period: int = 14) -> Optional[float]:
    """
    Calculate RSI (Relative Strength Index) for a symbol.
    Note: This is a stub for future implementation when historical data is available.
    """
    # For now, return None as we only have today's data
    # Future implementation would calculate RSI from historical prices
    logger.debug(f"RSI calculation not implemented yet for {symbol}")
    return None


def calculate_volatility(symbol: str, trade_date: date, days: int = 7) -> Optional[float]:
    """
    Calculate volatility (standard deviation of returns) over specified days.
    Note: This is a stub for future implementation when historical data is available.
    """
    # For now, return None as we only have today's data
    # Future implementation would calculate volatility from historical returns
    logger.debug(f"Volatility calculation not implemented yet for {symbol}")
    return None


def calculate_high_low_range(symbol: str, trade_date: date, days: int = 20) -> tuple[Optional[float], Optional[float]]:
    """
    Calculate highest high and lowest low over specified days.
    Note: This is a stub for future implementation when historical data is available.
    """
    # For now, return None as we only have today's data
    # Future implementation would find min/max from historical prices
    logger.debug(f"High/Low range calculation not implemented yet for {symbol}")
    return None, None


def compute_present_day_metrics(symbol: str, trade_date: date, 
                               open_price: float, close_price: float) -> Dict[str, Any]:
    """
    Compute all available metrics for present day data.
    
    Args:
        symbol: Stock symbol
        trade_date: Trading date
        open_price: Opening price
        close_price: Closing price
    
    Returns:
        Dictionary with computed metrics
    """
    metrics = {
        "trade_date": trade_date,
        "symbol": symbol,
        "return_pct": calculate_return_pct(open_price, close_price),
        "ma7": calculate_moving_average(symbol, trade_date, 7),
        "ma30": calculate_moving_average(symbol, trade_date, 30),
        "rsi14": calculate_rsi(symbol, trade_date, 14),
        "vol7": calculate_volatility(symbol, trade_date, 7),
        "high20": None,  # Will be calculated when historical data is available
        "low20": None    # Will be calculated when historical data is available
    }
    
    # Calculate 20-day high/low if we had historical data
    high20, low20 = calculate_high_low_range(symbol, trade_date, 20)
    metrics["high20"] = high20
    metrics["low20"] = low20
    
    return metrics


def get_portfolio_metrics_summary(trade_date: date) -> Dict[str, Any]:
    """
    Get a summary of portfolio metrics for analysis.
    
    Args:
        trade_date: Date to analyze
    
    Returns:
        Portfolio metrics summary
    """
    from .db import get_daily_metrics
    
    try:
        metrics = get_daily_metrics(str(trade_date))
        
        if not metrics:
            return {
                "date": str(trade_date),
                "total_symbols": 0,
                "symbols_with_data": 0,
                "avg_return": None,
                "positive_returns": 0,
                "negative_returns": 0,
                "best_performer": None,
                "worst_performer": None
            }
        
        # Filter symbols with valid return data
        valid_returns = [m for m in metrics if m["return_pct"] is not None]
        
        if not valid_returns:
            return {
                "date": str(trade_date),
                "total_symbols": len(metrics),
                "symbols_with_data": 0,
                "avg_return": None,
                "positive_returns": 0,
                "negative_returns": 0,
                "best_performer": None,
                "worst_performer": None
            }
        
        # Calculate summary statistics
        returns = [m["return_pct"] for m in valid_returns]
        avg_return = sum(returns) / len(returns)
        positive_returns = len([r for r in returns if r > 0])
        negative_returns = len([r for r in returns if r < 0])
        
        best_performer = max(valid_returns, key=lambda x: x["return_pct"])
        worst_performer = min(valid_returns, key=lambda x: x["return_pct"])
        
        return {
            "date": str(trade_date),
            "total_symbols": len(metrics),
            "symbols_with_data": len(valid_returns),
            "avg_return": round(avg_return, 2),
            "positive_returns": positive_returns,
            "negative_returns": negative_returns,
            "best_performer": {
                "symbol": best_performer["symbol"],
                "return_pct": round(best_performer["return_pct"], 2)
            },
            "worst_performer": {
                "symbol": worst_performer["symbol"],
                "return_pct": round(worst_performer["return_pct"], 2)
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get portfolio metrics summary: {e}")
        raise


# Future implementations for when historical data becomes available

def _calculate_sma(prices: List[float], period: int) -> Optional[float]:
    """Calculate Simple Moving Average."""
    if len(prices) < period:
        return None
    return sum(prices[-period:]) / period


def _calculate_rsi_from_prices(prices: List[float], period: int = 14) -> Optional[float]:
    """Calculate RSI from price series."""
    if len(prices) < period + 1:
        return None
    
    gains = []
    losses = []
    
    for i in range(1, len(prices)):
        change = prices[i] - prices[i-1]
        if change > 0:
            gains.append(change)
            losses.append(0)
        else:
            gains.append(0)
            losses.append(abs(change))
    
    if len(gains) < period:
        return None
    
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period
    
    if avg_loss == 0:
        return 100
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def _calculate_volatility_from_returns(returns: List[float]) -> Optional[float]:
    """Calculate volatility (standard deviation) from returns."""
    if len(returns) < 2:
        return None
    
    mean_return = sum(returns) / len(returns)
    variance = sum((r - mean_return) ** 2 for r in returns) / (len(returns) - 1)
    
    return variance ** 0.5
