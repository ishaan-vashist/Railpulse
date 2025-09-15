"""FastAPI routes for the market data service."""
import logging
from datetime import date
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query, Header, Depends
from fastapi.security import HTTPBearer
from pydantic import BaseModel
from .config import settings, today_ist_date
from .db import get_daily_prices, get_daily_metrics, get_daily_recommendations
from .etl import run_today
from .llm import generate_and_store_recommendations

logger = logging.getLogger(__name__)

# Create routers
router = APIRouter()
admin_router = APIRouter(prefix="/admin")
security = HTTPBearer(auto_error=False)


# Response models
class HealthResponse(BaseModel):
    status: str
    timestamp: str


class MetricsResponse(BaseModel):
    date: str
    symbols: List[str]
    prices: List[dict]
    metrics: List[dict]


class RecommendationsResponse(BaseModel):
    date: str
    scope: str
    summary: str
    recommendations: List[str]
    created_at: str


class ETLResponse(BaseModel):
    success: bool
    message: str
    results: dict


# Security dependency
def verify_app_secret(
    x_app_secret: Optional[str] = Header(None),
    app_secret: Optional[str] = Query(None)
) -> bool:
    """Verify APP_SECRET from header or query parameter."""
    provided_secret = x_app_secret or app_secret
    
    if not provided_secret:
        raise HTTPException(
            status_code=401,
            detail="APP_SECRET required in header (x-app-secret) or query parameter (app_secret)"
        )
    
    if provided_secret != settings.app_secret:
        raise HTTPException(
            status_code=403,
            detail="Invalid APP_SECRET"
        )
    
    return True


# Health check endpoint
@router.get("/healthz", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    from datetime import datetime
    return HealthResponse(
        status="ok",
        timestamp=datetime.utcnow().isoformat()
    )


# Get metrics endpoint
@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics(
    date: Optional[str] = Query(None, description="Date in YYYY-MM-DD format, defaults to today"),
    symbol: Optional[str] = Query(None, description="Comma-separated symbols, defaults to all")
):
    """
    Get daily prices and metrics for specified date and symbols.
    
    Args:
        date: Date in YYYY-MM-DD format (defaults to today IST)
        symbol: Comma-separated list of symbols (defaults to all available)
    
    Returns:
        Combined prices and metrics data
    """
    try:
        # Parse date parameter
        if date:
            try:
                query_date = date
                # Validate date format
                from datetime import datetime
                datetime.strptime(date, "%Y-%m-%d")
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid date format. Use YYYY-MM-DD"
                )
        else:
            query_date = str(today_ist_date())
        
        # Parse symbols parameter
        symbols_list = None
        if symbol:
            symbols_list = [s.strip() for s in symbol.split(",") if s.strip()]
        
        # Get data from database
        prices = get_daily_prices(query_date, symbols_list)
        metrics = get_daily_metrics(query_date, symbols_list)
        
        # Extract unique symbols
        price_symbols = {p["symbol"] for p in prices}
        metric_symbols = {m["symbol"] for m in metrics}
        all_symbols = sorted(price_symbols.union(metric_symbols))
        
        return MetricsResponse(
            date=query_date,
            symbols=all_symbols,
            prices=prices,
            metrics=metrics
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve metrics: {str(e)}"
        )


# Get recommendations endpoint
@router.get("/recommendations", response_model=RecommendationsResponse)
async def get_recommendations(
    date: Optional[str] = Query(None, description="Date in YYYY-MM-DD format, defaults to today")
):
    """
    Get daily recommendations for specified date.
    
    Args:
        date: Date in YYYY-MM-DD format (defaults to today IST)
    
    Returns:
        Daily recommendations and analysis
    """
    try:
        # Parse date parameter
        if date:
            try:
                query_date = date
                # Validate date format
                from datetime import datetime
                datetime.strptime(date, "%Y-%m-%d")
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid date format. Use YYYY-MM-DD"
                )
        else:
            query_date = str(today_ist_date())
        
        # Get recommendations from database
        recommendations = get_daily_recommendations(query_date, "portfolio")
        
        if not recommendations:
            raise HTTPException(
                status_code=404,
                detail=f"No recommendations found for {query_date}"
            )
        
        import json
        return RecommendationsResponse(
            date=query_date,
            scope=recommendations["scope"],
            summary=recommendations["summary"],
            recommendations=json.loads(recommendations["recommendations"]),
            created_at=recommendations["created_at"].isoformat() if recommendations["created_at"] else ""
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get recommendations: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve recommendations: {str(e)}"
        )


# Admin endpoint to run today's ETL
@admin_router.post("/run-today", response_model=ETLResponse)
async def run_today_etl(
    _: bool = Depends(verify_app_secret)
):
    """
    Run the complete ETL pipeline for today's data.
    Requires APP_SECRET authentication.
    
    Returns:
        ETL execution results and LLM analysis
    """
    try:
        logger.info("Starting admin-triggered ETL pipeline")
        
        # Get symbols from configuration
        symbols = settings.get_tickers_list()
        
        if not symbols:
            raise HTTPException(
                status_code=400,
                detail="No symbols configured in TICKERS environment variable"
            )
        
        # Run ETL pipeline
        etl_results = run_today(symbols)
        
        # Generate LLM recommendations
        trade_date = today_ist_date()
        
        try:
            llm_results = generate_and_store_recommendations(trade_date)
            etl_results["llm_analysis"] = {
                "success": True,
                "summary": llm_results["summary"],
                "recommendations_count": len(llm_results["recommendations"])
            }
        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            etl_results["llm_analysis"] = {
                "success": False,
                "error": str(e)
            }
        
        success = etl_results["symbols_processed"] > 0
        message = f"ETL completed. Processed {etl_results['symbols_processed']} symbols successfully."
        
        if etl_results["symbols_failed"] > 0:
            message += f" {etl_results['symbols_failed']} symbols failed."
        
        if etl_results["symbols_no_data"] > 0:
            message += f" {etl_results['symbols_no_data']} symbols had no data."
        
        return ETLResponse(
            success=success,
            message=message,
            results=etl_results
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"ETL pipeline failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"ETL pipeline failed: {str(e)}"
        )


# Include routers
def get_router():
    """Get the main router with all endpoints."""
    main_router = APIRouter()
    main_router.include_router(router)
    main_router.include_router(admin_router)
    return main_router
