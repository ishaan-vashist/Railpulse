"""FastAPI application entrypoint."""
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routes import get_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

# Create FastAPI application
app = FastAPI(
    title="RailPulse Market Data API",
    description="Production-ready FastAPI service for Alpha Vantage intraday data aggregation",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Next.js development
        "http://localhost:3001", 
        "https://*.vercel.app",   # Vercel deployments
        "https://*.netlify.app",  # Netlify deployments
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Health check endpoint - defined before startup event to ensure it works
# even if startup validation fails
@app.get("/healthz")
async def health_check():
    """Health check endpoint for monitoring and deployment checks."""
    # This endpoint always returns OK to allow Railway health checks to pass
    # regardless of application startup status
    return {"status": "ok"}

# Include routes
app.include_router(get_router())

# Startup event
@app.on_event("startup")
async def startup_event():
    """Application startup tasks."""
    logger.info("RailPulse Market Data API starting up...")
    
    # Validate configuration
    from .config import settings, validate_interval
    
    try:
        # Check required environment variables
        required_vars = [
            "database_url",
            "alphavantage_api_key", 
            "openai_api_key",
            "tickers",
            "app_secret"
        ]
        
        missing_vars = []
        for var in required_vars:
            if not getattr(settings, var, None):
                missing_vars.append(var.upper())
        
        if missing_vars:
            logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
        
        # Validate Alpha Vantage interval
        if not validate_interval(settings.alphavantage_interval):
            logger.error(f"Invalid Alpha Vantage interval: {settings.alphavantage_interval}")
            raise ValueError(f"Invalid Alpha Vantage interval: {settings.alphavantage_interval}")
        
        # Test database connection
        from .db import execute_query
        execute_query("SELECT 1")
        
        logger.info("Configuration validated successfully")
        logger.info(f"Configured for {len(settings.get_tickers_list())} symbols: {', '.join(settings.get_tickers_list())}")
        
    except Exception as e:
        logger.error(f"Startup validation failed: {e}")
        raise


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown tasks."""
    logger.info("RailPulse Market Data API shutting down...")


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "RailPulse Market Data API",
        "version": "1.0.0",
        "description": "Production-ready FastAPI service for Alpha Vantage intraday data aggregation",
        "endpoints": {
            "health": "/healthz",
            "metrics": "/metrics",
            "recommendations": "/recommendations", 
            "admin_etl": "/admin/run-today",
            "docs": "/docs"
        }
    }



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
