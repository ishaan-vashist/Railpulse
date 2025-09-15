"""Database connection and helper functions."""
import logging
from typing import Dict, Any, Optional
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
from .config import settings

logger = logging.getLogger(__name__)

# Create engine with connection pooling
engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,
    pool_recycle=300,
    echo=False
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db_session():
    """Get a database session."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def execute_query(query: str, params: Optional[Dict[str, Any]] = None) -> Any:
    """Execute a SQL query and return results."""
    with engine.connect() as conn:
        try:
            result = conn.execute(text(query), params or {})
            conn.commit()
            return result
        except SQLAlchemyError as e:
            logger.error(f"Database query failed: {e}")
            conn.rollback()
            raise


def upsert_daily_prices(row: Dict[str, Any]) -> None:
    """Upsert a row into daily_prices table."""
    query = """
    INSERT INTO daily_prices (
        trade_date, symbol, open, high, low, close, adj_close, 
        volume, source, raw_json, created_at
    ) VALUES (
        :trade_date, :symbol, :open, :high, :low, :close, :adj_close,
        :volume, :source, :raw_json, NOW()
    )
    ON CONFLICT (trade_date, symbol) 
    DO UPDATE SET
        open = EXCLUDED.open,
        high = EXCLUDED.high,
        low = EXCLUDED.low,
        close = EXCLUDED.close,
        adj_close = EXCLUDED.adj_close,
        volume = EXCLUDED.volume,
        source = EXCLUDED.source,
        raw_json = EXCLUDED.raw_json,
        created_at = NOW()
    """
    
    try:
        execute_query(query, row)
        logger.info(f"Upserted daily_prices for {row['symbol']} on {row['trade_date']}")
    except Exception as e:
        logger.error(f"Failed to upsert daily_prices for {row['symbol']}: {e}")
        raise


def upsert_daily_metrics(row: Dict[str, Any]) -> None:
    """Upsert a row into daily_metrics table."""
    query = """
    INSERT INTO daily_metrics (
        trade_date, symbol, return_pct, ma7, ma30, rsi14, 
        vol7, high20, low20, created_at
    ) VALUES (
        :trade_date, :symbol, :return_pct, :ma7, :ma30, :rsi14,
        :vol7, :high20, :low20, NOW()
    )
    ON CONFLICT (trade_date, symbol)
    DO UPDATE SET
        return_pct = EXCLUDED.return_pct,
        ma7 = EXCLUDED.ma7,
        ma30 = EXCLUDED.ma30,
        rsi14 = EXCLUDED.rsi14,
        vol7 = EXCLUDED.vol7,
        high20 = EXCLUDED.high20,
        low20 = EXCLUDED.low20,
        created_at = NOW()
    """
    
    try:
        execute_query(query, row)
        logger.info(f"Upserted daily_metrics for {row['symbol']} on {row['trade_date']}")
    except Exception as e:
        logger.error(f"Failed to upsert daily_metrics for {row['symbol']}: {e}")
        raise


def upsert_daily_recommendations(row: Dict[str, Any]) -> None:
    """Upsert a row into daily_recommendations table."""
    query = """
    INSERT INTO daily_recommendations (
        for_date, scope, summary, recommendations, model, raw_prompt, created_at
    ) VALUES (
        :for_date, :scope, :summary, :recommendations, :model, :raw_prompt, NOW()
    )
    ON CONFLICT (for_date, scope)
    DO UPDATE SET
        summary = EXCLUDED.summary,
        recommendations = EXCLUDED.recommendations,
        model = EXCLUDED.model,
        raw_prompt = EXCLUDED.raw_prompt,
        created_at = NOW()
    """
    
    try:
        execute_query(query, row)
        logger.info(f"Upserted daily_recommendations for {row['for_date']} scope {row['scope']}")
    except Exception as e:
        logger.error(f"Failed to upsert daily_recommendations: {e}")
        raise


def get_daily_metrics(trade_date: str, symbols: Optional[list] = None) -> list:
    """Get daily metrics for a specific date and symbols."""
    base_query = """
    SELECT trade_date, symbol, return_pct, ma7, ma30, rsi14, vol7, high20, low20
    FROM daily_metrics 
    WHERE trade_date = :trade_date
    """
    
    params = {"trade_date": trade_date}
    
    if symbols:
        placeholders = ",".join([f":symbol_{i}" for i in range(len(symbols))])
        base_query += f" AND symbol IN ({placeholders})"
        for i, symbol in enumerate(symbols):
            params[f"symbol_{i}"] = symbol
    
    try:
        result = execute_query(base_query, params)
        return [dict(row._mapping) for row in result.fetchall()]
    except Exception as e:
        logger.error(f"Failed to get daily_metrics: {e}")
        raise


def get_daily_prices(trade_date: str, symbols: Optional[list] = None) -> list:
    """Get daily prices for a specific date and symbols."""
    base_query = """
    SELECT trade_date, symbol, open, high, low, close, adj_close, volume, source
    FROM daily_prices 
    WHERE trade_date = :trade_date
    """
    
    params = {"trade_date": trade_date}
    
    if symbols:
        placeholders = ",".join([f":symbol_{i}" for i in range(len(symbols))])
        base_query += f" AND symbol IN ({placeholders})"
        for i, symbol in enumerate(symbols):
            params[f"symbol_{i}"] = symbol
    
    try:
        result = execute_query(base_query, params)
        return [dict(row._mapping) for row in result.fetchall()]
    except Exception as e:
        logger.error(f"Failed to get daily_prices: {e}")
        raise


def get_daily_recommendations(for_date: str, scope: str = "portfolio") -> Optional[Dict[str, Any]]:
    """Get daily recommendations for a specific date and scope."""
    query = """
    SELECT for_date, scope, summary, recommendations, model, created_at
    FROM daily_recommendations 
    WHERE for_date = :for_date AND scope = :scope
    """
    
    try:
        result = execute_query(query, {"for_date": for_date, "scope": scope})
        row = result.fetchone()
        return dict(row._mapping) if row else None
    except Exception as e:
        logger.error(f"Failed to get daily_recommendations: {e}")
        raise
