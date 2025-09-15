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
            # Log the query for debugging
            param_str = str(params) if params else "None"
            logger.debug(f"Executing query: {query}\nParams: {param_str}")
            
            # Execute the query
            result = conn.execute(text(query), params or {})
            
            # Explicitly commit the transaction
            conn.commit()
            
            logger.debug(f"Query executed successfully")
            return result
        except SQLAlchemyError as e:
            logger.error(f"Database query failed: {e}")
            conn.rollback()
            raise


def upsert_daily_prices(row: Dict[str, Any]) -> None:
    """Upsert a row into daily_prices table."""
    # Ensure created_at is included in the row
    if 'created_at' not in row:
        from datetime import datetime
        row['created_at'] = datetime.utcnow()
    
    # Log the data being inserted for debugging
    logger.info(f"Upserting daily_prices for {row['symbol']} on {row['trade_date']}: "
               f"open={row['open']}, close={row['close']}, volume={row['volume']}")
    
    query = """
    INSERT INTO daily_prices (
        trade_date, symbol, open, high, low, close, adj_close, 
        volume, source, raw_json, created_at
    ) VALUES (
        :trade_date, :symbol, :open, :high, :low, :close, :adj_close,
        :volume, :source, :raw_json, :created_at
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
        created_at = EXCLUDED.created_at
    """
    
    try:
        execute_query(query, row)
        logger.info(f"Successfully upserted daily_prices for {row['symbol']} on {row['trade_date']}")
        
        # Verify the data was inserted
        verify_query = """
        SELECT * FROM daily_prices 
        WHERE symbol = :symbol AND trade_date = :trade_date
        """
        result = execute_query(verify_query, {"symbol": row["symbol"], "trade_date": row["trade_date"]})
        if result.rowcount > 0:
            logger.info(f"Verified data insertion for {row['symbol']} on {row['trade_date']}")
        else:
            logger.warning(f"Data verification failed for {row['symbol']} on {row['trade_date']}")
            
    except Exception as e:
        logger.error(f"Failed to upsert daily_prices for {row['symbol']}: {e}")
        raise


def upsert_daily_metrics(row: Dict[str, Any]) -> None:
    """Upsert a row into daily_metrics table."""
    # Ensure created_at is included in the row
    if 'created_at' not in row:
        from datetime import datetime
        row['created_at'] = datetime.utcnow()
    
    # Log the data being inserted for debugging
    logger.info(f"Upserting daily_metrics for {row['symbol']} on {row['trade_date']}: "
               f"return_pct={row['return_pct']}")
    
    query = """
    INSERT INTO daily_metrics (
        trade_date, symbol, return_pct, ma7, ma30, rsi14, 
        vol7, high20, low20, created_at
    ) VALUES (
        :trade_date, :symbol, :return_pct, :ma7, :ma30, :rsi14,
        :vol7, :high20, :low20, :created_at
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
        created_at = EXCLUDED.created_at
    """
    
    try:
        execute_query(query, row)
        logger.info(f"Successfully upserted daily_metrics for {row['symbol']} on {row['trade_date']}")
        
        # Verify the data was inserted
        verify_query = """
        SELECT * FROM daily_metrics 
        WHERE symbol = :symbol AND trade_date = :trade_date
        """
        result = execute_query(verify_query, {"symbol": row["symbol"], "trade_date": row["trade_date"]})
        if result.rowcount > 0:
            logger.info(f"Verified metrics insertion for {row['symbol']} on {row['trade_date']}")
        else:
            logger.warning(f"Metrics verification failed for {row['symbol']} on {row['trade_date']}")
            
    except Exception as e:
        logger.error(f"Failed to upsert daily_metrics for {row['symbol']}: {e}")
        raise


def upsert_daily_recommendations(row: Dict[str, Any]) -> None:
    """Upsert a row into daily_recommendations table."""
    # Ensure created_at is included in the row
    if 'created_at' not in row:
        from datetime import datetime
        row['created_at'] = datetime.utcnow()
    
    # Log the data being inserted for debugging
    logger.info(f"Upserting daily_recommendations for {row['for_date']} scope {row['scope']}")
    
    query = """
    INSERT INTO daily_recommendations (
        for_date, scope, summary, recommendations, model, raw_prompt, created_at
    ) VALUES (
        :for_date, :scope, :summary, :recommendations, :model, :raw_prompt, :created_at
    )
    ON CONFLICT (for_date, scope)
    DO UPDATE SET
        summary = EXCLUDED.summary,
        recommendations = EXCLUDED.recommendations,
        model = EXCLUDED.model,
        raw_prompt = EXCLUDED.raw_prompt,
        created_at = EXCLUDED.created_at
    """
    
    try:
        execute_query(query, row)
        logger.info(f"Successfully upserted daily_recommendations for {row['for_date']} scope {row['scope']}")
        
        # Verify the data was inserted
        verify_query = """
        SELECT * FROM daily_recommendations 
        WHERE for_date = :for_date AND scope = :scope
        """
        result = execute_query(verify_query, {"for_date": row["for_date"], "scope": row["scope"]})
        if result.rowcount > 0:
            logger.info(f"Verified recommendations insertion for {row['for_date']} scope {row['scope']}")
        else:
            logger.warning(f"Recommendations verification failed for {row['for_date']} scope {row['scope']}")
            
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


def get_latest_daily_price(symbol: str) -> Optional[Dict[str, Any]]:
    """Get the latest daily price for a symbol."""
    query = """
    SELECT trade_date, symbol, open, high, low, close, adj_close, volume, source
    FROM daily_prices 
    WHERE symbol = :symbol
    ORDER BY trade_date DESC
    LIMIT 1
    """
    
    try:
        result = execute_query(query, {"symbol": symbol})
        row = result.fetchone()
        return dict(row._mapping) if row else None
    except Exception as e:
        logger.error(f"Failed to get latest daily price for {symbol}: {e}")
        return None
