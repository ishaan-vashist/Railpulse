#!/usr/bin/env python3
"""CLI script to run the complete ETL pipeline for today's data."""
import sys
import os
import logging
from datetime import datetime

# Add the parent directory to the path so we can import from app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import settings, today_ist_date
from app.etl import run_today
from app.llm import generate_and_store_recommendations

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def main():
    """Main CLI function to run the ETL pipeline."""
    try:
        logger.info("=" * 60)
        logger.info("RailPulse ETL Pipeline - CLI Runner")
        logger.info("=" * 60)
        
        # Validate configuration
        logger.info("Validating configuration...")
        
        symbols = settings.get_tickers_list()
        if not symbols:
            logger.error("No symbols configured in TICKERS environment variable")
            sys.exit(1)
        
        logger.info(f"Configured symbols: {', '.join(symbols)}")
        logger.info(f"Alpha Vantage interval: {settings.alphavantage_interval}")
        logger.info(f"Target date: {today_ist_date()} (Asia/Kolkata)")
        
        # Run ETL pipeline
        logger.info("\n" + "=" * 40)
        logger.info("STEP 1: Running ETL Pipeline")
        logger.info("=" * 40)
        
        etl_results = run_today(symbols)
        
        # Print ETL results
        logger.info(f"ETL Results for {etl_results['trade_date']}:")
        logger.info(f"  - Symbols requested: {etl_results['symbols_requested']}")
        logger.info(f"  - Symbols processed: {etl_results['symbols_processed']}")
        logger.info(f"  - Symbols failed: {etl_results['symbols_failed']}")
        logger.info(f"  - Symbols with no data: {etl_results['symbols_no_data']}")
        
        if etl_results['processed_symbols']:
            logger.info(f"  - Successfully processed: {', '.join(etl_results['processed_symbols'])}")
        
        if etl_results['failed_symbols']:
            logger.warning(f"  - Failed symbols: {', '.join(etl_results['failed_symbols'])}")
        
        if etl_results['no_data_symbols']:
            logger.warning(f"  - No data symbols: {', '.join(etl_results['no_data_symbols'])}")
        
        # Run LLM analysis
        logger.info("\n" + "=" * 40)
        logger.info("STEP 2: Generating LLM Analysis")
        logger.info("=" * 40)
        
        trade_date = today_ist_date()
        
        try:
            llm_results = generate_and_store_recommendations(trade_date)
            
            logger.info("LLM Analysis completed successfully:")
            logger.info(f"  - Summary: {llm_results['summary'][:100]}...")
            logger.info(f"  - Recommendations: {len(llm_results['recommendations'])} generated")
            
            # Print recommendations
            for i, rec in enumerate(llm_results['recommendations'], 1):
                logger.info(f"    {i}. {rec}")
            
        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            logger.warning("Continuing without LLM analysis...")
        
        # Final summary
        logger.info("\n" + "=" * 40)
        logger.info("PIPELINE SUMMARY")
        logger.info("=" * 40)
        
        success_rate = (etl_results['symbols_processed'] / etl_results['symbols_requested']) * 100
        logger.info(f"Success rate: {success_rate:.1f}% ({etl_results['symbols_processed']}/{etl_results['symbols_requested']})")
        
        if etl_results['symbols_processed'] > 0:
            logger.info("✅ Pipeline completed successfully")
            logger.info(f"Data available via API endpoints for date: {trade_date}")
        else:
            logger.error("❌ Pipeline failed - no symbols processed successfully")
            sys.exit(1)
        
    except KeyboardInterrupt:
        logger.info("\nPipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
