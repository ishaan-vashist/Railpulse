"""LLM integration for market analysis and recommendations."""
import json
import logging
from datetime import date
from typing import Dict, Any, Optional
from openai import OpenAI
from .config import settings
from .db import upsert_daily_recommendations
from .etl import get_portfolio_summary

logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = OpenAI(api_key=settings.openai_api_key)


def create_market_prompt(portfolio_data: Dict[str, Any]) -> str:
    """
    Create a structured prompt for market analysis.
    
    Args:
        portfolio_data: Portfolio summary data from get_portfolio_summary
    
    Returns:
        Formatted prompt string
    """
    trade_date = portfolio_data["trade_date"]
    
    prompt = f"""You are a market analyst. As of today {trade_date} (Asia/Kolkata timezone), here are intraday-aggregated KPIs:

Portfolio Summary:
- Date: {trade_date}
- Symbols analyzed: {portfolio_data['symbols_count']}
- Portfolio return (equal-weighted): {portfolio_data['portfolio_return']}%

"""
    
    if portfolio_data['top_gainer']:
        prompt += f"- Top gainer: {portfolio_data['top_gainer']['symbol']} (+{portfolio_data['top_gainer']['return_pct']}%)\n"
    
    if portfolio_data['top_loser']:
        prompt += f"- Top loser: {portfolio_data['top_loser']['symbol']} ({portfolio_data['top_loser']['return_pct']}%)\n"
    
    prompt += f"""
Individual Symbol Performance:
"""
    
    for metric in portfolio_data['metrics'][:10]:  # Limit to first 10 symbols to keep prompt concise
        if metric['return_pct'] is not None:
            prompt += f"- {metric['symbol']}: {metric['return_pct']:.2f}%\n"
    
    prompt += """
Write a 3-4 sentence summary in plain language and provide 3 succinct, actionable recommendations.
Focus on risk, momentum, and simple next steps. Avoid speculation beyond the data.

Format your response as JSON with the following structure:
{
    "summary": "Your 3-4 sentence market summary here",
    "recommendations": [
        "First actionable recommendation",
        "Second actionable recommendation", 
        "Third actionable recommendation"
    ]
}"""
    
    return prompt


def call_llm_analysis(portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Call OpenAI API to generate market analysis.
    
    Args:
        portfolio_data: Portfolio summary data
    
    Returns:
        Dictionary with summary and recommendations
    """
    try:
        prompt = create_market_prompt(portfolio_data)
        
        logger.info("Calling OpenAI API for market analysis...")
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional market analyst. Provide concise, data-driven insights and actionable recommendations. Always respond in valid JSON format."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            max_tokens=500,
            temperature=0.3,
            top_p=1.0
        )
        
        # Extract and parse the response
        content = response.choices[0].message.content.strip()
        
        try:
            # Try to parse as JSON
            analysis = json.loads(content)
            
            # Validate required fields
            if "summary" not in analysis or "recommendations" not in analysis:
                raise ValueError("Missing required fields in LLM response")
            
            if not isinstance(analysis["recommendations"], list):
                raise ValueError("Recommendations must be a list")
            
            return analysis
            
        except json.JSONDecodeError:
            # If JSON parsing fails, create a structured response
            logger.warning("LLM response was not valid JSON, creating structured response")
            return {
                "summary": content[:300] + "..." if len(content) > 300 else content,
                "recommendations": [
                    "Monitor portfolio performance closely",
                    "Consider rebalancing based on today's movements", 
                    "Review risk exposure for volatile positions"
                ]
            }
        
    except Exception as e:
        logger.error(f"Failed to get LLM analysis: {e}")
        # Return a fallback analysis
        return {
            "summary": f"Portfolio analysis for {portfolio_data['trade_date']}: {portfolio_data['symbols_count']} symbols analyzed with average return of {portfolio_data['portfolio_return']}%. Market data processing completed successfully.",
            "recommendations": [
                "Review individual symbol performance for outliers",
                "Monitor market conditions for continued volatility",
                "Consider portfolio rebalancing if needed"
            ]
        }


def generate_and_store_recommendations(trade_date: date) -> Dict[str, Any]:
    """
    Generate LLM recommendations and store them in the database.
    
    Args:
        trade_date: Date to analyze
    
    Returns:
        Dictionary with the generated analysis
    """
    try:
        logger.info(f"Generating LLM recommendations for {trade_date}")
        
        # Get portfolio summary data
        portfolio_data = get_portfolio_summary(trade_date)
        
        if portfolio_data['symbols_count'] == 0:
            logger.warning(f"No portfolio data available for {trade_date}")
            
            # Try to fetch some sample data for demonstration purposes
            from .db import get_latest_daily_price, execute_query
            from .config import settings
            
            # Get available symbols
            symbols = settings.get_tickers_list()
            sample_data = []
            
            for symbol in symbols[:3]:  # Limit to 3 symbols to avoid overloading
                try:
                    latest = get_latest_daily_price(symbol)
                    if latest:
                        sample_data.append({
                            "symbol": symbol,
                            "latest_date": str(latest["trade_date"]),
                            "latest_price": latest["close"]
                        })
                except Exception as e:
                    logger.error(f"Error fetching sample data for {symbol}: {e}")
            
            if sample_data:
                return {
                    "summary": f"No market data available for {trade_date}, but showing sample data from most recent available dates.",
                    "recommendations": [
                        f"Consider fetching data for {trade_date} using the ETL pipeline",
                        f"Latest data available for {', '.join([d['symbol'] for d in sample_data])}",
                        f"Check API rate limits and data availability for the requested date"
                    ]
                }
            else:
                return {
                    "summary": f"No market data available for analysis on {trade_date}.",
                    "recommendations": [
                        "Check data sources for availability",
                        "Verify market trading hours",
                        "Review system configuration"
                    ]
                }
            
        
        # Generate LLM analysis
        analysis = call_llm_analysis(portfolio_data)
        
        # Prepare data for database storage
        recommendations_row = {
            "for_date": trade_date,
            "scope": "portfolio",
            "summary": analysis["summary"],
            "recommendations": json.dumps(analysis["recommendations"]),
            "model": "gpt-3.5-turbo",
            "raw_prompt": create_market_prompt(portfolio_data)
        }
        
        # Store in database
        upsert_daily_recommendations(recommendations_row)
        
        logger.info(f"Successfully generated and stored recommendations for {trade_date}")
        
        return {
            "date": str(trade_date),
            "summary": analysis["summary"],
            "recommendations": analysis["recommendations"],
            "portfolio_data": portfolio_data
        }
        
    except Exception as e:
        logger.error(f"Failed to generate recommendations for {trade_date}: {e}")
        raise


def get_recommendations_summary(trade_date: date) -> Optional[Dict[str, Any]]:
    """
    Get stored recommendations for a specific date.
    
    Args:
        trade_date: Date to retrieve recommendations for
    
    Returns:
        Dictionary with recommendations or None if not found
    """
    from .db import get_daily_recommendations
    
    try:
        recommendations = get_daily_recommendations(str(trade_date), "portfolio")
        
        if not recommendations:
            return None
        
        return {
            "date": str(recommendations["for_date"]),
            "scope": recommendations["scope"],
            "summary": recommendations["summary"],
            "recommendations": json.loads(recommendations["recommendations"]),
            "model": recommendations["model"],
            "created_at": recommendations["created_at"]
        }
        
    except Exception as e:
        logger.error(f"Failed to get recommendations for {trade_date}: {e}")
        raise
