"""Configuration management for the FastAPI service."""
import os
from datetime import date
from typing import List
import pytz
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Database
    database_url: str
    
    # Alpha Vantage
    alphavantage_api_key: str
    alphavantage_interval: str = "5min"
    alphavantage_max_daily_calls: int = 20
    alphavantage_calls_per_minute: int = 5
    
    # OpenAI
    openai_api_key: str
    
    # Tickers
    tickers: str  # Comma-separated string
    
    # Timezone
    timezone: str = "Asia/Kolkata"
    
    # Security
    app_secret: str
    
    # Logging
    log_level: str = "INFO"
    
    # Development
    use_mock_data: bool = False
    use_mock_data_on_error: bool = True
    
    class Config:
        env_file = ".env"
        case_sensitive = False

    def get_tickers_list(self) -> List[str]:
        """Parse tickers string into a list."""
        return [ticker.strip() for ticker in self.tickers.split(",") if ticker.strip()]


# Global settings instance
settings = Settings()

# Timezone instance
IST = pytz.timezone(settings.timezone)


def today_ist_date() -> date:
    """Get today's date in IST timezone."""
    from datetime import datetime
    utc_now = datetime.utcnow().replace(tzinfo=pytz.UTC)
    ist_now = utc_now.astimezone(IST)
    return ist_now.date()


def validate_interval(interval: str) -> bool:
    """Validate Alpha Vantage interval parameter."""
    allowed_intervals = ["1min", "5min", "15min", "30min", "60min"]
    return interval in allowed_intervals
