# Deploying RailPulse to Railway

This guide explains how to deploy the RailPulse service to Railway with scheduled data collection.

## Prerequisites

1. [Railway account](https://railway.app/)
2. Git repository with your RailPulse code

## Deployment Steps

### 1. Initial Setup

1. Log in to [Railway](https://railway.app/)
2. Click "New Project" → "Deploy from GitHub repo"
3. Select your RailPulse repository
4. Railway will automatically detect your Python project

### 2. Configure Environment Variables

Add the following environment variables in the Railway dashboard:

```
DATABASE_URL=postgresql://username:password@hostname:port/database
ALPHAVANTAGE_API_KEY=your_api_key
OPENAI_API_KEY=your_openai_key
TICKERS_LIST=AAPL,MSFT,GOOGL,AMZN,TSLA
ALPHAVANTAGE_CALLS_PER_MINUTE=5
ALPHAVANTAGE_MAX_DAILY_CALLS=20
TIMEZONE=Asia/Kolkata
```

### 3. Set Up Scheduled Job

1. In your project dashboard, go to "Settings"
2. Scroll to "Cron Jobs"
3. Click "Add New Cron Job"
4. Configure as follows:
   - **Command**: `python scripts/run_today.py`
   - **Schedule**: `30 11 * * 1-5` (runs at 11:30 UTC / 5:00 PM IST on weekdays)
   - **Timezone**: `UTC`

### 4. Verify Deployment

1. Monitor the deployment logs in the Railway dashboard
2. Check that your scheduled job runs at the configured time
3. Verify data is being collected in your database

## Troubleshooting

If your scheduled job fails:

1. Check the logs in the Railway dashboard
2. Verify your environment variables are set correctly
3. Ensure your database is accessible from Railway
4. Check Alpha Vantage API limits

## Updating Your Deployment

To update your deployment:

1. Push changes to your GitHub repository
2. Railway will automatically redeploy your application

## Monitoring

Railway provides built-in monitoring for your application:

1. View logs in the Railway dashboard
2. Set up notifications for job failures
3. Monitor resource usage in the dashboard
