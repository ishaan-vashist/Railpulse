# RailPulse Frontend Integration Guide

## Table of Contents
1. [Introduction](#introduction)
2. [API Overview](#api-overview)
3. [Authentication](#authentication)
4. [API Endpoints](#api-endpoints)
   - [Health Check](#health-check)
   - [Market Metrics](#market-metrics)
   - [Recommendations](#recommendations)
   - [Admin Operations](#admin-operations)
5. [Error Handling](#error-handling)
6. [Data Visualization](#data-visualization)
7. [Sample Code](#sample-code)
   - [React/Next.js](#reactnextjs)
   - [Data Fetching](#data-fetching)
   - [State Management](#state-management)
8. [Best Practices](#best-practices)
9. [Deployment](#deployment)

## Introduction

This guide provides detailed instructions for integrating a frontend application with the RailPulse Market Data API. The API provides financial market data, metrics, and AI-powered recommendations that can be visualized in a dashboard.

### Technology Stack Recommendations

- **Frontend Framework**: Next.js (React)
- **Data Fetching**: SWR or React Query
- **UI Components**: Tailwind CSS with Headless UI or Chakra UI
- **Charts**: Recharts, Chart.js, or D3.js
- **State Management**: React Context API or Redux Toolkit

## API Overview

The RailPulse API is a RESTful service built with FastAPI that provides:

- Daily price data (OHLCV) for configured financial instruments
- Technical metrics and indicators (return %, moving averages, etc.)
- AI-generated market analysis and recommendations
- Admin operations for triggering data updates

The API follows standard REST conventions and returns JSON responses.

## Authentication

### Public Endpoints

The following endpoints are publicly accessible without authentication:
- `/healthz` - Health check
- `/` - API information
- `/metrics` - Market metrics data

### Protected Endpoints

Admin endpoints require authentication using an API key:

```
/admin/run-today
/admin/generate-recommendations
```

Authentication is performed using one of two methods:

1. **Header-based authentication**:
   ```
   x-app-secret: YOUR_APP_SECRET
   ```

2. **Query parameter**:
   ```
   ?app_secret=YOUR_APP_SECRET
   ```

The `APP_SECRET` is configured in the server's environment variables.

## API Endpoints

### Health Check

**Endpoint**: `GET /healthz`

**Description**: Checks if the API is operational.

**Response Example**:
```json
{
  "status": "ok",
  "timestamp": "2025-09-16T02:21:23+05:30"
}
```

### Market Metrics

**Endpoint**: `GET /metrics`

**Description**: Retrieves market metrics for specified date and symbols.

**Query Parameters**:
- `date` (optional): Date in YYYY-MM-DD format (defaults to today IST)
- `symbol` (optional): Comma-separated list of symbols (defaults to all available)

**Response Example**:
```json
{
  "date": "2025-09-16",
  "symbols": ["AAPL", "MSFT", "SPY", "INFY", "RELIANCE"],
  "prices": [
    {
      "trade_date": "2025-09-16",
      "symbol": "AAPL",
      "open": 222.50,
      "high": 224.95,
      "low": 221.60,
      "close": 222.77,
      "adj_close": 222.77,
      "volume": 46388779,
      "source": "alphavantage"
    },
    // More price entries...
  ],
  "metrics": [
    {
      "trade_date": "2025-09-16",
      "symbol": "AAPL",
      "return_pct": 0.84,
      "ma7": 224.9,
      "ma30": 219.2,
      "rsi14": 58.7,
      "vol7": 1.3,
      "high20": 229.1,
      "low20": 214.4
    },
    // More metric entries...
  ]
}
```

### Recommendations

**Endpoint**: `GET /recommendations`

**Description**: Retrieves AI-generated market analysis and recommendations.

**Query Parameters**:
- `date` (optional): Date in YYYY-MM-DD format (defaults to today IST)

**Response Example**:
```json
{
  "date": "2025-09-16",
  "scope": "portfolio",
  "summary": "Major indices closed higher with tech leadership; AAPL and MSFT showed strong momentum while INFY underperformed the broader market. Overall portfolio return was positive at 1.2% for the day.",
  "recommendations": [
    "Consider increasing allocation to tech sector given the strong momentum in AAPL and MSFT",
    "Monitor INFY closely as it continues to underperform relative to the portfolio average",
    "Set trailing stops on positions that have exceeded their 20-day high to protect gains"
  ],
  "created_at": "2025-09-16T17:30:45+05:30"
}
```

### Admin Operations

#### Run ETL Pipeline

**Endpoint**: `POST /admin/run-today`

**Description**: Triggers the ETL pipeline to fetch and process today's market data.

**Authentication**: Required (APP_SECRET)

**Query Parameters**:
- `force_refresh` (optional): Boolean to force refresh data from API instead of using cache

**Request Example**:
```http
POST /admin/run-today?app_secret=YOUR_APP_SECRET&force_refresh=true
```

**Response Example**:
```json
{
  "success": true,
  "message": "ETL completed. Processed 5 symbols successfully.",
  "results": {
    "trade_date": "2025-09-16",
    "symbols_requested": 5,
    "symbols_processed": 5,
    "symbols_failed": 0,
    "symbols_no_data": 0,
    "processed_symbols": ["AAPL", "MSFT", "SPY", "INFY", "RELIANCE"],
    "failed_symbols": [],
    "no_data_symbols": [],
    "llm_analysis": {
      "success": true,
      "summary": "Market showed positive momentum across sectors",
      "recommendations_count": 3
    }
  }
}
```

#### Generate Recommendations

**Endpoint**: `POST /admin/generate-recommendations`

**Description**: Generates recommendations for a specific date.

**Authentication**: Required (APP_SECRET)

**Query Parameters**:
- `date`: Date in YYYY-MM-DD format
- `force` (optional): Boolean to force regeneration even if recommendations already exist

**Request Example**:
```http
POST /admin/generate-recommendations?app_secret=YOUR_APP_SECRET&date=2025-09-16&force=true
```

**Response Example**:
```json
{
  "status": "success",
  "message": "Successfully generated recommendations for 2025-09-16",
  "recommendations": {
    "date": "2025-09-16",
    "summary": "Market showed positive momentum across sectors",
    "recommendations": [
      "Consider increasing allocation to tech sector given the strong momentum",
      "Monitor underperforming positions closely",
      "Set trailing stops on positions that have exceeded their 20-day high"
    ],
    "portfolio_data": {
      "symbols_count": 5,
      "portfolio_return": 1.2,
      "top_gainer": {
        "symbol": "AAPL",
        "return_pct": 2.5
      },
      "top_loser": {
        "symbol": "INFY",
        "return_pct": -0.8
      }
    }
  }
}
```

## Error Handling

The API uses standard HTTP status codes to indicate success or failure:

- **200 OK**: Request succeeded
- **400 Bad Request**: Invalid request parameters
- **401 Unauthorized**: Missing or invalid authentication
- **403 Forbidden**: Authentication provided but not authorized
- **404 Not Found**: Resource not found
- **500 Internal Server Error**: Server-side error

Error responses follow this format:

```json
{
  "detail": "Error message describing what went wrong"
}
```

### Frontend Error Handling Best Practices

1. **Graceful Degradation**: Display fallback UI when API calls fail
2. **Retry Logic**: Implement exponential backoff for transient errors
3. **User Feedback**: Show loading states and error messages
4. **Logging**: Log errors to help with debugging

Example error handling in React:

```jsx
const fetchData = async () => {
  setLoading(true);
  setError(null);
  
  try {
    const response = await fetch('/api/metrics');
    
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'An error occurred');
    }
    
    const data = await response.json();
    setData(data);
  } catch (error) {
    setError(error.message);
    console.error('Error fetching metrics:', error);
  } finally {
    setLoading(false);
  }
};
```

## Data Visualization

### Recommended Chart Types

1. **Price Data**:
   - Candlestick charts for OHLC data
   - Line charts for closing prices over time
   - Volume bars below price charts

2. **Technical Indicators**:
   - Line charts for moving averages overlaid on price charts
   - Oscillator charts for RSI
   - Bar charts for comparative returns

3. **Portfolio Analysis**:
   - Pie charts for portfolio allocation
   - Bar charts for comparing symbol performance
   - Heat maps for correlation analysis

### Visualization Libraries

1. **Recharts**: React-based, easy to use for simple charts
2. **Chart.js**: Canvas-based, good balance of features and simplicity
3. **D3.js**: Most powerful and flexible, but steeper learning curve
4. **TradingView Lightweight Charts**: Specialized for financial charts

## Sample Code

### React/Next.js

#### API Client Setup

Create a dedicated API client to handle all API interactions:

```jsx
// lib/api.js
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
const APP_SECRET = process.env.NEXT_PUBLIC_APP_SECRET;

export async function fetchMetrics(date, symbols) {
  const params = new URLSearchParams();
  if (date) params.append('date', date);
  if (symbols) params.append('symbol', symbols.join(','));
  
  const response = await fetch(`${API_BASE_URL}/metrics?${params}`);
  
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to fetch metrics');
  }
  
  return response.json();
}

export async function fetchRecommendations(date) {
  const params = new URLSearchParams();
  if (date) params.append('date', date);
  
  const response = await fetch(`${API_BASE_URL}/recommendations?${params}`);
  
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to fetch recommendations');
  }
  
  return response.json();
}

export async function runETLPipeline(forceRefresh = false) {
  const params = new URLSearchParams();
  params.append('app_secret', APP_SECRET);
  if (forceRefresh) params.append('force_refresh', 'true');
  
  const response = await fetch(`${API_BASE_URL}/admin/run-today?${params}`, {
    method: 'POST',
  });
  
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to run ETL pipeline');
  }
  
  return response.json();
}

export async function generateRecommendations(date, force = false) {
  const params = new URLSearchParams();
  params.append('app_secret', APP_SECRET);
  params.append('date', date);
  if (force) params.append('force', 'true');
  
  const response = await fetch(`${API_BASE_URL}/admin/generate-recommendations?${params}`, {
    method: 'POST',
  });
  
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to generate recommendations');
  }
  
  return response.json();
}
```

### Data Fetching

Using SWR for data fetching with caching and revalidation:

```jsx
// components/MarketMetrics.jsx
import useSWR from 'swr';
import { fetchMetrics } from '../lib/api';

const fetcher = (url, date, symbols) => fetchMetrics(date, symbols);

export default function MarketMetrics({ date, symbols }) {
  const { data, error, isLoading, mutate } = useSWR(
    ['metrics', date, symbols],
    () => fetcher('metrics', date, symbols),
    {
      revalidateOnFocus: false,
      refreshInterval: 300000, // Refresh every 5 minutes
    }
  );

  if (isLoading) return <div>Loading market data...</div>;
  if (error) return <div>Error loading market data: {error.message}</div>;

  return (
    <div>
      <h2>Market Metrics for {data.date}</h2>
      <button onClick={() => mutate()}>Refresh Data</button>
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {data.symbols.map(symbol => (
          <div key={symbol} className="p-4 border rounded shadow">
            <h3 className="text-xl font-bold">{symbol}</h3>
            
            {/* Find price data for this symbol */}
            {data.prices
              .filter(price => price.symbol === symbol)
              .map(price => (
                <div key={`${symbol}-price`}>
                  <p>Open: {price.open}</p>
                  <p>High: {price.high}</p>
                  <p>Low: {price.low}</p>
                  <p>Close: {price.close}</p>
                  <p>Volume: {price.volume.toLocaleString()}</p>
                </div>
              ))}
            
            {/* Find metrics for this symbol */}
            {data.metrics
              .filter(metric => metric.symbol === symbol)
              .map(metric => (
                <div key={`${symbol}-metrics`} className="mt-2 pt-2 border-t">
                  <p>Return: {metric.return_pct?.toFixed(2)}%</p>
                  <p>MA7: {metric.ma7?.toFixed(2)}</p>
                  <p>RSI14: {metric.rsi14?.toFixed(2)}</p>
                </div>
              ))}
          </div>
        ))}
      </div>
    </div>
  );
}
```

### State Management

Using React Context for global state management:

```jsx
// context/MarketContext.jsx
import { createContext, useContext, useState } from 'react';

const MarketContext = createContext();

export function MarketProvider({ children }) {
  const [selectedDate, setSelectedDate] = useState(
    new Date().toISOString().split('T')[0]
  );
  const [selectedSymbols, setSelectedSymbols] = useState([]);
  const [isAdmin, setIsAdmin] = useState(false);
  
  const value = {
    selectedDate,
    setSelectedDate,
    selectedSymbols,
    setSelectedSymbols,
    isAdmin,
    setIsAdmin,
  };
  
  return (
    <MarketContext.Provider value={value}>
      {children}
    </MarketContext.Provider>
  );
}

export function useMarket() {
  return useContext(MarketContext);
}
```

## Best Practices

### Performance Optimization

1. **Minimize API Calls**:
   - Use SWR or React Query for caching
   - Implement debouncing for user inputs
   - Batch related data requests

2. **Lazy Loading**:
   - Load components and data only when needed
   - Use React.lazy() and Suspense for code splitting

3. **Memoization**:
   - Use React.memo() for expensive components
   - Use useMemo() and useCallback() hooks for computed values and callbacks

### Responsive Design

1. **Mobile-First Approach**:
   - Design for mobile screens first, then enhance for larger screens
   - Use CSS Grid and Flexbox for responsive layouts

2. **Adaptive Charts**:
   - Adjust chart complexity based on screen size
   - Use responsive chart libraries that handle resizing

3. **Touch-Friendly UI**:
   - Ensure interactive elements are large enough for touch
   - Implement swipe gestures for mobile users

### Accessibility

1. **Semantic HTML**:
   - Use proper heading hierarchy (h1, h2, etc.)
   - Use semantic elements (nav, main, section, etc.)

2. **ARIA Attributes**:
   - Add aria-label and aria-describedby where needed
   - Ensure proper focus management

3. **Color Contrast**:
   - Maintain WCAG 2.1 AA compliance for text contrast
   - Don't rely solely on color to convey information

## Deployment

### Next.js Deployment

1. **Vercel** (Recommended):
   ```bash
   # Install Vercel CLI
   npm i -g vercel
   
   # Deploy
   vercel
   ```

2. **Netlify**:
   - Connect your GitHub repository
   - Set build command: `npm run build`
   - Set publish directory: `out`

### Environment Variables

Configure these environment variables in your deployment platform:

```
NEXT_PUBLIC_API_URL=https://your-railpulse-api.railway.app
NEXT_PUBLIC_APP_SECRET=your_app_secret_for_admin_functions
```

### CORS Configuration

The RailPulse API has CORS configured to allow requests from:
- `http://localhost:3000` (development)
- `https://*.vercel.app` (Vercel deployments)
- `https://*.netlify.app` (Netlify deployments)

If you're deploying to a different domain, you'll need to update the CORS configuration in the API's `main.py` file.
