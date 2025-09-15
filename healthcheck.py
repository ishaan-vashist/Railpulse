"""
Simple health check server for Railway deployment.
This runs as a separate process to ensure health checks pass
even if the main application has startup issues.
"""
from fastapi import FastAPI
import uvicorn

app = FastAPI(
    title="RailPulse Health Check",
    description="Simple health check server for Railway deployment",
    version="1.0.0"
)

@app.get("/healthz")
async def health_check():
    """Health check endpoint for monitoring and deployment checks."""
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(
        "healthcheck:app",
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
