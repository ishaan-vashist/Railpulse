FROM python:3.10-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements-railway.txt .

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-railway.txt

# Copy the rest of the application
COPY . .

# Hardcode port to avoid environment variable issues
EXPOSE 8000

# Command to run the application with hardcoded port
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
