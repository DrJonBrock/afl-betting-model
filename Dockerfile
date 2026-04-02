# Dockerfile for AFL Betting Model API
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*

# Copy dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port (Railway sets PORT env)
EXPOSE 8000

# Run the API
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
