FROM python:3.11-slim

WORKDIR /app

# Install system dependencies and create non-root user
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && addgroup --system appgroup \
    && adduser --system --ingroup appgroup appuser

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code (explicit paths to avoid copying sensitive files)
COPY app/ app/
COPY model_artifacts/ model_artifacts/

# Create directory for database if needed and ensure ownership
RUN mkdir -p /app/data \
    && chown -R appuser:appgroup /app

# Set environment variables
ENV SKIP_FINBERT_PRELOAD=true
ENV SCRAPER_ENABLED=false
ENV SENTIMENT_MODE=simple

# Run as non-root user
USER appuser

# Expose port (Hugging Face Spaces uses 7860)
EXPOSE 7860

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
