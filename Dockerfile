# Use official Python slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

# Azure App Service sets PORT env variable; default to 8000
# WORKERS controls the number of uvicorn worker processes (default: 4)
ENV PORT=8000
ENV WORKERS=4

EXPOSE 8000

# Start the application
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers ${WORKERS}
