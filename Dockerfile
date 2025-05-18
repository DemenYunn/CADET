# Use official Python image as base
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Make entrypoint script executable
RUN chmod +x /app/docker-entrypoint.sh

# Set the entrypoint script
ENTRYPOINT ["/app/docker-entrypoint.sh"]

# Default command
CMD ["python", "code_evolver.py"]
