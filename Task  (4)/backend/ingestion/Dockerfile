FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml .
COPY . .

# Install dependencies with pip
RUN pip install --upgrade pip && pip install -e .

# Create data directory
RUN mkdir -p /data

# Run the ingestion script
CMD ["python", "ingest.py", "--dual-mode"]