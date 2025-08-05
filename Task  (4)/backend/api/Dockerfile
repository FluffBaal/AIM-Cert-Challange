FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY api/pyproject.toml .
COPY api/README.md .
COPY api/ ./api/
COPY agents/ ./agents/
COPY ingestion/ ./ingestion/

# Install dependencies with pip
RUN pip install --upgrade pip && pip install -e .

# Install additional dependencies from requirements.txt if it exists
RUN if [ -f api/requirements.txt ]; then pip install -r api/requirements.txt; fi

# Create state directory
RUN mkdir -p /app/state

# Set Python path
ENV PYTHONPATH=/app:$PYTHONPATH

# Expose port
EXPOSE 8000

# Run the application with increased timeouts for AI processing
# Note: --reload is incompatible with --workers, so we use reload for dev
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload", "--timeout-keep-alive", "180"]