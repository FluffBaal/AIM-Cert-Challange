.PHONY: install install-dev format lint type-check test clean docker-build docker-up docker-down

# Install production dependencies for all services
install:
	cd backend/api && uv pip install -e .
	cd backend/ingestion && uv pip install -e .
	cd backend/evaluation && uv pip install -e .

# Install development dependencies
install-dev:
	uv pip install -e ".[dev]"
	cd backend/api && uv pip install -e ".[dev]"
	cd backend/ingestion && uv pip install -e ".[dev]"
	cd backend/evaluation && uv pip install -e ".[dev]"
	pre-commit install

# Format code
format:
	uv run black backend/
	uv run ruff check --fix backend/

# Lint code
lint:
	uv run ruff check backend/
	uv run black --check backend/

# Type checking
type-check:
	uv run mypy backend/

# Run tests
test:
	uv run pytest backend/api/tests -v
	uv run pytest backend/ingestion/tests -v
	uv run pytest backend/evaluation/tests -v

# Clean up
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +

# Docker commands
docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

# Development shortcuts
dev-api:
	cd backend/api && uv run uvicorn main:app --reload --port 8000

dev-frontend:
	cd frontend && npm run dev

# Data ingestion
ingest:
	docker-compose run ingestion

# Full setup
setup: install-dev docker-build
	@echo "Setup complete! Run 'make docker-up' to start all services."