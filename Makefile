.PHONY: help install test test-cov test-fast clean lint format setup dev run

help:
	@echo "Available commands:"
	@echo "  make install     Install production dependencies"
	@echo "  make dev         Install development dependencies"
	@echo "  make test        Run all tests"
	@echo "  make test-cov    Run tests with coverage report"
	@echo "  make test-fast   Run tests in parallel"
	@echo "  make test-watch  Run tests in watch mode"
	@echo "  make lint        Run linters"
	@echo "  make format      Format code with black and isort"
	@echo "  make clean       Clean up temporary files"
	@echo "  make setup       Complete setup (install + dev)"
	@echo "  make run         Run the application"

install:
	pip install -r requirements.txt

dev:
	pip install -r requirements-dev.txt

setup: install dev
	@echo "Setup complete!"

test:
	pytest tests/ -v

test-cov:
	pytest tests/ --cov=app --cov-report=html --cov-report=term-missing
	@echo "Coverage report: htmlcov/index.html"

test-fast:
	pytest tests/ -n auto

test-watch:
	pytest-watch

test-api:
	pytest tests/test_api_endpoints.py -v

test-services:
	pytest tests/test_services.py -v

test-models:
	pytest tests/test_models.py -v

test-integration:
	pytest tests/test_integration.py -v

lint:
	flake8 app tests
	mypy app --ignore-missing-imports

format:
	black app tests
	isort app tests

clean:
	rm -rf .pytest_cache htmlcov .coverage coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

run:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

docker-build:
	docker build -t oil-price-api .

docker-run:
	docker run -p 8000:8000 oil-price-api

check: lint test
	@echo "All checks passed!"
