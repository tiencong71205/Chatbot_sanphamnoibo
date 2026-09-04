.PHONY: help install dev lint test build up down logs health ingest

PYTHON := python3
COMPOSE := docker compose -p vhomenex-rag-v2

help:
	@echo "Available commands:"
	@echo "  make install      Install dependencies"
	@echo "  make lint         Run ruff + black check"
	@echo "  make test         Run pytest"
	@echo "  make build        Build Docker images (no-cache)"
	@echo "  make up           Start Docker services"
	@echo "  make down         Stop Docker services"
	@echo "  make logs         Show container logs"
	@echo "  make health       Check /health endpoint"
	@echo "  make ingest       Ingest documents"
	@echo "  make ingest-dry   Dry run ingest"

install:
	pip install -r requirements-dev.txt

lint:
	ruff check .
	black --check .

test:
	pytest -v tests/

build:
	$(COMPOSE) build --no-cache

up:
	cp .env.docker.example .env.docker 2>/dev/null || true
	$(COMPOSE) up -d

down:
	$(COMPOSE) down

logs:
	$(COMPOSE) logs --tail=200 -f

health:
	curl -s http://localhost:8000/health | python3 -m json.tool

ingest:
	$(PYTHON) scripts/ingest_documents.py

ingest-dry:
	$(PYTHON) scripts/ingest_documents.py --dry-run

check-services:
	$(PYTHON) scripts/check_services.py
