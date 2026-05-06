.PHONY: up down logs test fmt lint install shell-ts shell-pg

up:
	docker compose -f infra/docker-compose.yml up -d --build

down:
	docker compose -f infra/docker-compose.yml down -v

logs:
	docker compose -f infra/docker-compose.yml logs -f backend

install:
	cd backend && python3 -m venv .venv && .venv/bin/pip install -e "." && .venv/bin/pip install pytest pytest-asyncio pytest-cov ruff "testcontainers[postgres]"

test:
	cd backend && .venv/bin/pytest -v --cov=src/cryptoswarm --cov-report=term-missing

test-unit:
	cd backend && .venv/bin/pytest tests/unit/ -v

test-integration:
	cd backend && .venv/bin/pytest tests/integration/ -v

fmt:
	cd backend && .venv/bin/ruff format src tests

lint:
	cd backend && .venv/bin/ruff check src tests

shell-ts:
	docker compose -f infra/docker-compose.yml exec timescale psql -U postgres -d cryptoswarm_ts

shell-pg:
	docker compose -f infra/docker-compose.yml exec postgres psql -U postgres -d cryptoswarm
