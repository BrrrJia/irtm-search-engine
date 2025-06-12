# Makefile
.PHONY: up down dev test format

up:
	docker compose -p irtm -f docker/docker-compose.yml up --build

down:
	docker compose -f docker/docker-compose.yml down

dev:
	uvicorn src.api.main:app --reload --port=8000

test:
	pytest tests/

format:
	black src/ tests/