.PHONY: verify fix lint format format-check type-check install test test-cov test-matrix test-matrix-cov pysentry

verify: lint format-check type-check

fix:
	uv run ruff check --fix src tests
	uv run ruff format src tests

lint:
	uv run ruff check src tests

format-check:
	uv run ruff format --check src tests

format:
	uv run ruff format src tests

type-check:
	uv run ty check

install:
	uv sync --all-groups

test:
	uv run pytest tests/ -v

test-cov:
	uv run pytest --cov --cov-report=xml --cov-report=term-missing

test-matrix:
	uv run hatch test

test-matrix-cov:
	uv run hatch test --cover

pysentry:
	uv run pysentry-rs
