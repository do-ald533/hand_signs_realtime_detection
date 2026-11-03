.DEFAULT_GOAL := help
.PHONY: help sync lint format typecheck test check hooks

help:  ## Show this help
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-10s %s\n", $$1, $$2}'

sync:  ## Install the locked environment
	uv sync --locked

lint:  ## Lint (ruff)
	uv run ruff check .

format:  ## Auto-format (ruff)
	uv run ruff format .

typecheck:  ## Type-check (pyright)
	uv run pyright

test:  ## Run tests with coverage
	uv run pytest --cov

check:  ## Everything CI runs
	uv run ruff check .
	uv run ruff format --check .
	uv run pyright
	uv run pytest --cov

hooks:  ## Install git pre-commit hooks
	uv run pre-commit install
