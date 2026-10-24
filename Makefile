
.PHONY: test test-verbose format lint lint-check

test:
	pytest tests

test-verbose:
	pytest tests -v

format:
	ruff check --select I --fix .
	ruff format .

lint:
	ruff check --fix .
	ruff format .
	ruff check .

lint-check:
	ruff check .
	ruff format --check .
