
.PHONY: test test-verbose lint lint-check

test:
	pytest tests

test-verbose:
	pytest tests -v

lint:
	ruff check --fix .
	ruff format .
	ruff check .

lint-check:
	ruff check .
	ruff format --check .
