.PHONY: fmt test lint

fmt:
	uvx ruff format

test:
	uvx pytest

lint:
	uvx ruff check --select I --fix .
