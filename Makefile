PY = uv run python

.PHONY: install run debug clean lint lint-strict test

install:
	uv sync

run:
	$(PY) -m src

debug:
	$(PY) -m pdb -m src

clean:
	find . -path ./.venv -prune -o -name "__pycache__" -type d -exec rm -rf {} +
	rm -rf .mypy_cache .pytest_cache

lint:
	uv run flake8 .
	uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 .
	uv run mypy . --strict

test:
	uv run pytest -q
