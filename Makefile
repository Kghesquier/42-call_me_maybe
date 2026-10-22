PY = uv run python

# Keep every download inside the project folder (home quota is too small)
export UV_CACHE_DIR := $(CURDIR)/.cache/uv
export HF_HOME := $(CURDIR)/.cache/huggingface
export UV_LINK_MODE := copy

.PHONY: install run debug clean fclean lint lint-strict

install:
	uv sync

run:
	$(PY) -m src

debug:
	$(PY) -m pdb -m src

clean:
	find . \( -path ./.venv -o -path ./.cache \) -prune -o -name "__pycache__" -type d -exec rm -rf {} +
	rm -rf .mypy_cache

fclean: clean
	rm -rf .cache .venv

lint:
	uv run flake8 .
	uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 .
	uv run mypy . --strict
