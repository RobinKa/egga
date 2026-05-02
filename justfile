set shell := ["bash", "-eu", "-o", "pipefail", "-c"]

[default]
default:
    @just --list

sync:
    uv sync

sync-frozen:
    uv sync --frozen

test:
    uv run pytest -n auto

test-quiet:
    uv run pytest -q

format:
    uv run ruff format .

format-check:
    uv run ruff format --check .

lint:
    uv run ruff check .

examples:
    for example in examples/*.py; do \
        uv run python "$example"; \
    done

build:
    uv build

check: sync format-check lint test examples build

publish-test: build
    uv publish --publish-url https://test.pypi.org/legacy/

publish: build
    uv publish
