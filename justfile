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

examples:
    for example in examples/*.py; do \
        uv run python "$example"; \
    done

build:
    uv build

check: sync test examples build

publish-test: build
    uv publish --publish-url https://test.pypi.org/legacy/

publish: build
    uv publish
