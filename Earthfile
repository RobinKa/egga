VERSION 0.6

base-python:
    FROM python:3.8

    # Install Poetry
    ENV PIP_CACHE_DIR /pip-cache
    RUN --mount=type=cache,target=$PIP_CACHE_DIR \
        pip install poetry==1.6.1
    RUN --mount=type=cache,target=$PIP_CACHE_DIR \
        poetry config virtualenvs.create false
    
    # Install graphviz which the tests use
    RUN apt-get update && apt-get install -y graphviz && apt-get clean

build:
    FROM +base-python

    WORKDIR /app

    # Copy poetry files
    COPY pyproject.toml poetry.lock README.md .

    # We only want to install the dependencies once, so if we copied
    # our code here now, we'd reinstall the dependencies ever ytime
    # the code changes. Instead, comment out the line making us depend
    # on our code, install, then copy our code and install again
    # with the line not commented.
    RUN sed -e '/packages/ s/^#*/#/' -i pyproject.toml

    # Install dependencies
    RUN poetry install
    
    # Copy without the commented out packages line and install again
    COPY --dir egga .
    COPY pyproject.toml .
    RUN poetry install

test:
    FROM +build

    # Run tests
    COPY --dir tests .
    RUN poetry run pytest -n auto

test-examples:
    FROM +build

    # Run examples
    COPY --dir examples .
    FOR example IN $(ls examples/**/*.py)
        RUN poetry run python "$example"
    END

publish:
    FROM +build

    ARG --required REPOSITORY

    RUN poetry config repositories.pypi https://upload.pypi.org/legacy/
    RUN poetry config repositories.testpypi https://test.pypi.org/legacy/

    RUN --mount=type=cache,target=$PIP_CACHE_DIR \
        --secret PYPI_TOKEN=+secrets/PYPI_TOKEN \
        poetry publish \
            --build --skip-existing -r $REPOSITORY \
            -u __token__ -p $PYPI_TOKEN
