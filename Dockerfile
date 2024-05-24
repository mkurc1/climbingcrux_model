FROM python:3.11.8-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install dependencies
RUN apt-get update \
    && apt-get install curl build-essential -y \
    && curl -sSL https://install.python-poetry.org | python - --version 1.3.2

ENV PATH="/root/.local/bin:$PATH"

# Set work directory
WORKDIR /code

COPY poetry.lock pyproject.toml ./

RUN poetry config virtualenvs.create false \
    && poetry install --only main --no-interaction --no-ansi

# Copy project
COPY . ./
