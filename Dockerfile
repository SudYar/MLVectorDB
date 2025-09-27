FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    curl git vim build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

COPY pyproject.toml poetry.lock* ./

RUN poetry install --no-root

COPY src ./src

CMD ["bash"]
