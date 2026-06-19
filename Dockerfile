FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && curl -sSL https://install.python-poetry.org | python3 - \
    && rm -rf /var/lib/apt/lists/*

RUN poetry config virtualenvs.create false
COPY pyproject.toml poetry.lock* /app/
RUN poetry install --without dev --no-root

RUN mkdir -p /app/src
COPY src/api.py /app/src/
COPY src/detector_mitre.py /app/src/

ENV PYTHONPATH=/app/src
ENV MODEL_PATH=/models
ENV DATA_PATH=/data

EXPOSE 8080

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8080"]