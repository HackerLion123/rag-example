FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml .
COPY src/ ./src/
COPY data/ ./data/

RUN uv pip install --system .

RUN uv sync

EXPOSE 8000

ENV PYTHONUNBUFFERED=1
ENV OLLAMA_BASE_URL=http://host.docker.internal:11434

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
