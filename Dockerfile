FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml .
COPY uv.lock .
COPY src/ ./src/
COPY data/ ./data/

RUN uv sync --locked --no-cache

EXPOSE 8000

ENV PYTHONUNBUFFERED=1

# Ollama url is set to host machine's localhost server.
ENV OLLAMA_BASE_URL=http://host.docker.internal:11434 

CMD ["uv", "run", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
