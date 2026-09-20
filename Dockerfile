# The API, packaged for a runtime with no GPU behind it.
#
# The local path is Ollama on the machine's own card. Nothing serverless has one, so
# the hosted image runs the same code against Gemini through the LLMClient protocol --
# DARWINBOX_LLM=gemini is the only difference between this and a laptop.
FROM python:3.11-slim

# DuckDB, pandas and openpyxl all ship wheels, so no build toolchain is needed.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DARWINBOX_LLM=gemini

WORKDIR /app

# Dependencies resolve from pyproject alone, so they cache across source edits.
COPY pyproject.toml README.md ./
RUN mkdir -p backend/darwinbox && touch backend/darwinbox/__init__.py \
    && pip install --no-cache-dir ".[cloud]" \
    && rm -rf backend

COPY backend/ backend/
COPY demo/ demo/

# Sessions live in process memory, so the deployment pins itself to one instance.
# Two instances would mean a session created against one is a 404 against the other.
ENV PYTHONPATH=/app/backend \
    PORT=8080
EXPOSE 8080

CMD exec uvicorn darwinbox.api.app:app --host 0.0.0.0 --port ${PORT} --workers 1
