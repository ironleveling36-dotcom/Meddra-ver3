FROM python:3.11-slim

# curl only for the healthcheck. No compiler needed (all wheels are prebuilt).
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first for better layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Pre-download the embedding model into the image so the first request is instant
# and startup doesn't depend on network access to Hugging Face.
ENV FASTEMBED_CACHE_DIR=/app/.fastembed_cache
RUN python -c "from fastembed import TextEmbedding; \
list(TextEmbedding(model_name='BAAI/bge-small-en-v1.5', cache_dir='/app/.fastembed_cache').embed(['warmup']))"

# App code + prebuilt index (data/) + web UI (static/).
COPY . .

ENV PORT=8000
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD ["sh", "-c", "curl -f http://localhost:${PORT:-8000}/health || exit 1"]

# JSON exec form (proper signal handling); sh -c expands $PORT at runtime.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
