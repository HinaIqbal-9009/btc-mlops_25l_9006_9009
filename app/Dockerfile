# Build from the REPO ROOT:  docker build -f app/Dockerfile -t btc-forecast:latest .

# ---------- Stage 1: builder (installs dependencies; thrown away afterwards) ----------
FROM python:3.10-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /build
# Requirement files first -> this layer is cached until dependencies change
COPY requirements.txt requirements-ml.txt ./
RUN pip install --prefix=/install -r requirements.txt

# ---------- Stage 2: runtime (small, non-root) ----------
FROM python:3.10-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=5000 \
    MODEL_DIR=/app/models \
    LOG_LEVEL=INFO

# Unprivileged user
RUN groupadd --gid 10001 appuser \
 && useradd --uid 10001 --gid appuser --no-create-home --shell /usr/sbin/nologin appuser

WORKDIR /app
COPY --from=builder /install /usr/local
# Code changes most often -> copied LAST so earlier layers stay cached
COPY app/app.py ./app.py
COPY src/ ./src/
RUN mkdir -p /app/models /app/data && chown -R appuser:appuser /app

USER appuser
EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
  CMD python -c "import os,urllib.request as u; u.urlopen('http://127.0.0.1:%s/health' % os.environ.get('PORT','5000'), timeout=2)"

# sh -c so ${PORT} is expanded; exec makes gunicorn PID 1 so it gets SIGTERM for graceful shutdown
CMD ["sh", "-c", "exec gunicorn --bind 0.0.0.0:${PORT} --workers 2 --threads 2 --timeout 60 app:app"]
