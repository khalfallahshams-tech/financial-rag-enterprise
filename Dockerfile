# =============================================================================
# Multi-Stage Enterprise Dockerfile for FiQA Financial RAG Intelligence Engine
# Security: Runs as non-root user (appuser, UID 10001)
# =============================================================================

# --- Stage 1: Dependency Builder ---
FROM python:3.11-slim AS builder

WORKDIR /build

# Install system compilation dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and pre-build wheels
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip wheel --no-cache-dir --wheel-dir /build/wheels -r requirements.txt

# --- Stage 2: Hardened Production Runtime ---
FROM python:3.11-slim AS runtime

# Security: Create non-root system user and group
RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -s /bin/bash -m appuser

WORKDIR /app

# Install runtime system utilities (curl for healthchecks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install pre-built wheels from builder stage
COPY --from=builder /build/wheels /wheels
RUN pip install --no-cache-dir /wheels/* && rm -rf /wheels

# Copy application source code and configuration
COPY --chown=appuser:appgroup . /app

# Set non-root user
USER appuser

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

# Expose API (8000), UI (8501), and MLflow (5000) ports
EXPOSE 8000 8501 5000

# Health check against FastAPI /health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default entrypoint starts the FastAPI intelligence service
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
