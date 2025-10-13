# Multi-stage build for diaremot2-on
# CPU-only speech intelligence pipeline

ARG PYTHON_VERSION=3.11

# Stage 1: Builder - install dependencies
FROM python:${PYTHON_VERSION}-slim AS builder

# Set working directory
WORKDIR /build

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt requirements.in ./

# Upgrade pip and install dependencies
RUN python -m pip install --upgrade pip wheel setuptools && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY tests/ ./tests/

# Install package
RUN pip install --no-cache-dir -e .

# Stage 2: Runtime - minimal image
FROM python:${PYTHON_VERSION}-slim

# Set working directory
WORKDIR /app

# Install only runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python${PYTHON_VERSION}/site-packages /usr/local/lib/python${PYTHON_VERSION}/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy source code
COPY src/ ./src/
COPY pyproject.toml README.md ./

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Model directory (can be overridden at runtime)
    DIAREMOT_MODEL_DIR=/srv/models \
    # HuggingFace cache
    HF_HOME=/app/.cache \
    HUGGINGFACE_HUB_CACHE=/app/.cache/hub \
    TRANSFORMERS_CACHE=/app/.cache/transformers \
    TORCH_HOME=/app/.cache/torch \
    # CPU threading
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    NUMEXPR_MAX_THREADS=4 \
    TOKENIZERS_PARALLELISM=false

# Create directories
RUN mkdir -p /srv/models /app/outputs /app/.cache && \
    chmod -R 777 /app/outputs /app/.cache

# Expose volume mount points
VOLUME ["/srv/models", "/app/outputs"]

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import diaremot; print('OK')" || exit 1

# Default entrypoint
ENTRYPOINT ["python", "-m", "diaremot.cli"]

# Default command (shows help)
CMD ["--help"]

# Labels
LABEL maintainer="Timothy Leigh Troglin" \
      version="2.2.0" \
      description="CPU-only speech intelligence pipeline with diarization, ASR, and affect analysis" \
      org.opencontainers.image.source="https://github.com/your-org/diaremot2-on" \
      org.opencontainers.image.documentation="https://github.com/your-org/diaremot2-on/blob/main/README.md" \
      org.opencontainers.image.vendor="DiaRemot" \
      org.opencontainers.image.licenses="MIT"
