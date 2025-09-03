# =============================================================================
# RAGagument Multi-Stage Dockerfile
# Supports: development, production, builder, test stages
# =============================================================================

# Base stage - common dependencies
FROM python:3.9-slim-bullseye AS base

# Common environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN addgroup --system --gid 1000 appgroup \
    && adduser --system --uid 1000 --gid 1000 appuser

# Builder stage - dependency compilation and wheel caching
FROM base AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create wheelhouse directory for dependency caching
RUN mkdir -p /wheels

# Copy requirements first for optimal caching
COPY requirements.txt .

# Download all packages as wheels for faster installation
RUN pip wheel --wheel-dir=/wheels --no-cache-dir \
    -r requirements.txt

# Test stage - security scanning and testing
FROM base AS test

# Copy pre-downloaded wheels from builder stage
COPY --from=builder /wheels /wheels

# Install test dependencies using pre-built wheels
COPY requirements-test.txt .
RUN pip install --user --no-cache-dir --no-warn-script-location \
    --find-links=/wheels --no-index -r requirements-test.txt

# Copy application code
COPY --chown=appuser:appgroup . .

# Security scanning
RUN curl -sfL https://raw.githubusercontent.com/aquasecurity/trivy/main/contrib/install.sh | sh -s -- -b /usr/local/bin

# Development stage - with debugging tools
FROM base AS development

# Install development tools
RUN apt-get update && apt-get install -y \
    vim \
    git \
    procps \
    net-tools \
    htop \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install development Python packages
RUN pip install --user --no-cache-dir --no-warn-script-location \
    debugpy \
    ptvsd \
    black \
    flake8 \
    pytest \
    pytest-cov

WORKDIR /app

# Copy pre-downloaded wheels from builder stage
COPY --from=builder /wheels /wheels

# Copy requirements file for installation
COPY requirements.txt .

# Install application dependencies from pre-built wheels (much faster)
RUN pip install --user --no-cache-dir --no-warn-script-location \
    --find-links=/wheels --no-index -r requirements.txt

# Ensure PATH includes user packages
ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY --chown=appuser:appgroup . .

# Expose ports
EXPOSE 8501 5678

# Development entrypoint - simple command
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

# Production stage - optimized runtime
FROM base AS production

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder --chown=appuser:appgroup /root/.local /home/appuser/.local

# Copy application code
COPY --chown=appuser:appgroup . .

# Ensure PATH includes user packages
ENV PATH=/home/appuser/.local/bin:$PATH

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Expose port
EXPOSE 8501

# Production entrypoint
ENTRYPOINT ["streamlit", "run", "app.py"]

# Default command
CMD ["--server.port=8501", "--server.address=0.0.0.0", "--server.enableCORS=false", "--server.enableXsrfProtection=true"]