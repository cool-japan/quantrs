# QuantRS2 Docker Image
# Multi-stage build for optimized image size

# Build stage
FROM python:3.11-slim as builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    pkg-config \
    libssl-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Rust (required for building QuantRS2)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# Upgrade pip and install build tools
RUN pip install --upgrade pip setuptools wheel

# Copy requirements first for better caching
COPY requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt

# Copy source code
WORKDIR /app
COPY . .

# Build and install QuantRS2
RUN pip install -e .

# Production stage
FROM python:3.11-slim as production

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:${PATH}"

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Copy application code
WORKDIR /app
COPY --from=builder /app .

# Create non-root user for security
RUN groupadd -r quantrs && useradd -r -g quantrs quantrs
RUN chown -R quantrs:quantrs /app
USER quantrs

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import quantrs2; print('QuantRS2 healthy')" || exit 1

# Default command
CMD ["python", "-c", "import quantrs2; print('QuantRS2 Docker container is ready!')"]

# Labels for metadata
LABEL maintainer="QuantRS2 Team" \
      version="0.1.0" \
      description="QuantRS2 Quantum Computing Framework" \
      org.opencontainers.image.source="https://github.com/cool-japan/quantrs" \
      org.opencontainers.image.documentation="https://quantrs2.readthedocs.io" \
      org.opencontainers.image.title="QuantRS2" \
      org.opencontainers.image.description="High-performance quantum computing framework"