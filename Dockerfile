# Builds the Rust text_normalizer extension into a wheel.
FROM python:3.12-slim AS rust-builder

ENV CARGO_HOME="/usr/local/cargo"
ENV RUSTUP_HOME="/usr/local/rustup"
ENV PATH="/usr/local/cargo/bin:${PATH}"

# Install curl to get Rust, pip for Maturin, AND build-essential for the C linker (cc/gcc)
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl python3-pip python3-setuptools python3-wheel build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install Rust toolchain
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable

# Install Maturin
RUN pip install --no-cache-dir maturin

WORKDIR /build_rust
COPY ./text_normalizer /build_rust/text_normalizer

# Build the Rust wheel
RUN maturin build --release --manifest-path /build_rust/text_normalizer/Cargo.toml --out /dist_wheels

# Stage 2: Python Venv Builder
FROM python:3.12-slim AS python-venv-builder

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 VENV_PATH=/opt/venv

# Create a virtual environment
RUN python -m venv $VENV_PATH
ENV PATH="$VENV_PATH/bin:$PATH"

# Copy and install Python dependencies from requirements.txt
COPY ./requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc build-essential pkg-config libssl-dev && \
    # Upgrade pip
    pip install --no-cache-dir --upgrade pip && \
    # Install CPU-only PyTorch first if it's a large dependency
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    # Copy and install Python dependencies from requirements.txt
    # Note: COPY requirements.txt before this RUN block
    pip install --no-cache-dir -r /tmp/requirements.txt && \
    # Remove build-time system dependencies AFTER they are used
    apt-get purge -y --auto-remove gcc build-essential pkg-config libssl-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy and install the Rust wheel from the rust-builder stage
COPY --from=rust-builder /dist_wheels /tmp/dist_wheels
RUN pip install --no-cache-dir /tmp/dist_wheels/*.whl && \
    rm -rf /tmp/dist_wheels

# Final Runtime Image
FROM python:3.12-slim AS final

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 VENV_PATH=/opt/venv PATH="$VENV_PATH/bin:$PATH"

# Copy the virtual environment from the python-venv-builder stage
COPY --from=python-venv-builder $VENV_PATH $VENV_PATH

# Create a non-root user and group
RUN groupadd --gid 1001 appuser && useradd --uid 1001 --gid 1001 -ms /bin/bash appuser

WORKDIR /app

# Copy application code
COPY ./main.py /app/main.py
COPY ./app /app/app

# Create the lancedb_data directory and set its ownership to appuser
# This must be done BEFORE switching to the non-root user.
RUN mkdir /app/lancedb_data && \
    chown appuser:appuser /app/lancedb_data

# Switch to the non-root user
USER appuser
EXPOSE 8000

# Define the command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
