# Dockerfile

# 1. Use an official Python runtime
FROM python:3.12-slim

# 2. Set environment variables for Python and Rust
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV CARGO_HOME="/usr/local/cargo"
ENV RUSTUP_HOME="/usr/local/rustup"
ENV PATH="/usr/local/cargo/bin:${PATH}"

# 3. Set the working directory *inside the container*
WORKDIR /app

# 4. Install base system dependencies + Rust toolchain
RUN apt-get update && \
    apt-get install -y curl build-essential pkg-config libssl-dev && \
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 5. Install Python build tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# 6. Install Python dependencies, including Maturin
COPY ./requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# 7. Build the Rust extension using Maturin
COPY ./text_normalizer /app/text_normalizer
RUN maturin build --release --manifest-path /app/text_normalizer/Cargo.toml --out /wheels

# 8. Install the *just built* Rust wheel
RUN pip install --no-cache-dir /wheels/*.whl && \
    rm -rf /wheels # Clean up the temporary wheel directory

# 9. Download/Cache Sentence Transformer model
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# 10. Copy application code
COPY ./main.py /app/main.py
COPY ./app /app/app

# 11. Expose the application port
EXPOSE 8000

# 12. Define the command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
