# Dockerfile

# 1. Use an official Python
FROM python:3.12-slim

# 2. Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. Set the working directory in the container
WORKDIR /app

# 4. Install dependencies
# Copy only the requirements file first to leverage Docker cache
COPY ./requirements.txt /app/requirements.txt
# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# 5. Download and cache the Sentence Transformer model during build
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# 6. Copy the rest of your application code into the container
COPY . /app

# 7. Expose the port the app runs on (must match the CMD)
EXPOSE 8000

# 8. Define the command to run your application
# Use uvicorn to run the FastAPI app located at app.main:app
# --host 0.0.0.0 makes it accessible from outside the container
# --port 8000 matches the EXPOSE directive
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
