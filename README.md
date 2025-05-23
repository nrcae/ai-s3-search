# PDF Semantic Search Engine

A FastAPI-based semantic search engine that enables users to upload PDF documents and perform intelligent search queries using vector embeddings. The application features a clean web interface for document management and real-time search capabilities.

---

## Features

- **PDF Document Upload**: Upload PDF files through a web interface with automatic text extraction.
- **Semantic Search**: Perform intelligent queries using sentence transformers and vector similarity.
- **Real-time Indexing**: Background processing with automatic vector embedding generation.
- **Source Filtering**: Filter search results by specific PDF documents.
- **Modern Web UI**: Clean, responsive interface with a dark theme and real-time status updates.
- **Vector Storage**: Efficient document storage and retrieval using LanceDB.

---

## Architecture

The application is built with the following key components:

- **FastAPI Backend**: RESTful API with endpoints for upload, search, and monitoring.
- **Vector Store**: LanceDB-powered vector database for similarity search.
- **Embedding Engine**: SentenceTransformer models for text-to-vector conversion.
- **Background Processing**: Asynchronous document indexing with batch processing.
- **Web Interface**: Static HTML/CSS/JavaScript frontend for user interaction.

---

## API Endpoints

- `POST /upload_pdf` – Upload PDF documents for indexing.
- `GET /search` – Perform semantic search with optional source filtering.
- `GET /status` – Check indexing status and system information.
- `GET /sources` – List all available PDF sources.
- `GET /` – Serve the web interface.

---

## Installation & Setup

### Prerequisites

- Python 3.12+
- AWS S3 credentials
- Required Python packages (see `requirements.txt`)

### Configuration

Set the following environment variables:

```bash
AWS_ACCESS_KEY=your_access_key
AWS_SECRET_KEY=your_secret_key
S3_BUCKET=your_bucket_name
EMBED_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
```