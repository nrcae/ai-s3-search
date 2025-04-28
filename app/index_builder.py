import threading
import concurrent.futures
from app.vectorstore import FAISSVectorStore
from app.s3_loader import fetch_pdf_files, extract_text_from_pdf
from app.embedder import get_embeddings, chunk_text

vector_store = FAISSVectorStore()  # Initialize empty store

def build_index_background(batch_size: int = 2048):
    global vector_store

    files = fetch_pdf_files()
    all_chunks = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        # Map extract_text_from_pdf over files concurrently
        for text in executor.map(extract_text_from_pdf, files):
            if text:
                all_chunks.extend(chunk_text(text))

    if all_chunks:
        # Batch the chunks for embedding
        for i in range(0, len(all_chunks), batch_size):
            batch_chunks = all_chunks[i:i + batch_size]
            vectors = get_embeddings(batch_chunks)
            vector_store.add(vectors, batch_chunks)
    else:
        vector_store.is_ready = True

def start_background_indexing():
    thread = threading.Thread(target=build_index_background)
    thread.daemon = True
    thread.start()