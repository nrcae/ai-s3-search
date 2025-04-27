import threading
from app.vectorstore import FAISSVectorStore
from app.s3_loader import fetch_pdf_files, extract_text_from_pdf
from app.embedder import get_embeddings, chunk_text

vector_store = FAISSVectorStore()  # Initialize empty store

def build_index_background():
    global vector_store
    files = fetch_pdf_files()
    all_chunks = []
    for f in files:
        text = extract_text_from_pdf(f)
        chunks = chunk_text(text)
        all_chunks.extend(chunks)
    if all_chunks:
        vectors = get_embeddings(all_chunks)
        vector_store.add(vectors, all_chunks)
    else:
        vector_store.is_ready = True

def start_background_indexing():
    thread = threading.Thread(target=build_index_background)
    thread.daemon = True
    thread.start()