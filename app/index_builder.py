import threading
import concurrent.futures
from typing import List, Generator
from app.vectorstore import FAISSVectorStore
from app.s3_loader import fetch_pdf_files, extract_text_from_pdf
from app.embedder import get_embeddings, chunk_text

# --- ADD Import for Rust module with fallback ---
try:
    from text_normalizer import normalize_text as normalize_text_rust
    print("INFO:     Using Rust 'normalize_text' function.")
except ImportError:
    print("WARNING:  Rust module 'text_normalizer_lib' not found. Using Python fallback for normalization.")
    # Minimal Python fallback implementation
    def normalize_text_rust(text: str) -> str:
        return text.strip().lower()

vector_store = FAISSVectorStore()

def process_pdfs_to_chunks(files: List[str], max_workers: int = 5) -> Generator[str, None, None]:
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for text in executor.map(extract_text_from_pdf, files):
            if text:
                try:
                    # Yield each chunk from the list/iterable returned by chunk_text
                    for chunk in chunk_text(text):
                        yield chunk
                except Exception as e:
                    print(f"Error chunking text: {e}")

def optimized_batch_embedding(chunk_generator: Generator[str, None, None], batch_size: int):
    global vector_store
    batch: List[str] = []

    for chunk in chunk_generator:
        normalized_chunk = normalize_text_rust(chunk)
        batch.append(normalized_chunk)
        if len(batch) >= batch_size:
            try:
                vectors = get_embeddings(batch)
                vector_store.add(vectors, batch) # Assumes add sets is_ready eventually
            except Exception as e:
                print(f"Error processing batch: {e}")
            finally:
                 batch.clear()

    if batch:
        try:
            vectors = get_embeddings(batch)
            vector_store.add(vectors, batch)
        except Exception as e:
            print(f"Error processing final batch: {e}")

def build_index_background(batch_size: int = 2048, max_workers: int = 5):
    global vector_store
    initial_ready_state = vector_store.is_ready

    try:
        files = fetch_pdf_files()
        if not files:
            print("No PDF files found.")
            if not initial_ready_state: vector_store.is_ready = True # Mark ready if empty & wasn't ready
            return

        print(f"Processing {len(files)} PDF files...")
        chunk_gen = process_pdfs_to_chunks(files, max_workers=max_workers)
        optimized_batch_embedding(chunk_gen, batch_size)
        if not vector_store.is_ready: vector_store.is_ready = True

    except Exception as e:
        print(f"CRITICAL ERROR during indexing: {e}")
        # Ensure store is marked ready if it failed and wasn't ready before
        if not initial_ready_state: vector_store.is_ready = True

# --- Minimal Thread Starter (Essentially unchanged) ---
def start_background_indexing():
    print("Initiating background indexing thread...")
    thread = threading.Thread(
        target=build_index_background,
        kwargs={'batch_size': 2048, 'max_workers': 5},
        daemon=True
    )
    thread.start()