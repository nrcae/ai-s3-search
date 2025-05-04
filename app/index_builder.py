import threading
import concurrent.futures
from typing import List, Generator
from app.vectorstore import FAISSVectorStore
from app.shared_resources import vector_store
from app.s3_loader import fetch_pdf_files, extract_text_from_pdf
from app.embedder import get_embeddings, chunk_text

# Attempt to import the BATCH function first
try:
    from text_normalizer import normalize_text_batch as normalize_text_batch_rust
    print("INFO:     Using Rust 'normalize_text_batch' function.")
    use_rust_batch = True
except ImportError:
    print("WARNING:  Rust batch function 'normalize_text_batch' not found. Will attempt single normalization.")
    use_rust_batch = False
    # If batch fails, try importing the SINGLE function
    try:
        from text_normalizer import normalize_text as normalize_text_rust
        print("INFO:     Using Rust 'normalize_text' function.")
        use_rust_single = True
    except ImportError:
        print("WARNING:  Rust single function 'normalize_text' not found. Using Python fallback.")
        use_rust_single = False
        # Minimal Python fallback implementation if both Rust imports fail
        def normalize_text_fallback(text: str) -> str:
            return text.strip().lower()

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

def optimized_batch_embedding(
    chunk_generator: Generator[str, None, None],
    batch_size: int,
    vector_store: FAISSVectorStore
):
    batch: List[str] = []

    for chunk in chunk_generator:
        batch.append(chunk)
        if len(batch) >= batch_size:
            normalized_batch = []
            try:
                if use_rust_batch: normalized_batch = normalize_text_batch_rust(batch)
                elif use_rust_single: normalized_batch = [normalize_text_rust(c) for c in batch]
                else: normalized_batch = [normalize_text_fallback(c) for c in batch]

                vectors = get_embeddings(batch)
                vector_store.add(vectors, batch) # Assumes add sets is_ready eventually
            except Exception as e:
                print(f"Error processing batch: {e}")
            finally:
                 batch.clear()

    if batch:
        normalized_batch = []
        try:
            if use_rust_batch: normalized_batch = normalize_text_batch_rust(batch)
            elif use_rust_single: normalized_batch = [normalize_text_rust(c) for c in batch]
            else: normalized_batch = [normalize_text_fallback(c) for c in batch]

            # Embed and add the final normalized batch
            vectors = get_embeddings(normalized_batch)
            vector_store.add(vectors, normalized_batch)
        except Exception as e:
            print(f"Error processing final batch: {e}")

def build_index_background(
    vector_store: FAISSVectorStore,
    batch_size: int = 32,
    max_workers: int = 5
):

    initial_ready_state = vector_store.is_ready

    try:
        files = fetch_pdf_files()
        if not files:
            print("No PDF files found.")
            if not initial_ready_state: vector_store.is_ready = True # Mark ready if empty & wasn't ready
            return

        print(f"Processing {len(files)} PDF files...")
        chunk_gen = process_pdfs_to_chunks(files, max_workers=max_workers)
        optimized_batch_embedding(chunk_gen, batch_size, vector_store)
        if not vector_store.is_ready: vector_store.is_ready = True

    except Exception as e:
        print(f"CRITICAL ERROR during indexing: {e}")
        # Ensure store is marked ready if it failed and wasn't ready before
        if not initial_ready_state: vector_store.is_ready = True

def start_background_indexing(vector_store_instance: FAISSVectorStore):
    print("Initiating background indexing thread...")
    thread = threading.Thread(
        target=build_index_background,
        args=(vector_store_instance,),
        kwargs={'batch_size': 2048, 'max_workers': 5},
        daemon=True
    )
    thread.start()