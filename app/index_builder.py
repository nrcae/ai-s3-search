import threading
import concurrent.futures
from typing import List, Generator
from app.vectorstore import LanceDBVectorStore
from app.s3_loader import fetch_pdf_files, extract_text_from_pdf
from app.embedder import get_embeddings, chunk_text
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Attempt to import the BATCH function first
try:
    from text_normalizer import normalize_text_batch as normalize_text_batch_rust
    logger.debug(" Using Rust 'normalize_text_batch' function.")
    use_rust_batch = True
except ImportError:
    logger.debug(" Rust batch function 'normalize_text_batch' not found. Will attempt single normalization.")
    use_rust_batch = False
    # If batch fails, try importing the SINGLE function
    try:
        from text_normalizer import normalize_text as normalize_text_rust
        logger.debug(" Using Rust 'normalize_text' function.")
        use_rust_single = True
    except ImportError:
        logger.warning(" Rust single function 'normalize_text' not found. Using Python fallback.")
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
                    logger.error(f" Chunking text: {e}")

def _normalize_batch(batch: List[str], rust_batch_enabled: bool, rust_single_enabled: bool) -> List[str]:
    """Helper function to normalize a batch of texts."""
    if rust_batch_enabled:
        logger.debug(f"Normalizing batch of {len(batch)} texts using Rust batch.")
        return normalize_text_batch_rust(batch)
    elif rust_single_enabled:
        logger.debug(f"Normalizing batch of {len(batch)} texts using Rust single.")
        return [normalize_text_rust(c) for c in batch]
    else:
        logger.debug(f"Normalizing batch of {len(batch)} texts using Python fallback.")
        return [normalize_text_fallback(c) for c in batch]

def _process_and_add_batch(
    current_batch: List[str],
    vector_store: LanceDBVectorStore,
    rust_batch_enabled: bool,
    rust_single_enabled: bool
):
    """
    Helper function to normalize, embed, and add a single batch to the vector store.
    Returns True if successful, False otherwise.
    """
    if not current_batch:
        return True

    try:
        normalized_batch = _normalize_batch(current_batch, rust_batch_enabled, rust_single_enabled)
        
        if not normalized_batch: # Should not happen if current_batch was not empty
            logger.warning("Normalization resulted in an empty batch, skipping embedding and adding.")
            return True # Technically not an error, but nothing was added

        logger.debug(f"Embedding {len(normalized_batch)} normalized texts.")
        vectors = get_embeddings(normalized_batch)

        logger.debug(f"Adding {len(normalized_batch)} items (vectors and normalized texts) to vector store.")
        vector_store.add(vectors, normalized_batch)
        return True
        
    except Exception as e:
        logger.error(f"Failed to process batch of size {len(current_batch)}. Error: {e}", exc_info=True)
        return False # Indicate failure for this batch

def optimized_batch_embedding(
    chunk_generator: Generator[str, None, None],
    batch_size: int,
    vector_store: LanceDBVectorStore,
    use_rust_batch: bool = True, # Default to using Rust batch if available
    use_rust_single: bool = False
):
    """
    Efficiently processes text chunks in batches: normalizes, embeds, and adds to vector store.
    """
    if batch_size <= 0:
        logger.error("batch_size must be positive.")
        raise ValueError("batch_size must be a positive integer.")

    current_batch: List[str] = []
    processed_chunks_count = 0
    failed_batches_count = 0

    for chunk in chunk_generator:
        if not isinstance(chunk, str):
            logger.warning(f"Skipping non-string item from chunk_generator: {type(chunk)}")
            continue
        
        current_batch.append(chunk)
        
        if len(current_batch) >= batch_size:
            if _process_and_add_batch(current_batch, vector_store, use_rust_batch, use_rust_single):
                processed_chunks_count += len(current_batch)
            else:
                failed_batches_count += 1
            current_batch.clear()

    # Process any remaining chunks in the last batch
    if current_batch:
        if _process_and_add_batch(current_batch, vector_store, use_rust_batch, use_rust_single):
            processed_chunks_count += len(current_batch)
        else:
            failed_batches_count += 1
        current_batch.clear() # Good practice to clear it

    logger.info(f"Finished optimized batch embedding. Processed chunks: {processed_chunks_count}. Failed batches: {failed_batches_count}.")

def build_index_background(
    vector_store: LanceDBVectorStore,
    batch_size: int = 32,
    max_workers: int = 5
):

    initial_ready_state = vector_store.is_ready

    try:
        files = fetch_pdf_files()
        if not files:
            logger.error(" No PDF files found.")
            if not initial_ready_state: vector_store.is_ready = True # Mark ready if empty & wasn't ready
            return

        logger.info(f" Processing {len(files)} PDF files...")
        chunk_gen = process_pdfs_to_chunks(files, max_workers=max_workers)
        optimized_batch_embedding(chunk_gen, batch_size, vector_store)
        if not vector_store.is_ready: vector_store.is_ready = True

    except Exception as e:
        logger.critical(f" CRITICAL ERROR during indexing: {e}")
        # Ensure store is marked ready if it failed and wasn't ready before
        if not initial_ready_state: vector_store.is_ready = True

def start_background_indexing(vector_store_instance: LanceDBVectorStore):
    logger.debug(" Initiating background indexing thread...")
    thread = threading.Thread(
        target=build_index_background,
        args=(vector_store_instance,),
        kwargs={'batch_size': 2048, 'max_workers': 5},
        daemon=True
    )
    thread.start()