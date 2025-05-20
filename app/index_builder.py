import threading
import concurrent.futures
from typing import List, Generator, Tuple
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


try:
    from text_normalizer import normalize_text_batch as normalize_text_batch_rust
    logger.debug(" Using Rust 'normalize_text_batch' function.")
    use_rust_batch = True
except ImportError:
    logger.debug(" Rust batch function 'normalize_text_batch' not found. Will attempt single normalization.")
    use_rust_batch = False
    try:
        from text_normalizer import normalize_text as normalize_text_rust
        logger.debug(" Using Rust 'normalize_text' function.")
        use_rust_single = True
    except ImportError:
        logger.warning(" Rust single function 'normalize_text' not found. Using Python fallback.")
        use_rust_single = False
        def normalize_text_fallback(text: str) -> str:
            return text.strip().lower()


def process_pdfs_to_chunks(files: List[str], max_workers: int = 5) -> Generator[Tuple[str, str], None, None]:
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_s3_key = {executor.submit(extract_text_from_pdf, s3_file_key): s3_file_key for s3_file_key in files}
        for future in concurrent.futures.as_completed(future_to_s3_key):
            s3_key = future_to_s3_key[future]
            try:
                text = future.result()
                if text:
                    for chunk in chunk_text(text):
                        yield chunk, s3_key
            except Exception as e:
                logger.error(f"Processing or chunking text from {s3_key}: {e}")


def _normalize_batch(batch: List[str], rust_batch_enabled: bool, rust_single_enabled: bool) -> List[str]:
    if rust_batch_enabled:
        logger.debug(f"Normalizing batch of {len(batch)} texts using Rust batch.")
        return normalize_text_batch_rust(batch)
    elif rust_single_enabled:
        logger.debug(f"Normalizing batch of {len(batch)} texts using Rust single.")
        return [normalize_text_rust(c) for c in batch]
    else:
        logger.debug(f"Normalizing batch of {len(batch)} texts using Python fallback.")
        return [normalize_text_fallback(c) for c in batch]


def process_and_add_batch(
    current_batch_items: List[Tuple[str, str]],
    vector_store: LanceDBVectorStore,
    rust_batch_enabled: bool,
    rust_single_enabled: bool
):
    if not current_batch_items:
        return True
    try:
        current_batch_texts = [item[0] for item in current_batch_items]
        current_batch_source_ids = [item[1] for item in current_batch_items]
        logger.info(f"Processing batch with texts: {len(current_batch_texts)}, source_ids: {len(current_batch_source_ids)}")

        normalized_batch_texts = _normalize_batch(current_batch_texts, rust_batch_enabled, rust_single_enabled)
        if not normalized_batch_texts:
            logger.warning("Normalization resulted in an empty batch of texts.")
            return True
        
        # If normalization can filter out texts, source_ids would need corresponding adjustments.
        # For a minimal change, we assume normalization doesn't alter the list length in a way that misaligns source_ids.
        vectors = get_embeddings(normalized_batch_texts)
        vector_store.add(vectors, normalized_batch_texts, current_batch_source_ids)
        return True
    
    except Exception as e:
        logger.error(f"Failed to process batch. Error: {e}", exc_info=True)
        return False

def optimized_batch_embedding(
    chunk_generator: Generator[Tuple[str, str], None, None],
    batch_size: int,
    vector_store: LanceDBVectorStore,
    use_rust_batch: bool = True,
    use_rust_single: bool = False
):
    if batch_size <= 0:
        logger.error("batch_size must be positive.")
        raise ValueError("batch_size must be a positive integer.")

    current_batch: List[Tuple[str, str]] = []
    processed_chunks_count = 0
    failed_batches_count = 0

    for chunk in chunk_generator:
        if not (isinstance(chunk, tuple) and len(chunk) == 2 and isinstance(chunk[0], str) and isinstance(chunk[1], str)):
            logger.warning(f"Skipping non-string item from chunk_generator: {type(chunk)}")
            continue
        
        current_batch.append(chunk)
        
        # Process batch when it reaches the desired size
        if len(current_batch) >= batch_size:
            if process_and_add_batch(current_batch, vector_store, use_rust_batch, use_rust_single):
                processed_chunks_count += len(current_batch)
            else:
                failed_batches_count += 1
            current_batch.clear()

    # Process any remaining chunks in the last batch
    if current_batch:
        if process_and_add_batch(current_batch, vector_store, use_rust_batch, use_rust_single):
            processed_chunks_count += len(current_batch)
        else:
            failed_batches_count += 1
        current_batch.clear()

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
            if not initial_ready_state: vector_store.is_ready = True
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