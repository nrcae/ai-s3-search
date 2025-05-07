import numpy as np
import torch
import logging
from typing import List, Generator
from sentence_transformers import SentenceTransformer
from app.config import EMBED_MODEL_NAME

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

model = None
embedding_cache = {}

def get_embeddings(texts: list[str]) -> np.ndarray:
    global model, embedding_cache

    # Lazy load the model only once
    if model is None:
        logger.info("Loading SentenceTransformer model...")
        # Check if MPS (Apple Silicon GPU) is available, otherwise fallback to CPU
        if torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
        logger.debug(f"Embedder: Using device: '{device}' for model '{EMBED_MODEL_NAME}'")
        model = SentenceTransformer(EMBED_MODEL_NAME, device=device)
        logger.info("Model loaded.")

    num_texts = len(texts)
    results = [None] * num_texts

    texts_to_encode_map = {} # texts_to_encode_map: maps a unique text string to a list of its original indices in the input `texts`
    unique_texts_for_model_input = [] # unique_texts_for_model_input: a list of unique text strings that are not in cache and need encoding

    for idx, text in enumerate(texts):
        cached_embedding = embedding_cache.get(text)
        if cached_embedding is not None:
            results[idx] = cached_embedding
        else:
            # If this text is encountered for the first time among uncached texts in this call
            if text not in texts_to_encode_map:
                unique_texts_for_model_input.append(text)
            # Record the original index for this text
            texts_to_encode_map.setdefault(text, []).append(idx)

    if unique_texts_for_model_input:
        logger.debug(f"Encoding {len(unique_texts_for_model_input)} unique uncached texts.")
        new_vectors = model.encode(
            unique_texts_for_model_input,
            show_progress_bar=False,
            convert_to_tensor=False,
            normalize_embeddings=False
        )
        for i, text_encoded in enumerate(unique_texts_for_model_input):
            vector = new_vectors[i]
            embedding_cache[text_encoded] = vector
            # Results for all original occurrences of this text
            for original_idx in texts_to_encode_map[text_encoded]:
                results[original_idx] = vector
    
    return np.array(results)

def chunk_text_generator(text: str, size: int = 100, overlap: int = 50) -> Generator[str, None, None]:

    # Validate parameters to ensure meaningful operation and prevent infinite loops
    if not isinstance(text, str):
        raise TypeError("Input 'text' must be a string.")
    if not isinstance(size, int) or not isinstance(overlap, int):
        raise TypeError("'size' and 'overlap' must be integers.")

    if size <= 0:
        raise ValueError("Chunk size must be positive.")
    if overlap < 0:
        raise ValueError("Overlap cannot be negative.")
    # This is crucial: if size <= overlap, step is <= 0, leading to issues.
    if size <= overlap:
        raise ValueError("Chunk size must be strictly greater than overlap to ensure progress.")

    words = text.split()
    if not words:
        return

    start = 0
    step = size - overlap
    while start < len(words):
        end = min(start + size, len(words))
        chunk = " ".join(words[start:end])
        yield chunk
        start += step
        # No need to check 'if step <=0: break' here because 'size <= overlap' check ensures step > 0

def chunk_text(text: str, size: int = 100, overlap: int = 50) -> List[str]:
    return list(chunk_text_generator(text, size, overlap))