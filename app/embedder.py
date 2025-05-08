import numpy as np
import torch
import logging
import re
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

def chunk_text_generator(text: str, size: int=500, overlap: int=200) -> Generator[str, None, None]:
    sentences = re.split(r'(?<=[.!?])\s+', text)  # Split at sentence endings
    
    current_chunk = []
    current_length = 0
    overlap_buffer = []
    
    for sentence in sentences:
        sentence_word_count = len(sentence.split())
        
        if current_length + sentence_word_count > size:
            # Yield current chunk
            yield ' '.join(current_chunk)
            
            # Retain overlap as full sentences
            overlap_buffer = current_chunk[-int(overlap/10):]  # Keep last 15 sentences (if overlap=150)
            current_chunk = overlap_buffer.copy()
            current_length = sum(len(s.split()) for s in current_chunk)
        
        current_chunk.append(sentence)
        current_length += sentence_word_count
    
    if current_chunk:
        yield ' '.join(current_chunk)

def chunk_text(text: str, size: int = 500, overlap: int = 200) -> List[str]:
    return list(chunk_text_generator(text, size, overlap))