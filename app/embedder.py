import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer
from app.config import EMBED_MODEL_NAME

model = None
embedding_cache = {}

def get_embeddings(texts: list[str]) -> np.ndarray:
    global model, embedding_cache

    # Load the model only once (lazy loading)
    if model is None:
        print("Loading SentenceTransformer model...")
        model = SentenceTransformer(EMBED_MODEL_NAME)
        print("Model loaded.")

    results = []
    uncached = []
    uncached_indices = []

    # Check cache for each text
    for idx, text in enumerate(texts):
        if text in embedding_cache:
            results.append(embedding_cache[text])
        else:
            results.append(None)
            uncached.append(text)
            uncached_indices.append(idx)

    # Encode only uncached texts
    if uncached:
        new_vectors = model.encode(uncached, show_progress_bar=False, convert_to_tensor=False)
        for i, vec in enumerate(new_vectors):
            embedding_cache[uncached[i]] = vec
            results[uncached_indices[i]] = vec

    return np.array(results)

def chunk_text(text: str, size=300, overlap=50) -> List[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), size - overlap):
        chunk = " ".join(words[i:i+size])
        chunks.append(chunk)
    return chunks