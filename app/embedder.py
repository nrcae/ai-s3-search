import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer
from app.config import EMBED_MODEL_NAME

model = None

def get_embeddings(texts: List[str]) -> np.ndarray:
    global model
    if model is None:
        print("Loading SentenceTransformer model...")
        model = SentenceTransformer(EMBED_MODEL_NAME)
        print("Model loaded.")
    return np.array(model.encode(texts, show_progress_bar=False, convert_to_tensor=False))

def chunk_text(text: str, size=300, overlap=50) -> List[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), size - overlap):
        chunk = " ".join(words[i:i+size])
        chunks.append(chunk)
    return chunks