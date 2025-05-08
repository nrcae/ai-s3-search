import faiss
import numpy as np
from typing import List, Dict, Any, Tuple
from cachetools import LRUCache
import threading
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class FAISSVectorStore:
    def __init__(self, embedding_dim: int = 768, cache_size: int = 1024):
        self.embedding_dim: int = embedding_dim
        try:
            self.index: faiss.Index = faiss.IndexFlatL2(embedding_dim)
            logger.debug(f" Initialized Faiss IndexFlatL2 with dimension {embedding_dim}")
        except Exception as e:
            self.index = None
            logger.error(f" Initializing Faiss IndexFlatL2 with dim {embedding_dim}: {e}")
            
        self.text_chunks: List[str] = []
        self.is_ready: bool = False
        self.cache: LRUCache = LRUCache(maxsize=cache_size)
        self.cache_lock = threading.Lock()

    def add(self, vectors: np.ndarray, texts: List[str]):
        if self.index is None: logger.error(" Index not initialized."); return
        try:
            vectors_np = np.ascontiguousarray(vectors.astype(np.float32))
            if vectors_np.shape[1] != self.embedding_dim:
                 logger.error(f" Add vector dim {vectors_np.shape[1]} != index dim {self.embedding_dim}.")
                 return
            self.index.add(vectors_np)
            self.text_chunks.extend(texts)
            self.is_ready = True # Mark ready after first add
            with self.cache_lock: self.cache.clear()
        except Exception as e: logger.error(f" During add: {e}")


    def search(self, query_vector: np.ndarray, top_k: int = 2, **kwargs: Dict[str, Any]) -> List[Tuple[float, str]]:
        # 1. Readiness Check
        if not self.is_ready or self.index is None or self.index.ntotal == 0:
            return []

        # 2. Prepare Query Vector
        try:
            query_vector = query_vector # Use shorter name
            if not isinstance(query_vector, np.ndarray): query_vector = np.array(query_vector)
            if query_vector.dtype != np.float32: query_vector = query_vector.astype(np.float32)
            if query_vector.ndim == 1: query_vector = query_vector.reshape(1, -1)

            if query_vector.shape[1] != self.embedding_dim:
                logger.error(f" Query vector dim {query_vector.shape[1]} != index dim {self.embedding_dim}.")
                return []
        except Exception as e:
            logger.error(f" Processing query vector: {e}")
            return []

        # 3. Cache Key Logic (Kept for correctness if kwargs used)
        cache_key = None
        try:
            cache_key = (query_vector.tobytes(), top_k, tuple(sorted(kwargs.items())))
        except Exception as e:
            logger.warning(f" Could not create cache key ({e}).")

        # 4. Cache Check
        if cache_key: # Check if key creation succeeded
            with self.cache_lock: cached_result = self.cache.get(cache_key)
            if cached_result is not None: return cached_result

        # 5. Perform Search
        try:
            actual_k = min(top_k, self.index.ntotal)
            if actual_k <= 0: return []

            distances, indices = self.index.search(query_vector, actual_k, **kwargs)

            # 6. Assemble Results using List Comprehension (More concise)
            results = [
                (float(distances[0, i]), self.text_chunks[idx])
                for i, idx in enumerate(indices[0]) # Iterate through indices directly
                if 0 <= idx < len(self.text_chunks)
            ] if indices.size > 0 else []

            if results:
                scores = [score for score, _ in results]
                min_score = min(scores)
                max_score = max(scores)
                if max_score > min_score:
                    # Normalize in-place for just the results shown
                    results = [((score - min_score) / (max_score - min_score), text) for score, text in results]
                else:
                    results = [(1.0, text) for _, text in results]

        except Exception as e:
            logger.error(f" An error occurred during search/result assembly: {e}")
            return []

        # 7. Store in Cache
        if cache_key: # Check if key creation succeeded
            with self.cache_lock: self.cache[cache_key] = results

        return results