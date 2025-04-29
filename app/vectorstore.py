import faiss
import numpy as np
from typing import List, Iterable, Dict, Any
from cachetools import LRUCache

class FAISSVectorStore:
    def __init__(self, embedding_dim: int = 384, cache_size: int = 1024):
        self.embedding_dim: int = embedding_dim
        try:
            self.index: faiss.Index = faiss.IndexFlatL2(embedding_dim)
        except Exception as e:
             print(f"Error initializing Faiss IndexFlatL2 with dim {embedding_dim}: {e}")
             self.index = None

        self.text_chunks: List[str] = []
        self.is_ready: bool = False
        self.cache: LRUCache = LRUCache(maxsize=cache_size)

    def add(self, vectors: np.ndarray, texts: Iterable[str]):
        if self.index is None:
            # Essential check - cannot proceed without an index
            print("Error: Index not initialized.")
            return

        try:
            # Prepare vectors - convert type only if needed (memory optimization)
            vectors_to_add = vectors.astype(np.float32, copy=False)
            self.index.add(vectors_to_add)
            self.text_chunks.extend(texts)
            self.is_ready = True
            self.cache.clear()

        except Exception as e:
            print(f"Error during add: {e}")

    def search(self, query_vector: np.ndarray, top_k: int = 5, **kwargs: Dict[str, Any]) -> List[str]:
        if not self.is_ready or self.index is None or self.index.ntotal == 0:
            return ["Index is not ready or is empty. Please try again later."]

        try:
            if not isinstance(query_vector, np.ndarray):
                query_vector = np.array(query_vector, dtype=np.float32)
            # Ensure correct dtype (Faiss usually requires float32)
            elif query_vector.dtype != np.float32:
                query_vector = query_vector.astype(np.float32)

            # Ensure query_vector is 2D (most vector search libs expect a matrix)
            if query_vector.ndim == 1:
                query_vector = query_vector.reshape(1, -1)
        except Exception as e:
            print(f"Error processing query vector: {e}")

        # 3. Create Cache Key (using processed vector and parameters)
        try:
            query_bytes = query_vector.tobytes() # Hashable representation of vector
            # Ensure kwargs are hashable and order-independent for caching
            kwargs_tuple = tuple(sorted(kwargs.items()))
            cache_key = (query_bytes, top_k, kwargs_tuple)
        except Exception as e:
            # Handle errors during key creation (e.g., unhashable item in kwargs)
            print(f"Warning: Could not create cache key ({e}). Proceeding without cache for this query.")
            cache_key = None

        if cache_key is not None:
            cached_result = self.cache.get(cache_key)
            if cached_result is not None:
                return cached_result

        # 5. Perform Search (Cache Miss)
        try:
            # Perform the actual search using the underlying index
            distances, indices = self.index.search(query_vector, top_k, **kwargs)

            # Retrieve text chunks based on indices
            results = [self.text_chunks[idx] for idx in indices[0] if 0 <= idx < len(self.text_chunks)]

            # 6. Store Result in Cache (if key was valid)
            if cache_key is not None:
                self.cache[cache_key] = results

            return results

        except IndexError:
            # This might happen if indices returned are out of range for text_chunks
            print(f"Error: Search returned indices out of bounds for text_chunks list.")
            return ["Search result indexing error."]
        except Exception as e:
            print(f"An error occurred during the search operation: {e}")
            return ["An error occurred during the search process."]