# vectorstore.py
import faiss
import numpy as np
from typing import List, Iterable, Dict, Any, Tuple
from cachetools import LRUCache
import threading

class FAISSVectorStore:
    def __init__(self, embedding_dim: int = 768, cache_size: int = 1024):
        self.embedding_dim: int = embedding_dim
        try:
            self.index: faiss.Index = faiss.IndexFlatL2(embedding_dim)
            print(f"Initialized Faiss IndexFlatL2 with dimension {embedding_dim}")
        except Exception as e:
             print(f"Error initializing Faiss IndexFlatL2 with dim {embedding_dim}: {e}")
             self.index = None

        self.text_chunks: List[str] = []
        self.is_ready: bool = False
        self.cache: LRUCache = LRUCache(maxsize=cache_size)
        self.cache_lock = threading.Lock()

    def add(self, vectors: np.ndarray, texts: Iterable[str]):
        if self.index is None: print("Error: Index not initialized."); return
        try:
            vectors_to_add = vectors.astype(np.float32, copy=False)
            if vectors_to_add.shape[1] != self.embedding_dim:
                 print(f"Error: Vector dim {vectors_to_add.shape[1]} != index dim {self.embedding_dim}."); return
            self.index.add(vectors_to_add)
            self.text_chunks.extend(texts)
            self.is_ready = True
            with self.cache_lock: # Lock when clearing cache
                self.cache.clear()
            print(f"Added {len(vectors_to_add)} vectors. Index total: {self.index.ntotal}")
        except Exception as e: print(f"Error during add: {e}")


    def search(self, query_vector: np.ndarray, top_k: int = 5, **kwargs: Dict[str, Any]) -> List[Tuple[float, str]]: # Return score and text
        """Searches the index, returns list of (distance, text)."""
        if not self.is_ready or self.index is None or self.index.ntotal == 0:
            return []

        try:
            if not isinstance(query_vector, np.ndarray):
                query_vector = np.array(query_vector, dtype=np.float32)
            elif query_vector.dtype != np.float32:
                query_vector = query_vector.astype(np.float32)

            # Ensure query_vector is 2D and correct dimension
            if query_vector.ndim == 1:
                 if query_vector.shape[0] != self.embedding_dim:
                      print(f"Error: Query vector dim {query_vector.shape[0]} != index dim {self.embedding_dim}.")
                      return [] # Return empty list on error
                 query_vector = query_vector.reshape(1, -1)
            elif query_vector.shape[1] != self.embedding_dim:
                 print(f"Error: Query vector dim {query_vector.shape[1]} != index dim {self.embedding_dim}.")
                 return []
        except Exception as e:
            print(f"Error processing query vector: {e}")
            return []

        cache_key = None
        try:
            query_bytes = query_vector.tobytes()
            kwargs_tuple = tuple(sorted(kwargs.items()))
            cache_key = (query_bytes, top_k, kwargs_tuple)
        except Exception as e:
            print(f"Warning: Could not create cache key ({e}). Proceeding without cache.")

        # --- Cache Check (Thread Safe) ---
        if cache_key is not None:
            with self.cache_lock:
                cached_result = self.cache.get(cache_key)
            if cached_result is not None:
                return cached_result

        # --- Perform Search (Cache Miss) ---
        try:
            actual_k = min(top_k, self.index.ntotal)
            if actual_k <= 0: return [] # Cannot search if k=0 or index empty

            distances, indices = self.index.search(query_vector, actual_k, **kwargs)

            # --- Retrieve text chunks and distances ---
            results = []
            if indices.size > 0 and distances.size > 0:
                 for i in range(len(indices[0])):
                     idx = indices[0, i]
                     dist = distances[0, i]
                     if 0 <= idx < len(self.text_chunks):
                         results.append((float(dist), self.text_chunks[idx]))

            # --- Store Result in Cache (Thread Safe) ---
            if cache_key is not None:
                with self.cache_lock:
                    self.cache[cache_key] = results

            return results

        except IndexError:
            print(f"Error: Search returned indices out of bounds.")
            return []
        except Exception as e:
            print(f"An error occurred during the search operation: {e}")
            return []
