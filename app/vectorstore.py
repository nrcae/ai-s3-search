import faiss
import numpy as np
from typing import List, Iterable

class FAISSVectorStore:
    def __init__(self, dim=384):
        self.index = faiss.IndexFlatL2(dim)
        self.text_chunks = []
        self.is_ready = False

    def add(self, vectors: np.ndarray, texts: Iterable[str]):
        if self.index is None:
            # Essential check - cannot proceed without an index
            print("Error: Index not initialized.")
            return

        try:
            # Prepare vectors - convert type only if needed (memory optimization)
            vectors_to_add = vectors.astype(np.float32, copy=False)

            # Add to index and store texts
            self.index.add(vectors_to_add)
            self.text_chunks.extend(texts)
            self.is_ready = True

        except Exception as e:
            print(f"Error during add: {e}")

    def search(self, query_vector: np.ndarray, top_k: int = 5):
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

            distances, indices = self.index.search(query_vector, top_k)

            # 4. Efficient Result Retrieval (list comprehension is fine and readable)
            # Access indices for the first (and likely only) query vector in the batch
            results = [self.text_chunks[idx] for idx in indices[0]]
            return results

        except IndexError:
            # Handle cases where indices returned by search might be out of bounds
            print(f"Error: Search returned invalid indices: {indices[0]}")
            return ["Search result indexing error."]
        except Exception as e:
            # Catch other potential errors during search or type conversion
            print(f"An error occurred during search: {e}")
            return ["An error occurred during the search process."]