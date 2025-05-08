import lancedb
import numpy as np
import uuid
import logging
from typing import List, Tuple
from cachetools import LRUCache
import threading
from lancedb.pydantic import Vector, LanceModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


from lancedb.pydantic import LanceModel, Vector
from typing import Optional

class LanceDBVectorStore:
    def __init__(self, embedding_dim: int = 768, cache_size: int = 1024):
        self.embedding_dim = embedding_dim
        self.db = lancedb.connect("./lancedb_data")
        
        # Define schema as instance attribute
        self.Schema = self.create_schema(embedding_dim)
        
        self.table: Optional[lancedb.table.Table] = None
        self.text_chunks: List[str] = []
        self.is_ready: bool = False
        self.cache = LRUCache(maxsize=cache_size)
        self.cache_lock = threading.Lock()

    def create_schema(self, dim: int) -> type[LanceModel]:
        """Dynamically create schema class with proper dimension"""
        class VectorSchema(LanceModel):
            id: str
            vector: Vector(dim)
            text: str
        return VectorSchema

    def add(self, vectors: np.ndarray, texts: List[str]):
        try:
            vectors_np = np.ascontiguousarray(vectors.astype(np.float32))
            
            # Use the instance's schema class
            data = [self.Schema(
                id=str(uuid.uuid4()),
                vector=vector.tolist(),
                text=text
            ) for vector, text in zip(vectors_np, texts)]

            if self.table is None:
                self.table = self.db.create_table(
                    "vectors",
                    data=data,
                    schema=self.Schema,  # Explicit schema
                    mode="overwrite",
                    exist_ok=True
                )
            else:
                self.table.add(data)

            self.text_chunks.extend(texts)
            self.is_ready = True
            with self.cache_lock:
                self.cache.clear()

        except Exception as e:
            logger.error(f"Error adding vectors: {e}")
            self.is_ready = False

    def search(self, query_vector: np.ndarray, top_k: int = 2, **kwargs) -> List[Tuple[float, str]]:
        if not self.is_ready or not self.table:
            return []

        try:
            query_vector = np.ascontiguousarray(query_vector.astype(np.float32))
            if query_vector.ndim == 1:
                query_vector = query_vector.reshape(1, -1)

            # Check cache
            cache_key = (query_vector.tobytes(), top_k)
            with self.cache_lock:
                if cache_key in self.cache:
                    return self.cache[cache_key]

            # Perform search
            results = self.table.search(query_vector).limit(top_k).to_list()
            
            # Convert L2 distances to similarities
            processed = []
            for result in results:
                try:
                    score = 1 / (1 + result["_distance"])  # L2 to similarity
                    processed.append((score, result["text"]))
                except KeyError:
                    continue

            # Normalize scores
            if processed:
                scores = [s for s, _ in processed]
                min_score, max_score = min(scores), max(scores)
                if max_score > min_score:
                    processed = [((s - min_score)/(max_score - min_score), t) 
                                for s, t in processed]
                else:
                    processed = [(1.0, t) for _, t in processed]

            # Update cache
            with self.cache_lock:
                self.cache[cache_key] = processed

            return processed[:top_k]

        except Exception as e:
            logger.error(f"Search error: {e}")
            return []
