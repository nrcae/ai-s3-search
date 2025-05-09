import lancedb
import uuid
import logging
import threading
import numpy as np
import pyarrow as pa
from typing import List, Tuple, Annotated, Optional
from cachetools import LRUCache
from lancedb.pydantic import Vector, LanceModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class LanceDBVectorStore:
    def __init__(self, embedding_dim: int = 768, cache_size: int = 1024):
        self.embedding_dim = embedding_dim
        self.db = lancedb.connect("./lancedb_data")

        self.PydanticSchema = self.create_pydantic_schema(embedding_dim)

        # Define the explicit Arrow schema
        self.arrow_schema = pa.schema([
            pa.field("id", pa.string(), nullable=False),
            pa.field("vector", pa.list_(pa.float32(), self.embedding_dim), nullable=False),
            pa.field("text", pa.string(), nullable=False)
        ])

        self.table: Optional[lancedb.table.Table] = None
        self.text_chunks: List[str] = []
        self.is_ready: bool = False
        self.cache = LRUCache(maxsize=cache_size)
        self.cache_lock = threading.Lock()

    def create_pydantic_schema(self, dim: int) -> type[LanceModel]:
        class VectorSchema(LanceModel):
            id: str
            vector: Annotated[List[float], Vector(dim)]
            text: str
        return VectorSchema

    def add(self, vectors: np.ndarray, texts: List[str]):
        try:
            # Ensure input vectors are float32
            vectors_np = np.ascontiguousarray(vectors.astype(np.float32))

            # Prepare data using Pydantic models for structure.
            # When vector.tolist() is called, np.float32 becomes Python float (64-bit).
            # LanceDB will convert these to Arrow Float32 during ingestion
            # because the table's schema (self.arrow_schema) dictates it.
            data_pydantic_models = [self.PydanticSchema(
                id=str(uuid.uuid4()),
                vector=vector.tolist(),
                text=text
            ) for vector, text in zip(vectors_np, texts)]

            if not data_pydantic_models:
                logger.warning("No data to add after Pydantic model preparation.")
                return

            if self.table is None:
                # Step 3: Use the explicit self.arrow_schema for table creation
                self.table = self.db.create_table(
                    "vectors",
                    schema=self.arrow_schema,
                    mode="overwrite",
                    exist_ok=True
                )
                # Add data (list of Pydantic models). LanceDB converts based on table's Arrow schema.
                self.table.add(data_pydantic_models)
            else:
                # Table already exists, assume its schema is correct or was previously corrected.
                self.table.add(data_pydantic_models)

            self.text_chunks.extend([item.text for item in data_pydantic_models])
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
            # Ensure query vector is 1D float32 numpy array
            query_vector_np = np.ascontiguousarray(query_vector.astype(np.float32))
            if query_vector_np.ndim == 2 and query_vector_np.shape[0] == 1:
                query_vector_np = query_vector_np.flatten()
            elif query_vector_np.ndim != 1: # Check if it's not already 1D
                logger.error(f"Query vector has incorrect shape: {query_vector_np.shape}, must be 1D or (1, D).")
                return []
            
            if query_vector_np.shape[0] != self.embedding_dim:
                 logger.error(f"Query vector dimension {query_vector_np.shape[0]} does not match table dimension {self.embedding_dim}")
                 return []


            cache_key = (query_vector_np.tobytes(), top_k)
            with self.cache_lock:
                if cache_key in self.cache:
                    return self.cache[cache_key]

            # Perform search, ensuring vector_column_name is specified
            results_df = self.table.search(
                query_vector_np,
                vector_column_name="vector"
            ).limit(top_k).to_df()

            processed = []
            for _, row in results_df.iterrows():
                score = 1 / (1 + row["_distance"]) # L2 to similarity
                processed.append((score, row["text"]))

            # Normalize scores
            if processed:
                scores_only = [s for s, _ in processed] # Corrected to scores_only
                min_score, max_score = min(scores_only), max(scores_only)
                if (max_score - min_score) > 1e-9: # Avoid division by zero
                    processed = [((s - min_score)/(max_score - min_score), t)
                                for s, t in processed]
                else: # All scores are the same
                    processed = [(1.0, t) for _, t in processed]

            with self.cache_lock:
                self.cache[cache_key] = processed

            return processed

        except Exception as e:
            logger.error(f"Search error: {e}")
            return []
