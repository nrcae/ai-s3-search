import lancedb
import uuid
import logging
import threading
import numpy as np
import pyarrow as pa
from typing import List, Tuple, Annotated, Optional
from datetime import datetime, timezone
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
        self.last_indexed_time: Optional[datetime] = None
        self.PydanticSchema = self.create_pydantic_schema(embedding_dim)

        # Define the explicit Arrow schema
        self.arrow_schema = pa.schema([
            pa.field("id", pa.string(), nullable=False),
            pa.field("vector", pa.list_(pa.float32(), self.embedding_dim), nullable=False),
            pa.field("text", pa.string(), nullable=False),
            pa.field("source_id", pa.string(), nullable=True)
        ])

        self.table: Optional[lancedb.table.Table] = None
        self.is_ready: bool = False
        self.cache = LRUCache(maxsize=cache_size)
        self.cache_lock = threading.Lock()


    def create_pydantic_schema(self, dim: int) -> type[LanceModel]:
        class VectorSchema(LanceModel):
            id: str
            vector: Annotated[List[float], Vector(dim)]
            text: str
            source_id: str
        return VectorSchema


    def add(self, vectors: np.ndarray, texts: List[str], source_ids: List[str]):
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
                text=text,
                source_id=source_id
            ) for vector, text, source_id in zip(vectors_np, texts, source_ids)]

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

            self.is_ready = True
            self.last_indexed_time = datetime.now(timezone.utc)
            with self.cache_lock:
                self.cache.clear()
        except Exception as e:
            logger.error(f"Error adding vectors: {e}")
            self.is_ready = False


    def search(self, query_vector: np.ndarray, top_k: int = 2, source_id: Optional[str] = None, **kwargs) -> List[Tuple[float, str, str]]:
        # Ensure the search component is ready and the table exists
        if not self.is_ready or not self.table:
            return []

        try:
            # Prepare query vector: ensure it's a 1D float32 numpy array
            query_vector_np = np.ascontiguousarray(query_vector.astype(np.float32))
            if query_vector_np.ndim == 2 and query_vector_np.shape[0] == 1:
                query_vector_np = query_vector_np.flatten()
            elif query_vector_np.ndim != 1:
                return []
            
            # Check if query vector dimension matches the table's embedding dimension
            # This check was missing in the provided version but present in the original. Adding it back for robustness.
            if query_vector_np.shape[0] != self.embedding_dim:
                logger.error(f"Query vector dimension {query_vector_np.shape[0]} does not match table dimension {self.embedding_dim}")
                return []

            # Attempt to retrieve results from cache
            cache_key = (query_vector_np.tobytes(), top_k)
            with self.cache_lock:
                if cache_key in self.cache:
                    return self.cache[cache_key]

            query_builder = self.table.search(query_vector_np, vector_column_name="vector")
            if source_id:
                # Ensure source_id is properly quoted for the SQL WHERE clause
                # Replace single quotes within source_id to prevent SQL injection if source_id could contain them.
                safe_source_id = source_id.replace("'", "''")
                query_builder = query_builder.where(f"source_id = '{safe_source_id}'")

            # Perform the vector search (only top_k results)
            results_df = query_builder.limit(top_k).select(["text", "source_id", "_distance"]).to_df()
            if results_df.empty:
                return []

            # Extract data from DataFrame
            distances = results_df["_distance"].tolist()
            texts = results_df["text"].tolist()
            # Get source_ids, defaulting to "Unknown" if column is missing or value is NaN
            if "source_id" in results_df.columns:
                source_ids = results_df["source_id"].fillna("Unknown").tolist()
            else:
                source_ids = ["Unknown"] * len(texts)

            # Process results: calculate initial scores
            processed = []
            for dist, text, sid in zip(distances, texts, source_ids):
                score = 1 / (1 + dist)
                processed.append((score, text, sid))

            # Normalize scores to a 0-1 range if there's a variance
            if processed:
                scores_only = [s for s, _, _ in processed]
                min_score, max_score = min(scores_only), max(scores_only)
                if (max_score - min_score) > 1e-9:
                    processed = [((s - min_score)/(max_score - min_score), t, sid)
                                 for s, t, sid in processed]
                else:
                    # All scores are (nearly) identical, set to 1.0
                    processed = [(1.0, t, sid) for _, t, sid in processed]

            with self.cache_lock:
                self.cache[cache_key] = processed

            return processed

        except Exception as e:
            logger.error(f"Search error: {e}")
            return []

    def get_all_source_ids(self) -> List[str]:
    
            if not self.is_ready or self.table is None:
                logger.warning("Vector store not ready or table not initialized. Cannot get source IDs.")
                return []

            if self.table.count_rows() == 0:
                logger.info("Table 'vectors' is empty. No source IDs to return.")
                return []

            try:
                logger.debug("Fetching all source IDs...")
                # Use to_arrow() to get an Arrow Table, then operate on its columns
                # This is generally more memory-efficient for large datasets than to_pandas() directly
                arrow_table = self.table.to_lance().to_table(columns=["source_id"])

                if arrow_table.num_rows == 0:
                    return []

                # Get the source_id column as an Arrow Array
                source_id_column = arrow_table.column("source_id")

                # Get unique values from the Arrow Array
                unique_arrow_array = source_id_column.unique()

                # Convert unique Arrow Array to a Python list
                # Filter out None values and empty strings, convert to string, then ensure uniqueness again with set
                # and finally sort.
                raw_unique_sources = unique_arrow_array.to_pylist()

                valid_sources = set()
                for s in raw_unique_sources:
                    if s is not None:
                        s_str = str(s).strip()
                        if s_str: # Ensure non-empty after stripping
                            valid_sources.add(s_str)

                sorted_sources = sorted(list(valid_sources))
                logger.info(f"Retrieved {len(sorted_sources)} unique source IDs.")
                return sorted_sources

            except Exception as e:
                logger.error(f"Error getting all source IDs: {e}", exc_info=True)
                return []
