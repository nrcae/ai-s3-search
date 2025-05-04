from fastapi import APIRouter, Query, HTTPException
import numpy as np
from typing import List, Tuple, Dict, Any
from app.embedder import get_embeddings
from app.shared_resources import vector_store

router = APIRouter()

@router.get("/search")
async def search_minimal(
    q: str = Query(..., description="Semantic search query"),
    top_k: int = Query(5, ge=1, le=50, description="Number of results")
) -> Dict[str, Any]:
    # 1. Check Index Readiness -> Use HTTPException 503
    if not vector_store.is_ready:
        raise HTTPException(status_code=503, detail="Index not ready. Try again later.")

    # 2. Check Query -> Use HTTPException 400
    query = q.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        # 3. Get Embedding
        query_vector_array = get_embeddings([query])
        if not query_vector_array.size > 0 or query_vector_array.shape[1] != vector_store.embedding_dim:
             raise ValueError("Embedding failed or dimension mismatch.")

        # 4. Perform Search
        results: List[Tuple[float, str]] = vector_store.search(
            query_vector=query_vector_array,
            top_k=top_k
        )

        return {"results": results}

    except ValueError as ve: # Catch specific embedding error
         raise 

@router.get("/status")
def status():
    return {"index_ready": vector_store.is_ready, "index_size": vector_store.index.ntotal}
