from fastapi import APIRouter, Query
import numpy as np
from app.embedder import get_embeddings
from app.index_builder import vector_store

router = APIRouter()

@router.get("/search")
def search(q: str = Query(..., description="Semantic search query")):
    if not vector_store.is_ready:
        return {"results": ["Index is still being built. Please try again later."]}
    q_vector = get_embeddings([q])[0]
    results = vector_store.search(np.array([q_vector]))
    return {"results": results}

@router.get("/status")
def status():
    return {"index_ready": vector_store.is_ready, "index_size": vector_store.index.ntotal}
