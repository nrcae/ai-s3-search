from fastapi import APIRouter, Query, HTTPException, UploadFile, File
from typing import List, Tuple, Dict, Any, Optional
from app.embedder import get_embeddings
from app.shared_resources import vector_store
from app.config import EMBED_MODEL_NAME
from app.index_builder import build_index_background
from app.s3_loader import upload_pdf
import io

router = APIRouter()

@router.post("/upload_pdf")
async def upload_pdf_endpoint(file: UploadFile = File(...)):

    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    try:
        contents = await file.read()
        file_content = io.BytesIO(contents)

        # Upload the file to S3
        if not upload_pdf(file_content, file.filename):
            raise HTTPException(status_code=500, detail="Failed to upload file to S3.")

        # Rebuild the index in the background
        build_index_background(vector_store)

        return {"filename": file.filename, "message": "File uploaded successfully and indexing started."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/search")
async def search_minimal(
    q: str = Query(..., description="Semantic search query"),
    top_k: int = Query(5, ge=1, le=50, description="Number of results")
) -> Dict[str, Any]:
    # Check Index Readiness -> Use HTTPException 503
    if not vector_store.is_ready:
        raise HTTPException(status_code=503, detail="Index not ready. Try again later.")

    # Check Query -> Use HTTPException 400
    query = q.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        # Get Embedding
        query_vector_array = get_embeddings([query])
        if not query_vector_array.size > 0 or query_vector_array.shape[1] != vector_store.embedding_dim:
             raise ValueError("Embedding failed or dimension mismatch.")

        # Perform Search
        results: List[Tuple[float, str, str]] = vector_store.search(
            query_vector=query_vector_array,
            top_k=top_k
        )
        return {"results": results}

    except ValueError as ve:
         raise 


@router.get("/status")
def status() -> Dict[str, Any]:
    last_indexed_str: Optional[str] = None
    if vector_store.last_indexed_time:
        last_indexed_str = vector_store.last_indexed_time.isoformat()

    index_size: int = 0
    if vector_store.table:
        try:
            index_size = vector_store.table.count_rows()
        except Exception as e:
            index_size = -1
    elif vector_store.is_ready and not vector_store.table:
        # If the store is marked "ready" but there's no table, it implies an empty index.
        index_size = 0
    elif not vector_store.is_ready:
        # If not ready, we might not know the size or it's actively changing
        index_size = -1

    return {
        "index_ready": vector_store.is_ready,
        "index_size": index_size,
        "last_indexed_time": last_indexed_str,
        "embedding_model_name": EMBED_MODEL_NAME
    }

