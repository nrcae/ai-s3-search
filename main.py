import uvicorn
import logging
import os
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from app.api import router
from app.index_builder import start_background_indexing
from app.shared_resources import vector_store

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', 
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Define the path to the app directory for constructing static file paths
APP_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(APP_DIR, "app", "static")

@asynccontextmanager
async def lifespan(app: FastAPI):
    start_background_indexing(vector_store)
    logger.info("Lifespan startup: Background indexing thread started.")
    yield
    logger.debug("Lifespan shutdown: Application shutting down.")

app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

app.include_router(router)

@app.get("/", response_class=HTMLResponse)
async def get_minimal_ui_html(request: Request):
    html_file_path = os.path.join(STATIC_DIR, "ui", "index.html")
    try:
        return FileResponse(html_file_path, media_type="text/html")
    except RuntimeError as e: # FileResponse can raise RuntimeError if file not found in some cases
        logger.error(f"Error serving index.html: {e}")
        return HTMLResponse(content="<h1>Error: UI file not found.</h1>", status_code=500)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)