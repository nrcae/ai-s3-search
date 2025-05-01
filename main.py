import uvicorn
from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.api import router
from app.index_builder import start_background_indexing

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Code to run before the application starts accepting requests
    print("INFO:     Lifespan startup: Initiating background indexing...")
    start_background_indexing()
    print("INFO:     Lifespan startup: Background indexing thread started.")

    yield # The application runs while the context manager is active

    # Code to run when the application is shutting down (optional)
    print("INFO:     Lifespan shutdown: Application shutting down.")
    # Add any cleanup logic here if needed

app = FastAPI(lifespan=lifespan)
app.include_router(router)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)