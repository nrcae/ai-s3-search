from fastapi import FastAPI
from app.api import router
from app.index_builder import start_background_indexing

import uvicorn

app = FastAPI()
app.include_router(router)

@app.on_event("startup")
async def startup_event():
    start_background_indexing()

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)