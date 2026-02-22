# src/serve/main.py
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
import logging
from src.serve.utils import ModelWrapper
from fastapi import Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
import time
import logging
from fastapi import Request


import os
from pathlib import Path

MODEL_PATH = os.getenv("MODEL_PATH", "models/model.pt")

app = FastAPI(title="CatsDogs API")
logger = logging.getLogger("uvicorn.access")
model = None

@app.on_event("startup")
def startup():
    global model
    model = ModelWrapper(MODEL_PATH)
    logger.info("Model loaded from %s", MODEL_PATH)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        res = model.predict_from_bytes(contents)
        return JSONResponse(content=res)
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
def metrics():
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("inference")

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()

    response = await call_next(request)

    process_time = time.time() - start_time
    logger.info(
        f"{request.method} {request.url.path} "
        f"Status: {response.status_code} "
        f"Latency: {process_time:.4f}s"
    )

    return response