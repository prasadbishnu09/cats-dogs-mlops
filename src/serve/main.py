# src/serve/main.py
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
import logging
from serve.utils import ModelWrapper
from fastapi import Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

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