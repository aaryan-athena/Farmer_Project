"""
KisaanMitra FastAPI backend.

Endpoints:
  GET  /health   -> liveness + model status
  GET  /crops    -> list of crops the farmer can declare (derived from labels)
  POST /predict  -> disease prediction (+ soil/weather advisory when lat/lon sent)

Lives in backend/, next to:
  - agro_context.py           (the enrichment layer + single source of truth for labels)
  - labels.py                 (crop groups, derived from agro_context)
  - inference.py              (crop-masking helper)
  - kisaanmitra_phase4.keras  (the model weights, via Git LFS)
"""

import io
import threading
import numpy as np
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input

from labels import CROP_GROUPS, CROPS      # crop -> [class indices], derived from agro_context
from inference import restrict_to_crop     # masks logits to the declared crop
import agro_context                         # single source of truth for class names (0-16)

MODEL_PATH = "kisaanmitra_phase4.keras"

model = None
model_error = None  # set by the loader thread if the load throws; surfaced on /health
IMG_SIZE = (300, 300)  # EfficientNetB3 default; overwritten from the model at load


def build_label(idx: int):
    """Return (human label, is_healthy) from the verified alphabetical map."""
    info = agro_context.DISEASE_INFO.get(int(idx))
    if not info:
        return f"Class {idx}", False
    return f"{info['crop']} — {info['name']}", info["type"] == "healthy"


def _load_model():
    """Load the model off the request path so Uvicorn can bind its port instantly.
    Render's port scan times out (~60-90s) if anything blocks before the port opens,
    and a cold ~78MB Keras load blows past that budget. This runs in a daemon thread."""
    global model, IMG_SIZE, model_error
    try:
        m = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False,
            safe_mode=False,  # required: the model has a Lambda(preprocess_input) layer
            custom_objects={"preprocess_input": preprocess_input},
        )
        shape = m.input_shape  # (None, H, W, 3)
        if shape and shape[1] and shape[2]:
            IMG_SIZE = (int(shape[2]), int(shape[1]))  # PIL expects (width, height)
        model = m  # assign LAST: /predict gates on `model is not None`, so IMG_SIZE is ready first
        print(f"Model loaded. Input size {IMG_SIZE}, {len(agro_context.DISEASE_INFO)} classes.")
    except Exception as e:
        model_error = f"{type(e).__name__}: {e}"
        print(f"MODEL LOAD FAILED -> {model_error}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Kick the load onto a daemon thread and return immediately so the port binds now.
    # The thread fills in `model` when ready; /predict returns 503 until then.
    threading.Thread(target=_load_model, daemon=True).start()
    yield


app = FastAPI(title="KisaanMitra API", lifespan=lifespan)

# NOTE: with allow_origins=["*"] you must keep allow_credentials=False — the CORS
# spec forbids a wildcard origin together with credentials, and FastAPI then sends
# NO Access-Control-Allow-Origin header at all (the exact bug you saw before).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def preprocess(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize(IMG_SIZE)
    # EfficientNetB3 has preprocess_input baked in as a Lambda layer.
    # Feed RAW float32 in 0-255. Do NOT divide by 255 — that drops accuracy to ~0.
    arr = np.array(img, dtype=np.float32)
    return np.expand_dims(arr, axis=0)


@app.get("/crops")
def crops():
    return {"crops": CROPS}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_error": model_error,
    }


@app.get("/weather")
def weather(lat: float, lon: float):
    """Standalone weather lookup so the frontend can retry just the weather
    fetch (e.g. after a transient upstream failure) without re-running the
    whole diagnosis."""
    w = agro_context.get_weather(lat, lon)
    if not w or w.get("temp_c") is None:
        debug = (w or {}).get("_debug_error", "no response")
        print(f"WEATHER FETCH FAILED (lat={lat}, lon={lon}) -> {debug}")
        raise HTTPException(status_code=503, detail=f"Weather service unavailable: {debug}")
    return {"weather": w}


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    crop: str | None = Form(None),
    lat: float | None = Form(None),
    lon: float | None = Form(None),
):
    if model is None:
        detail = (
            f"Model failed to load: {model_error}"
            if model_error
            else "Model is still loading — retry in a few seconds"
        )
        raise HTTPException(status_code=503, detail=detail)

    # Fail fast on a bad crop name before doing any image work.
    if crop is not None and crop not in CROP_GROUPS:
        raise HTTPException(status_code=422, detail=f"Unknown crop '{crop}'. Valid: {CROPS}")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    image_bytes = await file.read()
    if len(image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image too large (max 10MB)")

    try:
        arr = preprocess(image_bytes)
        preds = model.predict(arr, verbose=0)[0]

        # If the farmer declared a crop, restrict to that crop's classes and
        # renormalize. This makes cross-crop errors (e.g. wheat -> sugarcane)
        # impossible by construction. With no crop, the full 17-way runs as before.
        if crop is not None:
            preds = restrict_to_crop(preds, CROP_GROUPS[crop])

        idx = int(np.argmax(preds))
        confidence = float(preds[idx])
        label, is_healthy = build_label(idx)

        top3_idx = np.argsort(preds)[::-1][:3]
        top3 = [
            {"label": build_label(int(i))[0], "confidence": round(float(preds[int(i)]) * 100, 2)}
            for i in top3_idx
        ]

        response = {
            "disease": label,
            "confidence": round(confidence * 100, 2),  # already 0-100; frontend does NOT re-scale
            "healthy": is_healthy,
            "class_index": idx,
            "crop_filtered": crop is not None,
            "top3": top3,
        }

        # Soil + weather + curated advisory — only when the app sent a location.
        if lat is not None and lon is not None:
            response["context"] = agro_context.build_advisory(idx, lat, lon)

        return response

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
