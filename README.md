# 🌾 KisaanMitra — AI Crop Disease Detection

KisaanMitra is an AI-powered crop disease detection system built for farmers in Haryana.

## Model
- **Architecture:** EfficientNetB3 (transfer learning)
- **Classes:** 17 across 5 crops — Corn, Potato, Rice, Wheat, Sugarcane
- **Dataset:** 25,828 images
- **Test Accuracy:** 98%

## Crops & Diseases Detected
| Crop | Diseases |
|------|----------|
| Corn | Common Rust, Northern Leaf Blight, Healthy |
| Potato | Early Blight, Late Blight, Healthy |
| Rice | Bacterial Blight, Blast, Healthy |
| Wheat | Yellow Rust, Brown Rust, Black Rust, Leaf Blight, Healthy |
| Sugarcane | Mosaic, Red Rot, Healthy |

## Training Phases
| Phase | Strategy | Val Accuracy |
|-------|----------|-------------|
| Phase 1 | Frozen base | 98.40% |
| Phase 2 | Fine-tuned top 30 layers | 98.27% |
| Phase 3 | Targeted augmentation | 98.45% |
| Phase 4 | Class-weighted loss (deployed) | 98.32% |

## Stack
- TensorFlow / Keras
- Google Colab (T4 GPU)
- FastAPI (backend, hosted on Render)
- Static HTML/JS frontend (hosted on Vercel)

## Project structure
```
backend/     FastAPI app, model weights, inference/label/context logic, requirements.txt, runtime.txt
frontend/    Static single-page UI (index.html)
notebooks/   Training notebook (Kissan_1.ipynb)
vercel.json  Deploys frontend/ as a static site
```

## Architecture

```
frontend/index.html  --POST /predict (image + crop + lat/lon)-->  backend/main.py
                                                                        |
                                                                        |-- inference.py    (crop-restricted argmax over the 17 classes)
                                                                        |-- labels.py       (crop -> class-index groups)
                                                                        `-- agro_context.py (soil + weather + advisory enrichment)
```

1. The farmer uploads/captures a leaf photo in the browser, optionally declares
   a crop, and (if location permission is granted) the browser sends `lat`/`lon`.
2. `backend/main.py` loads the EfficientNetB3 `.keras` model **once**, on a
   background thread at startup (`lifespan`), so Render's port scan doesn't
   time out waiting on the ~78MB load. `/predict` returns `503` until the
   model finishes loading, and `/health` reports `model_loaded` /
   `model_error` so you can see load state without digging through logs.
3. The image is preprocessed (resized to the model's input size, fed as raw
   0-255 floats — EfficientNet's `preprocess_input` is baked into the model
   as a `Lambda` layer) and run through the model.
4. If a crop was declared, `inference.py` zeroes out every class that doesn't
   belong to that crop and renormalizes, so a wheat photo can never be
   misclassified as a sugarcane disease.
5. If `lat`/`lon` were sent, `agro_context.py` builds a location-aware
   advisory (see below) and attaches it to the response as `context`.

### Soil data
- Fully **offline** — no external API or key. `agro_context.py` ships a
  hardcoded list of ~30 reference points across India's major agricultural
  regions (dense around Haryana, the target region), each tagged with a soil
  type (alluvial, arid/sandy, black/vertisol, red, laterite, forest/hill).
- Given the farmer's coordinates, it finds the **nearest reference point**
  via haversine distance and returns that point's soil profile (texture, pH
  tendency, drainage, and agronomic notes).
- This is a nearest-neighbor approximation, not a soil survey lookup — it
  degrades gracefully outside the densely-mapped belt but is only as precise
  as the nearest reference point.

### Weather data
- Fetched from **wttr.in** (`https://wttr.in/{lat},{lon}?format=j1`) — free,
  keyless, called via Python's stdlib `urllib` so it adds no dependency.
- Pulls current conditions (temperature, humidity, precipitation) plus a
  short forecast, and sums expected rainfall over the next 48 hours.
- Responses are cached in-process per ~1km grid cell for 15 minutes
  (`_WEATHER_CACHE_TTL`), so repeated diagnoses from the same field don't
  hammer the upstream service.
- Combined with the diagnosed disease's optimal temperature window, this
  drives a simple **disease pressure heuristic** (`assess_disease_pressure`):
  `high` / `moderate` / `low` based on how closely current temperature,
  humidity, and wetness match conditions the pathogen favors. Healthy and
  viral classes are always `n/a` (viral spread isn't weather-driven the same
  way; healthy has no pathogen to score).

### Advisory assembly
`build_advisory()` combines the three signals above into a single response:
disease-pressure framing, a per-disease baseline management action, a
spray-timing tip derived from rain/heat, and a soil-driven caveat (e.g. flag
waterlogging risk on black soil during a wet spell). Every response carries a
disclaimer pointing farmers to their local Krishi Vigyan Kendra (KVK) before
acting on dosage/product specifics.

## Running locally
```bash
# backend
cd backend
pip install -r requirements.txt
uvicorn main:app --reload

# frontend — just open frontend/index.html, or serve it statically
```

## Environment variables / keys to update on deploy
No API keys are used anywhere in this project — weather comes from wttr.in and
soil data is an offline lookup table, both keyless. The one thing that **must**
be updated when redeploying is:

- **`frontend/index.html`** — the `API_BASE` constant (top of the `<script>`
  block) is hardcoded to the current Render backend URL
  (`https://kisaanmitra-api-a4g7.onrender.com`). If you redeploy the backend
  to a new host/URL, update this value before deploying the frontend.

There is no `.env` file and no secrets in this repo.
