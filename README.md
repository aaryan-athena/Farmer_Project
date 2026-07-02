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
- Fetched from **OpenWeatherMap's 5-day/3-hour forecast endpoint**, using
  `urllib` (stdlib, no new dependency). Requires the `OPENWEATHERMAP_API_KEY`
  environment variable — see [Environment variables](#environment-variables--keys-to-update-on-deploy) below.
- This project originally used wttr.in, then Open-Meteo, both keyless. Both
  turned out to be unreliable in production: Render's outbound IP is shared
  across many customers, so their shared-IP free tiers would intermittently
  429 (rate limit) or time out for reasons unrelated to this app's own (tiny)
  call volume. OpenWeatherMap's key gives this app its own quota (free tier:
  1,000 calls/day) that other Render tenants can't exhaust.
- The forecast's first 3-hour entry is used as "current" conditions
  (temperature, humidity, precipitation); the next 16 entries (3h × 16 = 48h)
  are summed for expected rainfall. OpenWeatherMap's own condition codes are
  mapped to the WMO code scheme (`_owm_code_to_wmo`) so the frontend's
  icon/description table and its Hindi translations don't need to know which
  provider is behind them.
- Responses are cached in-process per ~1km grid cell for 15 minutes
  (`_WEATHER_CACHE_TTL`), so repeated diagnoses from the same field don't
  spend extra calls against the quota. On a 429, a global cooldown
  (`_RATE_LIMITED_UNTIL`) skips the network call entirely for a while instead
  of retrying straight into the same limit.
- Combined with the diagnosed disease's optimal temperature window, this
  drives a simple **disease pressure heuristic** (`assess_disease_pressure`):
  `high` / `moderate` / `low` based on how closely current temperature,
  humidity, and wetness match conditions the pathogen favors. Healthy and
  viral classes are always `n/a` (viral spread isn't weather-driven the same
  way; healthy has no pathogen to score).
- If the weather chip is missing from a result (fetch failed), the frontend
  shows a "Try weather again" button that calls `GET /weather?lat=&lon=`
  directly — a lightweight endpoint that re-fetches just the weather without
  re-running the full diagnosis.

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
- **`OPENWEATHERMAP_API_KEY`** (backend, required for weather) — get a free
  key at https://openweathermap.org/api (free tier: 1,000 calls/day, no
  credit card) and set it as an environment variable on Render (Dashboard →
  your service → Environment). Without it, `/predict` and `/weather` still
  work, but responses have no weather data (soil-only) and the frontend shows
  the "Try weather again" button. There is no `.env` file in this repo —
  set it directly in Render's dashboard, not in code.
- **`frontend/index.html`** — the `API_BASE` constant (top of the `<script>`
  block) is hardcoded to the current Render backend URL
  (`https://farmer-project-9yfw.onrender.com`). If you redeploy the backend
  to a new host/URL, update this value before deploying the frontend.

Soil data is a fully offline lookup table and needs no key. No other API keys
or secrets are used anywhere in this project.
