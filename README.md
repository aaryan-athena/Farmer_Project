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
