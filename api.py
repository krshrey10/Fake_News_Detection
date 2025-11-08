# --- quiet transformers warnings (must be before any transformers imports) ---

from __future__ import annotations

from __future__ import annotations

# --- quiet transformers warnings (must be before any transformers imports) ---
import warnings
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module="transformers.tokenization_utils_base",
)
from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()
# ---------------------------------------------------------------------------

import os
import json
import joblib
import warnings
from typing import Callable, Any

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
from transformers.utils import logging as hf_logging

# -------------------- Warning/Logging Tweaks --------------------
# Silence future-warning noise from transformers and lower HF verbosity
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module="transformers.tokenization_utils_base",
)
hf_logging.set_verbosity_error()

# -------------------- Paths --------------------
MODEL_TFIDF = os.path.join("models", "fake_news_pipeline.joblib")
META_TFIDF  = os.path.join("reports", "model_meta.json")
MODEL_SBERT = os.path.join("models", "fake_news_sbert.joblib")
META_SBERT  = os.path.join("reports", "sbert_meta.json")

# -------------------- Backend Loader --------------------
def load_backend() -> tuple[
    Callable[[list[str]], list[str]] | None,
    Callable[[list[str]], list[list[float]]] | None,
    dict[str, Any],
    list[str],
]:
    """
    Prefer SBERT (SentenceTransformer embeddings + classifier) if present.
    Fallback to TF-IDF sklearn pipeline otherwise.

    Returns:
        predict_fn(texts) -> list[str]
        proba_fn(texts) -> list[list[float]] | None
        meta (dict)
        classes (list[str])
    """
    # Try SBERT
    if os.path.exists(MODEL_SBERT):
        try:
            art = joblib.load(MODEL_SBERT)  # {"encoder_name","classifier","classes"?}
            from sentence_transformers import SentenceTransformer
            enc = SentenceTransformer(art["encoder_name"])
            clf = art["classifier"]
            classes = list(getattr(clf, "classes_", []))

            def predict(texts: list[str]) -> list[str]:
                embs = enc.encode(texts, convert_to_numpy=True)
                return clf.predict(embs).tolist()

            def proba(texts: list[str]) -> list[list[float]] | None:
                if hasattr(clf, "predict_proba"):
                    embs = enc.encode(texts, convert_to_numpy=True)
                    return clf.predict_proba(embs).tolist()
                return None

            meta: dict[str, Any] = {}
            if os.path.exists(META_SBERT):
                with open(META_SBERT, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            meta.setdefault("model_type", "sbert_logreg")
            if "classes" not in meta:
                meta["classes"] = classes
            return predict, proba, meta, classes
        except Exception as e:
            print("SBERT load failed, falling back to TF-IDF:", e)

    # Fallback: TF-IDF
    if os.path.exists(MODEL_TFIDF):
        pipe = joblib.load(MODEL_TFIDF)
        classes = list(getattr(pipe, "classes_", []))

        def predict(texts: list[str]) -> list[str]:
            return pipe.predict(texts).tolist()

        def proba(texts: list[str]) -> list[list[float]] | None:
            return pipe.predict_proba(texts).tolist() if hasattr(pipe, "predict_proba") else None

        meta: dict[str, Any] = {}
        if os.path.exists(META_TFIDF):
            with open(META_TFIDF, "r", encoding="utf-8") as f:
                meta = json.load(f)
        meta.setdefault("model_type", "tfidf_logreg")
        if "classes" not in meta:
            meta["classes"] = classes
        return predict, proba, meta, classes

    # Nothing available
    return None, None, {}, []

predict_fn, proba_fn, meta, classes = load_backend()

# -------------------- App & CORS --------------------
app = FastAPI(title="Fake News Detector API", version="1.0")

# Allow cross-origin calls (tighten 'allow_origins' in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # e.g., ["https://your-app.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- Optional API Key Auth --------------------
API_KEY = os.getenv("API_KEY", "")  # set in env for production

def require_key(x_api_key: str = Header(default="")) -> None:
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

# -------------------- Schemas --------------------
class Item(BaseModel):
    text: str

class Batch(BaseModel):
    texts: list[str]

# -------------------- Meta / Utility Routes --------------------
@app.get("/", tags=["meta"])
def root():
    """Simple landing endpoint with pointers."""
    return {"service": "Fake News Detector API", "docs": "/docs", "health": "/health", "version": "/version"}

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    """Return an empty favicon to avoid 404 noise."""
    return Response(content=b"", media_type="image/x-icon", status_code=204)

@app.get("/hybridaction/{rest_of_path:path}", include_in_schema=False)
def ignore_hybridaction(rest_of_path: str):
    """Silence random extension/analytics requests."""
    return Response(status_code=204)

@app.get("/health", tags=["meta"])
def health():
    return {
        "status": "ok",
        "has_model": predict_fn is not None,
        "meta": meta,
        "classes": classes,
    }

@app.get("/version", tags=["meta"])
def version():
    active = meta.get("model_type", "unknown")
    return {"active_model": active, "meta": meta, "classes": classes}

# -------------------- Inference Routes --------------------
@app.post("/predict")
def predict_one(item: Item, _: None = require_key()):
    if predict_fn is None:
        return {"error": "model not loaded"}
    pred = predict_fn([item.text])[0]
    resp: dict[str, Any] = {"prediction": pred}
    if proba_fn is not None:
        P = proba_fn([item.text])
        if P is not None:
            resp["probabilities"] = dict(zip(classes, map(float, P[0])))
    return resp

@app.post("/predict_batch")
def predict_many(batch: Batch, _: None = require_key()):
    if predict_fn is None:
        return {"error": "model not loaded"}
    preds = predict_fn(batch.texts)
    out: dict[str, Any] = {"predictions": preds}
    if proba_fn is not None:
        P = proba_fn(batch.texts)
        if P is not None:
            out["probabilities"] = [dict(zip(classes, map(float, row))) for row in P]
    return out
