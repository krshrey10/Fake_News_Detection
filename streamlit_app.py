# streamlit_app.py — UI for Fake News Detector (Local TF-IDF/SBERT or Remote API)
import os
import io
import re
import json
import joblib
import pandas as pd
import numpy as np
import requests
import streamlit as st
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple

# ---------------- Paths ----------------
MODEL_TFIDF = os.path.join("models", "fake_news_pipeline.joblib")
META_TFIDF  = os.path.join("reports", "model_meta.json")
MODEL_SBERT = os.path.join("models", "fake_news_sbert.joblib")
META_SBERT  = os.path.join("reports", "sbert_meta.json")
CM_PATH     = os.path.join("reports", "confusion_matrix.png")

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

# ---------------- Remote API client ----------------
class APIClient:
    def __init__(self, base_url: str, timeout: float = 15.0, api_key: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["X-API-KEY"] = api_key

    def health(self) -> dict:
        r = self.session.get(f"{self.base_url}/health", timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def predict_one(self, text: str) -> dict:
        r = self.session.post(
            f"{self.base_url}/predict", json={"text": text},
            headers=self.headers, timeout=self.timeout
        )
        r.raise_for_status()
        return r.json()

    def predict_batch(self, texts: List[str]) -> dict:
        r = self.session.post(
            f"{self.base_url}/predict_batch", json={"texts": texts},
            headers=self.headers, timeout=self.timeout
        )
        r.raise_for_status()
        return r.json()

# ---------------- Local loaders (TF-IDF / SBERT) ----------------
@st.cache_resource
def _load_local_tfidf():
    if not os.path.exists(MODEL_TFIDF):
        return None, {}
    pipe = joblib.load(MODEL_TFIDF)
    meta = {}
    if os.path.exists(META_TFIDF):
        with open(META_TFIDF, "r", encoding="utf-8") as f:
            meta = json.load(f)
    meta.setdefault("model_type", "tfidf_logreg")
    meta.setdefault("classes", list(getattr(pipe, "classes_", [])))
    meta["active"] = "TFIDF"
    return pipe, meta

@st.cache_resource
def _load_local_sbert():
    if not os.path.exists(MODEL_SBERT):
        return None, {}
    art = joblib.load(MODEL_SBERT)  # {"encoder_name","classifier"}
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer(art["encoder_name"])
    clf = art["classifier"]
    meta = {}
    if os.path.exists(META_SBERT):
        with open(META_SBERT, "r", encoding="utf-8") as f:
            meta = json.load(f)
    meta.setdefault("model_type", "sbert_logreg")
    meta.setdefault("classes", list(getattr(clf, "classes_", [])))
    meta["active"] = "SBERT"
    return (enc, clf), meta

# ---------------- TF-IDF explanation helpers ----------------
def explain_tfidf_example_from_disk(model_path: str, text: str, k: int = 8
) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]], str, str]:
    """Return top positive & negative feature contributions for a TF-IDF+LogReg model."""
    pipe = joblib.load(model_path)
    vec = pipe.named_steps["tfidf"]
    clf = pipe.named_steps["clf"]

    pos_label = str(clf.classes_[1])
    neg_label = str(clf.classes_[0])

    X = vec.transform([text])
    vocab = vec.get_feature_names_out()
    coef = clf.coef_[0]

    idxs = X.nonzero()[1]
    contribs = []
    for j in idxs:
        score = float(X[0, j] * coef[j])
        contribs.append((vocab[j], score))

    contribs.sort(key=lambda x: x[1], reverse=True)
    top_pos = contribs[:k]
    top_neg = list(reversed(contribs[-k:]))
    return top_pos, top_neg, pos_label, neg_label


def make_bar_fig(items: List[Tuple[str, float]], title: str):
    """
    items: list[(feature, contribution)]
    returns: (fig, png_bytes)
    """
    labels = [w for w, _ in items]
    vals   = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(5, 3))
    y = np.arange(len(labels))
    ax.barh(y, vals)  # no manual color to respect plotting rules
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel("Contribution")
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight")
    buf.seek(0)
    return fig, buf.read()


def tfidf_token_importance(model_path: str, text: str):
    """
    Convert TF-IDF (uni/bi-gram) contributions into token-level scores.
    Returns: list of (raw_piece, score_or_None) in original text order.
    """
    pipe = joblib.load(model_path)
    vec = pipe.named_steps["tfidf"]
    clf = pipe.named_steps["clf"]

    X = vec.transform([text])
    vocab = vec.get_feature_names_out()
    coef = clf.coef_[0]

    analyzer = vec.build_analyzer()
    tokens = analyzer(text)  # normalized tokens

    # term -> contrib
    idxs = X.nonzero()[1]
    term_contrib = {}
    for j in idxs:
        term = vocab[j]
        score = float(X[0, j] * coef[j])
        term_contrib[term] = term_contrib.get(term, 0.0) + score

    # token aggregation
    per_token = {t: 0.0 for t in tokens}
    for term, s in term_contrib.items():
        if " " in term:
            parts = term.split()
            if len(parts) == 2:
                a, b = parts
                if a in per_token:
                    per_token[a] += s / 2.0
                if b in per_token:
                    per_token[b] += s / 2.0
        else:
            if term in per_token:
                per_token[term] += s

    # reconstruct raw pieces and map to normalized tokens when possible
    word_re = re.compile(r"\w+|\W+")
    raw_parts = word_re.findall(text)
    norm_iter = iter(tokens)
    next_norm = None

    def get_next_norm():
        nonlocal next_norm
        if next_norm is None:
            try:
                next_norm = next(norm_iter)
            except StopIteration:
                next_norm = None
        return next_norm

    out = []
    for part in raw_parts:
        if part.strip() == "" or not re.match(r"\w+", part):
            out.append((part, None))
            continue
        norm = part.lower()
        tok = get_next_norm()
        if tok is not None and tok == norm:
            score = per_token.get(tok, 0.0)
            out.append((part, score))
            next_norm = None
        else:
            out.append((part, None))
    return out


def render_highlighted_text(parts_with_scores, pos_label: str, neg_label: str):
    """
    Render HTML with inline background color:
    green = positive contribution (towards pos_label)
    red   = negative contribution (towards neg_label)
    Intensity = |score| normalized.
    """
    max_abs = max((abs(s) for _, s in parts_with_scores if s is not None), default=0.0)

    def color_for(s):
        if s is None or max_abs == 0:
            return None
        alpha = min(0.85, abs(s) / max_abs)
        if s >= 0:
            return f"rgba(46, 204, 113, {alpha})"  # green
        return f"rgba(231, 76, 60, {alpha})"       # red

    html_parts = []
    for raw, sc in parts_with_scores:
        bg = color_for(sc)
        if bg is None:
            html_parts.append(f"<span>{raw}</span>")
        else:
            title = f"{'+' if sc>=0 else '-'}{abs(sc):.4f}"
            html_parts.append(
                f"<span title='{title}' style='background:{bg}; padding:2px 2px; border-radius:4px'>{raw}</span>"
            )

    legend = (
        f"<div style='margin:.25rem 0 .5rem 0; font-size:0.9rem'>"
        f"<b>Token contributions</b> — "
        f"<span style='background:rgba(46,204,113,.35); padding:2px 6px; border-radius:4px'>→ {pos_label}</span> "
        f"&nbsp; "
        f"<span style='background:rgba(231,76,60,.35); padding:2px 6px; border-radius:4px'>→ {neg_label}</span>"
        f"</div>"
    )
    html = (
        "<div style='line-height:1.9; font-size:1rem; font-family:ui-sans-serif,system-ui,Segoe UI,Roboto'>"
        f"{legend}"
        + "".join(html_parts)
        + "</div>"
    )
    st.markdown(html, unsafe_allow_html=True)

# ---------------- Sidebar: backend/source ----------------
st.sidebar.header("Inference source")
backend = st.sidebar.radio("Choose backend", ["Local (model files)", "Remote API"], index=0)
prefer_local = st.sidebar.radio("Preferred local model  ❓", ["SBERT", "TFIDF"], index=1)
st.sidebar.caption("This only switches the local UI backend (not the FastAPI).")

api_base = None
api_key  = None
if backend == "Remote API":
    api_base = st.sidebar.text_input("API base URL", value="http://127.0.0.1:8000")
    api_key  = st.sidebar.text_input("API key (optional)", value="", type="password")

# ---------------- Resolve active backend ----------------
predict_fn = None
proba_fn   = None
meta       = {}
active_tag = "NONE"
api_client = None

if backend == "Remote API":
    try:
        api_client = APIClient(api_base, api_key=api_key or None)
        health = api_client.health()
        meta = health.get("meta", {})
        meta["classes"] = health.get("classes", meta.get("classes", []))
        active_tag = "API"

        def predict_fn(texts: List[str]):
            if len(texts) == 1:
                r = api_client.predict_one(texts[0])
                return [r.get("prediction", "")]
            r = api_client.predict_batch(texts)
            return r.get("predictions", [])

        def proba_fn(texts: List[str]):
            if len(texts) == 1:
                r = api_client.predict_one(texts[0])
                probs = r.get("probabilities")
                if probs:
                    return np.array([[float(probs[c]) for c in meta.get("classes", [])]])
                return None
            r = api_client.predict_batch(texts)
            plist = r.get("probabilities")
            if plist:
                return np.array([[float(row[c]) for c in meta.get("classes", [])] for row in plist])
            return None

        st.sidebar.json({"backend": "API", **{k: health[k] for k in ["status","has_model"] if k in health}, "meta": meta, "classes": meta.get("classes", [])})
    except Exception as e:
        st.sidebar.error(f"API health failed: {e}")
        st.stop()

else:
    # Local
    if prefer_local == "SBERT":
        sbert, meta = _load_local_sbert()
        if sbert is not None:
            enc, clf = sbert
            def predict_fn(texts: List[str]):
                embs = enc.encode(texts, convert_to_numpy=True)
                return clf.predict(embs)
            def proba_fn(texts: List[str]):
                if hasattr(clf, "predict_proba"):
                    embs = enc.encode(texts, convert_to_numpy=True)
                    return clf.predict_proba(embs)
                return None
            active_tag = "SBERT"
    if active_tag == "NONE":
        # fallback TF-IDF
        pipe, meta = _load_local_tfidf()
        if pipe is not None:
            def predict_fn(texts: List[str]): return pipe.predict(texts)
            def proba_fn(texts: List[str]):  return pipe.predict_proba(texts) if hasattr(pipe, "predict_proba") else None
            active_tag = "TFIDF"

    st.sidebar.subheader("Model info")
    st.sidebar.json({
        "active": active_tag,
        "type": meta.get("model_type", "unknown"),
        "version": meta.get("version", "unknown"),
        "trained_at": meta.get("trained_at", "unknown"),
        "classes": meta.get("classes", []),
    })

if active_tag == "NONE":
    st.error("No model available. Train a model or connect an API.")
    st.stop()

# ---------------- Main UI ----------------
st.title("📰 Fake News Detector")
st.markdown("### Single prediction")
text = st.text_area("Paste a headline/article:", height=160, placeholder="Type here...")

cols = st.columns([1, 1.2, 3])
with cols[0]:
    do_pred = st.button("Predict", type="primary")
with cols[1]:
    do_explain = st.button("Explain prediction", help="Available for local TF-IDF (token highlights + charts)")

if do_pred:
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        pred = predict_fn([text])[0]
        st.success(f"Prediction: **{pred}**")

        P = proba_fn([text]) if proba_fn else None
        if P is not None:
            probs = P[0]
            classes = meta.get("classes", [])
            st.write("**Probabilities:**")
            for c, p in sorted(zip(classes, probs), key=lambda x: -x[1]):
                st.progress(float(p), text=f"{c}: {p:.3f}")
            st.dataframe(
                pd.DataFrame({"class": classes, "probability": probs})
                  .sort_values("probability", ascending=False)
                  .reset_index(drop=True),
                use_container_width=True
            )

# ---- Explain (ONLY local TF-IDF so we can access vectorizer for attribution)
if do_explain:
    if active_tag != "TFIDF" or backend == "Remote API":
        st.info("Explanation with token highlighting works for the **local TF-IDF** model (switch backend in the sidebar).")
    elif not text.strip():
        st.warning("Please enter text first.")
    else:
        try:
            with st.spinner("Computing explanation..."):
                top_pos, top_neg, pos_label, neg_label = explain_tfidf_example_from_disk(MODEL_TFIDF, text, k=8)

                c1, c2 = st.columns(2)
                with c1:
                    fig_pos, png_pos = make_bar_fig(top_pos, f"Top + features → {pos_label}")
                    st.pyplot(fig_pos, use_container_width=True)
                    st.download_button(
                        "⬇️ Download (+) chart (PNG)",
                        data=png_pos,
                        file_name="explain_positive.png",
                        mime="image/png",
                        use_container_width=True
                    )
                with c2:
                    fig_neg, png_neg = make_bar_fig(top_neg, f"Top − features → {neg_label}")
                    st.pyplot(fig_neg, use_container_width=True)
                    st.download_button(
                        "⬇️ Download (−) chart (PNG)",
                        data=png_neg,
                        file_name="explain_negative.png",
                        mime="image/png",
                        use_container_width=True
                    )

                st.markdown("#### Highlighted text")
                parts = tfidf_token_importance(MODEL_TFIDF, text)
                render_highlighted_text(parts, pos_label, neg_label)

        except Exception as e:
            st.error(f"Could not explain TF-IDF prediction: {e}")

st.divider()

# ---------------- Batch CSV ----------------
st.markdown("### 📥 Batch classify a CSV")
st.caption("Upload a CSV that contains a **`text`** column to get predictions (and probabilities).")
st.code("text\nThis is a news headline", language="text")
upl = st.file_uploader("Upload CSV", type=["csv"])

if upl is not None:
    try:
        df_in = pd.read_csv(upl)
        if "text" not in df_in.columns:
            st.error("CSV must have a 'text' column.")
        else:
            texts = df_in["text"].astype(str).tolist()
            preds = predict_fn(texts)
            df_out = df_in.copy()
            df_out["prediction"] = preds

            P = proba_fn(texts) if proba_fn else None
            if P is not None:
                classes = meta.get("classes", [])
                for i, c in enumerate(classes):
                    df_out[f"proba_{c}"] = P[:, i]

            st.success(f"Predicted {len(df_out)} rows.")
            st.dataframe(df_out.head(20), use_container_width=True)

            csv_bytes = df_out.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download predictions CSV",
                data=csv_bytes,
                file_name="predictions.csv",
                mime="text/csv"
            )
    except Exception as e:
        st.error(f"Failed to process CSV: {e}")

st.divider()

# ---------------- Confusion matrix preview (local only) ----------------
if backend == "Local (model files)" and os.path.exists(CM_PATH):
    st.image(CM_PATH, caption="Confusion Matrix")

# ---------------- Footer ----------------
st.caption(
    f"Model version: {meta.get('version', 'unknown')} • "
    f"Trained: {meta.get('trained_at', 'unknown')} • "
    f"Active: {active_tag}"
)
