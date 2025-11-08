import json, time
from pathlib import Path
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression  # or LinearSVC
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib
import numpy as np

def main(
    data="data/train.csv",
    model_out="models/fake_news_sbert.joblib",
    meta_out="reports/sbert_meta.json",
    test_size=0.2,
    seed=42
):
    df = pd.read_csv(data)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must have 'text' and 'label' columns")

    X, y = df["text"].tolist(), df["label"].tolist()

    enc_name = "sentence-transformers/all-MiniLM-L6-v2"  # small & fast
    enc = SentenceTransformer(enc_name)
    X_emb = enc.encode(X, batch_size=64, convert_to_numpy=True, show_progress_bar=True)

    stratify = y if len(set(y)) > 1 else None
    Xtr, Xte, ytr, yte = train_test_split(X_emb, y, test_size=test_size, random_state=seed, stratify=stratify)

    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(Xtr, ytr)
    preds = clf.predict(Xte)

    acc = accuracy_score(yte, preds)
    f1  = f1_score(yte, preds, average="weighted", zero_division=0)
    rep = classification_report(yte, preds, zero_division=0)
    print(rep)

    # Save artifact: encoder name + classifier (joblib)
    artifact = {"encoder_name": enc_name, "classifier": clf, "classes": getattr(clf, "classes_", None)}
    Path(model_out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_out)

    meta = {
        "model_type": "sbert_logreg",
        "version": "v2.0.0",
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "accuracy": acc,
        "f1_weighted": f1,
        "encoder": enc_name
    }
    Path(meta_out).parent.mkdir(parents=True, exist_ok=True)
    with open(meta_out, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("Saved:", model_out)
    print("Metadata:", meta_out)
    print(f"Accuracy: {acc:.3f} | F1(w): {f1:.3f}")

if __name__ == "__main__":
    main()
