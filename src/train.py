import argparse, yaml, joblib
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import json, time
from src.utils import set_seed

def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def safe_split(df, test_size, seed):
    y = df["label"]
    # decide if stratify is possible
    stratify_target = None
    if y.nunique() > 1:
        n_test = max(1, int(round(len(df) * test_size)))
        # need at least 1 sample of each class in test; and at least 2 per class in total to allow a split
        if n_test >= y.nunique() and (y.value_counts().min() >= 2):
            stratify_target = y

    return train_test_split(
        df["text"], y,
        test_size=test_size,
        random_state=seed,
        stratify=stratify_target
    )

def plot_confusion(y_true, y_pred, out_path):
    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
    labels = list(np.unique(y_true))
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, interpolation='nearest')
    ax.set_title('Confusion Matrix')
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def main(cfg_path="config.yaml"):
    cfg = load_config(cfg_path)
    set_seed(cfg["train"]["random_state"])

    data_path = Path(cfg["paths"]["data"])
    model_path = Path(cfg["paths"]["model"])
    metrics_path = Path(cfg["paths"]["metrics"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must have 'text' and 'label' columns")

    X_train, X_test, y_train, y_test = safe_split(
        df, cfg["train"]["test_size"], cfg["train"]["random_state"]
    )

    # ----- pipeline -----
    tfidf = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        max_features=50000,
        strip_accents="unicode"
    )
    clf = LogisticRegression(
        max_iter=cfg["train"]["max_iter"],
        solver="lbfgs",
        class_weight="balanced"
    )
    pipe = Pipeline([
        ("tfidf", tfidf),
        ("clf", clf)
    ])

    # ----- train -----
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    # ----- metrics -----
    acc = accuracy_score(y_test, preds)
    f1  = f1_score(y_test, preds, average="weighted", zero_division=0)
    report = classification_report(y_test, preds, zero_division=0)

    # save metrics text
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(f"accuracy: {acc:.4f}\n")
        f.write(f"f1_weighted: {f1:.4f}\n\n")
        f.write(report)

    # save confusion matrix image
    cm_path = metrics_path.parent / "confusion_matrix.png"
    plot_confusion(y_test, preds, cm_path)

    # ----- save model -----
    joblib.dump(pipe, model_path)

    # ----- NEW: save model metadata -----
    meta = {
        "model_type": "tfidf_logreg",
        "version": "v1.0.0",
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "classes": pipe.classes_.tolist(),
        "vectorizer": {
            "ngram_range": list(tfidf.ngram_range),
            "max_features": tfidf.max_features,
            "stop_words": tfidf.stop_words if isinstance(tfidf.stop_words, str) else None,
            "strip_accents": tfidf.strip_accents
        }
    }
    meta_path = metrics_path.parent / "model_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # ----- logs -----
    print("Saved model →", model_path)
    print(f"Accuracy: {acc:.3f}  |  F1(w): {f1:.3f}")
    print("Confusion matrix:", cm_path)
    print("Metadata:", meta_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    main(args.config)
