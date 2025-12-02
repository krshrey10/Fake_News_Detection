## Fake News Detection – End‑to‑End NLP Pipeline
Production‑ready fake news classifier using TF‑IDF + Logistic Regression and SBERT (MiniLM) sentence embeddings, with explainability, FastAPI inference API, and an interactive Streamlit UI.

## 🔍 What is this project?
This repository implements an end‑to‑end misinformation detection system that classifies news headlines/text as real or fake.
It includes both a classic TF‑IDF + Logistic Regression baseline and an SBERT (MiniLM) embedding model with a linear classifier.

## Key goals:

Provide a simple but solid baseline for fake news detection.

Show how to serve NLP models via FastAPI and Streamlit.

Demonstrate explainability for TF‑IDF models (word‑level contributions).

Offer a lightweight, production‑ready structure you can extend or deploy.

## 🤔 Why this project?
Fake news detection is a great playground for:

Comparing bag‑of‑words vs. sentence embedding approaches.

Deploying realistic NLP workflows (training → evaluation → API → UI).

Exploring model explainability (why the model predicts “fake” vs “real”).

Experimenting with local vs. remote inference and basic MLOps practices.

This repo is designed as a template you can adapt to other text classification problems with minimal changes.

## 🧱 Project structure

fake-news-detection/
├── api.py # FastAPI app
├── streamlit_app.py # Streamlit UI
├── config.yaml # Training config
├── requirements.txt
├── Makefile
│
├── src/
│ ├── train.py # TF-IDF training
│ ├── train_sbert.py # SBERT + classifier
│ ├── infer.py # CLI inference
│ ├── metrics.py
│ ├── features.py
│ ├── utils.py
│ └── data.py
│
├── models/
│ ├── fake_news_pipeline.joblib
│ └── fake_news_sbert.joblib
│
├── reports/
│ ├── confusion_matrix.png
│ ├── metrics.txt
│ ├── model_meta.json
│ └── sbert_meta.json
│
├── data/
│ └── train.csv
│
└── assets/
├── ui_home.png
├── ui_explain.png
└── api_docs.png
    


✨ Features
Models

TF‑IDF + Logistic Regression baseline.

SBERT (MiniLM) sentence embeddings + Logistic Regression classifier.

Explainability

Per‑sample top positive / negative words (TF‑IDF model).

Bar‑chart visualization and highlighted tokens in the UI.

Serving

FastAPI inference API (single and batch prediction).

Streamlit UI for interactive use (text box + batch CSV upload).

Local vs. remote inference toggle in the UI.

Evaluation

F1 score, accuracy, precision, recall.

Confusion matrix plot saved under reports/.

Model metadata and versioning JSON.

Utilities

Batch CSV prediction (CLI + API).

Config‑driven training via config.yaml.
    
## 🖼 Screenshots
✅ Streamlit UI
<img src="assets/ui_home.png" width="800"/>
✅ Explain Prediction
<img src="assets/ui_explain.png" width="800"/>
✅ FastAPI – Swagger Docs
<img src="assets/api_docs.png" width="800"/>

<img width="1896" height="919" alt="image" src="https://github.com/user-attachments/assets/76707b80-e06e-4194-886c-4ac2cc1e2a52" />
<img width="1909" height="919" alt="image" src="https://github.com/user-attachments/assets/3c9c735d-f4cb-4018-8ec5-65c02d54d9bc" />
<img width="1491" height="885" alt="image" src="https://github.com/user-attachments/assets/42065059-42c2-4f1f-ae1d-1b7c9ccd1de5" />


## 🚀 Installation
git clone https://github.com/krshrey10/Fake_News_Detection.git
cd Fake_News_Detection
pip install -r requirements.txt


## 🧠 Training
✅ TF-IDF model
python -m src.train

✅ SBERT + Logistic Regression model
python -m src.train_sbert

Outputs go into /models + /reports.


## 📊 Model Results
| Model  | Vectorizer | Classifier          | F1 Score | Notes                   |
| ------ | ---------- | ------------------- | -------: | ----------------------- |
| TF-IDF | BoW        | Logistic Regression |     1.00 | Baseline                |
| SBERT  | MiniLM     | Logistic Regression |     1.00 | Better semantic capture |

✅ Confusion matrix included in /reports/confusion_matrix.png

## 💡 Explainability

The TF-IDF model supports per-sample feature contribution:

✔ Top positive + negative words
✔ Bar-chart visualization
✔ Highlights why prediction was made

Shown in the Streamlit UI → Explain prediction

## 🖥 Streamlit UI

Run locally:
streamlit run streamlit_app.py

Features:

Single text inference

Explain prediction

Batch CSV upload

Local/Remote backend toggle

Probability bars

## ⚙️ FastAPI Inference Service

Start server:uvicorn api:app --host 0.0.0.0 --port 8000

Interactive docs:

http://127.0.0.1:8000/docs

## 🔌 API Usage
✅ Health Check
curl http://127.0.0.1:8000/health

✅ Predict single
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d "{\"text\":\"Government unveils new healthcare policy\"}"

✅ Predict batch
curl -X POST "http://127.0.0.1:8000/predict_batch" \
     -H "Content-Type: application/json" \
     -d "{\"texts\":[\"headline1\", \"headline2\"]}"

## 📦 Future Work

Fine-tuning full transformer model

MLflow experiment tracking

Docker support

Threshold tuning + calibration

Simple deployment on Render

## 📄 License

MIT License

## ✍ Author

Shreya K R
🔗 GitHub: https://github.com/krshrey10





