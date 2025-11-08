## 📰 Fake News Detection

End-to-end NLP system with TF-IDF + SBERT models, explainability, Streamlit UI, and FastAPI API

This project builds a modern Fake News Detection pipeline using both classical ML (TF-IDF + Logistic Regression) and Transformer-based embeddings (SBERT).
It features:

✅ Local + remote inference
✅ Explainability (feature attributions)
✅ Interactive web UI (Streamlit)
✅ Production-ready API (FastAPI)
✅ Batch prediction + CSV support
✅ Model metadata versioning


| Feature                             | ✅ Status |
| ----------------------------------- | :------: |
| TF-IDF model                        |     ✅    |
| SBERT embedding model (MiniLM)      |     ✅    |
| Explainability (top + / − features) |     ✅    |
| Streamlit UI                        |     ✅    |
| Remote vs Local inference switch    |     ✅    |
| FastAPI inference service           |     ✅    |
| Batch CSV processing                |     ✅    |
| Metadata & Versioning               |     ✅    |
| Model Metrics + Confusion Matrix    |     ✅    |
| UI + Swagger docs screenshots       |     ✅    |


## 📸 Screenshots
<img width="1896" height="919" alt="image" src="https://github.com/user-attachments/assets/76707b80-e06e-4194-886c-4ac2cc1e2a52" />
<img width="1909" height="919" alt="image" src="https://github.com/user-attachments/assets/3c9c735d-f4cb-4018-8ec5-65c02d54d9bc" />

fake-news-detection/
├── api.py                     # FastAPI app
├── streamlit_app.py           # Streamlit UI
├── config.yaml                # Training config
├── requirements.txt
├── src/
│   ├── train.py               # TF-IDF training
│   ├── train_sbert.py         # SBERT + classifier training
│   ├── infer.py               # CLI inference
│   ├── utils.py
│   └── ...
├── models/
│   ├── fake_news_pipeline.joblib
│   └── fake_news_sbert.joblib
├── reports/
│   ├── confusion_matrix.png
│   ├── metrics.txt
│   ├── model_meta.json
│   └── sbert_meta.json
├── data/
│   └── train.csv
└── assets/                    # screenshots

## 🔧 Installation
git clone https://github.com/<user>/Fake_News_Detection.git
cd Fake_News_Detection
pip install -r requirements.txt


## 🏋️‍♂️ Training
TF-IDF model
python -m src.train

SBERT model
python -m src.train_sbert



### 🖥️ Streamlit UI
| Home | Prediction + Explain |
|------|---------------------|
| ![UI Home](./assets/ui_home.png) | ![Explain](./assets/ui_explain.png) |

> *The Streamlit app allows both local + remote (API) inference, CSV batch classification, confidence bars, and TF-IDF explainability.*

---

### 🔌 FastAPI – Interactive Docs
![API Docs](./assets/api_docs.png)

> *Interactive Swagger UI available at `/docs` to test endpoints easily.*


<img width="1491" height="885" alt="image" src="https://github.com/user-attachments/assets/42065059-42c2-4f1f-ae1d-1b7c9ccd1de5" />

✔ `assets/ui_home.png`  
✔ `assets/ui_explain.png`  
✔ `assets/api_docs.png`  



```markdown
## 🔌 API Usage

### Health check
```bash
curl http://127.0.0.1:8000/health


