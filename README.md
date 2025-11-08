## 📸 Screenshots
<img width="1896" height="919" alt="image" src="https://github.com/user-attachments/assets/76707b80-e06e-4194-886c-4ac2cc1e2a52" />
<img width="1909" height="919" alt="image" src="https://github.com/user-attachments/assets/3c9c735d-f4cb-4018-8ec5-65c02d54d9bc" />


### 🖥️ Streamlit UI
| Home | Prediction + Explain |
|------|---------------------|
| ![UI Home](./assets/ui_home.png) | ![Explain](./assets/ui_explain.png) |

> *The Streamlit app allows both local + remote (API) inference, CSV batch classification, confidence bars, and TF-IDF explainability.*

---

### 🔌 FastAPI – Interactive Docs
![API Docs](./assets/api_docs.png)

> *Interactive Swagger UI available at `/docs` to test endpoints easily.*

---

### ⚙️ Project Structure
fake-news-detection/
├── api.py
├── config.yaml
├── streamlit_app.py
├── requirements.txt
├── src/
│ ├── train.py
│ ├── train_sbert.py
│ ├── infer.py
│ ├── utils.py
│ └── ...
├── reports/
│ ├── confusion_matrix.png
│ ├── model_meta.json
│ ├── metrics.txt
│ └── sbert_meta.json
├── models/
│ ├── fake_news_pipeline.joblib
│ └── fake_news_sbert.joblib
├── data/
│ ├── train.csv
│ └── ...
└── README.md


---

## ✅ 2) Instructions to Add Screenshots

### Create folder:
<img width="1491" height="885" alt="image" src="https://github.com/user-attachments/assets/42065059-42c2-4f1f-ae1d-1b7c9ccd1de5" />

✔ `assets/ui_home.png`  
✔ `assets/ui_explain.png`  
✔ `assets/api_docs.png`  

> You can rename file names — just update README paths accordingly.

---

## ✅ 3) Example API Usage Section (optional addition)

```markdown
## 🔌 API Usage

### Health check
```bash
curl http://127.0.0.1:8000/health

