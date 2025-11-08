import sys
import joblib
import os

MODEL_PATH = os.path.join("models", "fake_news_pipeline.joblib")

def predict(text: str):
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}. Run training first.")
    model = joblib.load(MODEL_PATH)
    pred = model.predict([text])[0]
    return pred

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/infer.py \"your text here\"")
        sys.exit(1)

    text = " ".join(sys.argv[1:])
    print("Input:", text)
    result = predict(text)
    print("Prediction:", result)
