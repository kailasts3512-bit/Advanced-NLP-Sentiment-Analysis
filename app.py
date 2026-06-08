from fastapi import FastAPI
from src.inference.predict import predict_sentiment

app = FastAPI()

@app.get("/")
def home():
    return {"message":"Sentiment API"}

@app.get("/health")
def health():
    return {"status":"healthy"}

@app.post("/predict")
def predict(data: dict):

    prediction = predict_sentiment(
        data["text"]
    )

    return {
        "prediction": int(prediction)
    }