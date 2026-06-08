# Load dataset
# Preprocess data
# Split train/test
# Build model
# Train model
# Save model
# Save tokenizer

import pandas as pd
import joblib

from src.models.lstm_model import build_model
from src.data_processing.preprocess import preprocess_text

df = pd.read_csv("data/raw/reviews.csv")

X, tokenizer = preprocess_text(df["review"])

y = df["sentiment"]

model = build_model()

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(X,y,epochs=10)

model.save("models/sentiment_model.h5")

joblib.dump(
    tokenizer,
    "models/tokenizer.pkl"
)