import tensorflow as tf
import joblib
import numpy as np

from tensorflow.keras.preprocessing.sequence import pad_sequences

model = tf.keras.models.load_model(
    "models/sentiment_model.h5"
)

tokenizer = joblib.load(
    "models/tokenizer.pkl"
)

def predict_sentiment(text):

    seq = tokenizer.texts_to_sequences([text])

    padded = pad_sequences(
        seq,
        maxlen=100
    )

    pred = model.predict(padded)

    return np.argmax(pred)