from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding,
    Bidirectional,
    LSTM,
    Dense,
    Dropout
)

def build_model():

    model = Sequential([
        Embedding(10000,64),

        Bidirectional(
            LSTM(64,return_sequences=True)
        ),

        Dropout(0.5),

        Bidirectional(
            LSTM(32)
        ),

        Dropout(0.3),

        Dense(24,activation="relu"),

        Dense(3,activation="softmax")
    ])

    return model