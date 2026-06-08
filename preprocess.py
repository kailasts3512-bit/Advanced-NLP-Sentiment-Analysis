from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_WORDS = 10000
MAX_LENGTH = 100

def preprocess_text(texts):

    tokenizer = Tokenizer(
        num_words=MAX_WORDS,
        oov_token="<OOV>"
    )

    tokenizer.fit_on_texts(texts)

    sequences = tokenizer.texts_to_sequences(texts)

    padded = pad_sequences(
        sequences,
        maxlen=MAX_LENGTH,
        padding="post"
    )

    return padded, tokenizer