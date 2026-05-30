import streamlit as st
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ---------------------------
# Load Model
# ---------------------------

model = load_model("models/imdb_lstm.keras")

# ---------------------------
# Load IMDb Word Index
# ---------------------------

word_index = imdb.get_word_index()

# IMDb uses index offset of +3
word_index = {
    word: (index + 3)
    for word, index in word_index.items()
}

word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

VOCAB_SIZE = 10000
MAX_LENGTH = 200

# ---------------------------
# Streamlit UI
# ---------------------------

st.set_page_config(
    page_title="IMDb Sentiment Analysis",
    page_icon="🎬",
    layout="centered"
)

st.title("🎬 IMDb Movie Review Sentiment Analysis")

st.write(
    "Enter a movie review and the LSTM model will predict whether the sentiment is Positive or Negative."
)

review = st.text_area(
    "Movie Review",
    height=200,
    placeholder="Example: This movie was amazing. The acting was fantastic and the story was very engaging."
)

# ---------------------------
# Prediction
# ---------------------------

if st.button("Predict Sentiment"):

    if review.strip() == "":
        st.warning("Please enter a movie review.")
    else:

        words = review.lower().split()

        sequence = []

        for word in words:

            idx = word_index.get(word, 2)

            # Keep only words inside training vocabulary
            if idx >= VOCAB_SIZE:
                idx = 2

            sequence.append(idx)

        padded = pad_sequences(
            [sequence],
            maxlen=MAX_LENGTH,
            padding="pre",
            truncating="pre"
        )

        prediction = model.predict(
            padded,
            verbose=0
        )

        score = float(prediction[0][0])

        st.subheader("Prediction Result")

        if score >= 0.5:

            st.success("😊 Positive Review")

            st.write(
                f"Confidence Score: **{score:.2%}**"
            )

        else:

            st.error("😞 Negative Review")

            st.write(
                f"Confidence Score: **{(1-score):.2%}**"
            )

        st.write("---")

        st.write("Raw Model Score:")

        st.write(round(score, 4))