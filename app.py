"""
RNN IMDB Sentiment Analyzer - Streamlit App
=============================================
This app takes a movie review text as input and predicts whether the
sentiment is Positive or Negative using a trained Bidirectional LSTM model.

Backend: TensorFlow/Keras LSTM model
Frontend: Streamlit
"""

import streamlit as st
import numpy as np
import pickle
import json
import os
import re
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="🎬 Movie Sentiment Analyzer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=Source+Sans+3:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Source Sans 3', sans-serif;
    }
    h1, h2, h3 { font-family: 'Playfair Display', serif; }

    .main { background: #0d0d0d; color: #f0f0f0; }

    .hero-title {
        font-family: 'Playfair Display', serif;
        font-size: 2.8rem; font-weight: 900;
        text-align: center;
        background: linear-gradient(90deg, #F7DC6F, #E74C3C);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .hero-sub { text-align: center; color: #aaa; margin-bottom: 2rem; }

    .result-positive {
        background: linear-gradient(135deg, #1e3a2f, #2ecc7133);
        border: 2px solid #2ECC71; border-radius: 16px;
        padding: 2rem; text-align: center;
        font-family: 'Playfair Display', serif;
        font-size: 2.2rem; color: #2ECC71;
        box-shadow: 0 0 30px #2ecc7133;
    }
    .result-negative {
        background: linear-gradient(135deg, #3a1e1e, #e74c3c33);
        border: 2px solid #E74C3C; border-radius: 16px;
        padding: 2rem; text-align: center;
        font-family: 'Playfair Display', serif;
        font-size: 2.2rem; color: #E74C3C;
        box-shadow: 0 0 30px #e74c3c33;
    }
    .word-chip {
        display: inline-block; padding: 3px 10px; border-radius: 20px;
        margin: 3px; font-size: 0.85rem; font-weight: 600;
    }
    .metric-box {
        background: #1a1a1a; border-radius: 12px;
        padding: 1rem; text-align: center; margin: 0.3rem 0;
    }
    .metric-val { font-size: 1.8rem; font-weight: 700; }
    .metric-lbl { font-size: 0.8rem; color: #888; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Model & Tokenizer Loading
# ─────────────────────────────────────────────
VOCAB_SIZE = 10000
MAX_LEN    = 200
MODEL_PATH = 'models/rnn_sentiment.keras'
META_PATH  = 'models/rnn_metadata.json'
WIDX_PATH  = 'models/word_index.pkl'

@st.cache_resource
def load_resources():
    """Load model, word index, and metadata. Cached to avoid reloading."""
    model = None
    word_index = {}
    meta = {'vocab_size': VOCAB_SIZE, 'max_len': MAX_LEN,
            'final_val_accuracy': None, 'final_roc_auc': None, 'epochs_trained': None}

    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
    if os.path.exists(WIDX_PATH):
        with open(WIDX_PATH, 'rb') as f:
            word_index = pickle.load(f)
    if os.path.exists(META_PATH):
        with open(META_PATH) as f:
            meta = json.load(f)
    return model, word_index, meta

model, word_index, meta = load_resources()

# ─────────────────────────────────────────────
# Text Preprocessing (mirrors notebook)
# ─────────────────────────────────────────────
def preprocess_text(text: str, word_index: dict, vocab_size: int, max_len: int):
    """
    Convert raw review text to a padded integer sequence.
    Same pipeline as used during training (word_index from IMDB dataset).
    """
    # Lowercase and remove punctuation
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    tokens = text.split()

    # Convert words to integers; unknown words map to 2 (<UNK>)
    # IMDB word_index uses offset 3 (0=pad, 1=start, 2=unk, 3+=words)
    encoded = [min(word_index.get(w, 2) + 3, vocab_size - 1) for w in tokens]
    padded  = pad_sequences([encoded], maxlen=max_len, padding='post', truncating='post')
    return padded, tokens, encoded

def highlight_sentiment_words(tokens: list, word_index: dict):
    """
    Return lists of words associated with positive/negative sentiment
    based on crude frequency buckets (top-1000 vs top-5000 for illustration).
    """
    pos_keywords = {'great','excellent','brilliant','amazing','love','wonderful',
                    'perfect','best','fantastic','superb','enjoyed','beautiful',
                    'outstanding','incredible','masterpiece','hilarious','touching'}
    neg_keywords = {'bad','terrible','awful','boring','poor','horrible','worst',
                    'waste','disappointing','dull','stupid','annoying','hate',
                    'ridiculous','pathetic','garbage','mediocre','failed'}
    high = [t for t in tokens if t in pos_keywords]
    low  = [t for t in tokens if t in neg_keywords]
    return high, low

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎬 Model Info")
    if meta.get('final_val_accuracy'):
        st.metric("Best Val Accuracy", f"{meta['final_val_accuracy']:.2%}")
    if meta.get('final_roc_auc'):
        st.metric("ROC-AUC Score", f"{meta['final_roc_auc']:.4f}")
    if meta.get('epochs_trained'):
        st.metric("Epochs Trained", meta['epochs_trained'])
    st.metric("Vocabulary Size", f"{VOCAB_SIZE:,}")
    st.metric("Max Sequence Length", MAX_LEN)

    st.markdown("---")
    st.markdown("## ⚙️ Settings")
    threshold = st.slider("Confidence Threshold", 0.3, 0.9, 0.5, 0.05)
    show_tokens = st.checkbox("Show Keyword Highlights", value=True)

    st.markdown("---")
    st.markdown("### 📁 Dataset")
    st.markdown("**IMDB** — 50,000 labeled movie reviews (Keras built-in).")

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.markdown('<div class="hero-title">🎬 Movie Review Sentiment Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Paste a movie review below — our LSTM will detect whether it\'s Positive or Negative.</div>', unsafe_allow_html=True)

if model is None:
    st.warning("⚠️ No trained model found. Please run `notebooks/rnn_sentiment.ipynb` first.")

# ─────────────────────────────────────────────
# Sample Reviews
# ─────────────────────────────────────────────
SAMPLES = {
    "✅ Positive Example": (
        "This film is absolutely brilliant. The performances were outstanding and the "
        "storyline kept me engaged throughout. A masterpiece of modern cinema that I "
        "thoroughly enjoyed and would recommend to everyone."
    ),
    "❌ Negative Example": (
        "Terrible movie. Boring plot, horrible acting, and a complete waste of time. "
        "I couldn't wait for it to end. One of the worst films I have ever seen."
    ),
    "🤔 Mixed Example": (
        "The movie had some great visual effects but the story was confusing. "
        "The acting was decent in parts, though the ending was rather disappointing."
    ),
}

# ─────────────────────────────────────────────
# Input Area
# ─────────────────────────────────────────────
col1, col2 = st.columns([3, 2], gap="large")

with col1:
    st.markdown("### ✍️ Enter a Movie Review")

    # Quick-fill buttons
    scol1, scol2, scol3 = st.columns(3)
    sample_choice = None
    if scol1.button("✅ Positive Sample"): sample_choice = "✅ Positive Example"
    if scol2.button("❌ Negative Sample"): sample_choice = "❌ Negative Example"
    if scol3.button("🤔 Mixed Sample"):    sample_choice = "🤔 Mixed Example"

    default_text = SAMPLES[sample_choice] if sample_choice else ""
    review_text = st.text_area(
        "Paste or type a movie review:",
        value=default_text,
        height=200,
        placeholder="e.g. 'This movie was absolutely brilliant...'"
    )

    word_count = len(review_text.split()) if review_text else 0
    st.caption(f"📝 Word count: **{word_count}** (model uses first {MAX_LEN} words)")

    analyze_btn = st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True)

with col2:
    st.markdown("### 🎯 Result")

    if analyze_btn and review_text.strip() and model is not None:
        # ── Preprocess ──
        padded, tokens, encoded = preprocess_text(
            review_text, word_index, VOCAB_SIZE, MAX_LEN
        )

        # ── Predict ──
        with st.spinner("Analyzing..."):
            prob_pos  = float(model.predict(padded, verbose=0)[0][0])
            prob_neg  = 1.0 - prob_pos
            sentiment = "Positive" if prob_pos >= threshold else "Negative"
            confidence = prob_pos if sentiment == "Positive" else prob_neg

        # ── Display ──
        box_cls = "result-positive" if sentiment == "Positive" else "result-negative"
        emoji   = "😍" if sentiment == "Positive" else "😞"
        st.markdown(f'<div class="{box_cls}">{emoji} {sentiment}</div>', unsafe_allow_html=True)

        st.markdown(f"**Confidence:** `{confidence:.2%}`")
        st.progress(confidence)

        if confidence < threshold + 0.1:
            st.info("🤔 Model is uncertain — the review may contain mixed sentiment.")

        # Gauge chart
        fig, ax = plt.subplots(figsize=(5, 2.5), facecolor='none')
        ax.set_facecolor('none')
        ax.barh(['Negative', 'Positive'], [prob_neg, prob_pos],
                color=['#E74C3C', '#2ECC71'], height=0.5)
        ax.set_xlim(0, 1)
        ax.axvline(0.5, color='white', linestyle='--', alpha=0.5)
        ax.set_xlabel('Probability', color='white')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_edgecolor('#444')
        ax.annotate(f'{prob_pos:.2%}', xy=(prob_pos, 1), xytext=(5, 0),
                    textcoords='offset points', color='white', fontsize=9)
        ax.annotate(f'{prob_neg:.2%}', xy=(prob_neg, 0), xytext=(5, 0),
                    textcoords='offset points', color='white', fontsize=9)
        st.pyplot(fig, transparent=True)
        plt.close()

        # Keyword highlights
        if show_tokens:
            pos_words, neg_words = highlight_sentiment_words(tokens, word_index)
            if pos_words or neg_words:
                st.markdown("#### 🔑 Sentiment Keywords Found")
                for w in pos_words:
                    st.markdown(f'<span class="word-chip" style="background:#1e3a2f;color:#2ECC71">✅ {w}</span>', unsafe_allow_html=True)
                for w in neg_words:
                    st.markdown(f'<span class="word-chip" style="background:#3a1e1e;color:#E74C3C">❌ {w}</span>', unsafe_allow_html=True)

    elif analyze_btn and not review_text.strip():
        st.warning("Please enter a review first.")
    elif analyze_btn and model is None:
        st.error("Model not loaded. Train the model first (run the notebook).")
    else:
        st.info("👈 Enter a review and click **Analyze Sentiment**")

# ─────────────────────────────────────────────
# About
# ─────────────────────────────────────────────
st.markdown("---")
with st.expander("📖 About this Project"):
    st.markdown("""
    ### RNN IMDB Sentiment Analyzer

    **Model Architecture:**
    - Embedding layer (128-dim word vectors)
    - SpatialDropout1D (0.2)
    - Bidirectional LSTM (64 units, reads forward + backward)
    - Dense(32, ReLU) → Dense(1, Sigmoid)

    **Training Details:**
    - Dataset: IMDB (Keras) — 50,000 labeled movie reviews
    - Vocabulary: Top 10,000 words
    - Max sequence length: 200 tokens
    - Optimizer: Adam | Loss: Binary Cross-Entropy

    **Tech Stack:** Python · TensorFlow/Keras · Streamlit
    """)