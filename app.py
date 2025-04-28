# import libraries
import streamlit as st
import pickle
import re

# Load model and vectorizer
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Preprocessing
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.strip()
    return text

# Streamlit page config
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="wide", initial_sidebar_state="auto")

# Load external CSS
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Initialize session state
if "news" not in st.session_state:
    st.session_state["news"] = ""

# Title
st.markdown("<h2>📰 Fake News Detection</h2>", unsafe_allow_html=True)

# Text input
news = st.text_area("✏️ Enter news text below:", value=st.session_state["news"], key="news")

# Functions
def clear_news():
    st.session_state["news"] = ""

# Buttons
col1, col2 = st.columns(2)

with col1:
    if st.button("🔍 Predict"):
        if st.session_state["news"].strip() == "":
            st.warning("Please enter some news text!")
        else:
            input_clean = preprocess_text(st.session_state["news"])
            input_vect = vectorizer.transform([input_clean])

            # Prediction
            prediction_proba = model.predict_proba(input_vect)[0]
            prediction = prediction_proba.argmax()
            confidence = prediction_proba[prediction]

            # Show Result
            if prediction == 1:
                st.success(f"✅ This is Real News. (Confidence: {confidence*100:.2f}%)")
            else:
                st.error(f"❌ This is Fake News. (Confidence: {confidence*100:.2f}%)")

            # Explainable Words
            feature_names = vectorizer.get_feature_names_out()
            coefficients = model.coef_[0]
            nonzero_indices = input_vect.nonzero()[1]

            word_importance = []
            for idx in nonzero_indices:
                word = feature_names[idx]
                weight = coefficients[idx]
                word_importance.append((word, weight))

            word_importance = sorted(word_importance, key=lambda x: abs(x[1]), reverse=True)
            top_words = [f"{word} ({'+' if weight > 0 else ''}{weight:.2f})" for word, weight in word_importance[:5]]

            st.info(f"🔍 Important words influencing decision: {', '.join(top_words)}")

with col2:
    st.button("🔄 Enter New News", on_click=clear_news)


