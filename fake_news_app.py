import streamlit as st
import joblib

# Load saved Naive Bayes model + vectorizer
model = joblib.load("fake_news_model_nb.joblib")
vectorizer = joblib.load("tfidf_vectorizer_nb.joblib")

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(to right, #83a4d4, #b6fbff);
        background-attachment: fixed;
        background-size: cover;
    }
    textarea {
        background-color: rgba(255, 255, 255, 0.9) !important;
        color: #000000 !important;
        border-radius: 8px;
        padding: 10px;
    }
    div.stButton > button {
        background-color: #007acc;
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
    }
    div.stButton > button:hover {
        background-color: #005f99;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.title("📰 Fake News Detection System ")
st.markdown("Paste the **title + article text** below:")
article = st.text_area("",height=150)

if st.button("Predict"):
    if article.strip():
        vec = vectorizer.transform([article])
        prediction = model.predict(vec)[0]
        confidence = model.predict_proba(vec).max()
        st.write(f"### Prediction: `{prediction}`")
        st.write(f"Confidence: **{round(confidence * 100, 2)}%**")
    else:
        st.warning("⚠ Please paste your article text.")


st.markdown(
    """
    <hr style="margin-top: 50px; margin-bottom: 10px;">
    <div style="text-align: center; color: black;">
       - Developed by <strong> Riya Yadav</strong>
    </div>
    """,
    unsafe_allow_html=True
)

