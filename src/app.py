import streamlit as st
import joblib
from preprocessing import clean_text

model = joblib.load("models/model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

st.title("Flipkart Review Sentiment Analyzer")

review = st.text_area("Enter a product review")

if st.button("Analyze Sentiment"):
    if review.strip() == "":
        st.warning("Please enter a review.")
    else:
        cleaned = clean_text(review)
        vec = vectorizer.transform([cleaned])
        pred = model.predict(vec)[0]

        if pred == 1:
            st.success("Positive Review 😊")
        else:
            st.error("Negative Review 😠")
