import streamlit as st
import joblib
from preprocessing import clean_text

st.markdown(
    """
    <style>
    /* Main background */
    .stApp {
        background-color: #F5F7FA;
    }

    /* Title */
    h1 {
        color: #E6E6FF;
        text-align: center;
    }

    /* Text area */
    textarea {
        border-radius: 10px;
        border: 1px solid #2874F0;
    }

    /* Button */
    div.stButton > button {
        background-color: #2874F0;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1.2rem;
        font-size: 16px;
    }

    div.stButton > button:hover {
        background-color: #1f5fd1;
        color: white;
    }

    /* Result boxes */
    .positive {
        background-color: #D4EDDA;
        padding: 15px;
        border-radius: 8px;
        color: #155724;
        font-weight: bold;
    }

    .negative {
        background-color: #F8D7DA;
        padding: 15px;
        border-radius: 8px;
        color: #721C24;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.image(
    "https://logos-world.net/wp-content/uploads/2020/11/Flipkart-Logo.png",
    width=180
)


st.title("Flipkart Review Sentiment Analyzer")

st.markdown(
    "<p style='text-align:center;'>Analyze customer reviews and understand sentiment instantly</p>",
    unsafe_allow_html=True
)

model = joblib.load("models/model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")


review = st.text_area("📝 Enter a product review", height=150)

if st.button("Analyze Sentiment"):
    if review.strip() == "":
        st.warning("Please enter a review.")
    else:
        cleaned = clean_text(review)
        vec = vectorizer.transform([cleaned])
        pred = model.predict(vec)[0]

        if pred == 1:
            st.markdown(
                "<div class='positive'>✅ Positive Review</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                "<div class='negative'>❌ Negative Review</div>",
                unsafe_allow_html=True
            )


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
