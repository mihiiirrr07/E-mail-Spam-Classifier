
import streamlit as st
import joblib
import re

# Load model and vectorizer
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

def clean_text(text):
    text = re.sub(r"\W", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.lower()

# Streamlit UI
st.title("📧 Email Spam Classifier")
st.write("Enter an email message below to check if it's spam or not.")

email_text = st.text_area("Email content")

if st.button("Classify"):
    cleaned = clean_text(email_text)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)[0]
    label = "🛑 Spam" if prediction == 0 else "✅ Not Spam"
    st.subheader(f"Prediction: {label}")
