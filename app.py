# app.py
import streamlit as st
import pickle

# Load model + vectorizer
with open("LogReg_Sentiment_Model.pkl", "rb") as f:
    model = pickle.load(f)

with open("Tfidf_Vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

label_map = {-1:"Negative", 0:"Neutral", 1:"Positive"}

# UI
st.title("Sentiment Analysis App")
st.write("Enter a sentence to analyze its sentiment")

user_input = st.text_area("✍️ Enter your text:")

if st.button("Analyze"):
    if user_input.strip() != "":
        clean = user_input.lower()
        prep = vectorizer.transform([clean])
        pred = model.predict(prep)[0]
        st.write("🔎 Prediction:", label_map[int(pred)])
    else:
        st.write("⚠️ Please enter some text first.")
