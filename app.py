# app.py

import streamlit as st
import pandas as pd
from pandas import DataFrame
import joblib

# Load the vectorizer and model (adjust filenames to yours)
cv = joblib.load("countervectorizer.joblib")

st.title("Text Classification Demo")

# User input
sentence = st.text_input("Enter a sentence:", "")

if st.button("Predict"):
    if sentence:
        # Convert the sentence to the vectorized form
        Snew = cv.transform([sentence])
        # Predict
        result = cv.predict(Snew)
        # Display result
        st.write(f"Prediction: **{result[0]}**")
    else:
        st.warning("Please enter some text.")
