import streamlit as st
from src.predict import classify_tfidf
from src.utils import tfidf_lr_model


st.title("Emotion Analyzer App for Goodreads reviews")
title = st.text_input("Enter the book title", " ")

if st.button("Submit"):
    tfidf_model = tfidf_lr_model()
    book, result = classify_tfidf(title, tfidf_model)
    final_result = {"Book Title": book, "Predictions": result}
    st.write("Analyzed output:", final_result)
