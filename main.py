import streamlit as st
import app  # This imports your existing model logic

st.title("Star PM: Quantitative Engine")

if st.button("Run Portfolio Optimization"):
    # This calls your logic and shows results
    st.write("Model is running...")
    # Your model's results will display here
