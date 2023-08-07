import pandas as pd
import streamlit as st

def display():
    st.write("Fixed inconsistences in country names.")
    st.write("")
    st.write("Below is the cleaned dataframe")
    
    df = pd.read_csv("data/processed/df_cleaned.csv")
    st.dataframe(df.sample(10))