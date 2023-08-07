import pandas as pd
import streamlit as st

def display_before_merging():
    df_2015 = pd.read_csv("data/raw/2015.csv")
    df_2016 = pd.read_csv("data/raw/2016.csv")
    df_2017 = pd.read_csv("data/raw/2017.csv")
    df_2018 = pd.read_csv("data/raw/2018.csv")
    df_2019 = pd.read_csv("data/raw/2019.csv")
    
    st.write("2015")
    st.dataframe(df_2015.head(3))
    st.write("2016")
    st.dataframe(df_2016.head(3))
    st.write("2017")
    st.dataframe(df_2017.head(3))
    st.write("2018")
    st.dataframe(df_2018.head(3))
    st.write("2019")
    st.dataframe(df_2019.head(3))
    
def display_after_merging():
    df = pd.read_csv("data/processed/world_happiness_merged.csv")
    st.dataframe(df.sample(5))