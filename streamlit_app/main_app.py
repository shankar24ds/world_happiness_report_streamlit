import streamlit as st
import data_merge
import data_cleaning
import EDA

def main():
    st.set_page_config(layout="wide")
    st.sidebar.title("Navigation")
    page_options = ["Project Overview", "Data Merging", "Data Cleaning", "EDA"]
    selected_page = st.sidebar.radio("Go to", page_options)

    if selected_page == "Project Overview":
        project_overview_page()
    elif selected_page == "Data Merging":
        data_merge_page()
    elif selected_page == "Data Cleaning":
        data_cleaning_page()
    elif selected_page == "EDA":
        eda_page()

def project_overview_page():
    st.title("World Happiness Report")
    text = """
    The World Happiness Report is a survey of global happiness levels published since 2012. It ranks countries based on happiness scores and factors like economic production, social support, life expectancy, freedom, absence of corruption, and generosity.

    **Some questions that were asked & answered,**

    - Which countries are happiest overall and in each factor?
    - How did rankings change between reports?
    - Did any country experience significant happiness changes?
    """
    st.write(text)
    st.write("Link to the kaggle project:(https://www.kaggle.com/datasets/unsdsn/world-happiness)")

def data_merge_page():
    st.header("Data Merging")
    st.write("Combining 5 dataframes into one.")
    st.write("")
    st.write("**Sample DataFrames before merging:**")
    data_merge.display_before_merging()
    st.write("")
    st.write("**Sample DataFrame after merging:**")
    data_merge.display_after_merging()

def data_cleaning_page():
    st.header("Data Cleaning")
    data_cleaning.display()

def eda_page():
    st.header("Exploratory Data Analysis (EDA)")
    EDA.display()

if __name__ == '__main__':
    main()