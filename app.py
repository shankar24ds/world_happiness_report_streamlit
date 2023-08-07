# app.py
from modules import data_merge, data_cleaning, EDA

data_merge.merge_data()
data_cleaning.clean_data()
EDA.explore_data()