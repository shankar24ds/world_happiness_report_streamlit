# app.py
from modules import data_merge, data_cleaning, EDA

def main():
    data_merge.merge_data()
    data_cleaning.clean_data()
    EDA.explore_data()

if __name__ == "__main__":
    main()