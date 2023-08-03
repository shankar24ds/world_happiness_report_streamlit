import pandas as pd

def merge_data():
    """
    Merge multiple World Happiness Report DataFrames and save the integrated DataFrame as a pickle file.

    This function reads data from multiple CSV files representing World Happiness Reports for different years.
    It renames the columns to have consistent names across all years and adds a 'year' column to each DataFrame.
    Then, it concatenates these DataFrames vertically (union all) to create an integrated DataFrame.
    The resulting DataFrame is saved as a pickle file.

    Parameters:
        None

    Returns:
        None
    """
    # List of file paths for each year's data
    file_paths = [
        "data/raw/2015.csv",
        "data/raw/2016.csv",
        "data/raw/2017.csv",
        "data/raw/2018.csv",
        "data/raw/2019.csv"
    ]

    # List of column names for renaming
    column_names = {
        "Overall rank": "rank",
        "Country or region": "country",
        "Score": "score",
        "GDP per capita": "gdp_per_capita",
        "Social support": "social_support",
        "Healthy life expectancy": "life_expectancy",
        "Freedom to make life choices": "freedom",
        "Generosity": "generosity",
        "Perceptions of corruption": "corruption",
        "Happiness.Rank": "rank",
        "Country": "country",
        "Happiness.Score": "score",
        "Economy..GDP.per.Capita.": "gdp_per_capita",
        "Family": "social_support",
        "Health..Life.Expectancy.": "life_expectancy",
        "Freedom": "freedom",
        "Generosity": "generosity",
        "Trust..Government.Corruption.": "corruption",
        "Happiness Rank": "rank",
        "Happiness Score": "score",
        "Economy (GDP per Capita)": "gdp_per_capita",
        "Family": "social_support",
        "Health (Life Expectancy)": "life_expectancy",
        "Freedom": "freedom",
        "Generosity": "generosity",
        "Trust (Government Corruption)": "corruption"
    }

    # Read data from CSV files into DataFrames and rename columns
    dfs = []
    for year, file_path in enumerate(file_paths, start=2015):
        df = pd.read_csv(file_path)
        df.rename(columns=column_names, inplace=True)
        df["year"] = year
        df = df[["year", "rank", "country", "score", "gdp_per_capita", "social_support", "life_expectancy", "freedom", "generosity", "corruption"]]
        dfs.append(df)

    # Concatenate DataFrames vertically (union all)
    df = pd.concat(dfs)

    # Save the integrated DataFrame as a pickle file
    df.to_pickle("data/processed/world_happiness_merged.pkl")