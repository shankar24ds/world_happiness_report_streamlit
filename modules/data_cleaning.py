import pandas as pd

def clean_data():
    """
    Clean the World Happiness DataFrame by handling inconsistencies in country names.

    This function reads the DataFrame from the specified input CSV file, checks for inconsistencies in country names among years,
    and performs necessary replacements to make the country names consistent.
    The resulting DataFrame with cleaned country names is then saved to a new CSV file.

    Parameters:
        None

    Returns:
        None
    """
    # Read the DataFrame from the input CSV file
    df = pd.read_pickle("data/processed/world_happiness_merged.pkl")

    # Replacements for inconsistent country names
    replacements = {
        'Hong Kong S.A.R., China': 'Hong Kong',
        'Northern Cyprus': 'North Cyprus',
        'Trinidad & Tobago': 'Trinidad and Tobago',
        'Taiwan Province of China': 'Taiwan'
    }

    # Replace inconsistent country names
    df['country'] = df['country'].replace(replacements)

    # Get the list of countries with consistent data for all years (occurs 5 times)
    country_count = df['country'].value_counts().reset_index().sort_values('index')
    country_list = set(country_count[country_count['country'] == 5]['index'])

    # Filter the DataFrame to keep only the rows with consistent country names
    df = df[df['country'].isin(country_list)]

    # Save the cleaned DataFrame to a new CSV file
    df.to_pickle("data/processed/df_cleaned.pkl")