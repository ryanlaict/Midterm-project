import pandas as pd
import re
import ast

def encode_tags(df):
    """
    One-hot encode the 'tags' column.

    Args:
        df (pandas.DataFrame): The DataFrame containing a 'tags' column.

    Returns:
        pandas.DataFrame: Modified with encoded tag columns.
    """
    # Convert lists to comma-separated strings
    df["tags"] = df["tags"].apply(lambda x: ", ".join(x) if isinstance(x, list) else x)

    # Apply one-hot encoding
    return df.join(df["tags"].str.get_dummies(sep=", ")).drop(columns=["tags"])

def encode_primary_photo(df):
    """
    Converts the 'primary_photo' column into a binary (True/False) column 
    based on whether an href exists.

    Args:
        df (pandas.DataFrame): The input DataFrame containing 'primary_photo'.

    Returns:
        pandas.DataFrame: Modified DataFrame with 'has_primary_photo' column.
    """
    # Create a new binary column: True if 'href' exists, False otherwise
    df['has_primary_photo'] = df['primary_photo'].apply(lambda x: isinstance(x, dict) and 'href' in x)

    # Drop the original 'primary_photo' column
    df = df.drop(columns=['primary_photo'])

    return df

def encode_source(df):
    """
    Creates two new binary columns:
    - 'agent': 1 if 'agents' exist and are not None, 0 otherwise.
    - 'mls': 1 if 'type' exists and is not None, 0 otherwise.
    
    Then, drops the original 'source' column.
    """
    # Convert string-based dictionaries to actual dictionaries
    df['source'] = df['source'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    # Check for 'agents' and 'type' keys
    df['agent'] = df['source'].apply(lambda x: 1 if isinstance(x, dict) and x.get('agents') else 0)
    df['mls'] = df['source'].apply(lambda x: 1 if isinstance(x, dict) and x.get('type') else 0)

    # Drop the original 'source' column
    df.drop(columns=['source'], inplace=True)

    return df

def extract_city_state(df):
    """
    Extracts City and State from the 'permalink' column.

    Args:
        df (pandas.DataFrame): The input DataFrame containing 'permalink'.

    Returns:
        pandas.DataFrame: Modified DataFrame with 'city' and 'state' columns.
    """

    # Function to extract City
    def get_city(permalink):
        match = re.search(r'_(.*?)_[A-Z]{2}_', permalink)
        return match.group(1) if match else None

    # Function to extract State
    def get_state(permalink):
        match = re.search(r'_([A-Z]{2})_\d{5}_', permalink)
        return match.group(1) if match else None

    # Apply extraction functions to permalink column
    df['city'] = df['permalink'].apply(lambda x: get_city(x) if isinstance(x, str) else None)
    df['state'] = df['permalink'].apply(lambda x: get_state(x) if isinstance(x, str) else None)

    # Drop the original 'permalink' column
    df.drop(columns=['permalink'], inplace=True)
    
    return df

def categorize_cost_of_living(score):
    if score < 100:
        return 'Low'
    elif 100 <= score <= 120:
        return 'Average'
    elif 120 < score <= 150:
        return 'High'
    else:
        return 'Very High'