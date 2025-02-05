def encode_tags(df):
    """
    One-hot encode the 'tags' column.

    Args:
        df (pandas.DataFrame): The DataFrame containing a 'tags' column.

    Returns:
        pandas.DataFrame: Modified with encoded tag columns.
    """
    return df.join(df["tags"].str.get_dummies(sep=", ")).drop(columns=["tags"])