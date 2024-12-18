import pandas as pd
import numpy as np


def transform_long_to_dict(
    df: pd.DataFrame, 
    mapping: dict, 
    id_col: str = 'unique_id', 
    date_col: str = 'ds', 
    actuals_col: str = 'y'
) -> dict:
    """
    Transforms a DataFrame in long format into a dictionary of DataFrames based on a provided mapping.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame in long format containing at least the column defined by `id_col`.
    mapping : dict
        A dictionary where each key specifies a concatenated set of column names (e.g., 'region/subregion'),
        and each value is a list of `id_col` values that define which rows belong to the group.
    id_col : str, optional
        Name of the column containing the unique identifiers. Default is 'unique_id'.
    date_col : str, optional
        Name of the column containing the date information. Default is 'ds'.
    actuals_col : str, optional
        Name of the column containing the dependent variable (e.g., observations). Default is 'y'.

    Returns
    -------
    dict
        A dictionary where:
        - The key is a tuple of column names (e.g., ('region', 'subregion')).
        - The value is a filtered DataFrame with corresponding rows and derived columns.
    """
    dataframes = {}

    for key, values in mapping.items():
        # Split the key into individual column names
        columns = key.split('/')

        # Filter the DataFrame to include only rows with IDs in 'values'
        filtered_df = df[df[id_col].isin(values)].copy()

        # Only proceed if there are valid rows in the filtered DataFrame
        if not filtered_df.empty:
            df_split = filtered_df.copy()

            # Add derived columns based on the split 'id_col' values
            for col in columns:
                col_index = columns.index(col)
                df_split[col] = df_split[id_col].apply(
                    lambda x: x.split('/')[col_index] if len(x.split('/')) > col_index else None
                )

            # Drop the original 'id_col' column
            df_split.drop(columns=id_col, inplace=True)

            # Use a tuple of column names as the dictionary key
            key_tuple = tuple(columns)

            # Add the resulting DataFrame to the output dictionary
            dataframes[key_tuple] = df_split

    return dataframes


def transform_dict_to_long(
    dataframes: dict, 
    id_col: str = 'unique_id', 
    date_col: str = 'ds', 
    actuals_col: str = 'y', 
    set_index: bool = False, 
    include_all_columns: bool = False
) -> tuple:
    """
    Transforms a dictionary of DataFrames back into a single long-format DataFrame.

    Parameters
    ----------
    dataframes : dict
        A dictionary where:
        - The key is a tuple of column names (e.g., ('region', 'subregion')).
        - The value is a DataFrame containing:
          - Columns defined by the key (e.g., 'region', 'subregion').
          - `date_col`: The column with date information.
          - `actuals_col`: The column with observations.
    id_col : str, optional
        Name of the column for unique identifiers, created by combining key columns. Default is 'unique_id'.
    date_col : str, optional
        Name of the column containing date information. Default is 'ds'.
    actuals_col : str, optional
        Name of the column containing the dependent variable (e.g., observations). Default is 'y'.
    set_index : bool, optional
        If True, sets the `id_col` as the DataFrame index. Default is False.
    include_all_columns : bool, optional
        If True, includes all columns from the input DataFrames in the final long-format DataFrame. Default is False.

    Returns
    -------
    tuple
        - mapping : dict
            A dictionary where the key is a string representation of the column combination (e.g., 'region/subregion'),
            and the value is a NumPy array of unique `id_col` values derived from the combined key columns.
        - combined_long_format_df : pd.DataFrame
            A DataFrame in long format containing:
            - `id_col`: Combined ID created from the key columns.
            - `date_col`: Date information.
            - `actuals_col`: Observations or dependent variable.
            - Other columns if `include_all_columns` is True.
    """
    mapping = {}
    long_format_list = []

    # Iterate through the dictionary of DataFrames
    for key, df in dataframes.items():
        df = pd.DataFrame(df)  # Ensure the input is a DataFrame

        # Extract the column names from the key
        columns = key if isinstance(key, tuple) else (key,)

        # Create a mapping key by joining the column names
        mapping_key = '/'.join(columns)

        # Combine key columns to generate unique ID values
        if all(col in df.columns for col in columns):
            combined_values = df[list(columns)].astype(str).agg('/'.join, axis=1).unique()
            mapping[mapping_key] = np.array(combined_values)

        # Build the long-format DataFrame
        df_long = pd.DataFrame()
        if all(col in df.columns for col in [date_col, actuals_col]):
            df_long[id_col] = df[list(columns)].astype(str).agg('/'.join, axis=1)
            df_long[date_col] = df[date_col]
            df_long[actuals_col] = df[actuals_col]

            # Include other columns if requested
            if include_all_columns:
                other_columns = df.columns.difference([id_col, date_col, actuals_col])
                for col in other_columns:
                    df_long[col] = df[col].values

        # Append to the list of long-format DataFrames
        long_format_list.append(df_long)

    # Combine all long-format DataFrames into one
    combined_long_format_df = pd.concat(long_format_list, ignore_index=True)

    # Set the index if requested
    if set_index:
        combined_long_format_df.set_index(id_col, inplace=True, drop=True)

    return mapping, combined_long_format_df

