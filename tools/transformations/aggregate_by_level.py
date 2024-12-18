import pandas as pd
from itertools import combinations
from hierarchicalforecast.utils import aggregate

def aggregate_by_levels(data, method='dictionary', exclude_variables=None, show_dict_infos=False):
    """
    Processes the data using the specified method and excluding certain variables.

    Parameters:
    df (pd.DataFrame): The input data.
    method (str): The method to use for processing ('aggregate' or 'prepare').
    exclude_variables (list): List of variable names to exclude from the dataset.
    show_dict_infos (bool): Whether to print additional information for each dictionary element (only for 'aggregate').

    Returns:
    dict: Processed data based on the chosen method.
    """
   
    if exclude_variables is not None:
        data["pandas_df"] = data["pandas_df"].drop(columns=exclude_variables, errors='ignore')

    if method == 'dictionary':
        return aggregate_to_dict(data, show_dict_infos=show_dict_infos)
    elif method == 'long':
        return aggregate_to_long(data)
    else:
        raise ValueError("Invalid method specified. Choose either 'dictionary' or 'long'.")


def aggregate_to_dict(data, show_dict_infos=False):
    """
    Aggregates the dataframe based on specified groupby columns and creates new time series accordingly.
    Ensures aggregation for each date, regardless of 'total' values.
    
    Parameters:
    dataframe (pd.DataFrame): Input data from function "prepare_data"
    show_dict_infos (bool): Whether to print additional information for each dictionary element,
                            including dataframe names (keys) and the number of time series.

    Returns:
    dict: Dictionary of DataFrames over all combinations of groupby columns.
    """
    # Make a copy of the input dataframe to avoid modifying it directly
    df = data["pandas_df"]
    
    # Remove 'ts_id' column if it exists
    if 'ts_id' in df.columns:
        df.drop(columns=['ts_id'], inplace=True)
    
    # Extract all column names except 'ts_id', 'date', and 'total'
    groupby_cols = df.columns[~df.columns.isin(['ts_id', 'date', 'total'])].tolist()
    groupby_cols.remove('dataset')

    # Generate all combinations of groupby columns
    groupby_combinations = []

    for r in range(1, len(groupby_cols) + 1):
        groupby_combinations.extend(combinations(groupby_cols, r))
    
    # Always include top-level aggregation
    groupby_combinations = [["dataset"]] + [["dataset"] + list(comb) for comb in groupby_combinations]

    # Create a dictionary of DataFrames for each valid combination
    result_dict = {}
    for combo in groupby_combinations:
        # Group by all columns in combo and 'date', summing 'total'
        temp_df = df.groupby(list(combo) + ['date'], as_index=False)['total'].sum()
        
        # Add ts_id based on the grouping columns
        temp_df['ts_id'] = temp_df.groupby(list(combo)).ngroup() + 1
        
        # Add the DataFrame to the result dictionary with the combo as a tuple
        result_dict[tuple(combo)] = temp_df

    return result_dict



def aggregate_to_long(data, exclude_variables=None):
    """
    Aggregates a pandas DataFrame into multiple levels of groupings and prepares it for time series analysis.

    Parameters:
    -----------
    data : dict
        A dictionary containing the input pandas DataFrame with a key 'pandas_df'.
        The DataFrame should include the following columns:
        - 'total': Numerical column representing the target variable.
        - 'date': Column containing the date values.
        - Optional additional columns to be used for groupby aggregations.
    
    exclude_variables : list, optional
        A list of column names to exclude from the DataFrame before aggregation. 
        If a column in the list does not exist, it will be ignored.

    Returns:
    --------
    dict
        A dictionary containing:
        - "groupby_combinations" : list of lists
            All generated combinations of groupby column names for aggregation.
        - "Y_df" : pandas DataFrame
            The aggregated DataFrame ready for further analysis.
        - "S_df" : pandas DataFrame
            Summary statistics or additional aggregated output returned by the `aggregate` function.
        - "tags" : list
            Tags or metadata returned by the `aggregate` function.

    Notes:
    ------
    - The function renames the 'total' column to 'y' and 'date' column to 'ds' for compatibility 
      with time series forecasting libraries.
    - It generates all combinations of the groupby columns (except 'dataset', 'ds', 'y') for 
      hierarchical aggregation.
    - The top-level aggregation (grouping only by 'dataset') is always included.
    - The function assumes the existence of an external `aggregate` function that performs the 
      actual aggregation logic.
    """
    # Extract the DataFrame from the input dictionary
    df = data["pandas_df"]

    # Make a copy of the DataFrame to avoid modifying the original data
    Y_df = df.copy()

    # Rename columns for time series compatibility
    Y_df = Y_df.rename({'total': 'y', 'date': 'ds'}, axis=1)

    # Remove excluded variables if specified
    if exclude_variables is not None:
        Y_df = Y_df.drop(columns=exclude_variables, errors='ignore')

    # Identify columns to use for groupby operations
    # Exclude 'ts_id', 'y', 'ds', and 'total' columns
    groupby_cols = Y_df.columns[~Y_df.columns.isin(['ts_id', 'y', 'ds', 'total'])].tolist()

    # Ensure 'dataset' is included
    if 'dataset' in groupby_cols:
        groupby_cols.remove('dataset')

    # Define the column order for the selected DataFrame
    selected_columns = ['dataset', 'ds', 'y'] + groupby_cols
    Y_df = Y_df[selected_columns]

    # Convert the 'ds' column to datetime format for time series processing
    Y_df['ds'] = pd.to_datetime(Y_df['ds'])

    # Update groupby_cols to exclude 'ds' and 'y'
    groupby_cols = Y_df.columns[~Y_df.columns.isin(['ds', 'y'])].tolist()
    groupby_cols = groupby_cols[1:]  # Exclude the 'dataset' column

    # Generate all possible combinations of the groupby columns for aggregation
    groupby_combinations = []
    max_comb_len = len(groupby_cols)
    for r in range(1, max_comb_len + 1):
        groupby_combinations.extend(combinations(groupby_cols, r))

    # Always include the top-level aggregation (grouping only by 'dataset')
    groupby_combinations = [['dataset']] + [['dataset'] + list(comb) for comb in groupby_combinations]

    # Perform the aggregation using the external 'aggregate' function
    # The 'aggregate' function is assumed to return aggregated DataFrames and tags
    Y_df, S_df, tags = aggregate(Y_df, groupby_combinations)

    # Reset the index of the final aggregated DataFrame
    Y_df = Y_df.reset_index()

    # Return the results as a dictionary
    return {
        "groupby_combinations": groupby_combinations,
        "Y_df": Y_df,
        "S_df": S_df,
        "tags": tags,
    }
