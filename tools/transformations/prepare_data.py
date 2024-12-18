import pandas as pd

def prepare_data(data, cutoff_date=None, fill_missing_rows=False):
    """
    Process the data based on the given conditions.

    Parameters:
    - data (dict): Dictionary containing the data from function "load_data_from_catalog".
    - cutoff_date (str or None): String in date format to apply a cutoff or None if no cutoff should be applied.
    - fill_missing_rows (bool): Whether to fill missing rows with a specific frequency.

    Returns:
    - dict: Dictionary with processed pandas DataFrame under key "pandas_df".
    """

    data = data.copy()

    # Convert Spark DataFrame to Pandas DataFrame
    if "Telefonica" in data["datasetNameGeneral"]:
        spark_df = data['original_dataset']
        pandas_df = prepare_dataframe(spark_df, data["target_column_name"]).copy()  # Copy to avoid modifying the original
    elif "test" in data["datasetNameGeneral"]:
        spark_df = data['original_dataset']
        pandas_df = prepare_dataframe(spark_df, data["target_column_name"]).copy()  # Copy to avoid modifying the original
    else:
        spark_df = data['original_dataset']
        pandas_df = spark_df.toPandas().copy()  # Ensure a deep copy is made

    # Apply cutoff if provided
    if cutoff_date is not None:
        cutoff_date = pd.to_datetime(cutoff_date)
        pandas_df['date'] = pd.to_datetime(pandas_df['date'])  # Ensure 'date' is a datetime column
        pandas_df = pandas_df[pandas_df['date'] <= cutoff_date].copy()  # Copy to ensure the filtered DataFrame is new

    # Fill NA's with zeros
    if fill_missing_rows:
        pandas_df = add_missing_rows(pandas_df, freq=data["freq"]).copy()  # Copy to ensure a new DataFrame is created
        #print(f"{len(pandas_df) - len_before} rows were added to the dataset")
        pandas_df['total'] = pandas_df['total'].fillna(0)

    # Create 'ts_id' based on grouping variables with integer values
    grouping_vars = data.get("grouping_variables", [])

    if grouping_vars:
        # Generate a unique integer ts_id for each combination of grouping variables
        pandas_df['ts_id'] = pandas_df.groupby(grouping_vars, observed =False ).ngroup() + 1
    else:
        raise ValueError("No grouping variables found in 'data' dictionary to create 'ts_id'.")

    # Store the processed DataFrame back in the data dictionary
    data["pandas_df"] = pandas_df  

    return data


def add_missing_rows(df, freq):
    """
    Adds missing time periods to a DataFrame based on the specified frequency.
    For missing periods, the 'total' column values are set to 0.

    Parameters:
    - df (pd.DataFrame): Input DataFrame containing the following columns:
        - 'ts_id' (int/str): Identifier for each time series.
        - 'date' (str/datetime): The date column representing the time periods.
        - 'total' (numeric): The values corresponding to each time period.
    - freq (str): Frequency of the time series. Examples include:
        - 'D' for daily
        - 'M' for monthly
        - 'Q' for quarterly

    Returns:
    - pd.DataFrame: A DataFrame with missing time periods added, where missing
      periods have 'total' set to 0.

    """
    
    # Ensure 'date' column is a datetime object
    df['date'] = pd.to_datetime(df['date'])

    # Sort the DataFrame by 'ts_id' and 'date'
    df_sorted = df.sort_values(by=['ts_id', 'date']).reset_index(drop=True)

    # Determine the full date range for the specified frequency
    min_date = df_sorted['date'].min()
    max_date = df_sorted['date'].max()
    date_range = pd.date_range(min_date, max_date, freq=freq)

    # Placeholder for new rows with missing dates
    updated_rows = []

    # Iterate over each time series group ('ts_id') and find missing dates
    for ts_id, df_ts in df_sorted.groupby('ts_id'):
        existing_dates = pd.to_datetime(df_ts['date'].unique())
        missing_dates = date_range.difference(existing_dates)

        for date in missing_dates:
            # Create a new row with 'total' set to 0 for the missing period
            new_row = df_ts.iloc[-1].copy()  # Copy the last row to retain structure
            new_row['date'] = date           # Replace the date with the missing period
            new_row['total'] = 0             # Set 'total' to 0
            updated_rows.append(new_row)

    # Append new rows to the original DataFrame and return the result
    updated_df = pd.concat([df_sorted, pd.DataFrame(updated_rows)], ignore_index=True)
    
    return updated_df


def prepare_dataframe(spark_df, target_column):
    """
    Converts a Spark DataFrame into a clean Pandas DataFrame suitable for time series analysis.

    Parameters:
    - spark_df (pyspark.sql.DataFrame): Input Spark DataFrame containing columns like:
        - 'Year' (str): Year in 4-digit format (e.g., '2024').
        - 'Period' (str): Month abbreviation (e.g., 'JAN', 'FEB', etc.).
        - Other categorical or grouping columns.
    - target_column (str): The name of the column containing target values to forecast.
        This column will be renamed to 'total'.

    Returns:
    - pd.DataFrame: A cleaned Pandas DataFrame with the following structure:
        - 'date' (datetime): The last day of each corresponding month.
        - 'total' (float): Target values renamed and converted to float.
        - 'ts_id' (int): Unique identifier for each time series group.
    """
    # Convert Spark DataFrame to Pandas DataFrame
    pandas_df = spark_df.toPandas()

    # Replace slashes in all fields with underscores
    pandas_df = pandas_df.replace(to_replace=r'/', value='_', regex=True)

    # Clean and convert 'Year' and 'Period' columns
    pandas_df['Year'] = pandas_df['Year'].str.extract(r'(\d{4})')
    month_map = {
        "JAN": "01", "FEB": "02", "MAR": "03", "APR": "04", "MAY": "05",
        "JUN": "06", "JUL": "07", "AUG": "08", "SEP": "09", "OCT": "10",
        "NOV": "11", "DEC": "12"
    }
    pandas_df['Period'] = pandas_df['Period'].str.strip().map(month_map)

    # Create the 'date' column as the last day of each month
    pandas_df['date'] = pd.to_datetime(pandas_df['Year'] + '-' + pandas_df['Period'])
    pandas_df['date'] = pandas_df['date'] + pd.offsets.MonthEnd(0)

    # Rename the target column to 'total' and convert to float
    pandas_df.rename(columns={target_column: 'total'}, inplace=True)

    # Drop unnecessary columns
    pandas_df = pandas_df.drop(columns=['Year', 'Period'])

    # Convert remaining columns (excluding 'date' and 'total') to categorical types
    columns_to_convert = pandas_df.columns[~pandas_df.columns.str.contains('date|total')]
    pandas_df[columns_to_convert] = pandas_df[columns_to_convert].astype('category')

    # Create a unique time series ID ('ts_id') based on grouping variables
    ts_vars = pandas_df.columns[~pandas_df.columns.str.contains('date|total|id')].tolist()
    pandas_df['ts_id'] = pandas_df.groupby(ts_vars, observed=False).ngroup()

    return pandas_df
