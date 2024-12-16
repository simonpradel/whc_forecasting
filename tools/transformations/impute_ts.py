import pandas as pd

def add_missing_rows(dataframes_dict):
    """
    Adds missing monthly rows to DataFrames in a dictionary.

    This function processes a dictionary of DataFrames, each containing time series data with 'ts_id' and 'date' columns. 
    It ensures that all time series ('ts_id') cover a continuous range of monthly periods from the earliest 
    to the latest date across all DataFrames. For missing months, a new row is created with the 'total' column 
    set to 0, and other columns populated based on the most recent data of the corresponding 'ts_id'.

    Parameters:
    dataframes_dict (dict): A dictionary where keys are DataFrame names and values are DataFrames. 
                            Each DataFrame must include 'ts_id', 'date', and 'total' columns.

    Returns:
    dict: A dictionary of updated DataFrames with missing monthly rows added.
    """
    updated_dataframes = {}
    
    # Combine all data from the input DataFrames into a single DataFrame
    all_data = pd.concat(dataframes_dict.values(), ignore_index=True)
    
    # Determine the minimum and maximum date across all DataFrames
    min_date = all_data['date'].min()
    max_date = all_data['date'].max()
    date_range = pd.date_range(min_date, max_date, freq='M') 
    
    for df_name, df in dataframes_dict.items():
        # Sort the DataFrame by 'ts_id' and 'date' for structured processing
        df_sorted = df.sort_values(by=['ts_id', 'date']).reset_index(drop=True)
        
        # Create a collection for the new rows to be added
        updated_rows = []
        
        # Iterate over each time series ID ('ts_id') to find and fill missing months
        for ts_id, df_ts in df_sorted.groupby('ts_id'):
            # Identify the dates already present for the current 'ts_id'
            existing_dates = df_ts['date'].unique()
            existing_dates = pd.to_datetime(existing_dates)  # Ensure dates are DateTime objects
            missing_dates = date_range.difference(existing_dates)
            
            for date in missing_dates:
                # Create a new row for each missing month
                new_row = df_ts.iloc[-1].copy()  # Copy the last available row for this 'ts_id'
                new_row['date'] = date  # Set the date to the missing month's end
                new_row['total'] = 0  # Initialize the 'total' value to 0
                
                # Add the new row to the collection
                updated_rows.append(new_row)
        
        # Append the updated rows to the original DataFrame
        updated_df = pd.concat([df_sorted, pd.DataFrame(updated_rows)], ignore_index=True)
        
        # Add the updated DataFrame to the result dictionary
        updated_dataframes[df_name] = updated_df
    
    return updated_dataframes


    