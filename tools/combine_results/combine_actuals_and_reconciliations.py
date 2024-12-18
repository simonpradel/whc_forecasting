import pandas as pd
from tools.transformations.transform_aggregated_data import transform_long_to_dict

def combine_actuals_and_reconciliations(aggregated_data, Y_rec_df):
    """
    Combines actuals and reconciliations data by performing a join operation for each key in Y_rec_df 
    and then applying a transformation to the merged data.

    Args:
        aggregated_data (dict): 
            A dictionary containing aggregated data. 
            Expected keys include:
            - 'Y_df': A DataFrame with actuals data, where columns include 'unique_id', 'ds', and 'y'.
            - 'tags': A mapping dictionary used for transforming the merged data.
        Y_rec_df (dict): 
            A dictionary containing DataFrames to be merged with the actuals data. 
            Each key corresponds to a forecast identifier, and each value is a DataFrame with:
            - 'unique_id': Identifier for time series groups.
            - 'ds': Date column, which will be renamed to 'date'.
            - Other prediction-related columns (e.g., 'pred', 'pred/base', or similar).

    Returns:
        dict: A dictionary where each key corresponds to the forecast identifier, and the value is the transformed 
        result (as a dictionary) for the corresponding merged DataFrame.
    """
    
    result_dict = {}

    # Iterate through all keys in Y_rec_df
    for forecast_key, forecast_df in Y_rec_df.items():
        # Rename the 'ds' column to 'date'
        forecast_df = forecast_df.rename(columns={'ds': 'date'})
        
        # Also rename the corresponding column in aggregated_data['Y_df'], if necessary
        Y_df_renamed = aggregated_data['Y_df'].rename(columns={'ds': 'date'})
        
        # Adjust column names: Remove "pred/" at the beginning and rename "pred" to "base"
        forecast_df.columns = [
            col.replace('pred/', '') if col.startswith('pred/') else ('base' if col == 'pred' else col)
            for col in forecast_df.columns
        ]
        
        # Perform the join operation
        merged_df = pd.merge(
            Y_df_renamed, 
            forecast_df, 
            on=['unique_id', 'date'], 
            how='outer'
        )
        
        # Transform the merged data into a dictionary
        dict_df = transform_long_to_dict(
            df=merged_df, 
            mapping=aggregated_data['tags'], 
            id_col='unique_id', 
            date_col='date', 
            actuals_col='y'
        )
        
        # Store the result for this key
        result_dict[forecast_key] = dict_df

    return result_dict
