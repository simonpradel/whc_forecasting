import pandas as pd

# Function to merge forecast_weighted and reconciliation_dct based on key grouping variables
def combine_weightedForecast_reconciliation(forecast_weighted, forecast_equal_weights, reconciliation_dct):
    """
    Merges DataFrames from forecast_weighted, forecast_equal_weights, and reconciliation_dct 
    based on shared keys.

    :param forecast_weighted: Dictionary containing weighted forecasts as DataFrames.
    :param forecast_equal_weights: Dictionary containing equally weighted forecasts as DataFrames.
    :param reconciliation_dct: Dictionary containing actual data (actuals) as DataFrames.
    
    :return: Updated forecast_weighted dictionary with merged data.
    """
    for key in forecast_weighted.keys():
        if key in reconciliation_dct:
            # Extract forecast and actuals data
            forecast_df = forecast_weighted[key]
            forecast_df_equal = forecast_equal_weights.get(key)  # Ensure the key exists
            actuals_df = reconciliation_dct[key]

            # Grouping variables include the key and the 'date' column
            group_by_cols = list(key) + ['date']

            # Merge forecast_equal_weights if available
            if forecast_df_equal is not None:
                forecast_df = pd.merge(
                    forecast_df,
                    forecast_df_equal[['equal_weights_pred'] + group_by_cols],
                    on=group_by_cols,
                    how='outer'
                )
                # Merge actuals and forecasts based on grouping variables using an outer join
                merged_df = pd.merge(
                    actuals_df, 
                    forecast_df[['weighted_pred', 'equal_weights_pred'] + group_by_cols],
                    on=group_by_cols,
                    how='outer'
                )
            else:
                # Merge actuals and forecasts based on grouping variables using an outer join
                merged_df = pd.merge(
                    actuals_df, 
                    forecast_df[['weighted_pred'] + group_by_cols],
                    on=group_by_cols,
                    how='outer'
                )

            forecast_weighted[key] = merged_df

    return forecast_weighted


def merge_forecast_with_reconciliation(weights_forecast_dict, reconciliation_dict):
    """
    Iteratively merges the 'forecast_weighted' entries in 'weights_forecast_dict' with their 
    corresponding entries in 'reconciliation_dict' based on key grouping variables.

    :param weights_forecast_dict: Dictionary containing different models and their weighted forecasts.
                                  Each model entry contains a nested dictionary with keys such as
                                  'forecast_key', 'forecast_weighted', and 'forecast_equal_weights'.
    :param reconciliation_dict: Dictionary containing actual data (actuals) as DataFrames.
    
    :return: The updated 'weights_forecast_dict' dictionary with merged results under 'combined_results'.
    """
    final_dict = weights_forecast_dict.copy()
    # Iterate over all entries in weights_forecast_dict
    for model_key, model_data in final_dict.items():
        # Extract the forecast_key to locate the corresponding entry in reconciliation_dict
        forecast_key = model_data.get('forecast_key')

        # Check if the forecast_key exists in reconciliation_dict
        if forecast_key and forecast_key in reconciliation_dict:
            # Extract data from forecast_weighted and reconciliation_dict
            forecast_weighted = model_data.get('forecast_weighted')
            forecast_equal_weights = model_data.get('forecast_equal_weights')
            reconciliation_dct = reconciliation_dict[forecast_key]

            # Ensure forecast_weighted is present
            if forecast_weighted:
                # Use combine_weightedForecast_reconciliation to merge data
                combined_results = combine_weightedForecast_reconciliation(forecast_weighted, forecast_equal_weights, reconciliation_dct)
                # Store the merged results under 'combined_results'
                model_data['combined_results'] = combined_results
                model_data["reconciliation_dct"] = reconciliation_dct

    return final_dict
