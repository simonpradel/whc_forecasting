from tools.transformations.transform_aggregated_data import transform_dict_to_long

def transform_multiple_dict_to_long(loaded_forecasts, id_col='unique_id', date_col='date', numeric_col=None):
    """
    Transforms multiple forecast datasets into a long-format representation.

    :param loaded_forecasts: dict
        A dictionary containing forecast datasets. Each key represents a forecast label, 
        and each value is a dictionary-like object with datasets.
    :param id_col: str, optional, default='unique_id'
        The name of the column to use as the unique identifier for observations.
    :param date_col: str, optional, default='date'
        The name of the column containing date values.
    :param numeric_col: str, required
        The name of the numeric column to be transformed. 
        Must be explicitly provided; no default value exists.
    :return: dict
        A dictionary containing the transformed DataFrames in long format. 
        Each key corresponds to the original forecast label, and the values are transformed DataFrames.
    :raises ValueError:
        - If `numeric_col` is not specified.
        - If `numeric_col` does not exist in the dataset for any forecast label.
    """
    if numeric_col is None:
        raise ValueError("The numeric column must be explicitly specified.")

    transformed_data = {}

    # Iterate through the loaded forecasts dictionary
    for label, forecast_dict in loaded_forecasts.items():
        # Check if the specified numeric column exists in the dataset
        if numeric_col not in forecast_dict[('dataset',)].columns:
            raise ValueError(f"The column '{numeric_col}' does not exist in the dataset for label '{label}'.")

        # Apply the transformation function to reshape the data into long format
        mapping, transformed_df = transform_dict_to_long(
            dataframes=forecast_dict, 
            id_col=id_col, 
            date_col=date_col, 
            actuals_col=numeric_col,
            set_index=True
        )
        
        # Rename the date column to 'ds' for consistency with standard time series conventions
        transformed_df.rename(columns={date_col: 'ds'}, inplace=True)
        transformed_df.index.names = ['unique_id']
  
        # Add the transformed DataFrame to the result dictionary
        transformed_data[label] = transformed_df

    return transformed_data
