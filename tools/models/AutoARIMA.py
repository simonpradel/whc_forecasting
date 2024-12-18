from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA

def train_AutoARIMA_and_forecast(train_df, test_periods, freq, date_col="ds", id_col="unique_id", actuals_col="y", set_index=True, enable_ensemble=True, eval_metric="MAE", verbosity=0, time_limit=60 * 60 * 24, random_seed=123):
    """
    Trains an AutoARIMA model on the given training data, and generates forecasts as well as fitted values for the training period.
    
    Parameters:
    ----------
    train_df : pd.DataFrame
        The training dataset containing time series data. It should have at least three columns: 'ds' (date), 'y' (target value), 
        and an ID column (e.g., 'unique_id'). These columns may be renamed to fit the required input format.
    
    test_periods : int
        The number of periods (time steps) into the future to forecast.
    
    freq : str
        The frequency of the time series data. Examples include 'D' for daily, 'W' for weekly, 'M' for monthly, etc.
    
    date_col : str, default="ds"
        The name of the column in `train_df` that contains the date information. Default is "ds".
    
    actuals_col : str, default="y"
        The name of the column in `train_df` that contains the target values (the dependent variable). Default is "y".
    
    id_col : str, default="unique_id"
        The name of the column in `train_df` that contains the time series ID. Default is "unique_id".
    
    set_index : bool, default=True
        Whether to set the 'id_col' as the index of the returned DataFrames. Default is True.
    
    enable_ensemble : bool, default=True
        A flag to enable ensemble learning for the forecast model. This parameter is not used in the current model setup, but could be extended.
    
    eval_metric : str, default="MAE"
        The evaluation metric to use for assessing the model's performance. Default is "MAE" (Mean Absolute Error).
    
    verbosity : int, default=0
        The verbosity level for logging during the training process. Default is 0 (no logging).
    
    time_limit : int, default=60 * 60 * 24
        The maximum time (in seconds) allowed for the training process. Default is 24 hours.
    
    random_seed : int, default=123
        The seed for random number generation, ensuring reproducibility of results. Default is 123.

    Returns:
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        A tuple containing two DataFrames:
        - The first DataFrame contains the forecast values for the test period (out-of-sample).
        - The second DataFrame contains the fitted values for the training period (in-sample).
        Both DataFrames will have their original column names restored.
    """
    
    # Internal function to handle timeouts during training
    def handler(signum, frame):
        raise TimeoutError("Training has exceeded the time limit.")
    
    # Save original column names for later renaming
    original_columns = {'date': date_col, 'ts_id': id_col, 'target': actuals_col}
    
    # Check if 'id_col' is already an index column, if so, reset it
    if id_col in train_df.index.names:
        train_df = train_df.reset_index(level=id_col)

    # Rename columns to match the required format for StatsForecast
    train_df = train_df.rename(columns={date_col: 'ds', actuals_col: 'y', id_col: 'unique_id'})

    # Retain only the relevant columns for modeling
    train_df = train_df[['unique_id', 'ds', 'y']]
    
    # Set the season length based on the frequency
    if freq in ["M", "ME"]:
        season_length = 12  # For monthly data, season length is 12
    elif freq == "D":
        season_length = 7   # For daily data, season length is 7 (weekly seasonality)
    elif freq in ["Q", "QE", "QS"]:
        season_length = 4   # For quarterly data, season length is 4

    # Suppress warnings about 'unique_id' column
    import os
    os.environ["NIXTLA_ID_AS_COL"] = "true"

    # Initialize the AutoARIMA model with the specified season length
    fcst_AutoARIMA = StatsForecast(
        models=[AutoARIMA(season_length=season_length)],
        freq=freq,
        n_jobs=-1  # Use all available CPU cores for parallel processing
    )

    # Fit the AutoARIMA model to the training data
    fcst_AutoARIMA = fcst_AutoARIMA.fit(train_df)  # Directly use train_df here for fitting

    # Generate forecasts for both out-of-sample and in-sample periods
    forecast_result = fcst_AutoARIMA.forecast(df=train_df, h=test_periods, fitted=True).reset_index()

    # Extract the last 'test_periods' dates for out-of-sample forecast
    last_dates = forecast_result["ds"].drop_duplicates().sort_values().tail(test_periods)
    
    # Separate out-of-sample forecast and in-sample fitted values
    Y_hat_df_AutoARIMA = forecast_result[forecast_result["ds"].isin(last_dates)].rename(columns={'AutoARIMA': 'pred'})
    Y_fitted_df_AutoARIMA = forecast_result[~forecast_result["ds"].isin(last_dates)].rename(columns={'AutoARIMA': 'pred'})

    # Rename columns back to their original names
    Y_hat_df_AutoARIMA = Y_hat_df_AutoARIMA.rename(columns={'ds': original_columns['date'], 'unique_id': original_columns['ts_id'], 'y': original_columns['target']})
    Y_fitted_df_AutoARIMA = Y_fitted_df_AutoARIMA.rename(columns={'ds': original_columns['date'], 'unique_id': original_columns['ts_id'], 'y': original_columns['target']})

    # Optionally set the 'id_col' as the index for the resulting DataFrames
    if set_index:
        Y_hat_df_AutoARIMA.set_index(original_columns['ts_id'], inplace=True)
        Y_fitted_df_AutoARIMA.set_index(original_columns['ts_id'], inplace=True)

    # Return the forecasted values (out-of-sample) and fitted values (in-sample)
    return Y_hat_df_AutoARIMA, Y_fitted_df_AutoARIMA
